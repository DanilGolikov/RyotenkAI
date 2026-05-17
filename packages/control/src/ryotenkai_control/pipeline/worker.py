"""Internal pipeline worker — spawned by CLI ``run start/resume/restart`` and Web API.

Launched as ``python -m src.pipeline.worker --run-dir R [--config C]
[--resume] [--restart-from-stage S]``. Reads project context from
environment variables (``RYOTENKAI_PROJECT_ID``, ``RYOTENKAI_ACTOR``,
``RYOTENKAI_CONFIG_VERSION_HASH``, ``RYOTENKAI_RUNS_BASE_DIR``) — the
launcher (CLI / API) sets these before spawn.

NOT a Typer subcommand — does not import the CLI registry. Keeps cold
start fast and the public CLI surface clean (users see only
``ryotenkai run start`` / ``run resume`` / ``run restart``).

Mirrors :mod:`src.training.run_training` (the worker spawned on remote
training pods) — same dedicated-module-entry pattern, but for the
local-launch path that the orchestrator runs in.

Error rendering: subprocess output is what users see at the terminal
(parent CLI streams worker stderr through). Worker therefore renders
typed :class:`RyotenkAIError` failures via
:func:`ryotenkai_control.cli.errors.print_ryotenkai_error` (the non-
raising twin of ``die_from_ryotenkai``), giving uniform kubectl-style
output identical to direct CLI commands. ``KeyboardInterrupt`` produces
``aborted`` + exit 130 (POSIX SIGINT convention). All other unhandled
exceptions render a short ``error: internal error`` line plus a
``trace=`` correlation id and a hint to check server logs — full
traceback only goes to the log file, never to the user terminal.

Phase H1 (Error Persistence, 2026-05-17)
---------------------------------------
``_log_pipeline_outcome`` writes a kubectl/Terraform-style summary
block to ``pipeline.log`` as the LAST entry of every run (success or
failure). ``tail pipeline.log`` shows outcome immediately without
scrolling 1000 lines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import TYPE_CHECKING, Any
from pathlib import Path

if TYPE_CHECKING:
    from ryotenkai_shared.errors import RyotenkAIError


# ---------------------------------------------------------------------------
# Phase H1 — Final outcome summary in pipeline.log
# ---------------------------------------------------------------------------

_SEPARATOR_WIDTH = 80
_SEPARATOR = "=" * _SEPARATOR_WIDTH


def _format_attempt_summary(
    exc: Exception | None,
    *,
    attempt_no: int | None,
    stage_name: str | None,
    stage_idx: int | None,
    stage_total: int | None,
    trace_id: str | None,
    request_id: str | None,
    detail: str | None,
    context: dict[str, Any] | None,
    duration_seconds: float | None,
    code: str | None,
    title: str | None,
    total_stages: int | None = None,
) -> str:
    """Build the kubectl/Terraform-style summary block.

    Returns a multi-line string ready to be passed to
    ``logger.error`` / ``logger.info``. Lines for absent fields are
    omitted so the block stays compact for short runs.
    """
    lines: list[str] = [_SEPARATOR]
    if exc is None:
        # Success path.
        attempt_part = (
            f" at attempt {attempt_no}" if attempt_no is not None else ""
        )
        wall_part = (
            f" (wall={duration_seconds:.1f}s"
            if duration_seconds is not None
            else " ("
        )
        stages_part = (
            f", stages={total_stages}/{total_stages})"
            if total_stages is not None and duration_seconds is not None
            else (")" if duration_seconds is not None else "")
        )
        # Special-case for partial info.
        if duration_seconds is None and total_stages is None:
            header = f"Pipeline COMPLETED{attempt_part}"
        else:
            header = f"Pipeline COMPLETED{attempt_part}{wall_part}{stages_part}"
        lines.append(header)
        lines.append(_SEPARATOR)
        return "\n".join(lines)

    # Failure path.
    attempt_part = (
        f" at attempt {attempt_no}" if attempt_no is not None else ""
    )
    lines.append(f"Pipeline FAILED{attempt_part}")
    lines.append(_SEPARATOR)
    if title:
        lines.append(f"  error:    {title}")
    if code:
        lines.append(f"  code:     {code}")
    if stage_name:
        if stage_idx is not None and stage_total is not None:
            lines.append(
                f"  stage:    {stage_name} (Stage {stage_idx + 1}/{stage_total})"
            )
        else:
            lines.append(f"  stage:    {stage_name}")
    lines.append(f"  trace:    {trace_id or '-'}")
    if request_id:
        lines.append(f"  request:  {request_id}")
    if detail:
        lines.append(f"  detail:   {detail}")
    if context:
        # Strip internal/private keys (leading underscore) from the
        # rendered block to keep it user-facing; the typed dataclass
        # (H2) preserves everything for tooling.
        clean = {k: v for k, v in context.items() if not str(k).startswith("_")}
        if clean:
            try:
                rendered = json.dumps(clean, default=str, sort_keys=True)
            except Exception:  # noqa: BLE001 — best-effort
                rendered = str(clean)
            lines.append(f"  context:  {rendered}")
    if duration_seconds is not None:
        lines.append(f"  duration: {duration_seconds:.1f}s")
    if attempt_no is not None:
        lines.append(f"  attempts: {attempt_no}")
    lines.append("  outcome:  FAILED")
    lines.append(_SEPARATOR)
    return "\n".join(lines)


def _log_pipeline_outcome(
    exc: Exception | None,
    *,
    attempt_no: int | None = None,
    stage_name: str | None = None,
    stage_idx: int | None = None,
    stage_total: int | None = None,
    trace_id: str | None = None,
    request_id: str | None = None,
    detail: str | None = None,
    context: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
    code: str | None = None,
    title: str | None = None,
    total_stages: int | None = None,
) -> None:
    """Phase H1 — emit a Terraform-style outcome block to ``pipeline.log``.

    Uses the ryotenkai logger so the block flows through the file
    handler attached by :func:`init_run_logging`. When no file handler
    is attached (startup-time failure, very early in main()) the block
    is only emitted to stderr.

    Always emits at ``error`` level for failure (red in coloured TTYs)
    and ``info`` for success. The choice survives the file handler too:
    pipeline.log records the level prefix for grep filtering.
    """
    from ryotenkai_shared.utils.logger import logger

    text = _format_attempt_summary(
        exc,
        attempt_no=attempt_no,
        stage_name=stage_name,
        stage_idx=stage_idx,
        stage_total=stage_total,
        trace_id=trace_id,
        request_id=request_id,
        detail=detail,
        context=context,
        duration_seconds=duration_seconds,
        code=code,
        title=title,
        total_stages=total_stages,
    )
    if exc is None:
        logger.info("\n" + text)
    else:
        logger.error("\n" + text)


def _extract_attempt_fields(
    exc: RyotenkAIError | None,
) -> dict[str, Any]:
    """Pull H1-stamped attempt context off a :class:`RyotenkAIError`.

    The execution loop calls
    :func:`ryotenkai_control.pipeline.execution.stage_execution_loop._stamp_attempt_context`
    before re-raising, so the keys ``stage_name`` / ``stage_idx`` /
    ``stage_total`` / ``attempt_no`` are populated for stage failures.
    Pre-stage failures (launch rejection, bootstrap errors) won't have
    them ⇒ the dict is empty and the outcome block omits those lines.
    """
    if exc is None:
        return {}
    ctx = getattr(exc, "context", None)
    if not isinstance(ctx, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("stage_name", "stage_idx", "stage_total", "attempt_no"):
        if key in ctx and ctx[key] is not None:
            out[key] = ctx[key]
    return out


# ---------------------------------------------------------------------------
# Worker body
# ---------------------------------------------------------------------------


def _config_from_run_dir(run_dir: Path) -> Path:
    """Lift ``config_path`` from a run-dir's ``pipeline_state.json``.

    Used for resume/restart paths where the launcher omits ``--config``;
    the original config path is recorded in the state file.
    """
    # Local imports keep ``worker --help`` cheap.
    from ryotenkai_control.pipeline.state import PipelineStateLoadError, PipelineStateStore

    store = PipelineStateStore(run_dir)
    if not store.exists():
        raise SystemExit(f"error: pipeline_state.json not found in run directory: {run_dir}")
    try:
        state = store.load()
    except PipelineStateLoadError as exc:
        raise SystemExit(f"error: cannot load pipeline state for {run_dir}: {exc}")
    if not state.config_path:
        raise SystemExit(f"error: run {run_dir.name!r} has no recorded config_path; " "pass --config explicitly")
    resolved = Path(state.config_path)
    if not resolved.exists():
        raise SystemExit(f"error: config recorded in state no longer exists: {resolved}")
    return resolved


def _run_pipeline(args: argparse.Namespace) -> int:
    """Pipeline body: parse args, build orchestrator, run, return 0 on success.

    Extracted from :func:`main` so the top-level exception handler can wrap
    a single function and keep the dispatch table tidy. Raises propagate to
    :func:`main` which renders them via the unified kubectl-style helper.
    """
    # Heavy imports stay lazy so ``--help`` doesn't pay torch/mlflow costs.
    from ryotenkai_shared.config.runtime import RuntimeSettings, load_runtime_settings
    from ryotenkai_control.pipeline.orchestrator import PipelineOrchestrator
    from ryotenkai_shared.config.loader import load_pipeline_config

    pipeline_start_time = time.time()

    run_dir: Path | None = args.run_dir.expanduser().resolve() if args.run_dir else None
    if args.config is not None:
        config_path = args.config.expanduser().resolve()
    elif run_dir is not None:
        config_path = _config_from_run_dir(run_dir)
    else:
        raise SystemExit(
            "error: either --config or --run-dir (with a recorded "
            "config_path in pipeline_state.json) is required"
        )

    cfg = load_pipeline_config(config_path)

    base = load_runtime_settings()
    runs_base_override = os.environ.get("RYOTENKAI_RUNS_BASE_DIR")
    settings = RuntimeSettings(
        runs_base_dir=Path(runs_base_override) if runs_base_override else base.runs_base_dir,
        log_level=base.log_level,
    )

    orchestrator = PipelineOrchestrator(
        config=cfg,
        run_directory=run_dir,
        settings=settings,
    )

    # Hook SIGINT/SIGTERM AFTER constructing the orchestrator (so it
    # exists when the handler tries to ``notify_signal``) but BEFORE
    # ``run()`` so a Ctrl+C during early stages already cooperates.
    # Pollers (PodSshWaiter, etc.) check the cancel event via
    # ``sleep_cancellable`` and raise PipelineCancelled at their own
    # boundaries — see ``src/utils/cancellation.py``.
    from ryotenkai_shared.utils.cancellation import (
        install_handler,
        set_active_orchestrator,
    )

    install_handler()
    set_active_orchestrator(orchestrator)
    try:
        orchestrator.run(
            run_dir=run_dir,
            resume=args.resume,
            restart_from_stage=args.restart_from_stage,
        )
    finally:
        set_active_orchestrator(None)

    # Phase H1 — success outcome block to pipeline.log.
    pipeline_duration = time.time() - pipeline_start_time
    try:
        total_stages = len(list(cfg.stages or []))
    except Exception:  # noqa: BLE001 — defensive
        total_stages = None
    attempt_no = _try_read_attempt_no(orchestrator)
    _log_pipeline_outcome(
        None,
        attempt_no=attempt_no,
        duration_seconds=pipeline_duration,
        total_stages=total_stages,
    )
    return 0


def _try_read_attempt_no(orchestrator: Any) -> int | None:
    """Best-effort recovery of the active attempt_no for outcome logging.

    Reads from the orchestrator's exposed state when available; falls
    back to None on any failure. Never raises.
    """
    try:
        state = getattr(orchestrator, "_pipeline_state", None)
        if state is None:
            return None
        attempts = getattr(state, "attempts", None) or []
        if attempts:
            return int(attempts[-1].attempt_no)
    except Exception:  # noqa: BLE001 — defensive
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.worker",
        description=(
            "Internal pipeline worker. Not invoked by users directly — "
            "use `ryotenkai run start/resume/restart` instead."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory under <project>/runs/ or <runs_base>/runs/. "
            "Omit for fresh runs — orchestrator allocates a new dir under "
            "settings.runs_base_dir."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Pipeline YAML. Omit for resume/restart — lifted from pipeline_state.json.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last completed stage.",
    )
    parser.add_argument(
        "--restart-from-stage",
        default=None,
        help="Restart at the named stage, discarding later attempts.",
    )
    args = parser.parse_args(argv)

    # Imports here (not module-level) so ``worker --help`` stays cold-start
    # cheap. Each handler is fast — only failure paths hit them.
    from ryotenkai_control.cli.errors import print_ryotenkai_error
    from ryotenkai_shared.api.request_id import current_request_id, generate_request_id
    from ryotenkai_shared.errors import InternalError, RyotenkAIError
    from ryotenkai_shared.utils.logger import logger

    # Mirror the CLI wrap_command: stamp a fresh request id so any log
    # lines the worker emits share a correlation id with the rendered
    # error output. Parent CLI may have already stamped one (via
    # ``wrap_command``) — inherit if so, else mint our own.
    request_id = current_request_id() or generate_request_id()

    try:
        return _run_pipeline(args)
    except KeyboardInterrupt:
        # POSIX SIGINT convention: exit 128 + 2 = 130. Mirror the CLI
        # wrap_command's short "aborted" message on stderr.
        print("aborted", file=sys.stderr)
        return 130
    except RyotenkAIError as exc:
        # Phase H1 — outcome summary to pipeline.log (always — when the
        # file handler is attached this lands in the persistent log,
        # otherwise it only reaches stderr).
        attempt_fields = _extract_attempt_fields(exc)
        _log_pipeline_outcome(
            exc,
            attempt_no=attempt_fields.get("attempt_no"),
            stage_name=attempt_fields.get("stage_name"),
            stage_idx=attempt_fields.get("stage_idx"),
            stage_total=attempt_fields.get("stage_total"),
            trace_id=getattr(exc, "trace_id", None),
            request_id=request_id,
            detail=exc.detail,
            context=dict(exc.context) if exc.context else None,
            code=exc.code.value,
            title=exc.title,
        )
        return print_ryotenkai_error(exc, request_id=request_id)
    except SystemExit:
        # argparse / explicit raise SystemExit — let it propagate with
        # its own message; the message text is already user-facing.
        raise
    except Exception as exc:  # pragma: no cover -- final safety net
        # Catch-all for truly unexpected bugs. Render as InternalError so
        # the user sees the unified format with a trace correlation id
        # instead of a raw traceback. Full traceback goes to logs only —
        # never to user-facing stderr (security: no path/env/local leak).
        logger.error("Unhandled exception in worker", exc_info=exc)
        internal = InternalError(
            detail=(
                "An unexpected error occurred. See server logs for "
                "the full traceback."
            ),
            context={"exception_type": type(exc).__name__},
        )
        _log_pipeline_outcome(
            internal,
            trace_id=getattr(internal, "trace_id", None),
            request_id=request_id,
            detail=internal.detail,
            context=dict(internal.context) if internal.context else None,
            code=internal.code.value,
            title=internal.title,
        )
        return print_ryotenkai_error(internal, request_id=request_id)


if __name__ == "__main__":
    sys.exit(main())
