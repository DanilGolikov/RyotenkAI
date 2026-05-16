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
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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
    return 0


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
        return print_ryotenkai_error(
            InternalError(
                detail=(
                    "An unexpected error occurred. See server logs for "
                    "the full traceback."
                ),
                context={"exception_type": type(exc).__name__},
            ),
            request_id=request_id,
        )


if __name__ == "__main__":
    sys.exit(main())
