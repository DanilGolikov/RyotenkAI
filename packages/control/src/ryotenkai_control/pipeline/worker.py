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
    from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

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

    # Heavy imports stay lazy so ``--help`` doesn't pay torch/mlflow costs.
    from src.config.runtime import RuntimeSettings, load_runtime_settings
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.workspace.integrations.loader import load_pipeline_config

    run_dir: Path | None = args.run_dir.expanduser().resolve() if args.run_dir else None
    if args.config is not None:
        config_path = args.config.expanduser().resolve()
    elif run_dir is not None:
        config_path = _config_from_run_dir(run_dir)
    else:
        raise SystemExit(
            "error: either --config or --run-dir (with a recorded " "config_path in pipeline_state.json) is required"
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
    from src.utils.cancellation import (
        install_handler,
        set_active_orchestrator,
    )

    install_handler()
    set_active_orchestrator(orchestrator)
    try:
        result = orchestrator.run(
            run_dir=run_dir,
            resume=args.resume,
            restart_from_stage=args.restart_from_stage,
        )
    finally:
        set_active_orchestrator(None)
    return 0 if result.is_success() else 1


if __name__ == "__main__":
    sys.exit(main())
