"""``ryotenkai run <verb>`` — start / resume / restart / interrupt / restart-points.

This is the *write* face of the run lifecycle: it spawns / resumes /
stops a pipeline. Read-only inspection commands live under ``runs``
(plural) — see :mod:`src.cli.commands.runs`.

All heavy imports (orchestrator, mlflow, torch) live inside command
bodies so ``ryotenkai run --help`` stays under the 300 ms budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.cli import _signals
from src.cli.common_options import (
    DryRunOpt,
    OptionalRunDirOpt,
    ProjectOpt,
    RunDirArg,
)
from src.cli.context import CLIContext
from src.cli.errors import die
from src.cli.renderer import get_renderer

run_app = typer.Typer(
    no_args_is_help=True,
    help="Start / resume / restart / interrupt training runs.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_config(config: Path | None, run_dir: Path | None) -> Path:
    """Resolve the config path — explicit ``--config`` wins, else lift
    it from the run directory's ``pipeline_state.json``."""
    if config is not None:
        return config
    if run_dir is not None:
        from src.pipeline.state import PipelineStateLoadError, PipelineStateStore

        store = PipelineStateStore(run_dir.expanduser().resolve())
        if not store.exists():
            raise die(f"pipeline_state.json not found in run directory: {run_dir}")
        try:
            state = store.load()
        except PipelineStateLoadError as exc:
            raise die(f"cannot load pipeline state for {run_dir}: {exc}")
        if not state.config_path:
            raise die(
                f"run {run_dir.name!r} has no recorded config_path; pass --config explicitly",
            )
        resolved = Path(state.config_path)
        if not resolved.exists():
            raise die(
                f"config recorded in state no longer exists: {resolved}",
                hint="pass --config explicitly to override",
            )
        return resolved
    raise die(
        "missing required argument",
        hint="provide --config <path> for a fresh run, or --run-dir to resume/restart",
    )


def _exec_orchestrator(
    config: Path,
    *,
    run_dir: Path | None,
    resume: bool,
    restart_from_stage: str | None,
    dry_run: bool,
) -> None:
    """Common path for ``run start / resume / restart``."""
    if dry_run:
        # Validate config + stage selection without spawning the run.
        from src.utils.config import load_config

        load_config(config)
        typer.echo(
            f"dry-run: config {config} OK; would start "
            f"(run_dir={run_dir}, resume={resume}, "
            f"restart_from_stage={restart_from_stage})",
        )
        return

    from src.pipeline.orchestrator import PipelineOrchestrator  # heavy: lazy

    orchestrator = PipelineOrchestrator(config, run_directory=run_dir)
    _signals.set_active_orchestrator(orchestrator)
    try:
        result = orchestrator.run(
            run_dir=run_dir, resume=resume, restart_from_stage=restart_from_stage,
        )
        if not result.is_success():
            raise die(f"pipeline failed: {result.unwrap_err()}")
        typer.echo("Pipeline completed successfully.")
    finally:
        _signals.set_active_orchestrator(None)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@run_app.command("start")
def start_cmd(
    ctx: typer.Context,  # noqa: ARG001 — kept for parity with future remote-mode
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to pipeline config YAML (required for fresh runs).",
            dir_okay=False, resolve_path=True,
        ),
    ] = None,
    run_dir: OptionalRunDirOpt = None,
    project: ProjectOpt = None,  # noqa: ARG001 — TODO Phase 2 wiring
    dry_run: DryRunOpt = False,
) -> None:
    """Start a fresh pipeline run (or resume one when --run-dir is given)."""
    resolved = _resolve_config(config, run_dir)
    _exec_orchestrator(
        resolved, run_dir=run_dir, resume=False,
        restart_from_stage=None, dry_run=dry_run,
    )


@run_app.command("resume")
def resume_cmd(
    run_dir: RunDirArg,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Override config (default: read from pipeline_state.json).",
            dir_okay=False, resolve_path=True,
        ),
    ] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Resume an interrupted run from its first failed / pending stage."""
    resolved = _resolve_config(config, run_dir)
    _exec_orchestrator(
        resolved, run_dir=run_dir, resume=True,
        restart_from_stage=None, dry_run=dry_run,
    )


@run_app.command("restart")
def restart_cmd(
    run_dir: RunDirArg,
    from_stage: Annotated[
        str,
        typer.Option(
            "--from-stage", "-s",
            help="Stage name or 1-based stage number to restart from "
                 "(see: ryotenkai run restart-points <run_dir>).",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Override config (default: read from pipeline_state.json).",
            dir_okay=False, resolve_path=True,
        ),
    ] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Restart an existing run from a specific stage onwards."""
    resolved = _resolve_config(config, run_dir)
    _exec_orchestrator(
        resolved, run_dir=run_dir, resume=False,
        restart_from_stage=from_stage, dry_run=dry_run,
    )


@run_app.command("interrupt")
def interrupt_cmd(
    run_dir: RunDirArg,
    dry_run: DryRunOpt = False,
) -> None:
    """Send SIGINT to a detached run's pid (graceful stop)."""
    from src.api.services import launch_service

    if dry_run:
        typer.echo(f"dry-run: would interrupt run at {run_dir}")
        return
    response = launch_service.interrupt_launch(run_dir)
    typer.echo(f"interrupted: pid={response.pid}, status={response.status}")


@run_app.command("restart-points")
def restart_points_cmd(
    ctx: typer.Context,
    run_dir: RunDirArg,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Config YAML (optional; falls back to pipeline_state.json).",
            dir_okay=False, resolve_path=True,
        ),
    ] = None,
) -> None:
    """List the stages this run can be restarted / resumed from."""
    from src.pipeline.launch.restart_options import load_restart_point_options

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        _, points = load_restart_point_options(run_dir, _resolve_config(config, run_dir))
    except typer.Exit:
        raise
    except Exception as exc:
        raise die(str(exc))

    if state.is_machine_readable:
        renderer.emit([
            {
                "index": idx,
                "stage": item.stage,
                "available": bool(item.available),
                "mode": item.mode,
                "reason": item.reason,
            }
            for idx, item in enumerate(points, start=1)
        ])
    else:
        renderer.table(
            headers=["#", "Stage", "Available", "Mode", "Reason"],
            rows=[
                (idx, item.stage, "yes" if item.available else "no",
                 item.mode, item.reason)
                for idx, item in enumerate(points, start=1)
            ],
        )
        renderer.text("")
        renderer.text("Use # or stage name with `ryotenkai run restart --from-stage`")
    renderer.flush()


__all__ = ["run_app"]
