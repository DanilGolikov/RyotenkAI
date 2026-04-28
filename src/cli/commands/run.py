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
    """Common path for ``run start / resume / restart``.

    Integration-resolver failures (project YAML references an
    integration that's not registered or has an empty/invalid
    ``current.yaml``) surface as clean CLI errors rather than raw
    Python tracebacks — see :mod:`src.config.integrations.exceptions`.
    """
    from src.config.integrations.exceptions import (
        IntegrationNotFoundError,
        IntegrationUnresolvedError,
    )

    if dry_run:
        from src.utils.config import load_config

        try:
            load_config(config)
        except IntegrationNotFoundError as exc:
            raise die(
                str(exc),
                hint=(
                    "create the integration via the Web UI "
                    "(http://localhost:5173/settings/integrations) or CLI"
                ),
            )
        except IntegrationUnresolvedError as exc:
            raise die(str(exc))
        typer.echo(
            f"dry-run: config {config} OK; would start "
            f"(run_dir={run_dir}, resume={resume}, "
            f"restart_from_stage={restart_from_stage})",
        )
        return

    from src.pipeline.orchestrator import PipelineOrchestrator  # heavy: lazy

    try:
        orchestrator = PipelineOrchestrator(config, run_directory=run_dir)
    except IntegrationNotFoundError as exc:
        raise die(
            str(exc),
            hint=(
                "create the integration via the Web UI "
                "(http://localhost:5173/settings/integrations) or CLI"
            ),
        )
    except IntegrationUnresolvedError as exc:
        raise die(str(exc))

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
    skip_pod_probe: Annotated[
        bool,
        typer.Option(
            "--skip-pod-probe",
            help=(
                "Skip the pre-flight pod availability check (Phase 11.C-2). "
                "Use when the pod was created outside RyotenkAI or when the "
                "probe is misbehaving — pipeline will fall back to its own "
                "SSH connect step to surface real status."
            ),
        ),
    ] = False,
    dry_run: DryRunOpt = False,
) -> None:
    """Resume an interrupted run from its first failed / pending stage.

    Phase 11.C-2: when the run's latest attempt has a recorded
    ``pod_metadata`` (Phase 11.C-1 schema), this command first
    probes the pod's availability via :class:`PodAvailabilityProbe`
    and, if it's sleeping (Phase 11.B ``podStop`` outcome), wakes it
    via :func:`resume_pod_with_retry` BEFORE re-spawning the
    pipeline. Without this pre-flight, the pipeline's own SSH
    connect step would surface the sleeping pod as an unreachable
    error.

    Use ``--skip-pod-probe`` to bypass — useful for legacy runs
    (no pod_metadata) or for runs created outside the standard
    GPUDeployer flow.
    """
    if not skip_pod_probe and not dry_run:
        _resume_pod_if_needed(run_dir)

    resolved = _resolve_config(config, run_dir)
    _exec_orchestrator(
        resolved, run_dir=run_dir, resume=True,
        restart_from_stage=None, dry_run=dry_run,
    )


def _resume_pod_if_needed(run_dir: Path) -> None:
    """Phase 14.C — thin CLI adapter over :class:`LaunchResumeService`.

    Translates the service's typed :class:`ResumeOutcome` into
    ``typer.echo`` progress + ``die()`` hints. Resume orchestration
    itself (probe, retry, capacity handling) is owned by the
    service; both this CLI and the REST surface in
    :func:`src.api.services.launch_service.resume_pod_for_run`
    delegate to it.

    Failure modes (all surface as ``die`` with operator-friendly hints):

    * ``GONE`` — pod terminated; pipeline can't continue from
      checkpoint without a fresh pod. Hint at ``run restart``.
    * ``SLEEPING_RESUME_FAILED`` (capacity exhausted) — show actionable
      error.
    * ``PROBE_FAILED`` — RunPod GraphQL outage; suggest retry or
      ``--skip-pod-probe``.

    Skipped no-op (silent / message-only) when:

    * Run has no pod_metadata (legacy / mock runs).
    * Provider doesn't support in-pod resume (single_node, future
      always-on providers).
    * RUNPOD_API_KEY missing.
    """
    from src.pipeline.launch.resume_service import (
        LaunchResumeService, ResumeProgress,
    )
    from src.pipeline.launch.pod_availability import PodAvailability

    def _echo(evt: ResumeProgress) -> None:
        # Probing line has no leading indent (matches pre-14.C UX);
        # subsequent verdict / resuming / resumed lines are indented
        # 2 spaces for readability.
        if evt.kind == "probing":
            typer.echo(evt.message)
        else:
            typer.echo(f"  {evt.message}")

    outcome = LaunchResumeService().resume(run_dir, on_progress=_echo)

    if outcome.ok:
        # Includes the "skipped" path — service emits no progress
        # for skipped (it's silent except in the verdict step).
        # Pre-14.C printed "(skipping pod probe: ...)" in a few
        # cases; service now returns the message in outcome.message
        # without echoing. For UX parity, surface the message when
        # availability == "skipped" and the run has metadata.
        if outcome.availability == "skipped" and outcome.message:
            typer.echo(f"  ({outcome.message})")
        return

    # Failure paths.
    if outcome.availability == PodAvailability.GONE.value:
        raise die(
            "Pod has been terminated; cannot resume in-place.",
            hint=(
                f"use 'ryotenkai run restart {run_dir} --from-stage <stage>' "
                "to recreate from a checkpoint"
            ),
        )

    if outcome.capacity_exhausted:
        raise die(
            f"Pod resume capacity unavailable: {outcome.message}",
            hint=(
                "RunPod has no GPU available right now in the "
                "pod's datacenter. Retry later, or use "
                "'ryotenkai run restart' to recreate from a "
                "checkpoint in a different region"
            ),
        )

    if outcome.availability == PodAvailability.PROBE_FAILED.value:
        raise die(
            f"Pod probe failed: {outcome.message}",
            hint=(
                "RunPod may be experiencing an outage. Retry in a few "
                "minutes, or pass --skip-pod-probe to let the pipeline "
                "discover the real state via SSH"
            ),
        )

    raise die(
        f"Pod resume failed: {outcome.message}",
        hint="check RunPod console; pass --skip-pod-probe to bypass",
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
