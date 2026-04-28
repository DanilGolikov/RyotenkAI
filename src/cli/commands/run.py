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
    config: Path | None,
    *,
    run_dir: Path | None,
    resume: bool,
    restart_from_stage: str | None,
    dry_run: bool,
    project_id: str | None = None,
) -> None:
    """Common path for ``run start / resume / restart``.

    When ``project_id`` is given, the adapter
    (:func:`src.workspace.projects.adapter.load_project_inputs`)
    supplies the orchestrator with ``(config, env, metadata)`` derived
    from the project's filesystem; ``config`` is treated as a config
    override rather than the source of truth.

    Integration-resolver failures (project YAML references an
    integration that's not registered or has an empty/invalid
    ``current.yaml``) and project-not-found surface as clean CLI errors
    rather than raw Python tracebacks — see
    :mod:`src.workspace.integrations.exceptions` and
    :mod:`src.workspace.projects.adapter`.
    """
    from src.workspace.integrations.exceptions import (
        IntegrationNotFoundError,
        IntegrationUnresolvedError,
    )
    from src.workspace.projects.adapter import (
        ProjectNotFoundError,
        load_project_inputs,
    )

    project_inputs = None
    if project_id is not None:
        try:
            # Bare ``run start --project X`` → use project's
            # ``configs/current.yaml``. ``run start -c Y --project X``
            # → use ``Y`` as override but keep project's env+metadata.
            project_inputs = load_project_inputs(
                project_id,
                config_override=config if config is not None else None,
            )
        except ProjectNotFoundError as exc:
            raise die(str(exc))
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

    if dry_run:
        if project_inputs is not None:
            typer.echo(
                f"dry-run: project={project_id} config OK; would start "
                f"(run_dir={run_dir}, resume={resume}, "
                f"restart_from_stage={restart_from_stage}, "
                f"actor={project_inputs.metadata.get('actor')})",
            )
            return
        # Legacy ad-hoc path — resolve_config above guarantees config
        # is set on this branch.
        assert config is not None
        from src.workspace.integrations.loader import load_pipeline_config

        try:
            load_pipeline_config(config)
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
        if project_inputs is not None:
            # Variant 1 path: orchestrator gets a pre-resolved config +
            # explicit env mapping + audit metadata. Adapter has already
            # called ``load_config`` under the hood. The project's own
            # ``runs/`` directory becomes the runs-base so a fresh
            # launch lands at ``<project>/runs/<run_id>/`` instead of
            # the global location.
            from src.config.runtime import RuntimeSettings, load_runtime_settings

            base = load_runtime_settings()
            project_settings = RuntimeSettings(
                runs_base_dir=project_inputs.runs_base_dir,
                log_level=base.log_level,
            )
            orchestrator = PipelineOrchestrator(
                config=project_inputs.config,
                env=project_inputs.env,
                metadata=project_inputs.metadata,
                run_directory=run_dir,
                settings=project_settings,
            )
        else:
            # Anonymous / ad-hoc path. CLI loads the YAML (which runs
            # the UX-layer integration resolver) and hands a fully-
            # resolved ``PipelineConfig`` to the orchestrator. There
            # is no legacy positional path-based constructor anymore.
            assert config is not None
            from src.workspace.integrations.loader import load_pipeline_config

            cfg_obj = load_pipeline_config(config)
            orchestrator = PipelineOrchestrator(
                config=cfg_obj, run_directory=run_dir,
            )
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
    project: ProjectOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Start a fresh pipeline run (or resume one when --run-dir is given).

    --project / RYOTENKAI_PROJECT semantics
        - bare ``--project X`` → use project's ``configs/current.yaml``.
        - ``--config Y --project X`` → use ``Y`` as override, env +
          metadata still come from project ``X``.
        - bare ``--config Y`` (no --project) → anonymous run.
        - bare command (neither flag) → falls back to ``project use``
          via ``cli_state.context_store.get_current_project()``.
    """
    # Resolve project from explicit --project > RYOTENKAI_PROJECT (handled
    # by typer's envvar) > persisted ``project use`` context.
    resolved_project = project
    if resolved_project is None and config is None and run_dir is None:
        # Only consult the persisted context when the user gave us
        # nothing else to anchor on — keeps "ad-hoc -c X.yaml" paths
        # untouched.
        from src.cli_state import context_store

        resolved_project = context_store.get_current_project()

    if resolved_project is not None:
        # Project mode: --config is treated as override, not source of
        # truth. ``_resolve_config`` would error on missing config for
        # bare ``--project X`` runs, so we skip it.
        _exec_orchestrator(
            config, run_dir=run_dir, resume=False,
            restart_from_stage=None, dry_run=dry_run,
            project_id=resolved_project,
        )
        return

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
    from src.pipeline.launch.pod_availability import PodAvailability
    from src.pipeline.launch.resume_service import (
        LaunchResumeService,
        ResumeProgress,
    )

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
