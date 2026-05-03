"""``ryotenkai run <verb>`` — start / resume / restart / interrupt / restart-points.

This is the *write* face of the run lifecycle: it spawns a pipeline
worker subprocess (``python -m src.pipeline.worker``) and waits for
it to exit. Read-only inspection commands live under ``runs``
(plural) — see :mod:`src.cli.commands.runs`.

All heavy imports (orchestrator, mlflow, torch) live inside the
worker subprocess — the parent CLI never imports them. ``ryotenkai
run --help`` stays under the 300 ms budget; the heavy load happens
once per launch in the spawned child.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from ryotenkai_control.cli.common_options import (
    DryRunOpt,
    OptionalRunDirOpt,
    ProjectOpt,
    RunDirArg,
)
from ryotenkai_control.cli.context import CLIContext
from ryotenkai_control.cli.errors import die, format_validation_errors
from ryotenkai_control.cli.renderer import get_renderer

run_app = typer.Typer(
    no_args_is_help=True,
    help="Start / resume / restart / interrupt training runs.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


# Actor stamped on a "bare CLI" run (no --actor flag, no env var, no
# OS user) — kept consistent with the worker-side default.
_CLI_ACTOR_FALLBACK = "unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_config(config: Path | None, run_dir: Path | None) -> Path:
    """Resolve the config path — explicit ``--config`` wins, else lift
    it from the run directory's ``pipeline_state.json``."""
    if config is not None:
        return config
    if run_dir is not None:
        from ryotenkai_control.pipeline.state import PipelineStateLoadError, PipelineStateStore

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


def _resolve_project_for_launch(
    project_id: str | None,
    *,
    config: Path | None,
    run_dir: Path | None,
):
    """Resolve project context for spawn.

    Precedence:
      1. Explicit ``--project`` flag (or ``RYOTENKAI_PROJECT`` env var
         routed through Typer) → :func:`resolve_project_launch_inputs`.
      2. Walk-up from ``run_dir`` (resume / restart paths) → finds
         enclosing ``project.json`` if any.
      3. None — anonymous / ad-hoc run.

    Returns a :class:`ResolvedProject` or ``None``. Failures
    (project not registered, malformed YAML) surface as ``die()``.
    """
    from pydantic import ValidationError

    from ryotenkai_control.workspace.projects.adapter import (
        ProjectNotFoundError,
        resolve_project_launch_inputs,
        resolve_project_launch_inputs_from_run_dir,
    )

    if project_id is not None:
        try:
            return resolve_project_launch_inputs(
                project_id,
                config_override=config if config is not None else None,
            )
        except ProjectNotFoundError as exc:
            raise die(str(exc))
        except ValidationError as exc:
            raise die(f"invalid config in project {project_id!r}\n" f"{format_validation_errors(exc)}")

    if run_dir is not None:
        # Resume/restart from a project-launched run — pick up env +
        # metadata from the enclosing project workspace so MLflow tags
        # carry through across attempts.
        return resolve_project_launch_inputs_from_run_dir(run_dir)

    return None


def _spawn_worker(
    *,
    mode: str,
    config: Path | None,
    run_dir: Path | None,
    restart_from_stage: str | None,
    project_id: str | None,
    dry_run: bool,
) -> None:
    """Spawn the pipeline worker subprocess and propagate exit code.

    Single launch path used by ``start / resume / restart``. Builds
    the ``LaunchRequest`` + ``extra_env``, then either prints the
    plan (``dry_run``) or forks ``python -m src.pipeline.worker``
    foreground (stdio inherited, blocking).
    """
    from ryotenkai_control.pipeline.launch import LaunchRequest
    from ryotenkai_control.workspace.projects.adapter import build_subprocess_extra_env

    resolved = _resolve_project_for_launch(
        project_id,
        config=config,
        run_dir=run_dir,
    )

    # ``config`` precedence: explicit --config flag wins; else use the
    # project's resolved config_path; else lift from run_dir's state.
    if config is not None:
        config_for_spawn = config
    elif resolved is not None:
        config_for_spawn = resolved.config_path
    elif mode in ("resume", "restart"):
        # Allow worker to lift from pipeline_state.json — it has the
        # same logic. Pass None to omit --config from the command.
        config_for_spawn = None
    else:
        # Fresh run with neither --config nor --project nor run_dir
        # context. This shouldn't happen because the dispatch above
        # validates inputs, but raise a clean error if it does.
        raise die(
            "missing required argument",
            hint="provide --config <path>, --project <id>, or --run-dir for resume/restart",
        )

    LaunchRequest(
        mode="fresh" if mode == "start" else mode,  # type: ignore[arg-type]
        run_dir=(run_dir or Path.cwd()),  # placeholder; overridden when run_dir is None
        config_path=config_for_spawn,
        restart_from_stage=restart_from_stage,
        log_level="INFO",
    )

    # Validate request shape (fresh requires config; restart requires stage).
    # Fresh run without run_dir: worker's RuntimeSettings creates a fresh dir.
    if run_dir is None:
        # LaunchRequest requires a run_dir. For fresh runs without an
        # explicit dir we let the worker allocate one — pass a sentinel
        # of CWD here and rely on settings.runs_base_dir downstream.
        # The worker module accepts --run-dir as optional and falls back.
        pass

    actor_default = _CLI_ACTOR_FALLBACK
    extra_env = build_subprocess_extra_env(resolved, default_actor=actor_default)

    if dry_run:
        plan = {
            "mode": mode,
            "run_dir": str(run_dir) if run_dir else None,
            "config_path": str(config_for_spawn) if config_for_spawn else None,
            "restart_from_stage": restart_from_stage,
            "project_id": resolved.metadata["project_id"] if resolved else None,
            "extra_env_keys": sorted(extra_env.keys()),
        }
        typer.echo(json.dumps(plan, indent=2))
        return

    # Foreground spawn — child shares parent's PG so kernel routes
    # SIGINT to it natively. Parent ignores SIGINT during wait so the
    # child handles it cleanly without a Python-side forwarder fight.
    if run_dir is None and config_for_spawn is None:
        raise die("spawn requires --run-dir or --config")

    # Construct the command directly (LaunchRequest validation expects
    # a real run_dir for fresh-mode; we shortcut around it for the
    # "no run_dir" case by building the command manually).
    cmd = ["python", "-m", "ryotenkai_control.pipeline.worker"]
    if run_dir is not None:
        cmd.extend(["--run-dir", str(run_dir.expanduser().resolve())])
    if config_for_spawn is not None:
        cmd.extend(["--config", str(config_for_spawn)])
    if mode == "resume":
        cmd.append("--resume")
    elif mode == "restart":
        cmd.extend(["--restart-from-stage", restart_from_stage or ""])

    # Use sys.executable to keep venv consistency across CLI and worker.
    import sys

    cmd[0] = sys.executable

    process_env = os.environ.copy()
    process_env["LOG_LEVEL"] = "INFO"
    for k, v in extra_env.items():
        if v != "":
            process_env[k] = v

    project_root = Path(__file__).resolve().parents[3]

    # Ignore SIGINT in parent — kernel still delivers it to the
    # child via the shared process group. Child handles cleanup;
    # parent just waits for the child to exit.
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=process_env,
            start_new_session=False,
        )
        return_code = proc.wait()
    finally:
        signal.signal(signal.SIGINT, old_handler)

    if return_code != 0:
        raise typer.Exit(code=return_code)
    typer.echo("Pipeline completed successfully.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@run_app.command("start")
def start_cmd(
    ctx: typer.Context,  # noqa: ARG001 — kept for parity with future remote-mode
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to pipeline config YAML (required for fresh runs).",
            dir_okay=False,
            resolve_path=True,
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
        from ryotenkai_control.cli_state import context_store

        resolved_project = context_store.get_current_project()

    if resolved_project is not None:
        # Project mode: --config is treated as override, not source of
        # truth. ``_resolve_config`` would error on missing config for
        # bare ``--project X`` runs, so we skip it.
        _spawn_worker(
            mode="start",
            config=config,
            run_dir=run_dir,
            restart_from_stage=None,
            project_id=resolved_project,
            dry_run=dry_run,
        )
        return

    resolved = _resolve_config(config, run_dir)
    _spawn_worker(
        mode="start",
        config=resolved,
        run_dir=run_dir,
        restart_from_stage=None,
        project_id=None,
        dry_run=dry_run,
    )


@run_app.command("resume")
def resume_cmd(
    run_dir: RunDirArg,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Override config (default: read from pipeline_state.json).",
            dir_okay=False,
            resolve_path=True,
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
    _spawn_worker(
        mode="resume",
        config=resolved,
        run_dir=run_dir,
        restart_from_stage=None,
        project_id=None,
        dry_run=dry_run,
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
    from ryotenkai_control.pipeline.launch.pod_availability import PodAvailability
    from ryotenkai_control.pipeline.launch.resume_service import (
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
            hint=(f"use 'ryotenkai run restart {run_dir} --from-stage <stage>' " "to recreate from a checkpoint"),
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
            "--from-stage",
            "-s",
            help="Stage name or 1-based stage number to restart from " "(see: ryotenkai run restart-points <run_dir>).",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Override config (default: read from pipeline_state.json).",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Restart an existing run from a specific stage onwards."""
    resolved = _resolve_config(config, run_dir)
    _spawn_worker(
        mode="restart",
        config=resolved,
        run_dir=run_dir,
        restart_from_stage=from_stage,
        project_id=None,
        dry_run=dry_run,
    )


@run_app.command("interrupt")
def interrupt_cmd(
    run_dir: RunDirArg,
    dry_run: DryRunOpt = False,
) -> None:
    """Send SIGINT to a detached run's pid (graceful stop)."""
    from ryotenkai_control.api.services import launch_service

    if dry_run:
        typer.echo(f"dry-run: would interrupt run at {run_dir}")
        return
    response = launch_service.interrupt(run_dir)
    status = "interrupted" if response.interrupted else (response.reason or "noop")
    typer.echo(f"interrupted: pid={response.pid}, status={status}")


@run_app.command("restart-points")
def restart_points_cmd(
    ctx: typer.Context,
    run_dir: RunDirArg,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Config YAML (optional; falls back to pipeline_state.json).",
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """List the stages this run can be restarted / resumed from."""
    from ryotenkai_control.pipeline.launch.restart_options import load_restart_point_options

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        _, points = load_restart_point_options(run_dir, _resolve_config(config, run_dir))
    except typer.Exit:
        raise
    except Exception as exc:
        raise die(str(exc))

    if state.is_machine_readable:
        renderer.emit(
            [
                {
                    "index": idx,
                    "stage": item.stage,
                    "available": bool(item.available),
                    "mode": item.mode,
                    "reason": item.reason,
                }
                for idx, item in enumerate(points, start=1)
            ]
        )
    else:
        renderer.table(
            headers=["#", "Stage", "Available", "Mode", "Reason"],
            rows=[
                (idx, item.stage, "yes" if item.available else "no", item.mode, item.reason)
                for idx, item in enumerate(points, start=1)
            ],
        )
        renderer.text("")
        renderer.text("Use # or stage name with `ryotenkai run restart --from-stage`")
    renderer.flush()


__all__ = ["run_app"]
