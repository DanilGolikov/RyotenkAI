from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.api.schemas.launch import (
    InterruptResponse,
    LaunchRequestSchema,
    LaunchResponse,
    RestartPoint,
    RestartPointsResponse,
)
from src.pipeline.launch import (
    LaunchRequest,
    interrupt_launch_process,
    is_process_alive,
    load_restart_point_options,
    pick_default_launch_mode,
    read_lock_pid,
    spawn_launch_detached,
    validate_resume_run,
)
from src.workspace.projects.store import ProjectStore, ProjectStoreError
from src.pipeline.state import PipelineStateStore, remove_stale_lock


def _project_env_for_run_dir(run_dir: Path) -> dict[str, str]:
    """Walk up from ``run_dir`` until we find a ``project.json`` sibling and
    return that workspace's env.json overrides. Returns {} if the run isn't
    inside a project workspace (legacy runs, ad-hoc dirs).
    """
    for candidate in (run_dir, *run_dir.parents):
        if (candidate / "project.json").is_file():
            try:
                return ProjectStore(candidate).read_env()
            except ProjectStoreError:
                return {}
    return {}


def list_restart_points(run_dir: Path, config_path: Path | None = None) -> RestartPointsResponse:
    resolved_config, options = load_restart_point_options(run_dir, config_path)
    return RestartPointsResponse(
        config_path=str(resolved_config),
        points=[
            RestartPoint(stage=opt.stage, available=opt.available, mode=opt.mode, reason=opt.reason)
            for opt in options
        ],
    )


def default_launch_mode(run_dir: Path) -> str:
    return pick_default_launch_mode(run_dir)


def launch(run_dir: Path, request: LaunchRequestSchema) -> LaunchResponse:
    lock_pid = read_lock_pid(run_dir)
    if lock_pid is not None and is_process_alive(lock_pid):
        raise LaunchAlreadyRunningError(lock_pid)

    config_path = Path(request.config_path).expanduser().resolve() if request.config_path else None

    if request.mode == "resume":
        # Surfaces ValueError if resume isn't valid (config drift, nothing to resume, etc.).
        validate_resume_run(run_dir, config_path)

    launch_request = LaunchRequest(
        mode=request.mode,
        run_dir=run_dir,
        config_path=config_path,
        restart_from_stage=request.restart_from_stage,
        log_level=request.log_level,
    )
    # Surface validation errors (missing config, illegal restart stage, etc.) as 422.
    launch_request = launch_request.validate()
    project_env = _project_env_for_run_dir(run_dir)
    pid, command, launcher_log = spawn_launch_detached(
        launch_request, extra_env=project_env
    )
    return LaunchResponse(
        pid=pid,
        launched_at=datetime.now(UTC).isoformat(),
        command=list(command),
        launcher_log=str(launcher_log),
        run_dir=str(run_dir),
    )


def interrupt(run_dir: Path) -> InterruptResponse:
    pid = read_lock_pid(run_dir)
    if pid is None:
        return InterruptResponse(interrupted=False, pid=None, reason="no_lock_file")
    if not is_process_alive(pid):
        # Stale lock — route removal through PipelineStateStore so the sibling-client
        # invariant (only state.store owns run.lock lifecycle) is preserved.
        # remove_stale_lock re-reads the pid right before unlink; if another process
        # legitimately acquired the lock in the meantime the file is left untouched.
        remove_stale_lock(PipelineStateStore(run_dir).lock_path, expected_pid=pid)
        return InterruptResponse(interrupted=False, pid=pid, reason="process_not_found")
    ok = interrupt_launch_process(pid)
    if not ok:
        return InterruptResponse(interrupted=False, pid=pid, reason="signal_failed")
    return InterruptResponse(interrupted=True, pid=pid, reason=None)


class LaunchAlreadyRunningError(RuntimeError):
    def __init__(self, pid: int) -> None:
        super().__init__(f"run already active with pid={pid}")
        self.pid = pid


# ---------------------------------------------------------------------------
# Phase 11.C-2 — pod resume service
# ---------------------------------------------------------------------------


class ResumePodResponse:
    """Result of a ``resume_pod`` API call.

    Plain class rather than a Pydantic schema for now — the surface
    is small (3 fields) and used by exactly one router. If a second
    consumer arrives, lift to ``src.api.schemas.launch`` like the
    other responses.
    """

    __slots__ = ("availability", "ok", "message")

    def __init__(
        self, *, availability: str, ok: bool, message: str,
    ) -> None:
        self.availability = availability
        self.ok = ok
        self.message = message

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "availability": self.availability,
            "ok": self.ok,
            "message": self.message,
        }


def resume_pod_for_run(run_dir: Path) -> ResumePodResponse:
    """Phase 11.C-2 — wake the run's pod if it's sleeping.

    Used by:

    * ``ryotenkai run resume`` (CLI) — invoked indirectly via
      ``_resume_pod_if_needed`` helper that has its own progress
      output. CLI calls the underlying probe + retry directly so
      it can stream progress to stdout.
    * ``POST /api/v1/runs/{run_id}/resume-pod`` (Web UI) — invokes
      this service and returns the response shape.

    Behaviour:

    * No pod_metadata → ``RUNNING`` (legacy attempt — let the
      pipeline's connect step surface real errors).
    * Pod ``RUNNING`` → no-op success.
    * Pod ``SLEEPING_RESUMABLE`` → wake via ``resume_pod_with_retry``
      with the standard 5-min capacity-aware budget.
    * Pod ``GONE`` / ``SLEEPING_RESUME_FAILED`` / ``PROBE_FAILED``
      → return non-ok with operator-friendly message.

    Synchronous facade — the underlying retry runs in
    ``asyncio.run``. The router wraps the call in
    :func:`run_in_threadpool` so FastAPI's event loop stays free.
    """
    import asyncio
    import os

    from src.pipeline.launch.pod_availability import (
        PodAvailability,
        PodAvailabilityProbe,
        load_pod_metadata_for_run,
        resume_pod_with_retry,
    )

    metadata = load_pod_metadata_for_run(run_dir)
    if metadata is None:
        return ResumePodResponse(
            availability=PodAvailability.RUNNING.value,
            ok=True,
            message=(
                "No pod metadata recorded for this run (legacy attempt). "
                "Continue with normal resume flow."
            ),
        )

    if metadata.provider != "runpod":
        return ResumePodResponse(
            availability=PodAvailability.RUNNING.value,
            ok=True,
            message=(
                f"Provider {metadata.provider!r} doesn't have an in-pod "
                "resume mechanism; continue with normal flow."
            ),
        )

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        return ResumePodResponse(
            availability=PodAvailability.PROBE_FAILED.value,
            ok=False,
            message="RUNPOD_API_KEY not in environment",
        )

    from src.providers.runpod.training.api_client import RunPodAPIClient

    client = RunPodAPIClient(api_key=api_key)

    def _query_pod(pod_id: str) -> dict:
        result = client.query_pod(pod_id)
        if result.is_failure():
            err = result.unwrap_err()
            raise RuntimeError(err.message)
        return result.unwrap()

    probe = PodAvailabilityProbe(query_pod=_query_pod)
    verdict = probe.probe(metadata)

    if verdict.availability == PodAvailability.RUNNING:
        return ResumePodResponse(
            availability=verdict.availability.value,
            ok=True,
            message=verdict.message or "Pod is already running",
        )

    if verdict.availability == PodAvailability.GONE:
        return ResumePodResponse(
            availability=verdict.availability.value,
            ok=False,
            message=verdict.message,
        )

    if verdict.availability == PodAvailability.PROBE_FAILED:
        return ResumePodResponse(
            availability=verdict.availability.value,
            ok=False,
            message=verdict.message,
        )

    if verdict.availability != PodAvailability.SLEEPING_RESUMABLE:
        return ResumePodResponse(
            availability=verdict.availability.value,
            ok=False,
            message=f"Unexpected pod state: {verdict.availability.value}",
        )

    # Wake.
    from src.providers.runpod.sdk_adapter import is_capacity_error_message

    async def _resume_call(pod_id: str) -> bool:
        result = client.resume_pod(pod_id)
        if result.is_failure():
            err = result.unwrap_err()
            raise RuntimeError(err.message)
        return True

    outcome = asyncio.run(
        resume_pod_with_retry(
            metadata.pod_id,
            resume_call=_resume_call,
            is_capacity_error=is_capacity_error_message,
        ),
    )

    if outcome.ok:
        return ResumePodResponse(
            availability=PodAvailability.RUNNING.value,
            ok=True,
            message=(
                f"Pod resumed in {outcome.elapsed_seconds:.1f}s "
                f"({outcome.attempts} attempt(s))"
            ),
        )

    return ResumePodResponse(
        availability=(
            PodAvailability.SLEEPING_RESUME_FAILED.value
            if outcome.capacity_exhausted
            else PodAvailability.PROBE_FAILED.value
        ),
        ok=False,
        message=outcome.error_message,
    )


__all__ = [
    "LaunchAlreadyRunningError",
    "ResumePodResponse",
    "default_launch_mode",
    "interrupt",
    "launch",
    "list_restart_points",
    "resume_pod_for_run",
]
