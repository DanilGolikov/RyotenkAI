from __future__ import annotations

import contextlib
import os
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
from src.pipeline.project.store import ProjectStore, ProjectStoreError


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
        # Stale lock — remove it (best-effort) so subsequent launches aren't blocked.
        with contextlib.suppress(OSError):
            (run_dir / "run.lock").unlink()
        return InterruptResponse(interrupted=False, pid=pid, reason="process_not_found")
    ok = interrupt_launch_process(pid)
    if not ok:
        return InterruptResponse(interrupted=False, pid=pid, reason="signal_failed")
    return InterruptResponse(interrupted=True, pid=pid, reason=None)


class LaunchAlreadyRunningError(RuntimeError):
    def __init__(self, pid: int) -> None:
        super().__init__(f"run already active with pid={pid}")
        self.pid = pid


__all__ = [
    "LaunchAlreadyRunningError",
    "default_launch_mode",
    "interrupt",
    "launch",
    "list_restart_points",
]
