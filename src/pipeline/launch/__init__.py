"""Launch-time preparation for a pipeline run.

This package owns the "everything that must happen before the stage loop starts"
slice of the orchestrator: loading/initialising pipeline state, deriving the
start stage, validating config drift, building the attempt record, computing
per-attempt directories, and recording launch-time rejections.

The orchestrator is responsible for acquiring the run lock, setting up MLflow,
and forking the execution context — those stay outside this package because
they are cross-cutting concerns with their own lifecycle.
"""

from src.pipeline.launch.launch_preparator import (
    LaunchPreparationError,
    LaunchPreparator,
    PreparedAttempt,
)

# Re-export the subprocess launch/interrupt mechanics that used to live in
# `src/pipeline/launch.py`. Keeping them accessible at the package root so
# existing callers (`from src.pipeline.launch import spawn_launch_detached`,
# `read_lock_pid`, etc.) don't need to update their imports after the
# module → package refactor.
from src.pipeline.launch.runtime import (
    MODE_FRESH,
    MODE_NEW_RUN,
    MODE_RESTART,
    MODE_RESUME,
    ActiveLaunch,
    LaunchLogLevel,
    LaunchMode,
    LaunchRequest,
    LaunchResult,
    LaunchStatus,
    RestartPointOption,
    build_train_command,
    execute_launch_subprocess,
    interrupt_launch_process,
    is_process_alive,
    load_restart_point_options,
    pick_default_launch_mode,
    read_lock_pid,
    resolve_config_path_for_run,
    spawn_launch_detached,
    validate_resume_run,
)

__all__ = [
    "MODE_FRESH",
    "MODE_NEW_RUN",
    "MODE_RESTART",
    "MODE_RESUME",
    "ActiveLaunch",
    "LaunchLogLevel",
    "LaunchMode",
    "LaunchPreparationError",
    "LaunchPreparator",
    "LaunchRequest",
    "LaunchResult",
    "LaunchStatus",
    "PreparedAttempt",
    "RestartPointOption",
    "build_train_command",
    "execute_launch_subprocess",
    "interrupt_launch_process",
    "is_process_alive",
    "load_restart_point_options",
    "pick_default_launch_mode",
    "read_lock_pid",
    "resolve_config_path_for_run",
    "spawn_launch_detached",
    "validate_resume_run",
]
