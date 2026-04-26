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

# Subprocess launch/interrupt mechanics — process spawning, PID tracking,
# launch-log tail readers.
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
    build_train_command,
    execute_launch_subprocess,
    interrupt_launch_process,
    is_process_alive,
    read_lock_pid,
    spawn_launch_detached,
)

# Restart-point + resume queries — pure read-only inspections of run state.
# Defined in launch_queries.py so they stay light enough for the CLI's lazy
# imports (no orchestrator/torch in the chain).
from src.pipeline.launch_queries import (
    RestartPointOption,
    load_restart_point_options,
    pick_default_launch_mode,
    resolve_config_path_for_run,
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
