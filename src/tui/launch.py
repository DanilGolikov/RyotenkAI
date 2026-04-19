"""Backward-compatibility shim. Real module lives in src/pipeline/launch.

Kept to avoid churn in existing TUI imports. New code should import from
src.pipeline.launch directly.
"""

from __future__ import annotations

from src.pipeline.launch import (
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
    "ActiveLaunch",
    "LaunchLogLevel",
    "LaunchMode",
    "LaunchRequest",
    "LaunchResult",
    "LaunchStatus",
    "MODE_FRESH",
    "MODE_NEW_RUN",
    "MODE_RESTART",
    "MODE_RESUME",
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
