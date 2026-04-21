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

__all__ = [
    "LaunchPreparationError",
    "LaunchPreparator",
    "PreparedAttempt",
]
