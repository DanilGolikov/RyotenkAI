"""
Stage managers - helper classes for pipeline stages.

These are NOT stages themselves, but utilities used by stages.
"""

from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
from src.pipeline.stages.managers.log_manager import LogManager, RunPodLogManager

__all__ = [
    "LogManager",
    "RunPodLogManager",  # Backward compatibility alias
    "TrainingDeploymentManager",
]
