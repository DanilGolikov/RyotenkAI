"""
Training Managers

Single Responsibility managers for training operations.
Each manager handles ONE specific aspect of training.

Follows Single Responsibility Principle (SOLID).

Managers:
- DataBuffer: Checkpoint management for multi-phase training
- DataLoaderManager: Dataset loading and preparation
- ModelSaverManager: Model saving and checkpoint management
- MLflowManager: MLflow experiment tracking (self-hosted, free)
"""

from src.training.managers.data_buffer import (
    DataBuffer,
    PhaseState,
    PhaseStatus,
    PipelineState,
    list_available_runs,
)
from src.training.managers.data_loader import DataLoaderManager
from src.training.managers.mlflow_manager import MLflowManager, get_mlflow_manager
from src.training.managers.model_saver import ModelSaverManager

__all__ = [
    # DataBuffer (multi-phase checkpoint management)
    "DataBuffer",
    # Other managers
    "DataLoaderManager",
    "MLflowManager",
    "ModelSaverManager",
    "PhaseState",
    "PhaseStatus",
    "PipelineState",
    "get_mlflow_manager",
    "list_available_runs",
]
