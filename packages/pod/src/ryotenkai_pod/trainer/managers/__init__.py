"""
Training Managers

Single Responsibility managers for training operations.
Each manager handles ONE specific aspect of training.

Follows Single Responsibility Principle (SOLID).

Managers:
- DataBuffer: Checkpoint management for multi-phase training
- DataLoaderManager: Dataset loading and preparation
- ModelSaverManager: Model saving and checkpoint management

(``MLflowManager`` deleted in M7 — Pattern A delegates MLflow logging
to HF MLflowCallback; lifecycle is owned by control's
``pipeline/mlflow/lifecycle`` package.)
"""

from ryotenkai_pod.trainer.managers.data_buffer import (
    DataBuffer,
    PhaseState,
    PhaseStatus,
    PipelineState,
    list_available_runs,
)
from ryotenkai_pod.trainer.managers.data_loader import DataLoaderManager
from ryotenkai_pod.trainer.managers.model_saver import ModelSaverManager

__all__ = [
    # DataBuffer (multi-phase checkpoint management)
    "DataBuffer",
    # Other managers
    "DataLoaderManager",
    "ModelSaverManager",
    "PhaseState",
    "PhaseStatus",
    "PipelineState",
    "list_available_runs",
]
