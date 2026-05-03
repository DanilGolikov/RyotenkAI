"""
mlflow_manager — backward-compatible package facade.

All names previously importable from
`src.training.managers.mlflow_manager` remain importable unchanged.

Internal structure:
    setup.py         ← MLflowSetupMixin
    run_lifecycle.py ← MLflowRunLifecycleMixin
    logging_core.py  ← MLflowLoggingMixin
    manager.py       ← MLflowManager (thin coordinator + all delegation)
"""

from __future__ import annotations

from src.training.managers.mlflow_manager.manager import MLflowManager, get_mlflow_manager
from src.training.managers.mlflow_manager.setup import MLflowSetupMixin
from src.training.managers.mlflow_manager.run_lifecycle import MLflowRunLifecycleMixin
from src.training.managers.mlflow_manager.logging_core import MLflowLoggingMixin

__all__ = [
    "MLflowManager",
    "get_mlflow_manager",
    "MLflowSetupMixin",
    "MLflowRunLifecycleMixin",
    "MLflowLoggingMixin",
]
