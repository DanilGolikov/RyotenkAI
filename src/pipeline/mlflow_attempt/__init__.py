"""Per-attempt MLflow lifecycle management.

Encapsulates MLflow setup, preflight, root/attempt run open/close, and
teardown — extracted from PipelineOrchestrator.
"""

from src.pipeline.mlflow_attempt.manager import (
    MLflowAttemptManager,
    MLflowPreflightError,
)

__all__ = [
    "MLflowAttemptManager",
    "MLflowPreflightError",
]
