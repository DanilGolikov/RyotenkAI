"""Per-attempt MLflow lifecycle management.

Encapsulates MLflow setup, preflight, root/attempt run open/close, and
teardown — extracted from PipelineOrchestrator.

.. deprecated:: Phase M7 (cleanup backlog)
    Phase M2 introduced
    :mod:`ryotenkai_control.pipeline.mlflow.lifecycle` as the
    canonical home for the open/preflight/finalize flow. This
    package remains because :class:`PipelineOrchestrator` still
    instantiates :class:`MLflowAttemptManager` (via importlib
    indirection that loads
    :class:`ryotenkai_pod.trainer.managers.mlflow_manager.MLflowManager`)
    for the legacy wide ``IMLflowManager`` protocol.

    TODO(M7-cleanup): replace ``MLflowAttemptManager`` callsites in
    :class:`PipelineOrchestrator` with direct use of
    ``pipeline/mlflow/lifecycle/`` (``RunLifecycleCoord`` +
    ``MlflowFinalizer``), then delete this directory and the legacy
    ``trainer/managers/mlflow_manager/`` package together.
"""

from ryotenkai_control.pipeline.mlflow_attempt.manager import (
    MLflowAttemptManager,
    MLflowManagerNotInitializedError,
)

__all__ = [
    "MLflowAttemptManager",
    "MLflowManagerNotInitializedError",
]
