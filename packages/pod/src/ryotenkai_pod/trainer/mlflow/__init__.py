"""pod/trainer/mlflow — thin facade after M7 cleanup.

Only HF wiring + metrics buffer + model publisher remain. The old
``MLflowManager`` god-class (and all its subcomponents) has been
deleted per ``docs/plans/vectorized-fluttering-mist.md``.

Canonical write-path Protocols (``ITrackingClient``, ``IMetricSink``,
``IArtifactSink``, ``IRunQuery``, ``IModelRegistry``,
``IJournalUploader``, ``IPromptRegistry``) live in
``ryotenkai_shared.infrastructure.mlflow.protocols`` and should be used
for all new code.
"""

from __future__ import annotations

from ryotenkai_pod.trainer.mlflow.hf_wiring import HFMlflowWiring
from ryotenkai_pod.trainer.mlflow.model_publisher import ModelPublisher

__all__ = [
    "HFMlflowWiring",
    "ModelPublisher",
]
