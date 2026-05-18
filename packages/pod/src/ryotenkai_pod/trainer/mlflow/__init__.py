"""pod/trainer/mlflow — thin façade after M7 cleanup.

Only HF wiring + metrics buffer + model publisher remain. The old
``MLflowManager`` god-class (and all its subcomponents) has been
deleted per ``docs/plans/vectorized-fluttering-mist.md``.

``IMLflowManager`` is preserved as a typing alias for ``Any`` so the
small number of remaining call sites with ``mlflow_manager: IMLflowManager | None``
signatures can keep compiling while they are migrated to typed events.

Canonical write-path Protocols (``ITrackingClient``, ``IMetricSink``,
``IArtifactSink``, ``IRunQuery``, ``IModelRegistry``,
``IJournalUploader``, ``IPromptRegistry``) live in
``ryotenkai_shared.infrastructure.mlflow.protocols`` and should be used
for all new code.
"""

from __future__ import annotations

from typing import Any

from ryotenkai_pod.trainer.mlflow.hf_wiring import HFMlflowWiring
from ryotenkai_pod.trainer.mlflow.model_publisher import ModelPublisher

# Type alias kept as ``Any`` so remaining ``mlflow_manager: IMLflowManager | None``
# parameters compile. All real code now passes ``None`` (Pattern A).
IMLflowManager = Any


__all__ = [
    "HFMlflowWiring",
    "IMLflowManager",
    "ModelPublisher",
]
