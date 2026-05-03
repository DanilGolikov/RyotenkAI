"""Stage artifact infrastructure — Typed Envelope pattern.

Each pipeline stage writes its own JSON artifact with a unified envelope:
    {stage, status, started_at, duration_seconds, error, data}

The `data` field is typed via TypedDict schemas defined in schemas.py.
"""

from src.pipeline.artifacts.base import (
    StageArtifactCollector,
    StageArtifactEnvelope,
    save_stage_artifact,
)
from src.pipeline.artifacts.schemas import (
    DeploymentArtifactData,
    EvalArtifactData,
    EvalPluginData,
    InferenceArtifactData,
    ModelArtifactData,
    TrainingArtifactData,
    ValidationArtifactData,
    ValidationDatasetData,
    ValidationPluginData,
)

__all__ = [
    "DeploymentArtifactData",
    "EvalArtifactData",
    "EvalPluginData",
    "InferenceArtifactData",
    "ModelArtifactData",
    "StageArtifactCollector",
    "StageArtifactEnvelope",
    "TrainingArtifactData",
    "ValidationArtifactData",
    "ValidationDatasetData",
    "ValidationPluginData",
    "save_stage_artifact",
]
