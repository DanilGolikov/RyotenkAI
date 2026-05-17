"""DatasetValidator stage and its artifact-accumulator collaborator.

Both live together because the stage owns the per-plugin / per-dataset
lifecycle that :class:`ValidationArtifactManager` records. After the
Phase 4 event-system unification (2026-05-16) the legacy
``DatasetValidatorEventCallbacks`` dataclass is gone — typed events
flow through :class:`IEventEmitter` and the artifact recorder is
passed as a typed collaborator (no callback wrapping).
"""

from ryotenkai_control.pipeline.stages.dataset_validator.artifact_manager import ValidationArtifactManager
from ryotenkai_control.pipeline.stages.dataset_validator.stage import DatasetValidator

__all__ = [
    "DatasetValidator",
    "ValidationArtifactManager",
]
