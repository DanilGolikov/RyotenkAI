"""DatasetValidator stage and its artifact-accumulator collaborator.

Both live together because the stage owns the validation event protocol
(``DatasetValidatorEventCallbacks``) and ``ValidationArtifactManager`` is
the only callback consumer the orchestrator wires up. Keeping them as
sibling modules in one package makes the contract obvious — nobody has
to grep across ``stages/`` and a separate ``validation/`` package to
understand the flow.
"""

from src.pipeline.stages.dataset_validator.artifact_manager import ValidationArtifactManager
from src.pipeline.stages.dataset_validator.stage import (
    DatasetValidator,
    DatasetValidatorEventCallbacks,
)

__all__ = [
    "DatasetValidator",
    "DatasetValidatorEventCallbacks",
    "ValidationArtifactManager",
]
