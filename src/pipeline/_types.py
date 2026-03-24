"""
Lightweight stage name constants — no src.* imports, stdlib only.

Kept as a separate leaf module so that lightweight commands (e.g. list-restart-points)
can import stage names without triggering the full stages package init chain
(DatasetValidator → data loaders → utils.container → training stack).
"""

from __future__ import annotations

from enum import StrEnum


class StageNames(StrEnum):
    """
    Stage names used as keys in pipeline context.

    Using StrEnum allows:
    - IDE autocomplete
    - Type checking
    - Refactoring safety
    - Direct use as dict keys (str-compatible)

    Example:
        context[StageNames.GPU_DEPLOYER] = {...}
        deployer_ctx = context.get(StageNames.GPU_DEPLOYER, {})
    """

    DATASET_VALIDATOR = "Dataset Validator"
    GPU_DEPLOYER = "GPU Deployer"
    TRAINING_MONITOR = "Training Monitor"
    MODEL_RETRIEVER = "Model Retriever"
    MODEL_EVALUATOR = "Model Evaluator"
    INFERENCE_DEPLOYER = "Inference Deployer"


CANONICAL_STAGE_ORDER: tuple[StageNames, ...] = (
    StageNames.DATASET_VALIDATOR,
    StageNames.GPU_DEPLOYER,
    StageNames.TRAINING_MONITOR,
    StageNames.MODEL_RETRIEVER,
    StageNames.INFERENCE_DEPLOYER,
    StageNames.MODEL_EVALUATOR,
)
