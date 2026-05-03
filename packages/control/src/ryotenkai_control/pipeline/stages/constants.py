"""Constants for pipeline stages.

Eliminates magic strings for context keys and stage names. Importing this
module is intentionally cheap — no third-party deps, no stage classes —
so light-weight callers (CLI ``list-restart-points``, restart-point
queries, config validators) can pull these values without dragging the
training stack into sys.modules.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["CANONICAL_STAGE_ORDER", "PipelineContextKeys", "StageNames"]


class StageNames(StrEnum):
    """Stage names used as keys in pipeline context.

    Using StrEnum gives IDE autocomplete, type checking, and refactoring
    safety while still allowing direct use as dict keys (str-compatible).

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


class PipelineContextKeys(StrEnum):
    """Flat (root-level) context keys written by the orchestrator or consumed by
    multiple stages. Stage-namespaced sub-dicts live in context[StageNames.*].

    Using StrEnum provides the same benefits as StageNames:
    IDE autocomplete, type checking, and refactoring safety.

    Example:
        run = context.get(PipelineContextKeys.RUN)
        context[PipelineContextKeys.MLFLOW_MANAGER] = manager
    """

    RUN = "run"
    CONFIG_PATH = "config_path"
    MLFLOW_PARENT_RUN_ID = "mlflow_parent_run_id"
    MLFLOW_MANAGER = "mlflow_manager"
    DOCKER_IMAGE_SHA = "docker_image_sha"
    LOGICAL_RUN_ID = "logical_run_id"
    ATTEMPT_ID = "attempt_id"
    ATTEMPT_NO = "attempt_no"
    RUN_DIRECTORY = "run_directory"
    ATTEMPT_DIRECTORY = "attempt_directory"
    FORCED_STAGES = "forced_stages"
