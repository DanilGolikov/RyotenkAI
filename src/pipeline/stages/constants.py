"""
Constants for pipeline stages.

Eliminates magic strings for context keys and stage names.
"""

from __future__ import annotations

from enum import StrEnum

from src.pipeline._types import CANONICAL_STAGE_ORDER, StageNames

__all__ = ["CANONICAL_STAGE_ORDER", "PipelineContextKeys", "StageNames"]


class PipelineContextKeys(StrEnum):
    """
    Flat (root-level) context keys written by the orchestrator or consumed by
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
