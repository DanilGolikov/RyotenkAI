from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ...base import StrictBaseModel
from ..constants import (
    SINGLE_NODE_TRAINING_START_TIMEOUT_DEFAULT,
    SINGLE_NODE_TRAINING_START_TIMEOUT_MAX,
    SINGLE_NODE_TRAINING_START_TIMEOUT_MIN,
)


class SingleNodeTrainingConfig(StrictBaseModel):
    """
    Training-specific settings for single_node provider.

    Example:
        training:
          workspace_path: /home/user/workspace
          training_start_timeout: 30
    """

    workspace_path: str = Field(..., description="Remote workspace base directory")
    training_start_timeout: int = Field(
        SINGLE_NODE_TRAINING_START_TIMEOUT_DEFAULT,
        ge=SINGLE_NODE_TRAINING_START_TIMEOUT_MIN,
        le=SINGLE_NODE_TRAINING_START_TIMEOUT_MAX,
        description="Max wait for training start",
    )
    gpu_type: str | None = Field(None, description="GPU type for logging. Auto-detected if not set.")
    mock_mode: bool = Field(False, description="Mock mode for testing")

    # =========================================================================
    # EXECUTION MODE: docker-only (training runs in container runtime)
    # =========================================================================
    execution_mode: str = Field(
        "docker",
        description="How training is executed on single_node. Docker-only: must be 'docker'.",
    )

    # Docker runtime settings
    docker_image: str = Field(..., description="Docker image for training runtime (required).")
    docker_shm_size: str = Field(
        "16g",
        description="Docker shared memory size (--shm-size). Increase for dataloaders / large batches if needed.",
    )
    docker_container_name_prefix: str = Field(
        "ryotenkai_training",
        description="Docker container name prefix. Must start with 'ryotenkai_training' for TrainingMonitor compatibility.",
    )

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError(f"workspace_path must be absolute path, got '{v}'")
        return v

    @field_validator("execution_mode")
    @classmethod
    def validate_execution_mode(cls, v: str) -> str:
        allowed = {"docker"}
        v_norm = v.strip().lower()
        if v_norm not in allowed:
            raise ValueError(f"execution_mode must be one of {sorted(allowed)}, got '{v}'")
        return v_norm

    @model_validator(mode="after")
    def validate_docker_settings(self) -> SingleNodeTrainingConfig:
        if not self.docker_container_name_prefix.startswith("ryotenkai_training"):
            raise ValueError(
                "providers.single_node.training.docker_container_name_prefix must start with 'ryotenkai_training' "
                "to be detectable by TrainingMonitor"
            )
        return self


__all__ = [
    "SingleNodeTrainingConfig",
]
