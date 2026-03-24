from __future__ import annotations

from pydantic import Field, field_validator

from ...base import StrictBaseModel
from ..constants import (
    RUNPOD_TRAINING_VOLUME_DISK_GB_DEFAULT,
)


class RunPodTrainingConfig(StrictBaseModel):
    """
    Training-specific settings for RunPod provider.

    New structure (aligns with single_node):
        providers:
          runpod:
            connect: ...
            cleanup: ...
            training:
              gpu_type: "NVIDIA A40"
              cloud_type: "ALL"
              image_name: "ryotenkai/ryotenkai-training-runtime:latest"
              container_disk_gb: 100
              volume_disk_gb: 20
              ports: "8888/http,22/tcp"
              template_id: null
    """

    # GPU configuration
    gpu_type: str = Field(..., description="REQUIRED: GPU type to request (RunPod GPU ID, e.g. 'NVIDIA A40').")
    cloud_type: str = Field("ALL", description="Cloud type: ALL, SECURE, COMMUNITY")

    # Container configuration (pod runtime image)
    # IMPORTANT: training is docker-only. The pod image MUST include all required deps + runtime contract checker.
    image_name: str = Field(..., description="REQUIRED: Docker image for training pod (prebuilt runtime).")
    container_disk_gb: int = Field(100, ge=10, description="Container disk size in GB")
    volume_disk_gb: int = Field(
        RUNPOD_TRAINING_VOLUME_DISK_GB_DEFAULT,
        ge=0,
        description="Volume disk size in GB",
    )
    ports: str = Field("8888/http,22/tcp", description="Port mappings")
    template_id: str | None = Field(None, description="Optional template ID")

    @field_validator("cloud_type")
    @classmethod
    def validate_cloud_type(cls, v: str) -> str:
        """Validate cloud type."""
        allowed = ["ALL", "SECURE", "COMMUNITY"]
        if v.upper() not in allowed:
            raise ValueError(f"cloud_type must be one of {allowed}")
        return v.upper()


__all__ = [
    "RunPodTrainingConfig",
]
