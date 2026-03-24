"""
RunPod provider configuration schema.

Source of truth for RunPod provider config lives in this package.
Runtime code should import `RunPodProviderConfig` from `src.config.providers.runpod`.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...base import StrictBaseModel

# NOTE: These are runtime imports required for Pydantic nested-model validation.
# Ruff's TC001 ("move import into TYPE_CHECKING") is not applicable here.
from .cleanup import RunPodCleanupConfig  # noqa: TC001
from .connect import RunPodConnectConfig  # noqa: TC001
from .inference import RunPodPodsInferenceConfig  # noqa: TC001
from .training import RunPodTrainingConfig  # noqa: TC001


class RunPodProviderConfig(StrictBaseModel):
    """
    Configuration for RunPod cloud provider.

    Example YAML:
        providers:
          runpod:
            connect:
              ssh:
                key_path: ~/.ssh/id_ed25519_runpod
            cleanup:
              auto_delete_pod: true
              keep_pod_on_error: false
              on_interrupt: true
            training:
              gpu_type: "NVIDIA A40"
              cloud_type: ALL
              image_name: ryotenkai/ryotenkai-training-runtime:latest
              container_disk_gb: 100
              volume_disk_gb: 20
            inference:
              volume: { name: helix-hf-cache, data_center_id: US-KS-2, size_gb: 200 }
              pod: { image_name: "ryotenkai/inference-vllm:latest", gpu_count: 1, ports: ["22/tcp"] }
              serve: { port: 8000 }
    """

    # Connection configuration (SSH)
    connect: RunPodConnectConfig

    # Cleanup policy
    cleanup: RunPodCleanupConfig = Field(..., description="Cleanup policy")

    # Training settings
    training: RunPodTrainingConfig

    # Inference (Pods) settings (may be empty when RunPod is used for training only)
    inference: RunPodPodsInferenceConfig = Field(
        ...,
        description="Inference settings for RunPod Pods (may be empty when unused).",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunPodProviderConfig:
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)


__all__ = [
    "RunPodProviderConfig",
]
