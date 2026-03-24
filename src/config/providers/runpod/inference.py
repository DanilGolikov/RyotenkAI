from __future__ import annotations

from pydantic import Field, model_validator

from ...base import StrictBaseModel
from ...inference.common import InferenceLLMConfig
from ..constants import (
    RUNPOD_CONTAINER_DISK_GB_DEFAULT,
    RUNPOD_CONTAINER_DISK_GB_MAX,
    RUNPOD_CONTAINER_DISK_GB_MIN,
    RUNPOD_POD_VOLUME_GB_DEFAULT,
    RUNPOD_SERVE_PORT_DEFAULT,
    RUNPOD_SERVE_PORT_MAX,
    RUNPOD_SERVE_PORT_MIN,
    RUNPOD_VOLUME_SIZE_GB_DEFAULT,
    RUNPOD_VOLUME_SIZE_GB_MAX,
    RUNPOD_VOLUME_SIZE_GB_MIN,
)


class RunPodNetworkVolumeConfig(StrictBaseModel):
    """
    Persistent RunPod Network Volume configuration (for HF cache + inference artifacts).

    Notes (RunPod semantics):
    - Network volumes live independently of Pods.
    - For Pods, network volumes are only available in Secure Cloud.
    - Network volumes must be attached during Pod deployment.
    """

    id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional existing network volume id (preferred for stability).",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-friendly volume name (used for lookup; if not found, used for auto-create).",
    )
    data_center_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "RunPod datacenter id for the Network Volume.\n\n"
            "Required when `id` is not provided (RunPod REST API requires `dataCenterId` at creation time). "
            "Example: 'US-KS-2', 'EU-RO-1'."
        ),
    )
    size_gb: int = Field(
        RUNPOD_VOLUME_SIZE_GB_DEFAULT,
        ge=RUNPOD_VOLUME_SIZE_GB_MIN,
        le=RUNPOD_VOLUME_SIZE_GB_MAX,
        description="Network volume size in GB (used for auto-create; can be increased later, cannot be decreased).",
    )

    @model_validator(mode="after")
    def _run_model_validators(self) -> RunPodNetworkVolumeConfig:
        # Normalize whitespace-only values to None.
        if isinstance(self.id, str):
            self.id = self.id.strip() or None
        if isinstance(self.data_center_id, str):
            self.data_center_id = self.data_center_id.strip() or None

        # If id is not provided, we may need to auto-create, which requires a datacenter.
        if not self.id and not self.data_center_id:
            raise ValueError(
                "RunPod network volume config requires either providers.runpod.inference.volume.id "
                "or providers.runpod.inference.volume.data_center_id."
            )
        return self


class RunPodInferencePodConfig(StrictBaseModel):
    """RunPod Pod settings for vLLM inference runtime."""

    name_prefix: str = Field(
        "ryotenkai-vllm-pod",
        min_length=1,
        description="Prefix for Pod name (a deterministic suffix will be added).",
    )

    image_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Docker image for the Pod inference runtime. "
            "Recommended: push `docker/inference/Dockerfile` as `<user>/inference-vllm:<tag>`."
        ),
    )

    # Compute / placement (REST API: gpuTypeIds + gpuCount)
    gpu_type_ids: list[str] = Field(
        default_factory=lambda: [
            "NVIDIA RTX 2000 Ada Generation",
            "NVIDIA RTX 4000 Ada Generation",
            "NVIDIA RTX A4000",
            "NVIDIA RTX A4500",
            "NVIDIA L4",
            "NVIDIA RTX A5000",
            "NVIDIA GeForce RTX 3090",
        ],
        description="Preferred GPU types (first available will be used).",
    )
    gpu_count: int = Field(1, ge=1, le=8, description="GPUs per Pod.")
    allowed_cuda_versions: list[str] | None = Field(
        default=None,
        description="Optional CUDA version filter (e.g. ['12.8','12.7']).",
    )

    # Storage / networking
    container_disk_gb: int = Field(
        RUNPOD_CONTAINER_DISK_GB_DEFAULT,
        ge=RUNPOD_CONTAINER_DISK_GB_MIN,
        le=RUNPOD_CONTAINER_DISK_GB_MAX,
        description="Container disk size in GB.",
    )
    volume_disk_gb: int = Field(
        RUNPOD_POD_VOLUME_GB_DEFAULT,
        ge=0,
        description=(
            "Pod Volume size in GB (mounted at /workspace; persists between stop/start of the same pod). "
            "Used only when network volume is not configured. "
            "When a network volume is also configured, RunPod gives it priority and this field is ignored."
        ),
    )
    ports: list[str] = Field(
        default_factory=lambda: ["22/tcp"],
        description="Exposed ports, e.g. ['22/tcp'] or ['22/tcp','8000/http'].",
    )


class RunPodInferenceServeConfig(StrictBaseModel):
    """vLLM server settings inside the Pod container."""

    port: int = Field(
        RUNPOD_SERVE_PORT_DEFAULT,
        ge=RUNPOD_SERVE_PORT_MIN,
        le=RUNPOD_SERVE_PORT_MAX,
        description="vLLM port inside the Pod (SSH tunnel target).",
    )


class RunPodPodsInferenceConfig(StrictBaseModel):
    """
    Inference settings for RunPod Pods (Pod + Network Volume).

    Lives under:
      providers:
        runpod:
          inference: { ... }

    NOTE:
    `volume` and `pod` are required only when `inference.provider='runpod'` is active.
    They may be omitted when RunPod is used for training only.
    """

    volume: RunPodNetworkVolumeConfig | None = Field(
        default=None,
        description="Persistent network volume settings (required for runpod inference).",
    )
    pod: RunPodInferencePodConfig | None = Field(
        default=None,
        description="Pod runtime settings (required for runpod inference).",
    )
    serve: RunPodInferenceServeConfig = Field(
        default_factory=RunPodInferenceServeConfig,  # type: ignore[arg-type]
        description="vLLM serve settings inside the Pod.",
    )
    llm: InferenceLLMConfig = Field(
        default_factory=InferenceLLMConfig,  # type: ignore[arg-type]
        description="LLM execution settings (system prompt, etc.).",
    )


__all__ = [
    "RunPodInferencePodConfig",
    "RunPodInferenceServeConfig",
    "RunPodNetworkVolumeConfig",
    "RunPodPodsInferenceConfig",
]
