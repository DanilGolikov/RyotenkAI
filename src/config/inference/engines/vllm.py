from __future__ import annotations

from pydantic import Field

from ...base import StrictBaseModel
from ..constants import (
    GPU_MEMORY_UTILIZATION_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MAX_MODEL_LEN_MIN,
)


class InferenceVLLMEngineConfig(StrictBaseModel):
    """
    vLLM runtime config.

    Docker images (two-container strategy, used by inference.provider='single_node'):
    - merge_image: ephemeral container for LoRA merge (transformers+peft+accelerate)
    - serve_image: long-running container for vLLM serve (vLLM runtime only)

    Notes:
    - For runpod (Pods + unified image) these fields are not used at runtime.
    - We keep them in the schema for compatibility / explicit version tracking, but require them conditionally via InferenceConfig
      validators only when the selected provider needs them.

    OPTIONAL fields (runtime parameters with best-practice defaults):
    - tensor_parallel_size, max_model_len, etc.
    """

    # =========================================================================
    # Docker images (two-container strategy; required for inference.provider='single_node')
    # =========================================================================
    merge_image: str | None = Field(
        default=None,
        description="Docker image for LoRA merge job. Required for inference.provider='single_node'.",
    )
    serve_image: str | None = Field(
        default=None,
        description="Docker image for vLLM serve. Required for inference.provider='single_node'.",
    )

    # vLLM runtime config
    tensor_parallel_size: int = Field(
        1, ge=1, description="Optional: Number of GPUs for tensor parallelism (default: 1)"
    )
    max_model_len: int = Field(
        MAX_MODEL_LEN_DEFAULT,
        ge=MAX_MODEL_LEN_MIN,
        description="Optional: Maximum sequence length (default: 4096)",
    )
    gpu_memory_utilization: float = Field(
        GPU_MEMORY_UTILIZATION_DEFAULT,
        gt=0.0,
        le=1.0,
        description="Optional: GPU memory utilization (default: 0.90)",
    )
    quantization: str | None = Field(None, description="Optional: Quantization (null, 'awq', 'gptq')")
    enforce_eager: bool = Field(False, description="Optional: Enforce eager execution for debugging (default: False)")


__all__ = [
    "InferenceVLLMEngineConfig",
]
