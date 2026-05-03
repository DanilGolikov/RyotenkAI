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

    The docker image is pinned in
    :data:`src.inference.__about__.INFERENCE_IMAGES` (one unified
    image per :file:`docker/inference/Dockerfile` covers both LoRA
    merge and vLLM serve — the legacy two-container strategy is
    gone). Image versions are tied to releases — no user-facing
    field. Override via env
    ``RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_VLLM`` for CI / dev only.

    All fields here are runtime tuning knobs:
    - ``merge_before_deploy``: lifted from ``inference.common.lora``
      because whether/how to merge an adapter is engine-specific
      (vLLM supports both merged and on-the-fly LoRA; future engines
      may force one or the other).
    - ``tensor_parallel_size``, ``max_model_len``,
      ``gpu_memory_utilization``, ``quantization``, ``enforce_eager``:
      vLLM-specific options with best-practice defaults.
    """

    # LoRA handling (engine-specific — merging is something the
    # serving runtime decides, not a generic LoRA setting).
    merge_before_deploy: bool = Field(
        True,
        description=(
            "Merge the LoRA adapter into the base model checkpoint "
            "before serving via vLLM. ``True`` produces a single "
            "merged model (lower latency, larger disk); ``False`` "
            "loads the adapter at vLLM startup (higher latency, "
            "smaller disk)."
        ),
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
