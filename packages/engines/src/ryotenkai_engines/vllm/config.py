"""``VLLMEngineConfig`` — typed config for the vLLM engine.

Ported from ``packages/shared/.../config/inference/engines/vllm.py``
(``InferenceVLLMEngineConfig``). Differences:

  * Renamed ``InferenceVLLMEngineConfig`` → ``VLLMEngineConfig``
    (engines are inference by definition; the prefix was redundant).
  * Added ``kind: Literal["vllm"]`` discriminator field.
  * Subclasses :class:`BaseEngineConfig` (gives ``model_config =
    {"extra": "forbid"}`` for free).
  * vLLM-specific constants are inlined — engines is a leaf workspace
    member and cannot import from ``ryotenkai_shared.config.inference``.

The legacy class stays in shared/config until PR-6 wires this one in
and PR-9 deletes the shim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ryotenkai_engines.interfaces import BaseEngineConfig

# vLLM engine constants — inlined here (engines is a leaf, can't import
# from shared.config.inference.constants where these originated).
MAX_MODEL_LEN_MIN = 256
MAX_MODEL_LEN_DEFAULT = 4096
GPU_MEMORY_UTILIZATION_DEFAULT = 0.90


class VLLMEngineConfig(BaseEngineConfig):
    """vLLM runtime config.

    The container image is resolved by the engine registry (convention
    default: ``ryotenkai/inference-vllm:{engine_version}``; overridable
    via env / provider / manifest). This class carries only runtime
    tuning knobs that the engine accepts as CLI flags.

    LoRA handling:
      * ``merge_before_deploy=True`` (default) merges the adapter into
        the base checkpoint before serving — single merged model, lower
        latency, larger disk.
      * ``merge_before_deploy=False`` defers the merge — vLLM loads the
        adapter at startup. MVP: ``False`` is rejected by
        :meth:`VLLMEngineRuntime.validate_config`.
    """

    kind: Literal["vllm"] = "vllm"

    # LoRA handling — engine-specific (vLLM supports both modes; future
    # engines may force one).
    merge_before_deploy: bool = Field(
        True,
        description=(
            "Merge LoRA adapter into the base model before serving. "
            "True (default): single merged model — lower inference "
            "latency. False: vLLM loads the adapter at startup — "
            "smaller disk but higher launch latency. MVP rejects False."
        ),
    )

    # vLLM-specific runtime tuning
    tensor_parallel_size: int = Field(
        1,
        ge=1,
        description="Number of GPUs for tensor parallelism.",
    )
    max_model_len: int = Field(
        MAX_MODEL_LEN_DEFAULT,
        ge=MAX_MODEL_LEN_MIN,
        description="Maximum sequence length the engine will accept.",
    )
    gpu_memory_utilization: float = Field(
        GPU_MEMORY_UTILIZATION_DEFAULT,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU memory the engine may occupy.",
    )
    quantization: str | None = Field(
        None,
        description=(
            "Optional quantization mode (e.g. 'awq', 'gptq', 'fp8'). "
            "Validated against :attr:`EngineCapabilities.supported_quantizations` "
            "at engine.validate_config time."
        ),
    )
    enforce_eager: bool = Field(
        False,
        description="Disable CUDA graphs — debugging knob (slower, more verbose).",
    )


__all__ = (
    "VLLMEngineConfig",
    "MAX_MODEL_LEN_MIN",
    "MAX_MODEL_LEN_DEFAULT",
    "GPU_MEMORY_UTILIZATION_DEFAULT",
)
