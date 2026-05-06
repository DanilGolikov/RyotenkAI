"""``EngineCapabilities`` — runtime mirror of ``engine.toml [capabilities]``.

The Pydantic model is the single source of truth for what an engine
declares it can do. Cross-field invariants (e.g. "supports_quantization=False
implies supported_quantizations is empty") are enforced via
``@model_validator``.

The drift detector ``packages/engines/scripts/check_engine_manifests.py``
(PR-10) cross-checks that every shipped engine's runtime ``get_capabilities()``
exactly matches its ``engine.toml [capabilities]`` block — preventing the
runtime and the manifest from getting out of sync.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


#: Wire-protocol dialect the engine speaks.
#:
#:   * ``"openai_compatible"`` — vLLM, SGLang, MAX, Ollama, llama-cpp-server.
#:     Our ``OpenAICompatibleInferenceClient`` talks to all of these.
#:   * ``"custom"`` — engine speaks its own protocol. PipelineConfig
#:     validator rejects this until/unless ``ModelClientFactory`` gains a
#:     non-OpenAI builder. Forward-looking flag.
ApiDialect = Literal["openai_compatible", "custom"]


class EngineCapabilities(BaseModel):
    """Declarative engine capabilities.

    Exactly mirrors ``engine.toml [capabilities]``. Authors define this
    once in TOML; ``EngineManifest`` parses it; the runtime class returns
    the same object from ``get_capabilities()``. Drift detector enforces
    parity at CI time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_dialect: ApiDialect = Field(
        description=(
            "Wire protocol the engine exposes once running. "
            "OpenAI-compatible engines work with our existing model client; "
            "'custom' is reserved for future engines that need a bespoke client."
        ),
    )
    supports_lora: bool = Field(
        description=(
            "True iff the engine accepts LoRA adapter paths at launch. "
            "vLLM today: True (with ``--enable-lora`` + ``--lora-modules``)."
        ),
    )
    supports_quantization: bool = Field(
        description="True iff the engine accepts a quantization mode CLI flag.",
    )
    supports_streaming: bool = Field(
        description=(
            "True iff the engine streams tokens via SSE / chunked HTTP. "
            "Affects whether evaluation plugins can use streaming-mode tests."
        ),
    )
    supports_tensor_parallel: bool = Field(
        description=(
            "True iff the engine supports multi-GPU tensor-parallel sharding. "
            "Drives ``tensor_parallel_size`` validation."
        ),
    )
    supported_quantizations: tuple[str, ...] = Field(
        default=(),
        description=(
            "Quantization modes the engine accepts (e.g. ['awq', 'gptq', 'fp8']). "
            "Used for fail-fast validation when user picks an unsupported mode."
        ),
    )
    supported_dtypes: tuple[str, ...] = Field(
        description=(
            "DType strings the engine accepts (e.g. ['bfloat16', 'float16']). "
            "Validated against user config at PipelineConfig load."
        ),
    )
    default_port: int = Field(
        ge=1,
        le=65535,
        description=(
            "Default TCP port the engine binds to inside its container. "
            "Provider may override via port mapping; this is just the engine's "
            "natural choice (vLLM defaults to 8000, TGI to 80, etc.)."
        ),
    )

    # ---- cross-field invariants ----

    @model_validator(mode="after")
    def _check_invariants(self) -> EngineCapabilities:
        # Quantization parity: claiming False but listing modes is a bug.
        if not self.supports_quantization and self.supported_quantizations:
            raise ValueError(
                f"capabilities.supports_quantization=False but "
                f"supported_quantizations is non-empty "
                f"({list(self.supported_quantizations)!r}). Either flip the "
                f"flag or clear the list."
            )
        # Reverse: claiming True with empty list is suspicious but not invalid
        # (some engines accept any string, validation deferred to runtime).
        # We don't reject — drift detector flags it as a warning.

        # Tuples must not contain duplicates (data integrity).
        if len(set(self.supported_quantizations)) != len(self.supported_quantizations):
            raise ValueError(
                f"capabilities.supported_quantizations contains duplicates: "
                f"{list(self.supported_quantizations)!r}"
            )
        if len(set(self.supported_dtypes)) != len(self.supported_dtypes):
            raise ValueError(
                f"capabilities.supported_dtypes contains duplicates: "
                f"{list(self.supported_dtypes)!r}"
            )
        # supported_dtypes must be non-empty — every engine must declare at
        # least one dtype it supports (otherwise no model can run).
        if not self.supported_dtypes:
            raise ValueError(
                "capabilities.supported_dtypes must declare at least one dtype "
                "(e.g. ['bfloat16']). An engine with no dtypes can launch nothing."
            )
        return self


__all__ = ("ApiDialect", "EngineCapabilities")
