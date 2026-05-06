"""``VLLMEngineRuntime`` — IInferenceEngine implementation for vLLM.

Ported from ``packages/providers/.../inference/vllm/engine.py`` (the
legacy ``VLLMEngine`` class). Differences:

  * Implements :class:`IInferenceEngine` Protocol.
  * Returns structured :class:`LaunchSpec` instead of a docker shell
    string. Provider formats the spec into ``docker run …`` (or k8s
    ContainerSpec, etc.) — engines no longer assume docker.
  * Adds :meth:`get_capabilities` (mirrors ``engine.toml``).
  * Adds :meth:`validate_config` carrying the
    ``merge_before_deploy=False`` MVP gate that used to live in the
    provider __init__.
  * No ``json.dumps(model_path)``  — ``LaunchSpec.args`` is already a
    tuple of pre-split arguments; provider quotes for shell when
    formatting (or for k8s, just passes them through).

Pure command-builder; no IO, no async.
"""

from __future__ import annotations

from typing import ClassVar

from ryotenkai_engines.capabilities import EngineCapabilities
from ryotenkai_engines.interfaces import (
    BaseEngineConfig,
    IInferenceEngine,  # noqa: F401  — runtime-checkable Protocol; isinstance check uses it
    LaunchSpec,
)
from ryotenkai_engines.vllm.config import VLLMEngineConfig
from ryotenkai_shared.utils.result import AppError, Err, Ok, Result


class VLLMEngineRuntime:
    """vLLM OpenAI-compatible server runtime.

    See :class:`IInferenceEngine` for the Protocol contract.
    """

    engine_id: ClassVar[str] = "vllm"
    config_class: ClassVar[type[BaseEngineConfig]] = VLLMEngineConfig

    # ---- capabilities ----

    def get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities — MUST mirror engine.toml [capabilities]."""
        return EngineCapabilities(
            api_dialect="openai_compatible",
            supports_lora=True,
            supports_quantization=True,
            supports_streaming=True,
            supports_tensor_parallel=True,
            supported_quantizations=("awq", "gptq", "fp8", "bitsandbytes"),
            supported_dtypes=("bfloat16", "float16", "float32"),
            default_port=8000,
        )

    # ---- launch spec ----

    def build_launch_spec(
        self,
        *,
        cfg: BaseEngineConfig,
        image: str,
        container_name: str,
        port: int,
        workspace_host_path: str,
        model_path_in_container: str,
    ) -> LaunchSpec:
        """Build a vLLM launch spec.

        Args mirror :meth:`IInferenceEngine.build_launch_spec`. The
        ``cfg`` is narrowed to :class:`VLLMEngineConfig` (the registry
        guarantees this at validation time).
        """
        if not isinstance(cfg, VLLMEngineConfig):
            raise TypeError(
                f"VLLMEngineRuntime.build_launch_spec expected VLLMEngineConfig, "
                f"got {type(cfg).__name__}"
            )

        args: list[str] = [
            "serve",
            model_path_in_container,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(cfg.tensor_parallel_size),
            "--max-model-len",
            str(cfg.max_model_len),
            "--gpu-memory-utilization",
            str(cfg.gpu_memory_utilization),
        ]
        if cfg.quantization:
            args.extend(["--quantization", cfg.quantization])
        if cfg.enforce_eager:
            args.append("--enforce-eager")

        # vLLM's HF-cache plumbing — providers don't need to know.
        env = {
            "HF_HOME": "/workspace/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/workspace/hf_cache",
            "TRANSFORMERS_CACHE": "/workspace/hf_cache",
        }

        return LaunchSpec(
            image=image,
            container_name=container_name,
            args=tuple(args),
            env=env,
            port=port,
            volumes=((workspace_host_path, "/workspace"),),
        )

    # ---- healthcheck ----

    def build_healthcheck_command(self, *, host: str, port: int) -> str:
        """curl-based readiness probe.

        Exit code 0 ⇒ engine is up. Echoes ``1`` on success / ``0`` on
        failure (legacy convention preserved for provider parsing).
        """
        url = f"http://{host}:{port}/v1/models"
        return f"curl -s -f -m 5 {url} >/dev/null 2>&1 && echo 1 || echo 0"

    # ---- endpoint url ----

    def build_default_endpoint_url(self, *, host: str, port: int) -> str:
        """OpenAI-compatible base URL — what ModelClientFactory dials."""
        return f"http://{host}:{port}/v1"

    # ---- validate_config ----

    def validate_config(self, cfg: BaseEngineConfig) -> Result[None, AppError]:
        """Engine-side invariants (post-Pydantic).

        Today: rejects ``merge_before_deploy=False`` (MVP gate, lifted
        from the legacy ``SingleNodeInferenceProvider.__init__`` so
        every provider gets the same enforcement for free).

        Future: ``cfg.quantization`` cross-check against capabilities
        is also enforced here once the registry's ``validate_config``
        wiring lands in PR-7.
        """
        if not isinstance(cfg, VLLMEngineConfig):
            return Err(
                AppError(
                    message=(
                        f"VLLMEngineRuntime.validate_config expected "
                        f"VLLMEngineConfig, got {type(cfg).__name__}"
                    ),
                    code="VLLM_CONFIG_TYPE_MISMATCH",
                )
            )

        if not cfg.merge_before_deploy:
            return Err(
                AppError(
                    message=(
                        "vLLM engine MVP requires merge_before_deploy=True. "
                        "Live LoRA adapter loading (False) is on the roadmap."
                    ),
                    code="VLLM_LIVE_LORA_NOT_SUPPORTED",
                    details={"merge_before_deploy": cfg.merge_before_deploy},
                )
            )

        # Quantization cross-check: if user picked a mode, it must be
        # in ``supported_quantizations``. Defensive — Pydantic doesn't
        # restrict the string today, so we catch typos here.
        if cfg.quantization is not None:
            caps = self.get_capabilities()
            if cfg.quantization not in caps.supported_quantizations:
                return Err(
                    AppError(
                        message=(
                            f"vLLM does not support quantization "
                            f"{cfg.quantization!r}. Supported modes: "
                            f"{list(caps.supported_quantizations)!r}."
                        ),
                        code="VLLM_QUANTIZATION_UNSUPPORTED",
                        details={
                            "requested": cfg.quantization,
                            "supported": list(caps.supported_quantizations),
                        },
                    )
                )

        return Ok(None)


__all__ = ("VLLMEngineRuntime",)
