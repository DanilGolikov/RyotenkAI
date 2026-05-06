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
    PreparePlan,
    PrepareStep,
)
from ryotenkai_engines.vllm.config import VLLMEngineConfig
from ryotenkai_shared.utils.result import AppError, Err, Ok, Result

# ---------------------------------------------------------------------------
# Module-level prepare constants — engine concerns (lifted out of the
# legacy SingleNodeInferenceProvider during PR-16).
# ---------------------------------------------------------------------------

#: Hard wall-clock timeout for the LoRA merge container.
_MERGE_TIMEOUT_S = 3600

#: stdout marker that ``merge_lora.py`` prints on success. Provider checks
#: for this substring as defense-in-depth on top of exit-code-only.
_MERGE_SUCCESS_MARKER = "MERGE_SUCCESS"

#: Path inside the vLLM image where ``merge_lora.py`` is baked
#: (PR-15 Dockerfile: ``COPY merge_lora.py /opt/helix/merge_lora.py``).
_MERGE_SCRIPT_IN_CONTAINER = "/opt/helix/merge_lora.py"

#: Container path for HF model cache (shared by merge + serve).
_HF_CACHE_DIR_IN_CONTAINER = "/workspace/hf_cache"


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
            requires_prepare=True,
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

    # ---- prepare_model (PR-16) ----

    def prepare_model(
        self,
        *,
        cfg: BaseEngineConfig,
        base_model: str,
        adapter_path_in_container: str | None,
        workspace_host_path: str,
        run_id: str,
        trust_remote_code: bool,
    ) -> Result[PreparePlan, AppError]:
        """Build a 1-step LoRA-merge plan, or an empty plan if no merge needed.

        Three branches:

        1. ``cfg.merge_before_deploy=False`` — engine is configured to load
           LoRA on the fly (post-MVP). Returns ``Ok(PreparePlan.empty())``.
           ``validate_config`` rejects this today; the branch is here for
           when the MVP gate is lifted.

        2. ``adapter_path_in_container is None`` — no adapter was supplied;
           serve the base model directly. Returns ``Ok(PreparePlan.empty())``.

        3. Adapter present + merge required — return a single ``merge_lora``
           step that runs ``/opt/helix/merge_lora.py`` (baked into the
           image at PR-15) inside the same vLLM image, writing to
           ``/workspace/runs/{run_id}/model``.

        The provider:
          * formats this step into ``docker run …`` via ``format_prepare_step``
          * polls logs for ``MERGE_SUCCESS``
          * verifies ``{output}/config.json`` exists
          * passes ``plan.final_model_path`` to ``build_launch_spec``
        """
        if not isinstance(cfg, VLLMEngineConfig):
            return Err(
                AppError(
                    message=(
                        f"VLLMEngineRuntime.prepare_model expected "
                        f"VLLMEngineConfig, got {type(cfg).__name__}"
                    ),
                    code="VLLM_CONFIG_TYPE_MISMATCH",
                )
            )

        # Branch 1 + 2 — empty plan, no merge.
        if not cfg.merge_before_deploy or adapter_path_in_container is None:
            return Ok(PreparePlan.empty())

        # Branch 3 — single merge step.
        output_in_container = f"/workspace/runs/{run_id}/model"

        args: list[str] = [
            _MERGE_SCRIPT_IN_CONTAINER,
            "--base-model", base_model,
            "--adapter", adapter_path_in_container,
            "--output", output_in_container,
            "--cache-dir", _HF_CACHE_DIR_IN_CONTAINER,
        ]
        if trust_remote_code:
            args.append("--trust-remote-code")

        merge_step = PrepareStep(
            name="merge_lora",
            image=None,  # provider falls through to serve image
            entrypoint=("python3",),
            args=tuple(args),
            env={
                "HF_HOME": _HF_CACHE_DIR_IN_CONTAINER,
                "HUGGINGFACE_HUB_CACHE": _HF_CACHE_DIR_IN_CONTAINER,
                "TRANSFORMERS_CACHE": _HF_CACHE_DIR_IN_CONTAINER,
            },
            volumes=((workspace_host_path, "/workspace"),),
            inputs=(adapter_path_in_container,),
            outputs=(output_in_container,),
            success_marker=_MERGE_SUCCESS_MARKER,
            success_artifact=f"{output_in_container}/config.json",
            timeout_seconds=_MERGE_TIMEOUT_S,
        )

        return Ok(
            PreparePlan(
                steps=(merge_step,),
                final_model_path=output_in_container,
            )
        )

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
