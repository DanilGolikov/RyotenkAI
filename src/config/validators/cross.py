"""
Cross-config validators.

This module is reserved for rules that validate relationships across multiple config blocks,
e.g. model × training × providers × datasets.

Implementation notes:
- Avoid importing config models at runtime to prevent circular imports (TYPE_CHECKING only).
- Cross-config validators may be pure (schema-only) or "dynamic" (best-effort runtime checks).
  Keep I/O (FS/network) out of these functions unless explicitly intended.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

from src.config.providers.registry import PROVIDER_TYPES
from src.constants import PROVIDER_RUNPOD, PROVIDER_SINGLE_NODE

from .constants import ERR_FALLBACK_LOC, ERR_FALLBACK_MSG, ERR_KEY_LOC, ERR_KEY_MSG

if TYPE_CHECKING:
    from ..pipeline.schema import PipelineConfig
    from src.utils.result import ConfigError, Result


def _config_error(message: str, code: str) -> Result[None, ConfigError]:
    from src.utils.result import ConfigError, Err

    return Err(ConfigError(message=message, code=code))


def _validate_provider_schema(
    *,
    provider_cfg: dict[str, Any],
    schema_cls: type,
    code: str,
    context: str,
) -> Result[None, ConfigError] | None:
    """Common helper: run Pydantic schema validation against a provider dict.

    Returns ``None`` on success (so callers can chain to ``Ok``), or a
    ``Result.Err`` with the shared error-formatting style used throughout
    this module when validation fails.
    """
    from pydantic import ValidationError

    from src.utils.result import Ok

    del Ok  # imported for side effect — Ok is returned by callers
    try:
        schema_cls(**provider_cfg)
    except ValidationError as e:
        errors = e.errors()
        if errors:
            err0 = errors[0]
            loc = ".".join(str(p) for p in cast("tuple[Any, ...]", err0.get(ERR_KEY_LOC) or ())) or ERR_FALLBACK_LOC
            msg = str(err0.get(ERR_KEY_MSG) or "").strip() or ERR_FALLBACK_MSG
        else:
            loc = ERR_FALLBACK_LOC
            msg = str(e).strip() or ERR_FALLBACK_MSG
        return _config_error(f"{context}. First error at {loc}: {msg}", code)
    except Exception as e:
        return _config_error(f"{context}: {e!s}", code)
    return None


def validate_pipeline_providers_config(cfg: PipelineConfig) -> Result[None, ConfigError]:
    """
    Validate provider configuration (schema-only).

    Checks:
    - providers section exists and not empty
    - training.provider is set
    - training.provider references existing provider in providers registry
    """

    from src.utils.result import Ok

    # Must have at least one provider
    if not cfg.providers:
        return _config_error("No providers configured. Add 'providers:' section to config.", "CONFIG_PROVIDERS_MISSING")

    # training.provider must be set
    if not cfg.training.provider:
        return _config_error(
            "training.provider not set. Specify which provider to use.",
            "CONFIG_TRAINING_PROVIDER_MISSING",
        )

    # Validate training.provider reference
    if cfg.training.provider not in cfg.providers:
        available = list(cfg.providers.keys())
        return _config_error(
            f"training.provider='{cfg.training.provider}' not found. Available: {available}",
            "CONFIG_TRAINING_PROVIDER_NOT_FOUND",
        )

    # Validate active training provider schema (fail-fast; schema-only).
    active = cfg.training.provider
    provider_cfg = cfg.providers[active]

    provider_type = PROVIDER_TYPES.get(active)
    if provider_type is not None:
        err = _validate_provider_schema(
            provider_cfg=provider_cfg,
            schema_cls=provider_type.schema,
            code=provider_type.training_error_code,
            context=(
                f"training.provider={active!r} but providers.{active} is invalid for "
                f"{provider_type.schema_name}"
            ),
        )
        if err is not None:
            return err

    return Ok(None)


def validate_pipeline_active_provider_is_registered(cfg: PipelineConfig) -> Result[None, ConfigError]:
    """
    Validate that `training.provider` is a registered provider in GPUProviderFactory.

    This validation is intentionally dynamic:
    - It queries GPUProviderFactory.get_available_providers() (no hardcoded names).
    - It is best-effort: some runtimes may not ship `src.providers`.
    """

    # Local import to avoid heavy side-effects at module import time.
    from src.utils.logger import logger
    from src.utils.result import Ok

    # Only validate when provider is explicitly set.
    if not cfg.training.provider:
        return Ok(None)

    # First validate structural provider registry constraints.
    providers_validation = validate_pipeline_providers_config(cfg)
    if providers_validation.is_failure():
        return providers_validation

    try:
        # Importing src.providers.training triggers provider auto-registration
        # (the package's __init__ imports each provider module). The
        # importlib indirection lets modular runtimes that don't ship the
        # providers tree skip this check via the ModuleNotFoundError branch.
        providers_mod = importlib.import_module("src.providers.training")
        gpu_provider_factory = providers_mod.GPUProviderFactory
        available = gpu_provider_factory.get_available_providers()
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or ""
        if missing.startswith("src.providers"):
            logger.debug(
                "[CFG:PROVIDER_REGISTRY] Skipping provider factory validation: "
                "src.providers is not available in this runtime"
            )
            return Ok(None)
        return _config_error(
            f"Failed to load provider registry for validation: {e!s}",
            "CONFIG_PROVIDER_REGISTRY_LOAD_FAILED",
        )
    except Exception as e:
        return _config_error(
            f"Failed to load provider registry for validation: {e!s}",
            "CONFIG_PROVIDER_REGISTRY_LOAD_FAILED",
        )

    active = cfg.training.provider
    if active not in available:
        return _config_error(
            (
                f"Unknown provider: '{active}'. Available providers: {available}. "
                "Did you forget to import the provider module?"
            ),
            "CONFIG_PROVIDER_NOT_REGISTERED",
        )

    return Ok(None)


def validate_pipeline_strategy_dataset_references(cfg: PipelineConfig) -> Result[None, ConfigError]:
    """
    Validate that all training.strategy dataset references exist in datasets registry.

    Checks:
    - datasets has at least one entry
    - if strategy.dataset is set → it must exist in cfg.datasets
    """

    from src.utils.result import Ok

    # Must have at least one dataset
    if not cfg.datasets:
        return _config_error("datasets must contain at least one entry", "CONFIG_DATASETS_EMPTY")

    available = list(cfg.datasets.keys())
    for i, strategy in enumerate(cfg.training.strategies):
        if strategy.dataset and strategy.dataset not in cfg.datasets:
            return _config_error(
                (
                    f"Strategy {i} ({strategy.strategy_type}) references "
                    f"dataset '{strategy.dataset}' which is not in datasets registry. "
                    f"Available: {available}"
                ),
                "CONFIG_STRATEGY_DATASET_MISSING",
            )
    return Ok(None)


def validate_pipeline_inference_provider_config(cfg: PipelineConfig) -> Result[None, ConfigError]:
    """
    Validate inference provider prerequisites.

    Motivation:
    - Inference deployment is optional (`inference.enabled`), but when enabled we should fail-fast
      if the selected inference provider cannot be configured with the current config.

    Current behavior:
    - Inference providers are guarded by `InferenceConfig` validators (SUPPORTED_* lists).
    - Some providers require provider-specific config blocks under `providers:*`.

    Checks:
    - When inference.enabled=true and inference.provider=='single_node':
      - `providers.single_node` exists
      - `providers.single_node` is a valid `SingleNodeConfig` (schema validation only)
    """

    from src.utils.result import Ok

    if not getattr(cfg, "inference", None) or not cfg.inference.enabled:
        return Ok(None)

    provider: str = str(cfg.inference.provider)

    # Only validate provider-config presence for providers that are configured via `providers:` registry.
    # Some providers may eventually keep all config under `inference:*` (no `providers.*` block).
    if provider not in {PROVIDER_SINGLE_NODE, PROVIDER_RUNPOD}:
        return Ok(None)

    provider_type = PROVIDER_TYPES.get(provider)
    if provider_type is None:
        return Ok(None)

    if provider not in cfg.providers:
        available = list(cfg.providers.keys())
        return _config_error(
            (
                f"inference.enabled=true but providers.{provider} is missing. "
                f"Inference provider {provider!r} requires providers.{provider} config. "
                f"Available providers: {available}"
            ),
            "CONFIG_INFERENCE_PROVIDER_MISSING",
        )

    provider_cfg = cfg.providers[provider]
    schema_err = _validate_provider_schema(
        provider_cfg=provider_cfg,
        schema_cls=provider_type.schema,
        code=provider_type.inference_error_code,
        context=(
            f"inference.enabled=true but providers.{provider} is invalid for "
            f"{provider_type.schema_name}"
        ),
    )
    if schema_err is not None:
        return schema_err

    # Provider-specific additional checks beyond schema validation.
    if provider == PROVIDER_RUNPOD:
        from src.config.providers.runpod import RunPodProviderConfig

        parsed = RunPodProviderConfig(**provider_cfg)
        if parsed.inference.pod is None:
            return _config_error(
                (
                    "inference.enabled=true but providers.runpod.inference.pod is missing. "
                    "RunPod inference requires at least: providers.runpod.connect.ssh.key_path, "
                    "providers.runpod.inference.pod (volume is optional)."
                ),
                "CONFIG_RUNPOD_INFERENCE_POD_MISSING",
            )

    return Ok(None)


def validate_pipeline_evaluation_requires_inference(cfg: PipelineConfig) -> Result[None, ConfigError]:
    """
    Fail-fast: evaluation.enabled=true requires inference.enabled=true.

    Evaluation needs an active inference endpoint to collect model answers.
    If inference is disabled, there is nothing to evaluate against.
    """
    from src.utils.result import Ok

    eval_cfg = getattr(cfg, "evaluation", None)
    if not eval_cfg or not getattr(eval_cfg, "enabled", False):
        return Ok(None)

    inference_cfg = getattr(cfg, "inference", None)
    if not inference_cfg or not getattr(inference_cfg, "enabled", False):
        return _config_error(
            (
                "evaluation.enabled=true requires inference.enabled=true. "
                "The evaluation stage needs a live inference endpoint to collect model answers. "
                "Either enable inference deployment or disable evaluation."
            ),
            "CONFIG_EVALUATION_REQUIRES_INFERENCE",
        )

    return Ok(None)


def validate_pipeline_adapter_cache_hf_config(cfg: PipelineConfig) -> Result[None, ConfigError]:
    """
    Validate adapter cache configuration against HF Hub integration settings.

    Rules:
    - If any phase has adapter_cache.enabled=true:
      → experiment_tracking.huggingface must be configured and enabled
      → adapter_cache.repo_id must differ from experiment_tracking.huggingface.repo_id
        (to prevent mixing intermediate adapters with the final merged model)
    """
    from src.utils.result import Ok

    cache_phases = [
        s for s in cfg.training.strategies
        if hasattr(s, "adapter_cache") and getattr(s.adapter_cache, "enabled", False)
    ]
    if not cache_phases:
        return Ok(None)

    hf_cfg = getattr(cfg.experiment_tracking, "huggingface", None)
    if hf_cfg is None or not hf_cfg.integration:
        return _config_error(
            (
                "adapter_cache.enabled=true requires experiment_tracking.huggingface to be configured with an integration. "
                "Add experiment_tracking.huggingface section with integration (pointing at a Settings → Integrations entry), repo_id, and private fields."
            ),
            "CONFIG_ADAPTER_CACHE_HF_REQUIRED",
        )

    final_repo_id = hf_cfg.repo_id
    for i, phase in enumerate(cache_phases):
        if phase.adapter_cache.repo_id == final_repo_id:
            return _config_error(
                (
                    f"Strategy {i} ({phase.strategy_type}): adapter_cache.repo_id='{phase.adapter_cache.repo_id}' "
                    f"must differ from experiment_tracking.huggingface.repo_id='{final_repo_id}'. "
                    "The adapter cache repository stores intermediate adapters; "
                    "the HF Hub repo_id is reserved for the final merged model."
                ),
                "CONFIG_ADAPTER_CACHE_REPO_CONFLICT",
            )

    return Ok(None)


__all__ = [
    "validate_pipeline_active_provider_is_registered",
    "validate_pipeline_adapter_cache_hf_config",
    "validate_pipeline_evaluation_requires_inference",
    "validate_pipeline_inference_provider_config",
    "validate_pipeline_providers_config",
    "validate_pipeline_strategy_dataset_references",
]
