"""
Cross-config validators.

This module is reserved for rules that validate relationships across multiple config blocks,
e.g. model × training × providers × datasets.

Implementation notes:
- Avoid importing config models at runtime to prevent circular imports (TYPE_CHECKING only).
- Cross-config validators may be pure (schema-only) or "dynamic" (best-effort runtime checks).
  Keep I/O (FS/network) out of these functions unless explicitly intended.

Error model: each validator raises :class:`ConfigInvalidError` on failure
(typed-exception migration, Phase A2 Batch 1). ``context["code"]`` carries
the legacy ``CONFIG_*`` subcode for callers that still branch on the cause.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

from ryotenkai_shared.config.providers.registry import PROVIDER_TYPES
from ryotenkai_shared.constants import PROVIDER_RUNPOD, PROVIDER_SINGLE_NODE
from ryotenkai_shared.errors import ConfigInvalidError

from .constants import ERR_FALLBACK_LOC, ERR_FALLBACK_MSG, ERR_KEY_LOC, ERR_KEY_MSG

if TYPE_CHECKING:
    from ..pipeline.schema import PipelineConfig


def _raise_config_invalid(message: str, code: str) -> None:
    """Raise :class:`ConfigInvalidError` with the legacy subcode preserved.

    The legacy ``CONFIG_*`` code is stored both in ``context["code"]`` for
    programmatic access and embedded in the detail string so existing
    assertions that match against the rendered Pydantic ``ValidationError``
    message (which only carries ``str(detail)``) continue to surface it.
    """
    raise ConfigInvalidError(detail=f"[{code}] {message}", context={"code": code})


def _validate_provider_schema(
    *,
    provider_cfg: dict[str, Any],
    schema_cls: type,
    code: str,
    context: str,
) -> None:
    """Common helper: run Pydantic schema validation against a provider dict.

    Returns ``None`` on success; on validation failure raises
    :class:`ConfigInvalidError` with the shared error-formatting style used
    throughout this module.
    """
    from pydantic import ValidationError

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
        _raise_config_invalid(f"{context}. First error at {loc}: {msg}", code)
    except Exception as e:
        _raise_config_invalid(f"{context}: {e!s}", code)


def validate_pipeline_providers_config(cfg: PipelineConfig) -> None:
    """
    Validate provider configuration (schema-only).

    Checks:
    - providers section exists and not empty
    - training.provider is set
    - training.provider references existing provider in providers registry

    Raises:
        ConfigInvalidError: on any validation failure (legacy subcode in
            ``context["code"]``).
    """

    # Must have at least one provider
    if not cfg.providers:
        _raise_config_invalid(
            "No providers configured. Add 'providers:' section to config.",
            "CONFIG_PROVIDERS_MISSING",
        )

    # training.provider must be set
    if not cfg.training.provider:
        _raise_config_invalid(
            "training.provider not set. Specify which provider to use.",
            "CONFIG_TRAINING_PROVIDER_MISSING",
        )

    # Validate training.provider reference
    if cfg.training.provider not in cfg.providers:
        available = list(cfg.providers.keys())
        _raise_config_invalid(
            f"training.provider='{cfg.training.provider}' not found. Available: {available}",
            "CONFIG_TRAINING_PROVIDER_NOT_FOUND",
        )

    # Validate active training provider schema (fail-fast; schema-only).
    active = cfg.training.provider
    provider_cfg = cfg.providers[active]

    provider_type = PROVIDER_TYPES.get(active)
    if provider_type is not None:
        _validate_provider_schema(
            provider_cfg=provider_cfg,
            schema_cls=provider_type.schema,
            code=provider_type.training_error_code,
            context=(
                f"training.provider={active!r} but providers.{active} is invalid for "
                f"{provider_type.schema_name}"
            ),
        )


def validate_pipeline_active_provider_is_registered(cfg: PipelineConfig) -> None:
    """
    Validate that `training.provider` is registered in the manifest-driven
    :class:`ProviderRegistry`.

    Replaces the legacy ``GPUProviderFactory.get_available_providers()``
    query (Phase 14.D+F refactor). The registry walks ``provider.toml``
    manifests from disk; the list of "available providers" is exactly
    :meth:`ProviderRegistry.list`.

    Best-effort: modular runtimes that don't ship the ``ryotenkai_providers``
    package skip this check via the ``ModuleNotFoundError`` branch.

    Raises:
        ConfigInvalidError: on validation failure.
    """

    # Local import to avoid heavy side-effects at module import time.
    from ryotenkai_shared.utils.logger import logger

    # Only validate when provider is explicitly set.
    if not cfg.training.provider:
        return

    # First validate structural provider registry constraints (raises on fail).
    validate_pipeline_providers_config(cfg)

    try:
        # Pull the manifest-driven registry. importlib indirection lets
        # modular runtimes that don't ship ``ryotenkai_providers`` skip
        # this check via the ModuleNotFoundError branch.
        registry_mod = importlib.import_module("ryotenkai_providers.registry")
        registry = registry_mod.get_registry()
        available = list(registry.list(role="training"))
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or ""
        if missing.startswith("ryotenkai_providers"):
            logger.debug(
                "[CFG:PROVIDER_REGISTRY] Skipping provider registry validation: "
                "ryotenkai_providers is not available in this runtime"
            )
            return
        _raise_config_invalid(
            f"Failed to load provider registry for validation: {e!s}",
            "CONFIG_PROVIDER_REGISTRY_LOAD_FAILED",
        )
    except Exception as e:
        _raise_config_invalid(
            f"Failed to load provider registry for validation: {e!s}",
            "CONFIG_PROVIDER_REGISTRY_LOAD_FAILED",
        )

    active = cfg.training.provider
    if active not in available:
        _raise_config_invalid(
            (
                f"Unknown provider: '{active}'. Available providers: {available}. "
                "Did you forget to import the provider module?"
            ),
            "CONFIG_PROVIDER_NOT_REGISTERED",
        )


def validate_pipeline_strategy_dataset_references(cfg: PipelineConfig) -> None:
    """
    Validate that all training.strategy dataset references exist in datasets registry.

    Checks:
    - datasets has at least one entry
    - if strategy.dataset is set → it must exist in cfg.datasets

    Raises:
        ConfigInvalidError: on validation failure.
    """

    # Must have at least one dataset
    if not cfg.datasets:
        _raise_config_invalid("datasets must contain at least one entry", "CONFIG_DATASETS_EMPTY")

    available = list(cfg.datasets.keys())
    for i, strategy in enumerate(cfg.training.strategies):
        if strategy.dataset and strategy.dataset not in cfg.datasets:
            _raise_config_invalid(
                (
                    f"Strategy {i} ({strategy.strategy_type}) references "
                    f"dataset '{strategy.dataset}' which is not in datasets registry. "
                    f"Available: {available}"
                ),
                "CONFIG_STRATEGY_DATASET_MISSING",
            )


def validate_pipeline_inference_provider_config(cfg: PipelineConfig) -> None:
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
      - `providers.single_node` is a valid `SingleNodeProviderConfig` (schema validation only)

    Raises:
        ConfigInvalidError: on validation failure.
    """

    if not getattr(cfg, "inference", None) or not cfg.inference.enabled:
        return

    provider: str = str(cfg.inference.provider)

    # Phase 14.D+F — registry-membership check (was hardcoded
    # ``{PROVIDER_SINGLE_NODE, PROVIDER_RUNPOD}`` set). Adding a
    # third provider = update :data:`PROVIDER_TYPES` once; this
    # gate then accepts it automatically.
    if provider not in PROVIDER_TYPES:
        return

    provider_type = PROVIDER_TYPES.get(provider)
    if provider_type is None:
        return

    if provider not in cfg.providers:
        available = list(cfg.providers.keys())
        _raise_config_invalid(
            (
                f"inference.enabled=true but providers.{provider} is missing. "
                f"Inference provider {provider!r} requires providers.{provider} config. "
                f"Available providers: {available}"
            ),
            "CONFIG_INFERENCE_PROVIDER_MISSING",
        )

    provider_cfg = cfg.providers[provider]
    _validate_provider_schema(
        provider_cfg=provider_cfg,
        schema_cls=provider_type.schema,
        code=provider_type.inference_error_code,
        context=(
            f"inference.enabled=true but providers.{provider} is invalid for "
            f"{provider_type.schema_name}"
        ),
    )

    # Provider-specific additional checks beyond schema validation.
    if provider == PROVIDER_RUNPOD:
        from ryotenkai_shared.config.providers.runpod import RunPodProviderConfig

        parsed = RunPodProviderConfig(**provider_cfg)
        if parsed.inference.pod is None:
            _raise_config_invalid(
                (
                    "inference.enabled=true but providers.runpod.inference.pod is missing. "
                    "RunPod inference requires at least: providers.runpod.connect.ssh.key_path, "
                    "providers.runpod.inference.pod (volume is optional)."
                ),
                "CONFIG_RUNPOD_INFERENCE_POD_MISSING",
            )


def validate_pipeline_evaluation_requires_inference(cfg: PipelineConfig) -> None:
    """
    Fail-fast: evaluation.enabled=true requires inference.enabled=true.

    Evaluation needs an active inference endpoint to collect model answers.
    If inference is disabled, there is nothing to evaluate against.

    Raises:
        ConfigInvalidError: on validation failure.
    """
    eval_cfg = getattr(cfg, "evaluation", None)
    if not eval_cfg or not getattr(eval_cfg, "enabled", False):
        return

    inference_cfg = getattr(cfg, "inference", None)
    if not inference_cfg or not getattr(inference_cfg, "enabled", False):
        _raise_config_invalid(
            (
                "evaluation.enabled=true requires inference.enabled=true. "
                "The evaluation stage needs a live inference endpoint to collect model answers. "
                "Either enable inference deployment or disable evaluation."
            ),
            "CONFIG_EVALUATION_REQUIRES_INFERENCE",
        )


def validate_pipeline_adapter_cache_hf_config(cfg: PipelineConfig) -> None:
    """
    Validate adapter cache configuration against HF Hub integration settings.

    Rules:
    - If any phase has adapter_cache.enabled=true:
      → integrations.huggingface must be configured (repo_id set)
      → adapter_cache.repo_id must differ from integrations.huggingface.repo_id
        (to prevent mixing intermediate adapters with the final merged model)

    Raises:
        ConfigInvalidError: on validation failure.
    """
    cache_phases = [
        s for s in cfg.training.strategies
        if hasattr(s, "adapter_cache") and getattr(s.adapter_cache, "enabled", False)
    ]
    if not cache_phases:
        return

    hf_cfg = getattr(cfg.integrations, "huggingface", None)
    if hf_cfg is None or not hf_cfg.repo_id:
        _raise_config_invalid(
            (
                "adapter_cache.enabled=true requires integrations.huggingface.repo_id to be set. "
                "Add the integrations.huggingface section with repo_id (and optionally private)."
            ),
            "CONFIG_ADAPTER_CACHE_HF_REQUIRED",
        )

    final_repo_id = hf_cfg.repo_id
    for i, phase in enumerate(cache_phases):
        if phase.adapter_cache.repo_id == final_repo_id:
            _raise_config_invalid(
                (
                    f"Strategy {i} ({phase.strategy_type}): adapter_cache.repo_id='{phase.adapter_cache.repo_id}' "
                    f"must differ from integrations.huggingface.repo_id='{final_repo_id}'. "
                    "The adapter cache repository stores intermediate adapters; "
                    "the HF Hub repo_id is reserved for the final merged model."
                ),
                "CONFIG_ADAPTER_CACHE_REPO_CONFLICT",
            )


__all__ = [
    "validate_pipeline_active_provider_is_registered",
    "validate_pipeline_adapter_cache_hf_config",
    "validate_pipeline_evaluation_requires_inference",
    "validate_pipeline_inference_provider_config",
    "validate_pipeline_providers_config",
    "validate_pipeline_strategy_dataset_references",
]
