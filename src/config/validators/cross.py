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

from src.constants import PROVIDER_RUNPOD, PROVIDER_SINGLE_NODE

from .constants import ERR_FALLBACK_LOC, ERR_FALLBACK_MSG, ERR_KEY_LOC, ERR_KEY_MSG

if TYPE_CHECKING:
    from ..pipeline.schema import PipelineConfig


def validate_pipeline_providers_config(cfg: PipelineConfig) -> tuple[bool, str]:
    """
    Validate provider configuration (schema-only).

    Checks:
    - providers section exists and not empty
    - training.provider is set
    - training.provider references existing provider in providers registry
    """

    # Must have at least one provider
    if not cfg.providers:
        return False, "No providers configured. Add 'providers:' section to config."

    # training.provider must be set
    if not cfg.training.provider:
        return False, "training.provider not set. Specify which provider to use."

    # Validate training.provider reference
    if cfg.training.provider not in cfg.providers:
        available = list(cfg.providers.keys())
        return False, f"training.provider='{cfg.training.provider}' not found. Available: {available}"

    # Validate active training provider schema (fail-fast; schema-only).
    active = cfg.training.provider
    provider_cfg = cfg.providers[active]

    if active == PROVIDER_SINGLE_NODE:
        try:
            from pydantic import ValidationError

            from src.config.providers.single_node import SingleNodeConfig

            _ = SingleNodeConfig(**provider_cfg)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                err0 = errors[0]
                loc = ".".join(str(p) for p in cast("tuple[Any, ...]", err0.get(ERR_KEY_LOC) or ())) or ERR_FALLBACK_LOC  # noqa: WPS226
                msg = str(err0.get(ERR_KEY_MSG) or "").strip() or ERR_FALLBACK_MSG
            else:
                loc = ERR_FALLBACK_LOC
                msg = str(e).strip() or ERR_FALLBACK_MSG
            return (
                False,
                f"training.provider={PROVIDER_SINGLE_NODE!r} but providers.{PROVIDER_SINGLE_NODE} is invalid for SingleNodeConfig. "
                f"First error at {loc}: {msg}",
            )
        except Exception as e:
            return (
                False,
                f"training.provider={PROVIDER_SINGLE_NODE!r} but providers.{PROVIDER_SINGLE_NODE} is invalid for SingleNodeConfig: {e!s}",
            )

    if active == PROVIDER_RUNPOD:
        try:
            from pydantic import ValidationError

            from src.config.providers.runpod import RunPodProviderConfig

            _ = RunPodProviderConfig(**provider_cfg)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                err0 = errors[0]
                loc = ".".join(str(p) for p in cast("tuple[Any, ...]", err0.get(ERR_KEY_LOC) or ())) or ERR_FALLBACK_LOC  # noqa: WPS226
                msg = str(err0.get(ERR_KEY_MSG) or "").strip() or ERR_FALLBACK_MSG
            else:
                loc = ERR_FALLBACK_LOC
                msg = str(e).strip() or ERR_FALLBACK_MSG
            return (
                False,
                f"training.provider={PROVIDER_RUNPOD!r} but providers.{PROVIDER_RUNPOD} is invalid for RunPodProviderConfig. "
                f"First error at {loc}: {msg}",
            )
        except Exception as e:
            return (
                False,
                f"training.provider={PROVIDER_RUNPOD!r} but providers.{PROVIDER_RUNPOD} is invalid for RunPodProviderConfig: {e!s}",
            )

    return True, ""


def validate_pipeline_active_provider_is_registered(cfg: PipelineConfig) -> tuple[bool, str]:
    """
    Validate that `training.provider` is a registered provider in GPUProviderFactory.

    This validation is intentionally dynamic:
    - It queries GPUProviderFactory.get_available_providers() (no hardcoded names).
    - It is best-effort: some runtimes may not ship `src.pipeline`.
    """

    # Local import to avoid heavy side-effects at module import time.
    from src.utils.logger import logger

    # Only validate when provider is explicitly set.
    if not cfg.training.provider:
        return True, ""

    # First validate structural provider registry constraints.
    ok, err = validate_pipeline_providers_config(cfg)
    if not ok:
        return ok, err

    try:
        # Import triggers provider auto-registration (see src/pipeline/providers/__init__.py).
        providers_mod = importlib.import_module("src.pipeline.providers")
        gpu_provider_factory = providers_mod.GPUProviderFactory
        available = gpu_provider_factory.get_available_providers()
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or ""
        if missing.startswith("src.pipeline"):
            logger.debug(
                "[CFG:PROVIDER_REGISTRY] Skipping provider factory validation: "
                "src.pipeline is not available in this runtime"
            )
            return True, ""
        return False, f"Failed to load provider registry for validation: {e!s}"
    except Exception as e:
        return False, f"Failed to load provider registry for validation: {e!s}"

    active = cfg.training.provider
    if active not in available:
        return (
            False,
            f"Unknown provider: '{active}'. Available providers: {available}. "
            "Did you forget to import the provider module?",
        )

    return True, ""


def validate_pipeline_strategy_dataset_references(cfg: PipelineConfig) -> tuple[bool, str]:
    """
    Validate that all training.strategy dataset references exist in datasets registry.

    Checks:
    - datasets has at least one entry
    - if strategy.dataset is set → it must exist in cfg.datasets
    """

    # Must have at least one dataset
    if not cfg.datasets:
        return False, "datasets must contain at least one entry"

    available = list(cfg.datasets.keys())
    for i, strategy in enumerate(cfg.training.strategies):
        if strategy.dataset and strategy.dataset not in cfg.datasets:
            return False, (
                f"Strategy {i} ({strategy.strategy_type}) references "
                f"dataset '{strategy.dataset}' which is not in datasets registry. "
                f"Available: {available}"
            )
    return True, ""


def validate_pipeline_inference_provider_config(cfg: PipelineConfig) -> tuple[bool, str]:
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

    if not getattr(cfg, "inference", None) or not cfg.inference.enabled:
        return True, ""

    provider: str = str(cfg.inference.provider)

    # Only validate provider-config presence for providers that are configured via `providers:` registry.
    # Some providers may eventually keep all config under `inference:*` (no `providers.*` block).
    if provider not in {PROVIDER_SINGLE_NODE, PROVIDER_RUNPOD}:
        return True, ""

    if provider == PROVIDER_SINGLE_NODE:
        if PROVIDER_SINGLE_NODE not in cfg.providers:
            available = list(cfg.providers.keys())
            return (
                False,
                f"inference.enabled=true but providers.{PROVIDER_SINGLE_NODE} is missing. "
                f"Inference provider {PROVIDER_SINGLE_NODE!r} requires providers.{PROVIDER_SINGLE_NODE} config (connect/training/inference). "
                f"Available providers: {available}",
            )

        provider_cfg = cfg.providers[PROVIDER_SINGLE_NODE]
        try:
            from pydantic import ValidationError

            from src.config.providers.single_node import SingleNodeConfig

            # Schema-only validation (no I/O).
            _ = SingleNodeConfig(**provider_cfg)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                err0 = errors[0]
                loc = ".".join(str(p) for p in cast("tuple[Any, ...]", err0.get(ERR_KEY_LOC) or ())) or ERR_FALLBACK_LOC  # noqa: WPS226
                msg = str(err0.get(ERR_KEY_MSG) or "").strip() or ERR_FALLBACK_MSG
            else:
                loc = ERR_FALLBACK_LOC
                msg = str(e).strip() or ERR_FALLBACK_MSG
            return (
                False,
                f"inference.enabled=true but providers.{PROVIDER_SINGLE_NODE} is invalid for SingleNodeConfig. "
                f"First error at {loc}: {msg}",
            )
        except Exception as e:
            return (
                False,
                f"inference.enabled=true but providers.{PROVIDER_SINGLE_NODE} is invalid for SingleNodeConfig: {e!s}",
            )

        return True, ""

    if provider == PROVIDER_RUNPOD:
        if PROVIDER_RUNPOD not in cfg.providers:
            available = list(cfg.providers.keys())
            return (
                False,
                f"inference.enabled=true but providers.{PROVIDER_RUNPOD} is missing. "
                f"Inference provider {PROVIDER_RUNPOD!r} requires providers.{PROVIDER_RUNPOD} config. "
                f"Available providers: {available}",
            )

        provider_cfg = cfg.providers[PROVIDER_RUNPOD]
        try:
            from pydantic import ValidationError

            from src.config.providers.runpod import RunPodProviderConfig

            # Schema-only validation (no I/O).
            parsed = RunPodProviderConfig(**provider_cfg)
            if parsed.inference.pod is None:
                return (
                    False,
                    "inference.enabled=true but providers.runpod.inference.pod is missing. "
                    "RunPod inference requires at least: providers.runpod.connect.ssh.key_path, "
                    "providers.runpod.inference.pod (volume is optional).",
                )
        except ValidationError as e:
            errors = e.errors()
            if errors:
                err0 = errors[0]
                loc = ".".join(str(p) for p in cast("tuple[Any, ...]", err0.get(ERR_KEY_LOC) or ())) or ERR_FALLBACK_LOC  # noqa: WPS226
                msg = str(err0.get(ERR_KEY_MSG) or "").strip() or ERR_FALLBACK_MSG
            else:
                loc = ERR_FALLBACK_LOC
                msg = str(e).strip() or ERR_FALLBACK_MSG
            return (
                False,
                f"inference.enabled=true but providers.{PROVIDER_RUNPOD} is invalid for RunPodProviderConfig. "
                f"First error at {loc}: {msg}",
            )
        except Exception as e:
            return (
                False,
                f"inference.enabled=true but providers.{PROVIDER_RUNPOD} is invalid for RunPodProviderConfig: {e!s}",
            )

        return True, ""

    return True, ""


def validate_pipeline_evaluation_requires_inference(cfg: PipelineConfig) -> tuple[bool, str]:
    """
    Fail-fast: evaluation.enabled=true requires inference.enabled=true.

    Evaluation needs an active inference endpoint to collect model answers.
    If inference is disabled, there is nothing to evaluate against.
    """
    eval_cfg = getattr(cfg, "evaluation", None)
    if not eval_cfg or not getattr(eval_cfg, "enabled", False):
        return True, ""

    inference_cfg = getattr(cfg, "inference", None)
    if not inference_cfg or not getattr(inference_cfg, "enabled", False):
        return (
            False,
            "evaluation.enabled=true requires inference.enabled=true. "
            "The evaluation stage needs a live inference endpoint to collect model answers. "
            "Either enable inference deployment or disable evaluation.",
        )

    return True, ""


__all__ = [
    "validate_pipeline_active_provider_is_registered",
    "validate_pipeline_evaluation_requires_inference",
    "validate_pipeline_inference_provider_config",
    "validate_pipeline_providers_config",
    "validate_pipeline_strategy_dataset_references",
]
