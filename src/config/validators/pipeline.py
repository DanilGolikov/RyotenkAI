from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.schema import PipelineConfig


def validate_pipeline_config_references(cfg: PipelineConfig) -> None:
    """Validate cross-block references (providers, datasets, etc)."""

    # Local import to avoid import-time cycles.
    from .cross import (
        validate_pipeline_active_provider_is_registered,
        validate_pipeline_adapter_cache_hf_config,
        validate_pipeline_evaluation_requires_inference,
        validate_pipeline_inference_provider_config,
        validate_pipeline_strategy_dataset_references,
    )

    # Provider validation:
    # - schema-only: providers registry contains training.provider
    # - best-effort: training.provider is registered in GPUProviderFactory (when available)
    ok, err = validate_pipeline_active_provider_is_registered(cfg)
    if not ok:
        raise ValueError(err)

    # Dataset validation: strategy.dataset references must exist in datasets registry.
    ok, err = validate_pipeline_strategy_dataset_references(cfg)
    if not ok:
        raise ValueError(err)

    # Inference validation (cross-block):
    # - when inference is enabled, ensure selected inference provider is configurable
    ok, err = validate_pipeline_inference_provider_config(cfg)
    if not ok:
        raise ValueError(err)

    # Evaluation fail-fast: evaluation.enabled=true requires inference.enabled=true
    ok, err = validate_pipeline_evaluation_requires_inference(cfg)
    if not ok:
        raise ValueError(err)

    # Adapter cache: HF integration must be enabled; repo_id must differ from final model repo
    ok, err = validate_pipeline_adapter_cache_hf_config(cfg)
    if not ok:
        raise ValueError(err)


__all__ = [
    "validate_pipeline_config_references",
]
