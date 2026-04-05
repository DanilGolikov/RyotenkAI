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
    providers_validation = validate_pipeline_active_provider_is_registered(cfg)
    if providers_validation.is_failure():
        raise ValueError(str(providers_validation.unwrap_err()))

    # Dataset validation: strategy.dataset references must exist in datasets registry.
    datasets_validation = validate_pipeline_strategy_dataset_references(cfg)
    if datasets_validation.is_failure():
        raise ValueError(str(datasets_validation.unwrap_err()))

    # Inference validation (cross-block):
    # - when inference is enabled, ensure selected inference provider is configurable
    inference_validation = validate_pipeline_inference_provider_config(cfg)
    if inference_validation.is_failure():
        raise ValueError(str(inference_validation.unwrap_err()))

    # Evaluation fail-fast: evaluation.enabled=true requires inference.enabled=true
    evaluation_validation = validate_pipeline_evaluation_requires_inference(cfg)
    if evaluation_validation.is_failure():
        raise ValueError(str(evaluation_validation.unwrap_err()))

    # Adapter cache: HF integration must be enabled; repo_id must differ from final model repo
    cache_validation = validate_pipeline_adapter_cache_hf_config(cfg)
    if cache_validation.is_failure():
        raise ValueError(str(cache_validation.unwrap_err()))


__all__ = [
    "validate_pipeline_config_references",
]
