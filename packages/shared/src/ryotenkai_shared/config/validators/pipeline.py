from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.schema import PipelineConfig


def validate_pipeline_config_references(cfg: PipelineConfig) -> None:
    """Validate cross-block references (providers, datasets, etc).

    Each cross-validator raises :class:`ConfigInvalidError` on its own
    failure. Pydantic ``model_validator(mode="after")`` callers expect a
    ``ValueError`` (so that Pydantic can wrap it into a ``ValidationError``
    with the proper ``ctx``); we translate at this boundary.
    """

    # Local import to avoid import-time cycles.
    from ryotenkai_shared.errors import ConfigInvalidError

    from .cross import (
        validate_pipeline_active_provider_is_registered,
        validate_pipeline_adapter_cache_hf_config,
        validate_pipeline_evaluation_requires_inference,
        validate_pipeline_inference_provider_config,
        validate_pipeline_strategy_dataset_references,
    )

    try:
        # Provider validation:
        # - schema-only: providers registry contains training.provider
        # - best-effort: training.provider is registered in GPUProviderFactory (when available)
        validate_pipeline_active_provider_is_registered(cfg)

        # Dataset validation: strategy.dataset references must exist in datasets registry.
        validate_pipeline_strategy_dataset_references(cfg)

        # Inference validation (cross-block):
        # - when inference is enabled, ensure selected inference provider is configurable
        validate_pipeline_inference_provider_config(cfg)

        # Evaluation fail-fast: evaluation.enabled=true requires inference.enabled=true
        validate_pipeline_evaluation_requires_inference(cfg)

        # Adapter cache: HF integration must be enabled; repo_id must differ from final model repo
        validate_pipeline_adapter_cache_hf_config(cfg)
    except ConfigInvalidError as exc:
        # Pydantic model_validator boundary: re-raise as ValueError so
        # Pydantic can wrap into a proper ValidationError. The detail
        # text is preserved verbatim for existing assertions.
        raise ValueError(str(exc)) from exc


__all__ = [
    "validate_pipeline_config_references",
]
