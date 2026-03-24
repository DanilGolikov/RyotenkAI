from __future__ import annotations

from pydantic import Field, model_validator

from src.constants import (
    INFERENCE_ENGINE_VLLM,
    PROVIDER_SINGLE_NODE,
    InferenceEngineName,
    InferenceProviderName,
)

from ..base import StrictBaseModel
from .common import InferenceCommonConfig
from .engines import InferenceVLLMEngineConfig


class InferenceEnginesConfig(StrictBaseModel):
    """Inference engine configs registry."""

    vllm: InferenceVLLMEngineConfig = Field(  # type: ignore[arg-type]
        default_factory=InferenceVLLMEngineConfig,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    )


class InferenceConfig(StrictBaseModel):
    """Inference deployment configuration (Provider × Engine)."""

    enabled: bool = False
    provider: InferenceProviderName = PROVIDER_SINGLE_NODE
    engine: InferenceEngineName = INFERENCE_ENGINE_VLLM

    engines: InferenceEnginesConfig = Field(default_factory=InferenceEnginesConfig)
    common: InferenceCommonConfig = Field(default_factory=InferenceCommonConfig)  # pyright: ignore[reportArgumentType]

    @model_validator(mode="after")
    def _run_model_validators(self) -> InferenceConfig:
        """
        Centralized cross-field validators for this config.

        Convention:
        - Keep ONE `@model_validator(mode="after")` method per config model.
        - Delegate validation logic to `src.config.validators.*`.
        """
        # Local import to avoid circular imports.
        from ..validators.inference import (
            validate_inference_enabled_is_supported,
            validate_inference_images_required_for_provider,
        )

        validate_inference_enabled_is_supported(self)
        validate_inference_images_required_for_provider(self)
        return self


__all__ = [
    "InferenceConfig",
    "InferenceEnginesConfig",
]
