from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator

from src.constants import ALL_STRATEGIES

from ...base import StrictBaseModel
from ..adapter_cache import AdapterCacheConfig  # noqa: TC001
from ..hyperparams import PhaseHyperparametersConfig


class StrategyPhaseConfig(StrictBaseModel):
    """
    Configuration for a single training strategy phase.

    Used in multi-phase training pipelines like: CPT → SFT → DPO

    Each phase can have its own:
    - Strategy type (cpt, sft, cot, dpo, orpo, sapo)
    - Hyperparameters overrides (in hyperparams block)
    - Dataset (reference by name OR inline config)
    - Adapter cache (optional HF Hub caching for trained adapters)
    """

    strategy_type: str = Field(..., description="Strategy type: cpt, sft, cot, dpo, orpo")

    # Dataset: can be name reference (str) OR inline DatasetConfig
    # None = use "default" dataset from registry
    dataset: str | None = Field(default=None, description="Dataset name from registry (None = 'default')")

    # Hyperparameters overrides
    hyperparams: PhaseHyperparametersConfig = Field(
        default_factory=PhaseHyperparametersConfig,  # pyright: ignore[reportCallIssue]  # type: ignore[call-arg]
        description="Phase-specific hyperparameters overrides",
    )

    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit strategy parameters for plugins/providers (no hidden defaults).",
    )

    adapter_cache: AdapterCacheConfig = Field(
        default_factory=AdapterCacheConfig,
        description=(
            "HF Hub adapter caching. When enabled: loads adapter if cached for current dataset, "
            "trains and uploads otherwise. Requires dataset to be set for fingerprinting."
        ),
    )

    @field_validator("strategy_type")
    @classmethod
    def validate_strategy_type(cls, v: str) -> str:
        allowed = ALL_STRATEGIES
        if v.lower() not in allowed:
            raise ValueError(f"Strategy type must be one of {allowed}")
        return v.lower()

    @model_validator(mode="after")
    def _run_model_validators(self) -> StrategyPhaseConfig:
        """
        Centralized cross-field validators for this config.

        Convention:
        - Keep ONE `@model_validator(mode="after")` method per config model.
        - Delegate validation logic to `src.config.validators.*`.
        """
        # Local import to avoid circular imports.
        from ...validators.training import validate_strategy_phase_config

        validate_strategy_phase_config(self)
        return self


__all__ = [
    "StrategyPhaseConfig",
]
