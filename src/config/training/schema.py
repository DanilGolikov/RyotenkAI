from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, field_validator, model_validator

from src.constants import STRATEGY_SFT

from ..base import StrictBaseModel
from .constants import TRAINING_TYPE_ADALORA, TRAINING_TYPE_QLORA

# NOTE: Runtime import is required for Pydantic field type.
from .hyperparams import GlobalHyperparametersConfig  # noqa: TC001
from .lora import AdaLoraConfig, LoraConfig  # noqa: TC001
from .strategies import StrategyPhaseConfig, validate_strategy_chain

if TYPE_CHECKING:
    from src.utils.result import Result, StrategyError


class TrainingOnlyConfig(StrictBaseModel):
    """
    Training configuration with unified lora section and adalora support.

    Architecture:
    - type: "qlora" | "lora" | "adalora"
    - hyperparams: GlobalHyperparametersConfig (global defaults)
    - lora: LoraConfig    (use when type='lora')
    - qlora: LoraConfig   (use when type='qlora', same structure, enables 4-bit quantization)
    - adalora: AdaLoraConfig (use when type='adalora')
    - provider: Reference to provider from providers registry

    Each type has its own named config block:
    - type: "qlora"   → qlora: ... (4-bit quantization + LoRA, uses bnb_4bit_* settings)
    - type: "lora"    → lora: ...  (full precision + LoRA)
    - type: "adalora" → adalora: ... (adaptive rank allocation)

    Example QLoRA:
        training:
          type: qlora
          hyperparams:
            epochs: 3
            learning_rate: 2e-4
            per_device_train_batch_size: 4
          qlora:
            r: 16
            lora_alpha: 32
          strategies:
            - strategy_type: sft
    """

    # =========================================================================
    # PROVIDER SELECTION
    # =========================================================================
    provider: str | None = Field(None, description="Provider name from 'providers' registry.")

    type: str = Field(
        TRAINING_TYPE_QLORA,
        description="Training type: qlora, lora, adalora",
    )

    # =========================================================================
    # ADAPTER CONFIGURATIONS
    # Each training type has its own named block:
    #   type: lora    → lora: ...
    #   type: qlora   → qlora: ...
    #   type: adalora → adalora: ...
    # =========================================================================
    lora: LoraConfig | None = Field(
        None,
        description="LoRA config (use when type='lora')",
    )
    qlora: LoraConfig | None = Field(
        None,
        description="QLoRA config (use when type='qlora'). Same structure as lora, uses bnb_4bit_* settings.",
    )
    adalora: AdaLoraConfig | None = Field(
        None,
        description="AdaLoRA config (use when type='adalora')",
    )

    # =========================================================================
    # QUANTIZATION (auto-set based on type, but can be overridden)
    # =========================================================================
    load_in_8bit: bool = Field(False, description="8-bit quantization (alternative to 4-bit)")

    # =========================================================================
    # STRATEGY CHAIN (required)
    # =========================================================================
    strategies: list[StrategyPhaseConfig] = Field(
        default_factory=lambda: [StrategyPhaseConfig(strategy_type=STRATEGY_SFT)],
        description="Training strategy chain (single or multi-phase)",
    )

    # =========================================================================
    # GLOBAL HYPERPARAMETERS (defaults)
    # =========================================================================
    hyperparams: GlobalHyperparametersConfig = Field(
        ...,  # REQUIRED: User must explicitly set core hyperparams
        description="Global training hyperparameters (5 core required + optional advanced)",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = ["qlora", "lora", "adalora"]
        if v.lower() not in allowed:
            raise ValueError(f"Training type must be one of {allowed}")
        return v.lower()

    @field_validator("strategies")
    @classmethod
    def validate_strategies_chain(cls, v: list[StrategyPhaseConfig]) -> list[StrategyPhaseConfig]:
        """
        Fail-fast validation for critical strategy-chain issues.

        Invalid ordering is warning-only; structural issues must still fail at config load time.
        """
        validation = validate_strategy_chain(v)
        if validation.is_failure():
            raise ValueError(str(validation.unwrap_err()))
        return v

    @model_validator(mode="after")
    def _run_model_validators(self) -> TrainingOnlyConfig:
        """
        Centralized cross-field validators for this config.

        Convention:
        - Keep ONE `@model_validator(mode="after")` method per config model.
        - Enumerate and call pure validator functions from `src.config.validators.*`.
        """
        # Local import to avoid circular imports.
        from ..validators.training import validate_training_adapter_requires_block

        validate_training_adapter_requires_block(self)
        return self

    def get_effective_load_in_4bit(self) -> bool:
        """Get effective 4-bit quantization setting based on type."""
        return self.type == TRAINING_TYPE_QLORA

    def get_effective_optimizer(self) -> str:
        """Get effective optimizer based on training type."""
        if self.hyperparams.optim is not None:
            return self.hyperparams.optim
        return "paged_adamw_8bit" if self.type == TRAINING_TYPE_QLORA else "adamw_torch"

    def get_adapter_config(self) -> LoraConfig | AdaLoraConfig:
        """
        Get the adapter config matching the training type.

        Returns:
            LoraConfig for type="lora"   (from lora: block)
            LoraConfig for type="qlora"  (from qlora: block)
            AdaLoraConfig for type="adalora" (from adalora: block)
        """
        if self.type == TRAINING_TYPE_ADALORA:
            if self.adalora is None:
                raise ValueError("type='adalora' requires 'training.adalora:' section in config")
            return self.adalora
        if self.type == TRAINING_TYPE_QLORA:
            if self.qlora is None:
                raise ValueError("type='qlora' requires 'training.qlora:' section in config")
            return self.qlora
        # type='lora'
        if self.lora is None:
            raise ValueError("type='lora' requires 'training.lora:' section in config")
        return self.lora

    def get_strategy_chain(self) -> list[StrategyPhaseConfig]:
        """Get the strategy chain."""
        return self.strategies

    def is_multi_phase(self) -> bool:
        """Check if this is a multi-phase training configuration."""
        return len(self.strategies) > 1

    def get_total_epochs(self) -> int:
        """Get total epochs across all strategies."""
        total = 0
        default_epochs = self.hyperparams.epochs
        for phase in self.strategies:
            epochs = phase.hyperparams.epochs
            if epochs is None:
                epochs = default_epochs
            total += epochs
        return total

    def validate_chain(self) -> Result[None, StrategyError]:
        """Validate the strategy chain transitions."""
        return validate_strategy_chain(self.strategies)

    @staticmethod
    def has_adapter() -> bool:
        """Check if training uses an adapter."""
        return True  # All types use adapters now (no full_ft)


# Backward-compatible alias (used across codebase)
TrainingConfig = TrainingOnlyConfig


__all__ = [
    "TrainingConfig",
    "TrainingOnlyConfig",
]
