from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, field_validator, model_validator

from ryotenkai_shared.constants import STRATEGY_SFT

from ..base import StrictBaseModel
from .adapter import AdapterConfigUnion
from .constants import TRAINING_TYPE_QLORA  # noqa: F401  — re-exported for callers (PR-9 cleanup)

# NOTE: Runtime import is required for Pydantic field type.
from .hyperparams import GlobalHyperparametersConfig  # noqa: TC001
from .lora import AdaLoraConfig, LoraConfig, QloraConfig  # noqa: TC001
from .metrics_buffer import MetricsBufferConfig
from .strategies import StrategyPhaseConfig, validate_strategy_chain

if TYPE_CHECKING:
    from ryotenkai_shared.utils.result import Result, StrategyError


class TrainingOnlyConfig(StrictBaseModel):
    """Training configuration with discriminated adapter union.

    Architecture (post-discriminated-unions refactor):
      * ``adapter: AdapterConfigUnion`` — single Tag-based discriminated
        union over LoraConfig / QloraConfig / AdaLoraConfig. The
        ``kind`` discriminator on each variant tells Pydantic which
        class to validate against.
      * ``hyperparams: GlobalHyperparametersConfig`` — global defaults
        consumed by the trainer.
      * ``strategies: list[StrategyPhaseConfig]`` — training strategy
        chain (single or multi-phase).
      * ``provider: str | None`` — reference into the providers
        registry.

    Example QLoRA YAML::

        training:
          adapter:
            kind: qlora
            r: 16
            lora_alpha: 32
            lora_dropout: 0.05
            bias: none
            target_modules: all-linear
            use_dora: false
            use_rslora: false
            init_lora_weights: gaussian
          hyperparams:
            epochs: 3
            learning_rate: 2e-4
            per_device_train_batch_size: 4
          strategies:
            - strategy_type: sft

    Backward compat:
      * ``type`` is preserved as a read-only property forwarding to
        ``self.adapter.kind`` until PR-9 finalizes call-site migration.
    """

    # =========================================================================
    # PROVIDER SELECTION
    # =========================================================================
    provider: str | None = Field(None, description="Provider name from 'providers' registry.")

    # =========================================================================
    # ADAPTER CONFIGURATION (Tag-based discriminated union)
    # =========================================================================
    adapter: AdapterConfigUnion = Field(  # type: ignore[valid-type]
        ...,
        description=(
            "Adapter config. Pydantic dispatches on ``kind`` "
            "(``lora`` | ``qlora`` | ``adalora``) to the matching class."
        ),
    )

    # =========================================================================
    # QUANTIZATION (auto-set based on adapter.kind, but can be overridden)
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

    # =========================================================================
    # METRICS BUFFER (Phase 12.A.2)
    # =========================================================================
    metrics_buffer: MetricsBufferConfig = Field(
        default_factory=MetricsBufferConfig,
        description=(
            "Controls MetricsBuffer decimation policy. By default every "
            "metric is preserved losslessly (keep_all=true); flip to "
            "keep_all=false on very long runs where buffer disk / "
            "replay overhead matters more than per-step granularity."
        ),
    )

    # =========================================================================
    # BACKWARD-COMPAT: read-only ``type`` property
    # =========================================================================
    @property
    def type(self) -> str:
        """Read-only forwarder to ``self.adapter.kind``.

        Pre-discriminated-union code paths read ``cfg.training.type`` for
        logging / branching. Property keeps those working until PR-9
        migrates them to ``cfg.training.adapter.kind`` and deletes this.

        Deprecated — new code should use ``self.adapter.kind`` directly.
        """
        return self.adapter.kind

    @field_validator("strategies")
    @classmethod
    def validate_strategies_chain(cls, v: list[StrategyPhaseConfig]) -> list[StrategyPhaseConfig]:
        """Fail-fast validation for critical strategy-chain issues."""
        validation = validate_strategy_chain(v)
        if validation.is_failure():
            raise ValueError(str(validation.unwrap_err()))
        return v

    @model_validator(mode="after")
    def _run_model_validators(self) -> TrainingOnlyConfig:
        """Cross-field validators (precision-consistency post-discriminated-unions)."""
        # Local import to avoid circular imports.
        from ..validators.training import validate_training_adapter_requires_block

        validate_training_adapter_requires_block(self)
        return self

    # =========================================================================
    # ACCESSORS
    # =========================================================================
    def get_effective_load_in_4bit(self) -> bool:
        """4-bit quantization is implied by ``adapter.kind == 'qlora'``."""
        return self.adapter.kind == "qlora"

    def get_effective_optimizer(self) -> str:
        """Optimizer default depends on adapter kind (qlora ⇒ paged_adamw_8bit)."""
        if self.hyperparams.optim is not None:
            return self.hyperparams.optim
        return "paged_adamw_8bit" if self.adapter.kind == "qlora" else "adamw_torch"

    def get_adapter_config(self) -> LoraConfig | QloraConfig | AdaLoraConfig:
        """Return the typed adapter config — already discriminator-narrowed."""
        return self.adapter

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
        return True  # All adapter kinds use adapters now (no full_ft).


# Backward-compatible alias (used across codebase)
TrainingConfig = TrainingOnlyConfig


__all__ = [
    "TrainingConfig",
    "TrainingOnlyConfig",
]
