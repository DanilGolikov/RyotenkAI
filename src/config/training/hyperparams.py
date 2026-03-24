from __future__ import annotations

from types import MappingProxyType
from typing import Any

from pydantic import Field

from ..base import StrictBaseModel
from .constants import (
    EPOCHS_MAX,
    LOGGING_STEPS_DEFAULT,
    SAVE_STEPS_DEFAULT,
    WEIGHT_DECAY_DEFAULT,
)


class GlobalHyperparametersConfig(StrictBaseModel):
    """
    Global training hyperparameters (training.hyperparams).

    Core fields are REQUIRED - user must explicitly choose values.
    Advanced fields are OPTIONAL with reasonable defaults.

    These values apply to ALL strategies unless overridden in phase-specific config.
    """

    # =========================================================================
    # REQUIRED: Core training parameters
    # =========================================================================
    per_device_train_batch_size: int = Field(..., ge=1, description="REQUIRED: Batch size per GPU device")
    gradient_accumulation_steps: int = Field(..., ge=1, description="REQUIRED: Gradient accumulation steps")
    learning_rate: float = Field(..., gt=0, lt=1, description="REQUIRED: Learning rate (2e-4 typical for SFT)")
    warmup_ratio: float = Field(..., ge=0, le=1, description="REQUIRED: Warmup ratio (0.05 typical)")
    epochs: int = Field(..., ge=1, le=EPOCHS_MAX, description="REQUIRED: Number of training epochs")

    # =========================================================================
    # OPTIONAL: Advanced training parameters (reasonable defaults)
    # =========================================================================
    weight_decay: float = Field(
        WEIGHT_DECAY_DEFAULT,
        ge=0,
        description="Optional: Weight decay for regularization",
    )
    lr_scheduler_type: str = Field("cosine", description="Optional: LR scheduler (cosine, linear, etc.)")
    neftune_noise_alpha: float | None = Field(None, ge=0, description="Optional: NEFTune noise alpha")
    optim: str | None = Field(None, description="Optional: Optimizer (auto-detect if not specified)")
    bf16: bool = Field(True, description="Optional: Use BF16 precision (recommended for A100+)")
    fp16: bool = Field(False, description="Optional: Use FP16 precision")
    gradient_checkpointing: bool = Field(True, description="Optional: Use gradient checkpointing (saves memory)")
    logging_steps: int = Field(
        LOGGING_STEPS_DEFAULT,
        ge=1,
        description="Optional: Log metrics every N steps",
    )
    save_steps: int = Field(
        SAVE_STEPS_DEFAULT,
        ge=1,
        description="Optional: Save checkpoint every N steps",
    )
    eval_steps: int | None = Field(
        None, ge=1, description="Optional: Evaluate every N steps (when eval dataset exists)"
    )

    # =========================================================================
    # OPTIONAL: Strategy-specific parameters (can be set globally)
    # =========================================================================
    beta: float | None = Field(None, description="Optional: DPO/ORPO beta parameter")
    max_length: int | None = Field(None, ge=1, description="Optional: Max sequence length (SFT/DPO/ORPO)")
    packing: bool | None = Field(None, description="Optional: Use dataset packing (SFT only)")

    # SAPO / GRPO specific
    sapo_temperature_pos: float | None = Field(None, ge=0.0, description="Optional: SAPO positive temperature")
    sapo_temperature_neg: float | None = Field(None, ge=0.0, description="Optional: SAPO negative temperature")
    num_generations: int | None = Field(None, ge=1, description="Optional: Number of generations for GRPO/SAPO")
    max_prompt_length: int | None = Field(None, ge=1, description="Optional: SAPO maximum prompt length")
    max_completion_length: int | None = Field(None, ge=1, description="Optional: SAPO maximum completion length")


class PhaseHyperparametersConfig(StrictBaseModel):
    """
    Phase-specific training hyperparameters (strategies[].hyperparams).

    ALL fields are OPTIONAL - allows selective overrides of global config.
    """

    epochs: int | None = Field(None, ge=1, le=EPOCHS_MAX, description="Override: Number of epochs")
    per_device_train_batch_size: int | None = Field(None, ge=1, description="Override: Batch size per device")
    gradient_accumulation_steps: int | None = Field(None, ge=1, description="Override: Gradient accumulation")
    learning_rate: float | None = Field(None, gt=0, lt=1, description="Override: Learning rate")
    warmup_ratio: float | None = Field(None, ge=0, le=1, description="Override: Warmup ratio")

    weight_decay: float | None = Field(None, ge=0, description="Override: Weight decay")
    lr_scheduler_type: str | None = Field(None, description="Override: LR scheduler type")
    neftune_noise_alpha: float | None = Field(None, ge=0, description="Override: NEFTune noise alpha")
    optim: str | None = Field(None, description="Override: Optimizer")
    bf16: bool | None = Field(None, description="Override: Use BF16 precision")
    fp16: bool | None = Field(None, description="Override: Use FP16 precision")
    gradient_checkpointing: bool | None = Field(None, description="Override: Gradient checkpointing")
    logging_steps: int | None = Field(None, ge=1, description="Override: Logging frequency")
    save_steps: int | None = Field(None, ge=1, description="Override: Checkpoint save frequency")
    eval_steps: int | None = Field(None, ge=1, description="Override: Evaluation frequency")
    beta: float | None = Field(None, description="Strategy-specific: DPO/ORPO beta")

    # Strategy-specific parameters
    max_length: int | None = Field(None, ge=1, description="Strategy-specific: Max sequence length (SFT/DPO/ORPO)")
    packing: bool | None = Field(None, description="Strategy-specific: Use packing (SFT)")

    # SAPO / GRPO specific hyperparameters
    sapo_temperature_pos: float | None = Field(None, ge=0.0, description="Strategy-specific: SAPO positive temperature")
    sapo_temperature_neg: float | None = Field(None, ge=0.0, description="Strategy-specific: SAPO negative temperature")
    num_generations: int | None = Field(None, ge=1, description="Strategy-specific: GRPO/SAPO generations")
    max_prompt_length: int | None = Field(None, ge=1, description="Strategy-specific: SAPO max prompt length")
    max_completion_length: int | None = Field(None, ge=1, description="Strategy-specific: SAPO max completion length")


# Default values for global hyperparameters (system defaults)
# NOTE: currently kept for reference only (not wired into merging/validation).
# WPS407: use MappingProxyType for immutable module-level constant
_DEFAULT_HYPERPARAMS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_steps": 500,
        "max_length": 2048,
        "packing": False,
        # SAPO Defaults
        "sapo_temperature_pos": 1.0,
        "sapo_temperature_neg": 1.0,
        "num_generations": 4,
        "max_prompt_length": 1536,
        "max_completion_length": 512,
    }
)


__all__ = [
    "GlobalHyperparametersConfig",
    "PhaseHyperparametersConfig",
]
