from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import Field, field_validator

from ...base import StrictBaseModel
from ..constants import (
    ADALORA_BETA_DEFAULT,
    ADALORA_TFINAL_DEFAULT,
    ADALORA_TINIT_DEFAULT,
    LORA_RANK_MAX,
)


class AdaLoraConfig(StrictBaseModel):
    """
    AdaLoRA (Adaptive Low-Rank Adaptation) configuration.

    AdaLoRA dynamically allocates rank budget during training,
    pruning less important singular values while keeping important ones.

    Used with training.type: "adalora"

    REQUIRED fields (explicit config v6.0):
    - AdaLoRA core: init_r, target_r
    - Common LoRA: lora_alpha, lora_dropout, bias, target_modules

    OPTIONAL fields (scheduling + advanced):
    - tinit, tfinal, delta_t, beta1, beta2
    """

    # =========================================================================
    # REQUIRED: AdaLoRA-specific parameters
    # =========================================================================
    init_r: int = Field(..., ge=1, le=LORA_RANK_MAX, description="REQUIRED: Initial rank for all adapters")
    target_r: int = Field(..., ge=1, le=LORA_RANK_MAX, description="REQUIRED: Target average rank (after pruning)")
    total_step: int = Field(
        ...,
        ge=1,
        description="REQUIRED: Total training steps for the pruning schedule (dataset_size / batch_size * epochs)",
    )

    # =========================================================================
    # OPTIONAL: AdaLoRA scheduling (reasonable defaults)
    # =========================================================================
    tinit: int = Field(
        default=ADALORA_TINIT_DEFAULT,
        ge=0,
        description="Optional: Steps before rank pruning begins (default: 200)",
    )
    tfinal: int = Field(
        default=ADALORA_TFINAL_DEFAULT,
        ge=1,
        description="Optional: Steps when rank pruning ends (default: 1000)",
    )
    delta_t: int = Field(
        default=10, ge=1, description="Optional: Interval between pruning steps (default: 10)", alias="deltaT"
    )
    beta1: float = Field(
        default=ADALORA_BETA_DEFAULT,
        ge=0.0,
        le=1.0,
        description="Optional: EMA coefficient for importance (default: 0.85)",
    )
    beta2: float = Field(
        default=ADALORA_BETA_DEFAULT,
        ge=0.0,
        le=1.0,
        description="Optional: EMA coefficient for second moment (default: 0.85)",
    )

    # =========================================================================
    # REQUIRED: Common LoRA parameters
    # =========================================================================
    lora_alpha: int = Field(..., ge=1, description="REQUIRED: LoRA alpha (scaling factor, typically 32)")
    lora_dropout: float = Field(..., ge=0.0, le=0.5, description="REQUIRED: Dropout for LoRA layers (0.05 recommended)")
    bias: Literal["none", "all", "lora_only"] = Field(..., description="REQUIRED: Bias type ('none' is standard)")
    target_modules: str | list[str] = Field(..., description="REQUIRED: Target modules ('all-linear' recommended)")

    @field_validator("bias", mode="before")
    @classmethod
    def validate_bias(cls, v: Any) -> Literal["none", "all", "lora_only"]:
        allowed = ["none", "all", "lora_only"]
        if v not in allowed:
            raise ValueError(f"bias must be one of {allowed}")
        return cast("Literal['none', 'all', 'lora_only']", v)

    @field_validator("target_modules", mode="before")
    @classmethod
    def validate_target_modules(cls, v: str | list[str] | None) -> str | list[str]:
        if v is None:
            return "all-linear"
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return v
        raise ValueError("target_modules must be 'all-linear' or list of module names")


__all__ = [
    "AdaLoraConfig",
]
