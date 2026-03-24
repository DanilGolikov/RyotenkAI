from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from ...base import StrictBaseModel
from ..constants import LORA_RANK_MAX


class LoraConfig(StrictBaseModel):
    """
    Unified LoRA/QLoRA configuration.

    Used for all LoRA-based methods:
    - type: "lora"  → full precision model + LoRA adapters
    - type: "qlora" → 4-bit quantized model + LoRA adapters

    REQUIRED fields (explicit config v6.0):
    - Base LoRA: r, lora_alpha, lora_dropout, bias, target_modules
    - Advanced: use_dora, use_rslora, init_lora_weights

    OPTIONAL fields (QLoRA quantization - best practice defaults):
    - bnb_4bit_quant_type, bnb_4bit_compute_dtype, bnb_4bit_use_double_quant
    """

    # =========================================================================
    # REQUIRED: Base LoRA parameters
    # =========================================================================
    r: int = Field(
        ...,
        ge=1,
        le=LORA_RANK_MAX,
        description="REQUIRED: LoRA rank (8-64 typical, higher = more capacity)",
    )
    lora_alpha: int = Field(..., ge=1, description="REQUIRED: LoRA alpha (scaling factor, typically 2*r)")
    lora_dropout: float = Field(..., ge=0.0, le=0.5, description="REQUIRED: Dropout for LoRA layers (0.05 recommended)")
    bias: Literal["none", "all", "lora_only"] = Field(
        ..., description="REQUIRED: Bias type ('none' is standard for LoRA)"
    )
    target_modules: str | list[str] = Field(
        ..., description="REQUIRED: Target modules ('all-linear' recommended for auto-detection)"
    )

    # =========================================================================
    # REQUIRED: Advanced LoRA variants
    # =========================================================================
    use_dora: bool = Field(..., description="REQUIRED: Enable DoRA (Weight-Decomposed LoRA, false for standard)")
    use_rslora: bool = Field(..., description="REQUIRED: Enable rsLoRA (Rank-Stabilized LoRA, false for standard)")
    init_lora_weights: str = Field(
        ...,
        description="REQUIRED: Weight initialization ('gaussian' is standard, options: eva, pissa, olora, loftq)",
    )

    # =========================================================================
    # OPTIONAL: QLoRA quantization (used when training.type == \"qlora\")
    # Best-practice defaults, rarely need to change
    # =========================================================================
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="Optional: Quantization type ('nf4' recommended, or 'fp4')",
    )
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16",
        description="Optional: Compute dtype for 4-bit base ('bfloat16' or 'float16')",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Optional: Use nested quantization for memory efficiency (recommended: true)",
    )

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

    @field_validator("bnb_4bit_quant_type")
    @classmethod
    def validate_quant_type(cls, v: str) -> str:
        allowed = ["nf4", "fp4"]
        if v not in allowed:
            raise ValueError(f"bnb_4bit_quant_type must be one of {allowed}")
        return v

    @field_validator("bnb_4bit_compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, v: str) -> str:
        allowed = ["bfloat16", "float16"]
        if v not in allowed:
            raise ValueError(f"bnb_4bit_compute_dtype must be one of {allowed}")
        return v

    @field_validator("init_lora_weights")
    @classmethod
    def validate_init_weights(cls, v: str) -> str:
        allowed = ["gaussian", "eva", "pissa", "pissa_niter_[number]", "olora", "loftq", "true", "false"]
        # Allow pissa_niter_N pattern
        if v.startswith("pissa_niter_") or v in allowed:
            return v
        raise ValueError(f"init_lora_weights must be one of {allowed}")

    @model_validator(mode="after")
    def _run_model_validators(self) -> LoraConfig:
        """
        Centralized cross-field validators for this config.

        Convention:
        - Keep ONE `@model_validator(mode="after")` method per config model.
        - Delegate validation logic to `src.config.validators.*`.
        """
        # Local import to avoid circular imports.
        from ...validators.training import validate_lora_config

        validate_lora_config(self)
        return self


# Backward compatibility aliases (historical names used across codebase)
LoRAConfig = LoraConfig
QLoRAConfig = LoraConfig  # QLoRA uses same config, difference is in training.type


__all__ = [
    "LoRAConfig",
    "LoraConfig",
    "QLoRAConfig",
]
