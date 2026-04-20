from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from ...base import StrictBaseModel
from ..constants import LORA_RANK_MAX


class LoraConfig(StrictBaseModel):
    """Plain LoRA adapter configuration (used when ``training.type == 'lora'``).

    Carries only fields that apply to full-precision LoRA. QLoRA-specific
    quantization knobs (``bnb_4bit_*``) live on :class:`QloraConfig`, which
    subclasses this model. Splitting the two means a ``training.lora:`` block
    cannot accidentally accept 4-bit settings (strict base model), and the
    generated JSON Schema for the Web form only exposes quantization fields
    under the ``qlora:`` branch.
    """

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

    use_dora: bool = Field(..., description="REQUIRED: Enable DoRA (Weight-Decomposed LoRA, false for standard)")
    use_rslora: bool = Field(..., description="REQUIRED: Enable rsLoRA (Rank-Stabilized LoRA, false for standard)")
    init_lora_weights: str = Field(
        ...,
        description="REQUIRED: Weight initialization ('gaussian' is standard, options: eva, pissa, olora, loftq)",
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

    @field_validator("init_lora_weights")
    @classmethod
    def validate_init_weights(cls, v: str) -> str:
        allowed = ["gaussian", "eva", "pissa", "pissa_niter_[number]", "olora", "loftq", "true", "false"]
        if v.startswith("pissa_niter_") or v in allowed:
            return v
        raise ValueError(f"init_lora_weights must be one of {allowed}")

    @model_validator(mode="after")
    def _run_model_validators(self) -> LoraConfig:
        """Centralized cross-field validators for this config."""
        from ...validators.training import validate_lora_config

        validate_lora_config(self)
        return self


class QloraConfig(LoraConfig):
    """QLoRA adapter configuration (used when ``training.type == 'qlora'``).

    Adds 4-bit quantization knobs on top of the base :class:`LoraConfig`.
    Defaults reflect the recommended setup (``nf4`` + bfloat16 compute + nested
    double quantization), so users rarely need to touch these.
    """

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


# Backward-compatibility aliases (historical names used across the codebase).
LoRAConfig = LoraConfig
QLoRAConfig = QloraConfig


__all__ = [
    "LoRAConfig",
    "LoraConfig",
    "QLoRAConfig",
    "QloraConfig",
]
