from pydantic import Field, field_validator

from ..base import StrictBaseModel


class ModelConfig(StrictBaseModel):
    """
    Model configuration for HuggingFace AutoModel.

    AutoModel auto-detects model architecture - no need for type field.
    Quantization (4-bit/8-bit) is controlled by training.type (qlora/lora).

    REQUIRED fields (explicit config v6.0):
    - name: Model identifier
    - torch_dtype: Must match your GPU capabilities
    - trust_remote_code: Model-specific requirement
    """

    # =========================================================================
    # REQUIRED: Core model parameters
    # =========================================================================
    name: str = Field(..., description="REQUIRED: Model name/path on HuggingFace (e.g., 'Qwen/Qwen2.5-7B-Instruct')")
    torch_dtype: str = Field(
        ...,
        description=(
            "REQUIRED: Torch dtype. Choose based on GPU:\n"
            "  - 'bfloat16': A100, H100 (recommended for modern GPUs)\n"
            "  - 'float16': V100, T4, older GPUs\n"
            "  - 'float32': CPU or compatibility mode\n"
            "  - 'auto': Auto-detect (may be suboptimal)"
        ),
    )
    trust_remote_code: bool = Field(
        ...,
        description=(
            "REQUIRED: Trust remote code when loading model.\n"
            "  - true: Required for Qwen, Phi, GLM\n"
            "  - false: Standard for Llama, Mistral, Gemma"
        ),
    )

    # =========================================================================
    # OPTIONAL: Advanced parameters (reasonable defaults)
    # =========================================================================
    tokenizer_name: str | None = Field(None, description="Optional: Tokenizer name (defaults to model name)")
    device_map: str = Field("auto", description="Optional: Device map for model loading (default: auto)")
    flash_attention: bool = Field(
        False, description="Optional: Use Flash Attention 2 for faster training (requires A100+)"
    )

    @field_validator("torch_dtype")
    @classmethod
    def validate_torch_dtype(cls, v: str) -> str:
        allowed = ["auto", "bfloat16", "float16", "float32"]
        if v not in allowed:
            raise ValueError(
                f"torch_dtype must be one of {allowed}.\n"
                f"Choose based on your GPU:\n"
                f"  - 'bfloat16': A100, H100 (recommended)\n"
                f"  - 'float16': V100, T4\n"
                f"  - 'float32': CPU or compatibility\n"
                f"  - 'auto': Auto-detect"
            )
        return v


__all__ = [
    "ModelConfig",
]
