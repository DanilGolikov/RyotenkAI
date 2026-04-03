"""
Universal model loader - replaces model adapters.

Uses HuggingFace AutoModel for all architectures:
- Qwen, Llama, Gemma, DeepSeek, Phi, Mistral, etc.

Key simplification:
- No model-specific adapters needed
- AutoModel auto-detects architecture
- Quantization controlled by config

Example:
    from src.training.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch as torch_module
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.utils.config import PipelineConfig

logger = get_logger(__name__)

torch: Any = torch_module


def load_model_and_tokenizer(
    config: PipelineConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer using AutoModel.

    Replaces ModelFactory + ModelAdapters with single universal function.
    Works with ANY HuggingFace model without model-specific code.

    Quantization is determined by:
    - training.type == "qlora" → 4-bit quantization
    - training.type == "lora" → no quantization
    - training.type == "full_ft" → no quantization

    Args:
        config: Pipeline configuration with model and training settings

    Returns:
        tuple[model, tokenizer]: Loaded model and tokenizer

    Example:
        config = load_config("config/pipeline.yaml")
        model, tokenizer = load_model_and_tokenizer(config)
        print(model.config.model_type)  # "qwen2", "llama", etc.
    """
    model_name = config.model.name
    logger.info(f"📦 Loading model: {model_name}")

    # Determine quantization from training type
    use_4bit = config.training.get_effective_load_in_4bit()

    # Create quantization config for QLoRA
    bnb_config = None
    if use_4bit:
        adapter_cfg = config.training.get_adapter_config()
        quant_type = adapter_cfg.bnb_4bit_quant_type
        compute_dtype = _get_torch_dtype(adapter_cfg.bnb_4bit_compute_dtype)
        use_double_quant = adapter_cfg.bnb_4bit_use_double_quant

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
        logger.info(f"Using 4-bit quantization (QLoRA): quant={quant_type}, compute_dtype={compute_dtype}, double_quant={use_double_quant}")

    # Determine torch dtype
    torch_dtype = _get_torch_dtype(config.model.torch_dtype)

    # Load model
    logger.debug(f"[LOADER:MODEL] name={model_name}, dtype={torch_dtype}, 4bit={use_4bit}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # transformers>=4.57: `torch_dtype` kwarg is deprecated in favor of `dtype`
        dtype=torch_dtype,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if config.model.flash_attention else None,
    )

    # Prepare model for training
    model.config.use_cache = False  # Required for gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    logger.info(f"✅ Model loaded: {model.config.model_type}")
    logger.debug(f"[LOADER:MODEL_LOADED] type={model.config.model_type}, params={model.num_parameters():,}")

    # Load tokenizer
    tokenizer_name = config.model.tokenizer_name or model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )

    # Setup pad token (critical for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.debug("[LOADER:TOKENIZER] pad_token set to eos_token")

    tokenizer.padding_side = "right"  # Required for causal LM

    logger.info("✅ Tokenizer loaded")
    logger.debug(f"[LOADER:TOKENIZER_LOADED] vocab_size={len(tokenizer)}")

    return model, tokenizer


def _get_torch_dtype(dtype_str: str) -> torch.dtype:  # pyright: ignore[reportInvalidTypeForm]
    """
    Convert string dtype to torch.dtype.

    Args:
        dtype_str: One of "auto", "bfloat16", "float16", "float32"

    Returns:
        torch.dtype
    """
    dtype_map = {
        "auto": torch.bfloat16,  # Default to bfloat16 for modern GPUs
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


__all__ = ["load_model_and_tokenizer"]
