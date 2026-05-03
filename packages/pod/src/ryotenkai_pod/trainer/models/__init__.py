"""
Model Module - Universal Model Loading

Simplified architecture (v3):
- HuggingFace AutoModel handles all architectures
- No model-specific adapters needed
- Single function: load_model_and_tokenizer()

Example:
    from src.training.models import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
"""

from .loader import load_model_and_tokenizer

# Backward compat alias
ModelLoader = load_model_and_tokenizer

__all__ = [
    "ModelLoader",
    "load_model_and_tokenizer",
]
