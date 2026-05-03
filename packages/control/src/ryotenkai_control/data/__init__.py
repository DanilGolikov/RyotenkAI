"""
Data Module - Dataset Loading

Simplified architecture (v3):
- TRL SFTTrainer handles datasets natively
- No adapters needed for standard formats (messages, text)
- JsonDatasetLoader for file loading

Supported formats:
- `messages`: ChatML format (TRL applies chat_template automatically)
- `text`: Plain text (TRL uses directly)

Example:
    from ryotenkai_pod.trainer.data_loaders import JsonDatasetLoader
    loader = JsonDatasetLoader(config)
    dataset = loader.load("data/train.jsonl")
"""

from .loaders import JsonDatasetLoader

__all__ = [
    "JsonDatasetLoader",
]
