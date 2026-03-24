"""
Dataset Loaders package.

Provides pluggable dataset loading strategies:
- JsonDatasetLoader: Load from local JSON/JSONL files
- HuggingFaceDatasetLoader: Load from HuggingFace Hub
- DatasetLoaderFactory: Auto-select loader by source type

Usage:
    from src.data.loaders import DatasetLoaderFactory

    # Auto-select loader based on dataset config
    factory = DatasetLoaderFactory(config)
    loader = factory.create_for_dataset(dataset_config)

    # Or explicitly:
    from src.data.loaders import JsonDatasetLoader, HuggingFaceDatasetLoader

    # Local JSON files
    loader = JsonDatasetLoader(config)
    result = loader.load_for_phase(phase_config)

    # HuggingFace Hub
    loader = HuggingFaceDatasetLoader(config)
    result = loader.load("tatsu-lab/alpaca", split="train")
"""

from src.data.loaders.base import BaseDatasetLoader
from src.data.loaders.factory import DatasetLoaderFactory
from src.data.loaders.hf_loader import HuggingFaceDatasetLoader
from src.data.loaders.json_loader import JsonDatasetLoader
from src.data.loaders.multi_source_loader import MultiSourceDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "DatasetLoaderFactory",
    "HuggingFaceDatasetLoader",
    "JsonDatasetLoader",
    "MultiSourceDatasetLoader",
]
