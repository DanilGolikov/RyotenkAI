"""
Dataset Loaders package.

Provides pluggable dataset loading strategies:
- JsonDatasetLoader: Load from local JSON/JSONL files
- HuggingFaceDatasetLoader: Load from HuggingFace Hub
- DatasetLoaderFactory: Auto-select loader by source type

Usage:
    from ryotenkai_pod.trainer.data_loaders import DatasetLoaderFactory

    # Auto-select loader based on dataset config
    factory = DatasetLoaderFactory(config)
    loader = factory.create_for_dataset(dataset_config)

    # Or explicitly:
    from ryotenkai_pod.trainer.data_loaders import JsonDatasetLoader, HuggingFaceDatasetLoader

    # Local JSON files
    loader = JsonDatasetLoader(config)
    result = loader.load_for_phase(phase_config)

    # HuggingFace Hub
    loader = HuggingFaceDatasetLoader(config)
    result = loader.load("tatsu-lab/alpaca", split="train")
"""

from ryotenkai_pod.trainer.data_loaders.base import BaseDatasetLoader
from ryotenkai_pod.trainer.data_loaders.factory import DatasetLoaderFactory
from ryotenkai_pod.trainer.data_loaders.hf_loader import HuggingFaceDatasetLoader
from ryotenkai_pod.trainer.data_loaders.json_loader import JsonDatasetLoader
from ryotenkai_pod.trainer.data_loaders.multi_source_loader import MultiSourceDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "DatasetLoaderFactory",
    "HuggingFaceDatasetLoader",
    "JsonDatasetLoader",
    "MultiSourceDatasetLoader",
]
