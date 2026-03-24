"""
Dataset Loader Factory.

Creates appropriate loader based on dataset source type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.utils.config import DatasetConfig, PipelineConfig
    from src.utils.container import IDatasetLoader


class DatasetLoaderFactory:
    """
    Factory for creating dataset loaders based on source type.

    Supports:
    - local: JsonDatasetLoader for local JSON/JSONL files
    - huggingface: HuggingFaceDatasetLoader for HF Hub datasets

    Example:
        factory = DatasetLoaderFactory(config)
        loader = factory.create_for_dataset(dataset_config)
        dataset = loader.load(source)
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize factory.

        Args:
            config: Pipeline configuration
        """
        self._config = config
        logger.debug("[DL_FACTORY:INIT] DatasetLoaderFactory initialized")

    def create_for_dataset(self, dataset_config: DatasetConfig) -> IDatasetLoader:
        """
        Create appropriate loader for dataset config.

        Args:
            dataset_config: Dataset configuration with source_type

        Returns:
            IDatasetLoader instance
        """
        source_type = dataset_config.get_source_type()

        if source_type == "huggingface":
            return self._create_huggingface_loader()
        else:  # local
            return self._create_json_loader()

    def create_for_source_type(self, source_type: str) -> IDatasetLoader:
        """
        Create loader by source type string.

        Args:
            source_type: "local" or "huggingface"

        Returns:
            IDatasetLoader instance
        """
        if source_type == "huggingface":
            return self._create_huggingface_loader()
        return self._create_json_loader()

    def create_default(self) -> IDatasetLoader:
        """
        Create default loader (JsonDatasetLoader).

        For backward compatibility.

        Returns:
            JsonDatasetLoader instance
        """
        return self._create_json_loader()

    def _create_json_loader(self) -> IDatasetLoader:
        """Create JsonDatasetLoader for local files."""
        from src.data.loaders import JsonDatasetLoader

        loader = JsonDatasetLoader(self._config)
        logger.debug("[DL_FACTORY:CREATED] JsonDatasetLoader")
        return loader

    def _create_huggingface_loader(self) -> IDatasetLoader:
        """Create HuggingFaceDatasetLoader for HF Hub."""
        from src.data.loaders import HuggingFaceDatasetLoader

        loader = HuggingFaceDatasetLoader(self._config)
        logger.debug("[DL_FACTORY:CREATED] HuggingFaceDatasetLoader")
        return loader


__all__ = ["DatasetLoaderFactory"]
