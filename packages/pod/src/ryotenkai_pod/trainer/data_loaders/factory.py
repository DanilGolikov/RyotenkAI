"""
Dataset Loader Factory.

Creates appropriate loader based on dataset source type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_shared.config import DatasetConfig, PipelineConfig
    from ryotenkai_pod.trainer.container import IDatasetLoader


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
        from ryotenkai_shared.config import DatasetSourceHF, DatasetSourceLocal

        source = dataset_config.source
        if isinstance(source, DatasetSourceHF):
            return self._create_huggingface_loader()
        if isinstance(source, DatasetSourceLocal):
            return self._create_json_loader()
        # Discriminator covers all kinds today; defensive for future variants.
        raise ValueError(
            f"No loader registered for dataset source kind: {source.kind!r}"
        )

    def create_for_source_type(self, source_type: str) -> IDatasetLoader:
        """
        Create loader by source type string.

        Args:
            source_type: "local" or "huggingface"

        Returns:
            IDatasetLoader instance
        """
        if source_type == SOURCE_TYPE_HUGGINGFACE:
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
        from ryotenkai_pod.trainer.data_loaders import JsonDatasetLoader

        loader = JsonDatasetLoader(self._config)
        logger.debug("[DL_FACTORY:CREATED] JsonDatasetLoader")
        return loader

    def _create_huggingface_loader(self) -> IDatasetLoader:
        """Create HuggingFaceDatasetLoader for HF Hub."""
        from ryotenkai_pod.trainer.data_loaders import HuggingFaceDatasetLoader

        loader = HuggingFaceDatasetLoader(self._config)
        logger.debug("[DL_FACTORY:CREATED] HuggingFaceDatasetLoader")
        return loader


__all__ = ["DatasetLoaderFactory"]
