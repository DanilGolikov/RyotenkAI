"""
Base Dataset Loader - Abstract interface for dataset loading.

Provides common interface for different dataset sources:
- Local files (JSON, JSONL)
- HuggingFace Hub
- Custom sources

Single Responsibility: Define contract for dataset loading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset

    from src.utils.config import PipelineConfig, StrategyPhaseConfig
    from src.utils.result import DataLoaderError, Result


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    Implements IDatasetLoader interface with common functionality.

    Subclasses must implement:
        - load(): Load dataset from source
        - validate_source(): Validate source exists/accessible

    Attributes:
        config: Pipeline configuration with dataset paths

    Example:
        >>> class CustomLoader(BaseDatasetLoader):
        ...     def load(self, source, split="train", max_samples=None):
        ...         # Custom loading logic
        ...         pass
        ...
        ...     def validate_source(self, source):
        ...         # Custom validation
        ...         return True
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize base loader.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        logger.debug(f"[{self._log_prefix}:INIT] DatasetLoader initialized")

    @property
    def _log_prefix(self) -> str:
        """Log prefix for this loader type."""
        return "DL"

    # =========================================================================
    # ABSTRACT METHODS (must implement)
    # =========================================================================

    @abstractmethod
    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load dataset from source.

        Args:
            source: Path to file or dataset ID
            split: Dataset split to load (default: "train")
            max_samples: Limit number of samples (optional)

        Returns:
            Loaded dataset

        Raises:
            FileNotFoundError: If source not found
            ValueError: If source format invalid
        """
        ...

    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate that dataset source exists/is accessible.

        Args:
            source: Path or dataset ID

        Returns:
            True if source is valid
        """
        ...

    # =========================================================================
    # COMMON METHODS (shared implementation)
    # =========================================================================

    def load_for_phase(
        self,
        phase: StrategyPhaseConfig,
    ) -> Result[Dataset, DataLoaderError]:
        """
        Load dataset for a training phase.

        Uses config to resolve dataset path for the phase.

        Args:
            phase: Strategy phase configuration with dataset reference

        Returns:
            Result with loaded dataset or DataLoaderError

        Example:
            result = loader.load_for_phase(phase_config)
            if result.is_success():
                dataset = result.unwrap()
        """
        from src.utils.result import DataLoaderError, Err, Ok

        try:
            # Get dataset config for this phase
            dataset_config = self.config.get_dataset_for_strategy(phase)
            if dataset_config.get_source_type() != "local":
                return Err(
                    DataLoaderError(
                        message="BaseDatasetLoader.load_for_phase supports only local datasets (use MultiSourceDatasetLoader)",
                        code="DATA_LOADER_LOCAL_ONLY",
                    )
                )

            if dataset_config.source_local is None:
                return Err(
                    DataLoaderError(
                        message="source_type='local' requires source_local",
                        code="DATA_LOADER_LOCAL_SOURCE_MISSING",
                    )
                )

            train_path = dataset_config.source_local.local_paths.train
            resolved_train_path = self.config.resolve_path(train_path)
            assert resolved_train_path is not None
            train_path_str = str(resolved_train_path)
            max_samples = dataset_config.max_samples

            logger.debug(
                f"[{self._log_prefix}:PHASE_LOAD] phase={phase.strategy_type}, "
                f"path={train_path} -> resolved={train_path_str}"
            )

            # Validate source
            if not self.validate_source(train_path_str):
                error_msg = f"Dataset source not found: {train_path} (resolved: {train_path_str})"
                logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
                return Err(DataLoaderError(message=error_msg, code="DATA_LOADER_FILE_NOT_FOUND"))

            # Load dataset
            logger.info(f"   Loading dataset: {train_path} (resolved: {train_path_str})")
            dataset = self.load(train_path_str, split="train", max_samples=max_samples)

            logger.debug(f"[{self._log_prefix}:LOADED] samples={len(dataset)}")
            return Ok(dataset)

        except KeyError as e:
            error_msg = f"Dataset '{phase.dataset}' not found in config: {e}"
            logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
            return Err(DataLoaderError(message=error_msg, code="DATA_LOADER_DATASET_NOT_FOUND"))

        except Exception as e:
            error_msg = f"Failed to load dataset: {e}"
            logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
            return Err(DataLoaderError(message=error_msg, code="DATA_LOADER_LOAD_FAILED"))

    @staticmethod
    def _apply_max_samples(
        dataset: Dataset,
        max_samples: int | None,
    ) -> Dataset:
        """
        Apply max_samples limit to dataset.

        Args:
            dataset: Original dataset
            max_samples: Maximum samples (None = no limit)

        Returns:
            Dataset with limit applied (if specified)
        """
        if max_samples is None:
            return dataset

        original_size = len(dataset)
        if original_size <= max_samples:
            return dataset

        limited = dataset.select(range(max_samples))
        logger.info(f"   Limited to {max_samples} samples (from {original_size})")
        return limited


__all__ = ["BaseDatasetLoader"]
