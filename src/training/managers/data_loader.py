"""
Data Loader Manager - Single Responsibility: Dataset loading and preparation

Handles all dataset operations:
- Loading training and evaluation datasets
- Dataset validation
- Dataset limiting/sampling
- Dataset statistics

Follows Single Responsibility Principle (SOLID).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from datasets import Dataset, load_dataset

from src.constants import STRATEGY_SFT
from src.training.managers.constants import HF_SPLIT_TRAIN
from src.utils.logger import logger
from src.utils.result import DataLoaderError, Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.utils.config import PipelineConfig


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


@dataclass
class DataLoaderEventCallbacks:
    """Callbacks for DataLoader events."""

    on_dataset_loaded: Callable[[int, int | None], None] | None = None
    # Args: train_samples, eval_samples (or None)

    on_dataset_validated: Callable[[bool], None] | None = None
    # Args: is_valid


class DataLoaderManager:
    """
    Manager for dataset loading and preparation.

    Single Responsibility: Handle all dataset loading operations.

    Example:
        loader = DataLoaderManager(config)
        result = loader.load_datasets()
        if result.is_success():
            train_ds, eval_ds = result.unwrap()
    """

    def __init__(
        self,
        config: PipelineConfig,
        callbacks: DataLoaderEventCallbacks | None = None,
    ):
        """
        Initialize data loader manager.

        Args:
            config: Pipeline configuration
            callbacks: Optional event callbacks
        """
        self.config = config
        self._callbacks = callbacks or DataLoaderEventCallbacks()

    def load_datasets(
        self, strategy_type: str = STRATEGY_SFT
    ) -> Result[tuple[Dataset, Dataset | None], DataLoaderError]:
        """
        Load training and evaluation datasets.

        Args:
            strategy_type: Strategy type for auto-generating training paths (default: SFT)

        Returns:
            Result[Tuple[Dataset, Optional[Dataset]], str]: (train, eval) datasets or error
        """
        try:
            # Get default dataset config from registry
            dataset_config = self.config.get_primary_dataset()

            source_type = dataset_config.get_source_type()

            # Load training dataset
            eval_dataset = None
            if source_type == "huggingface":
                if dataset_config.source_hf is None:
                    return Err(
                        DataLoaderError(
                            message="Dataset source_type='huggingface' requires source_hf",
                            code="DATA_LOADER_HF_SOURCE_MISSING",
                        )
                    )
                logger.info(f"Loading HF training data: {dataset_config.source_hf.train_id}")
                train_dataset = cast(
                    "Dataset",  # noqa: WPS226
                    load_dataset(
                        dataset_config.source_hf.train_id,
                        split=HF_SPLIT_TRAIN,
                        trust_remote_code=True,
                    ),
                )
                if dataset_config.source_hf.eval_id:
                    logger.info(f"Loading HF evaluation data: {dataset_config.source_hf.eval_id}")
                    eval_dataset = cast(
                        "Dataset",
                        load_dataset(
                            dataset_config.source_hf.eval_id,
                            split=HF_SPLIT_TRAIN,
                            trust_remote_code=True,
                        ),
                    )
            else:
                if dataset_config.source_local is None:
                    return Err(
                        DataLoaderError(
                            message="Dataset source_type='local' requires source_local",
                            code="DATA_LOADER_LOCAL_SOURCE_MISSING",
                        )
                    )

                # Auto-generate training path from local_paths (v6.0)
                # Pattern: data/{strategy_type}/{basename}
                local_train = dataset_config.source_local.local_paths.train
                train_basename = Path(local_train).name
                train_path = f"data/{strategy_type}/{train_basename}"

                logger.info(f"Loading training data: {train_path} (auto-generated)")
                train_dataset = cast(
                    "Dataset",
                    load_dataset("json", data_files=train_path, split=HF_SPLIT_TRAIN),
                )

                local_eval = dataset_config.source_local.local_paths.eval
                if local_eval:
                    eval_basename = Path(local_eval).name
                    eval_path = f"data/{strategy_type}/{eval_basename}"
                    logger.info(f"Loading evaluation data: {eval_path} (auto-generated)")
                    eval_dataset = cast(
                        "Dataset",
                        load_dataset("json", data_files=eval_path, split=HF_SPLIT_TRAIN),
                    )

            # Limit samples if specified
            if dataset_config.max_samples:
                train_dataset = self._limit_samples(train_dataset, dataset_config.max_samples)
                if eval_dataset:
                    eval_limit = dataset_config.max_samples // 10
                    eval_dataset = self._limit_samples(eval_dataset, eval_limit)

            # Log statistics
            self._log_statistics(train_dataset, eval_dataset)

            # Fire callback
            if self._callbacks.on_dataset_loaded:
                eval_len = len(eval_dataset) if eval_dataset is not None else None
                self._callbacks.on_dataset_loaded(len(train_dataset), eval_len)

            return Ok((train_dataset, eval_dataset))

        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            return Err(DataLoaderError(message=f"Dataset loading failed: {e!s}", code="DATA_LOADER_LOAD_FAILED"))

    @staticmethod
    def _limit_samples(dataset: Dataset, max_samples: int) -> Dataset:
        """
        Limit dataset to max_samples.

        Args:
            dataset: Input dataset
            max_samples: Maximum number of samples

        Returns:
            Limited dataset
        """
        if len(dataset) <= max_samples:
            return dataset

        logger.info(f"Limiting dataset from {len(dataset)} to {max_samples} samples")
        return dataset.select(range(max_samples))

    @staticmethod
    def _log_statistics(train_dataset: Dataset, eval_dataset: Dataset | None) -> None:
        """
        Log dataset statistics.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        logger.info(f"Training samples: {len(train_dataset):,}")

        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset):,}")
        else:
            logger.info("No evaluation dataset provided")

        # Log sample keys
        if len(train_dataset) > 0:
            sample_keys = list(train_dataset[0].keys())
            logger.info(f"Dataset keys: {sample_keys}")

    def validate_datasets(self, train_dataset: Dataset, eval_dataset: Dataset | None) -> Result[bool, DataLoaderError]:
        """
        Validate datasets structure.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Result[bool, str]: True if valid, error message otherwise
        """
        try:
            # Check training dataset
            if len(train_dataset) == 0:
                return Err(DataLoaderError(message="Training dataset is empty", code="DATA_LOADER_EMPTY_TRAIN"))

            # Check evaluation dataset if provided
            if eval_dataset and len(eval_dataset) == 0:
                return Err(DataLoaderError(message="Evaluation dataset is empty", code="DATA_LOADER_EMPTY_EVAL"))

            # Validate first sample structure
            sample = train_dataset[0]
            if not isinstance(sample, dict):
                return Err(
                    DataLoaderError(
                        message="Dataset samples must be dictionaries",
                        code="DATA_LOADER_INVALID_SAMPLE_FORMAT",
                    )
                )

            logger.info("Datasets validated successfully")

            # Fire callback
            if self._callbacks.on_dataset_validated:
                self._callbacks.on_dataset_validated(True)

            return Ok(True)

        except Exception as e:
            return Err(DataLoaderError(message=f"Dataset validation failed: {e!s}", code="DATA_LOADER_VALIDATE_FAILED"))


__all__ = ["DataLoaderManager"]
