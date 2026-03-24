"""
DatasetLoader - Load and validate datasets for training phases.

Single Responsibility: Dataset loading and validation from config paths.

Supports:
- Direct file paths
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, load_dataset

from src.constants import STRATEGY_SFT
from src.training.managers.constants import HF_SPLIT_TRAIN
from src.utils.logger import logger
from src.utils.result import DataLoaderError, Err, Ok, Result

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, StrategyPhaseConfig


class DatasetLoader:
    """
    Handles dataset loading and validation for training phases.

    Responsibilities:
    - Load dataset from config paths
    - Apply max_samples limit if configured
    - Validate dataset exists

    Example:
        loader = DatasetLoader(config)
        result = loader.load_for_phase(phase_config)
        if result.is_success():
            dataset = result.unwrap()
    """

    def __init__(
        self,
        config: PipelineConfig,
    ):
        """
        Initialize DatasetLoader.

        Args:
            config: Pipeline configuration with dataset paths
        """
        self.config = config
        logger.debug("[DL:INIT] DatasetLoader initialized")

    def load_for_phase(
        self,
        phase: StrategyPhaseConfig,
    ) -> Result[tuple[Dataset, Dataset | None], DataLoaderError]:
        """
        Load dataset for a training phase.

        Args:
            phase: Strategy phase configuration with dataset reference

        Returns:
            Result with loaded dataset or error message
        """
        try:
            dataset_config = self.config.get_dataset_for_strategy(phase)

            # New schema supports: local (training_paths auto-generated) and HuggingFace (train_id/eval_id)
            source_type = dataset_config.get_source_type()

            if source_type == "huggingface":
                loaded_result = self._load_hf_datasets(dataset_config)
            else:
                # Pass strategy_type for auto-generating training paths
                loaded_result = self._load_local_datasets(dataset_config, strategy_type=phase.strategy_type)

            if loaded_result.is_failure():
                return loaded_result

            train_dataset, eval_dataset = loaded_result.unwrap()

            # Apply max_samples if configured
            if dataset_config.max_samples:
                original_train = len(train_dataset)
                train_dataset = train_dataset.select(range(min(len(train_dataset), dataset_config.max_samples)))
                logger.info(f"   Limited train to {len(train_dataset)} samples (from {original_train})")

                if eval_dataset is not None:
                    eval_limit = max(dataset_config.max_samples // 10, 1)
                    original_eval = len(eval_dataset)
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), eval_limit)))
                    logger.info(f"   Limited eval to {len(eval_dataset)} samples (from {original_eval})")

            logger.debug(
                f"[DL:LOADED] train={len(train_dataset)}, eval={len(eval_dataset) if eval_dataset is not None else 0}"
            )
            return Ok((train_dataset, eval_dataset))

        except KeyError as e:
            error_msg = f"Dataset '{phase.dataset}' not found in config: {e}"
            logger.error(f"[DL:ERROR] {error_msg}")
            return Err(DataLoaderError(message=error_msg, code="DATA_LOADER_DATASET_NOT_FOUND"))

        except Exception as e:
            error_msg = f"Failed to load dataset: {e}"
            logger.error(f"[DL:ERROR] {error_msg}")
            return Err(DataLoaderError(message=error_msg, code="DATA_LOADER_LOAD_FAILED"))

    @staticmethod
    def _load_hf_datasets(dataset_config: Any) -> Result[tuple[Dataset, Dataset | None], DataLoaderError]:
        """Load HuggingFace datasets (train + optional eval)."""
        if dataset_config.source_hf is None:
            return Err(
                DataLoaderError(
                    message="Dataset source_type='huggingface' requires source_hf",
                    code="DATA_LOADER_HF_SOURCE_MISSING",
                )
            )

        train_id = dataset_config.source_hf.train_id
        eval_id = dataset_config.source_hf.eval_id

        logger.info(f"   Loading HF train dataset: {train_id}")
        train_dataset = cast(
            "Dataset",  # noqa: WPS226
            load_dataset(train_id, split=HF_SPLIT_TRAIN, trust_remote_code=True),
        )
        eval_dataset: Dataset | None = None
        if eval_id:
            logger.info(f"   Loading HF eval dataset: {eval_id}")
            eval_dataset = cast(
                "Dataset",
                load_dataset(eval_id, split=HF_SPLIT_TRAIN, trust_remote_code=True),
            )

        return Ok((train_dataset, eval_dataset))

    @staticmethod
    def _load_local_datasets(
        dataset_config: Any, strategy_type: str = STRATEGY_SFT
    ) -> Result[tuple[Dataset, Dataset | None], DataLoaderError]:
        """
        Load local datasets from training_paths (auto-generated if missing).

        Args:
            dataset_config: Dataset configuration
            strategy_type: Current strategy type (used for auto-generating paths if needed)

        Returns:
            Result with (train_dataset, eval_dataset) or error

        Note (v6.0):
            training_paths removed from config - auto-generated as:
                data/{strategy_type}/{basename(local_paths.train)}
        """
        if dataset_config.source_local is None:
            return Err(
                DataLoaderError(
                    message="Dataset source_type='local' requires source_local",
                    code="DATA_LOADER_LOCAL_SOURCE_MISSING",
                )
            )

        # Auto-generate training path from local_paths
        # Pattern: data/{strategy_type}/{basename}
        local_train = dataset_config.source_local.local_paths.train
        train_basename = Path(local_train).name
        train_rel = f"data/{strategy_type}/{train_basename}"

        train_path = Path(train_rel)
        logger.debug(f"[DL:LOADING] train_path={train_path} (auto-generated from local_paths)")
        if not train_path.exists():
            error_msg = f"Dataset file not found: {train_path}"
            logger.error(f"[DL:ERROR] {error_msg}")
            return Err(DataLoaderError(message=error_msg, code="DATA_LOADER_FILE_NOT_FOUND"))

        logger.info(f"   Loading dataset: {train_path}")
        train_dataset = cast(
            "Dataset",
            load_dataset("json", data_files=str(train_path), split=HF_SPLIT_TRAIN),
        )

        eval_dataset: Dataset | None = None
        local_eval = dataset_config.source_local.local_paths.eval
        if local_eval:
            eval_basename = Path(local_eval).name
            eval_rel = f"data/{strategy_type}/{eval_basename}"
            eval_path = Path(eval_rel)
            if eval_path.exists():
                logger.info(f"   Loading eval dataset: {eval_path}")
                eval_dataset = cast(
                    "Dataset",
                    load_dataset("json", data_files=str(eval_path), split=HF_SPLIT_TRAIN),
                )
            else:
                logger.warning(f"[DL:EVAL_MISSING] eval dataset not found: {eval_path} (skipping)")

        return Ok((train_dataset, eval_dataset))

    @staticmethod
    def validate_exists(train_path: str) -> bool:
        """
        Check if dataset file exists.

        Args:
            train_path: Path to training data file

        Returns:
            True if file exists
        """
        exists = Path(train_path).exists()
        logger.debug(f"[DL:VALIDATE] path={train_path}, exists={exists}")
        return exists

    @staticmethod
    def load(
        source: str,
        split: str = HF_SPLIT_TRAIN,
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load dataset from source.

        Implements IDatasetLoader interface.

        Args:
            source: File path to dataset
            split: Dataset split (default: "train")
            max_samples: Maximum samples to load

        Returns:
            Loaded dataset
        """
        logger.debug(f"[DL:LOAD] source={source}")

        dataset = load_dataset("json", data_files=source, split=split)
        dataset = cast("Dataset", dataset)

        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        return dataset

    @staticmethod
    def validate_source(source: str) -> bool:
        """
        Validate dataset source exists.

        Implements IDatasetLoader interface.

        Args:
            source: File path to dataset

        Returns:
            True if source is valid
        """
        return Path(source).exists()


__all__ = ["DatasetLoader"]
