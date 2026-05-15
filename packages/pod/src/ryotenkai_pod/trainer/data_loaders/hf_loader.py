"""
HuggingFace Dataset Loader - Load datasets from HuggingFace Hub.

Supports:
- Public datasets from HuggingFace Hub
- Private datasets with HF token
- Dataset subsets and configurations

Single Responsibility: Load datasets from HuggingFace Hub.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

from datasets import load_dataset

from ryotenkai_pod.trainer.data_loaders.base import BaseDatasetLoader
from ryotenkai_shared.errors import (
    DatasetLoadFailedError,
    HFAuthFailedError,
    HFNotFoundError,
)
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset

    from ryotenkai_shared.config import PipelineConfig


# Tokens used to classify huggingface_hub exception messages into typed
# domain exceptions. Kept as module-level constants so the mapping is
# explicit (and mutation-testable).
_HF_AUTH_TOKENS = ("401", "unauthor", "forbidden", "403", "auth", "token")
_HF_NOT_FOUND_TOKENS = ("404", "not found", "does not exist", "no such")


def _classify_hf_failure(
    source: str,
    exc: Exception,
) -> DatasetLoadFailedError | HFAuthFailedError | HFNotFoundError:
    """Translate a raw huggingface_hub/datasets exception to a typed one.

    Keeps the classification in one place so both ``load()`` and
    ``load_for_phase()`` produce the same shape for the same underlying
    failure.

    Args:
        source: HF dataset identifier (for ``detail``/``context``).
        exc: original exception raised by the HF stack.

    Returns:
        A concrete typed exception (caller decides whether to raise).
    """
    msg = str(exc).lower()
    if any(tok in msg for tok in _HF_AUTH_TOKENS):
        return HFAuthFailedError(
            detail=f"HuggingFace authentication failed for dataset '{source}': {exc}",
            context={"dataset": source, "legacy_code": "DATA_LOADER_HF_AUTH_FAILED"},
            cause=exc,
        )
    if any(tok in msg for tok in _HF_NOT_FOUND_TOKENS):
        return HFNotFoundError(
            detail=f"HuggingFace dataset not found: '{source}': {exc}",
            context={"dataset": source, "legacy_code": "DATA_LOADER_HF_DATASET_NOT_FOUND"},
            cause=exc,
        )
    return DatasetLoadFailedError(
        detail=f"Failed to load HuggingFace dataset '{source}': {exc}",
        context={"dataset": source, "legacy_code": "DATA_LOADER_LOAD_FAILED"},
        cause=exc,
    )


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """
    Load datasets from HuggingFace Hub.

    Handles:
    - Public datasets (e.g., "tatsu-lab/alpaca")
    - Private datasets with HF_TOKEN
    - Dataset configurations/subsets

    Attributes:
        config: Pipeline configuration
        token: HuggingFace API token (optional)

    Example:
        loader = HuggingFaceDatasetLoader(config)

        # Load public dataset
        dataset = loader.load("tatsu-lab/alpaca", split="train")

        # Load with subset
        dataset = loader.load("databricks/dolly-15k", split="train", max_samples=1000)

        # Load for training phase (raises DatasetLoadFailedError on failure)
        dataset = loader.load_for_phase(phase_config)
    """

    def __init__(self, config: PipelineConfig, token: str | None = None):
        """
        Initialize HuggingFace loader.

        Args:
            config: Pipeline configuration
            token: HuggingFace API token (optional, can use HF_TOKEN env var)
        """
        super().__init__(config)
        self.token = token or os.getenv("HF_TOKEN")

        if self.token:
            logger.debug(f"[{self._log_prefix}:INIT] Using HF token (length={len(self.token)})")
        else:
            logger.debug(f"[{self._log_prefix}:INIT] No HF token, using public access")

    @property
    def _log_prefix(self) -> str:
        """Log prefix for HuggingFace loader."""
        return "HF_DL"

    # =========================================================================
    # INTERFACE IMPLEMENTATION
    # =========================================================================

    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
        subset: str | None = None,
    ) -> Dataset:
        """
        Load dataset from HuggingFace Hub.

        Args:
            source: HuggingFace dataset ID (e.g., "tatsu-lab/alpaca")
            split: Dataset split to load (default: "train")
            max_samples: Limit number of samples (optional)
            subset: Dataset configuration/subset name (optional)

        Returns:
            Loaded dataset

        Raises:
            HFAuthFailedError: token rejected / unauthorised access.
            HFNotFoundError: dataset id / revision does not exist.
            DatasetLoadFailedError: any other HF / parse failure.
        """
        logger.debug(f"[{self._log_prefix}:LOADING] dataset={source}, split={split}, subset={subset}")

        try:
            # Load from HuggingFace Hub
            dataset = load_dataset(
                source,
                name=subset,
                split=split,
                token=self.token,
                trust_remote_code=True,  # Required for some datasets
            )

            dataset = cast("Dataset", dataset)
            logger.debug(f"[{self._log_prefix}:RAW_LOADED] samples={len(dataset)}")

            # Apply max_samples limit if specified
            dataset = self._apply_max_samples(dataset, max_samples)

            return dataset

        except Exception as e:
            error_msg = f"Failed to load HuggingFace dataset '{source}': {e}"
            logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
            raise _classify_hf_failure(source, e) from e

    def validate_source(self, source: str) -> bool:
        """
        Validate that HuggingFace dataset exists.

        Attempts to get dataset info without downloading.

        Args:
            source: HuggingFace dataset ID

        Returns:
            True if dataset exists and is accessible
        """
        try:
            from huggingface_hub import dataset_info

            info = dataset_info(source, token=self.token)
            exists = info is not None

            logger.debug(f"[{self._log_prefix}:VALIDATE] dataset={source}, exists={exists}")
            return exists

        except Exception as e:
            logger.debug(f"[{self._log_prefix}:VALIDATE] dataset={source}, error={e}")
            return False

    # =========================================================================
    # OVERRIDE FOR HUGGINGFACE-SPECIFIC LOADING
    # =========================================================================

    def load_for_phase(self, phase) -> Dataset:
        """
        Load HuggingFace dataset for a training phase.

        Uses ``hf_dataset_id`` from ``DatasetConfig`` (not ``train_path``).

        Args:
            phase: Strategy phase configuration

        Returns:
            Loaded dataset.

        Raises:
            HFAuthFailedError: token rejected when loading dataset.
            HFNotFoundError: dataset id not found on the Hub.
            DatasetLoadFailedError: misconfigured source, missing
                ``train_id``, missing dataset config, or other load error.
        """
        from ryotenkai_shared.config import DatasetSourceHF

        try:
            # Get dataset config for this phase
            dataset_config = self.config.get_dataset_for_strategy(phase)
            source = dataset_config.source

            if not isinstance(source, DatasetSourceHF):
                raise DatasetLoadFailedError(
                    detail=(f"HuggingFaceDatasetLoader requires source.kind='huggingface'; got {source.kind!r}"),
                    context={
                        "legacy_code": "DATA_LOADER_HF_SOURCE_MISSING",
                        "source_kind": source.kind,
                    },
                )

            dataset_id = source.train_id
            max_samples = dataset_config.max_samples

            if not dataset_id:
                error_msg = "source_hf.train_id is required for HuggingFace source"
                logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
                raise DatasetLoadFailedError(
                    detail=error_msg,
                    context={"legacy_code": "DATA_LOADER_HF_TRAIN_ID_MISSING"},
                )

            logger.debug(f"[{self._log_prefix}:PHASE_LOAD] dataset={dataset_id}, split=train")

            # Validate dataset exists
            if not self.validate_source(dataset_id):
                error_msg = f"HuggingFace dataset not found: {dataset_id}"
                logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
                raise HFNotFoundError(
                    detail=error_msg,
                    context={
                        "dataset": dataset_id,
                        "legacy_code": "DATA_LOADER_HF_DATASET_NOT_FOUND",
                    },
                )

            # Load dataset
            logger.info(f"   Loading HuggingFace dataset: {dataset_id}")
            dataset = self.load(dataset_id, split="train", max_samples=max_samples, subset=None)

            logger.debug(f"[{self._log_prefix}:LOADED] samples={len(dataset)}")
            return dataset

        except (DatasetLoadFailedError, HFAuthFailedError, HFNotFoundError):
            # Already-typed exception — surface as-is.
            raise

        except KeyError as e:
            error_msg = f"Dataset '{phase.dataset}' not found in config: {e}"
            logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
            raise DatasetLoadFailedError(
                detail=error_msg,
                context={
                    "legacy_code": "DATA_LOADER_DATASET_NOT_FOUND",
                    "phase_dataset": getattr(phase, "dataset", None),
                },
                cause=e,
            )

        except Exception as e:
            error_msg = f"Failed to load HuggingFace dataset: {e}"
            logger.error(f"[{self._log_prefix}:ERROR] {error_msg}")
            raise DatasetLoadFailedError(
                detail=error_msg,
                context={"legacy_code": "DATA_LOADER_LOAD_FAILED"},
                cause=e,
            )

    # =========================================================================
    # ADDITIONAL METHODS
    # =========================================================================

    def get_dataset_info(self, dataset_id: str) -> dict | None:
        """
        Get information about a HuggingFace dataset.

        Args:
            dataset_id: HuggingFace dataset ID

        Returns:
            Dict with dataset info or None if not found
        """
        try:
            from huggingface_hub import dataset_info

            info = dataset_info(dataset_id, token=self.token)

            return {
                "id": info.id,
                "author": info.author,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "card_data": info.card_data,
            }

        except Exception as e:
            logger.debug(f"[{self._log_prefix}:INFO_ERROR] dataset={dataset_id}, error={e}")
            return None

    def list_splits(self, dataset_id: str, subset: str | None = None) -> list[str]:
        """
        List available splits for a dataset.

        Args:
            dataset_id: HuggingFace dataset ID
            subset: Dataset configuration (optional)

        Returns:
            List of available split names
        """
        try:
            from datasets import get_dataset_split_names

            splits = get_dataset_split_names(dataset_id, subset, token=self.token)
            logger.debug(f"[{self._log_prefix}:SPLITS] dataset={dataset_id}, splits={splits}")
            return splits

        except Exception as e:
            logger.debug(f"[{self._log_prefix}:SPLITS_ERROR] dataset={dataset_id}, error={e}")
            return []

    @staticmethod
    def is_hf_dataset_id(source: str) -> bool:
        """
        Check if source looks like a HuggingFace dataset ID.

        HuggingFace dataset IDs typically have format: "org/dataset" or "dataset"

        Args:
            source: Dataset source string

        Returns:
            True if source looks like HuggingFace dataset ID
        """
        # HF IDs don't have common file extensions
        file_extensions = [".json", ".jsonl", ".csv", ".txt", ".parquet"]
        if any(source.lower().endswith(ext) for ext in file_extensions):
            return False

        # Check for absolute paths (starts with / on Unix or drive letter on Windows)
        from pathlib import Path

        if Path(source).is_absolute():
            return False

        # Check if it looks like a relative file path (contains ./ or ../)
        if source.startswith("./") or source.startswith("../"):
            return False

        # Check for common data directory patterns
        data_prefixes = ["data/", "datasets/", "output/", "outputs/"]
        if any(source.startswith(prefix) for prefix in data_prefixes):
            return False

        # HuggingFace IDs: "org/dataset" or just "dataset"
        # They typically have 0-1 forward slashes (org/name)
        # and don't contain directory-like structures with file extensions
        slash_count = source.count("/")
        return slash_count <= 1  # Direct return


__all__ = ["HuggingFaceDatasetLoader"]
