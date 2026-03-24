"""
JSON Dataset Loader - Load datasets from local JSON/JSONL files.

Supports:
- JSON files with array of records
- JSONL files (one JSON object per line)
- Automatic format detection

Single Responsibility: Load datasets from local JSON files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from datasets import load_dataset

from src.data.loaders.base import BaseDatasetLoader
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset

    from src.utils.config import PipelineConfig


class JsonDatasetLoader(BaseDatasetLoader):
    """
    Load datasets from local JSON/JSONL files.

    Handles:
    - .json files (array of objects)
    - .jsonl files (newline-delimited JSON)

    Attributes:
        config: Pipeline configuration

    Example:
        loader = JsonDatasetLoader(config)
        dataset = loader.load("data/train.jsonl", max_samples=1000)
        print(len(dataset))
        # 1000

        # Load for training phase
        result = loader.load_for_phase(phase_config)
        if result.is_success():
            dataset = result.unwrap()
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize JSON loader.

        Args:
            config: Pipeline configuration
        """
        super().__init__(config)

    @property
    def _log_prefix(self) -> str:
        """Log prefix for JSON loader."""
        return "JSON_DL"

    # =========================================================================
    # INTERFACE IMPLEMENTATION
    # =========================================================================

    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load dataset from JSON/JSONL file.

        Args:
            source: Path to JSON or JSONL file
            split: Dataset split (ignored for files, always "train")
            max_samples: Limit number of samples (optional)

        Returns:
            Loaded dataset

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(source)
        _ = split  # Split is part of loader interface; for files we always use "train"

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {source}")

        logger.debug(f"[{self._log_prefix}:LOADING] path={source}")

        # Load using HuggingFace datasets library
        # It auto-detects json vs jsonl format
        dataset = load_dataset("json", data_files=str(path), split="train")
        dataset = cast("Dataset", dataset)

        logger.debug(f"[{self._log_prefix}:RAW_LOADED] samples={len(dataset)}")

        # Apply max_samples limit if specified
        dataset = self._apply_max_samples(dataset, max_samples)

        return dataset

    def validate_source(self, source: str) -> bool:
        """
        Validate that JSON file exists.

        Args:
            source: Path to JSON/JSONL file

        Returns:
            True if file exists and is readable
        """
        path = Path(source)
        exists = path.exists() and path.is_file()

        logger.debug(f"[{self._log_prefix}:VALIDATE] path={source}, exists={exists}")

        return exists

    # =========================================================================
    # ADDITIONAL METHODS
    # =========================================================================

    def load_with_validation(
        self,
        source: str,
        required_fields: list[str] | None = None,
        max_samples: int | None = None,
    ) -> Dataset:
        """
        Load dataset with field validation.

        Args:
            source: Path to JSON/JSONL file
            required_fields: List of required field names
            max_samples: Limit number of samples

        Returns:
            Loaded dataset

        Raises:
            ValueError: If required fields missing
        """
        dataset = self.load(source, max_samples=max_samples)

        if required_fields:
            columns = dataset.column_names
            missing = [f for f in required_fields if f not in columns]

            if missing:
                raise ValueError(f"Dataset missing required fields: {missing}. Available: {columns}")

            logger.debug(f"[{self._log_prefix}:VALIDATED] required_fields={required_fields}")

        return dataset

    @staticmethod
    def get_file_info(source: str) -> dict:
        """
        Get information about a JSON/JSONL file.

        Args:
            source: Path to file

        Returns:
            Dict with file info (size, lines, etc.)
        """
        path = Path(source)

        if not path.exists():
            return {"exists": False, "error": "File not found"}

        stat = path.stat()
        info = {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "extension": path.suffix,
        }

        # Count lines for JSONL
        if path.suffix.lower() == ".jsonl":
            with path.open() as f:
                info["lines"] = sum(1 for _ in f)

        return info


__all__ = ["JsonDatasetLoader"]
