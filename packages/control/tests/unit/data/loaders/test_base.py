"""
Tests for BaseDatasetLoader abstract class.

Coverage:
- Default _log_prefix property
- load_for_phase error handling (KeyError, general Exception)
- _apply_max_samples edge case (dataset size <= max_samples)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets import Dataset

from src.data.loaders.base import BaseDatasetLoader


class ConcreteLoader(BaseDatasetLoader):
    """Concrete implementation of BaseDatasetLoader for testing."""

    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """Simple mock load implementation."""
        return Dataset.from_dict({"text": ["sample"]})

    def validate_source(self, source: str) -> bool:
        """Simple mock validation."""
        return source == "valid_source"


class TestBaseDatasetLoader:
    """Test BaseDatasetLoader functionality."""

    def test_log_prefix_default(self) -> None:
        """Test: Default _log_prefix is 'DL'."""
        config = MagicMock()
        loader = ConcreteLoader(config)
        assert loader._log_prefix == "DL"

    def test_load_for_phase_key_error(self) -> None:
        """Test: load_for_phase handles KeyError gracefully."""
        config = MagicMock()
        config.get_dataset_for_strategy.side_effect = KeyError("dataset_not_found")

        loader = ConcreteLoader(config)
        phase = MagicMock()
        phase.dataset = "nonexistent"

        result = loader.load_for_phase(phase)

        assert result.is_failure()
        error_msg = str(result.unwrap_err())
        assert "not found in config" in error_msg
        assert "nonexistent" in error_msg

    def test_load_for_phase_general_exception(self) -> None:
        """Test: load_for_phase handles general exceptions."""
        config = MagicMock()
        dataset_config = MagicMock()
        dataset_config.get_source_type = MagicMock(return_value="local")
        dataset_config.source_local = MagicMock()
        dataset_config.source_local.local_paths = MagicMock()
        dataset_config.source_local.local_paths.train = "data/train.jsonl"
        dataset_config.max_samples = None
        config.get_dataset_for_strategy.return_value = dataset_config
        config.resolve_path = lambda p: Path("valid_source")  # type: ignore[assignment]

        loader = ConcreteLoader(config)
        phase = MagicMock()

        # Mock validate_source to raise exception
        with patch.object(loader, "validate_source", side_effect=Exception("Unexpected error")):
            result = loader.load_for_phase(phase)

        assert result.is_failure()
        error_msg = str(result.unwrap_err())
        assert "Failed to load dataset" in error_msg

    def test_apply_max_samples_no_limit_needed(self) -> None:
        """Test: _apply_max_samples returns dataset as-is if already smaller."""
        config = MagicMock()
        loader = ConcreteLoader(config)

        # Dataset with 5 samples, max_samples=10
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(5)]})
        result = loader._apply_max_samples(dataset, max_samples=10)

        # Should return the same dataset
        assert len(result) == 5
        assert result == dataset

    def test_apply_max_samples_exact_match(self) -> None:
        """Test: _apply_max_samples when dataset size equals max_samples."""
        config = MagicMock()
        loader = ConcreteLoader(config)

        # Dataset with 10 samples, max_samples=10
        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(10)]})
        result = loader._apply_max_samples(dataset, max_samples=10)

        # Should return the same dataset
        assert len(result) == 10
        assert result == dataset
