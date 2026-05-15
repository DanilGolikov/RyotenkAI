"""
Tests for BaseDatasetLoader abstract class.

Post-Batch 15 contract: ``load_for_phase`` raises typed exceptions
(``DatasetLoadFailedError``) instead of returning ``Result``.

Coverage:
- Default ``_log_prefix`` property
- ``load_for_phase`` error handling (KeyError → DatasetLoadFailedError,
  general Exception → DatasetLoadFailedError, non-local source rejection,
  validate_source returning False)
- ``load_for_phase`` happy path (returns Dataset directly)
- ``_apply_max_samples`` edge cases (no limit, exact match, limited path)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from ryotenkai_pod.trainer.data_loaders.base import BaseDatasetLoader
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors import DatasetLoadFailedError


class ConcreteLoader(BaseDatasetLoader):
    """Concrete implementation of BaseDatasetLoader for testing."""

    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """Simple mock load implementation — returns a single-row dataset."""
        return Dataset.from_dict({"text": ["sample"]})

    def validate_source(self, source: str) -> bool:
        """Simple mock validation — accepts ``"valid_source"`` only."""
        return source == "valid_source"


def _local_dataset_config(train: str = "data/train.jsonl", max_samples: int | None = None) -> SimpleNamespace:
    from ryotenkai_shared.config import DatasetSourceLocal
    from ryotenkai_shared.config.datasets.sources import DatasetLocalPaths

    return SimpleNamespace(
        source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train=train)),
        max_samples=max_samples,
    )


class TestBaseDatasetLoader:
    """Test BaseDatasetLoader functionality."""

    def test_log_prefix_default(self) -> None:
        """Default ``_log_prefix`` is ``'DL'``."""
        config = SimpleNamespace()
        loader = ConcreteLoader(config)
        assert loader._log_prefix == "DL"

    def test_load_for_phase_key_error_raises_typed(self) -> None:
        """``load_for_phase`` translates ``KeyError`` to ``DatasetLoadFailedError``."""
        config = MagicMock()
        config.get_dataset_for_strategy.side_effect = KeyError("dataset_not_found")

        loader = ConcreteLoader(config)
        phase = SimpleNamespace(dataset="nonexistent")

        with pytest.raises(DatasetLoadFailedError) as excinfo:
            loader.load_for_phase(phase)

        assert excinfo.value.code == ErrorCode.DATASET_LOAD_FAILED
        assert "not found in config" in (excinfo.value.detail or "")
        assert "nonexistent" in (excinfo.value.detail or "")
        assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_DATASET_NOT_FOUND"
        assert excinfo.value.context.get("phase_dataset") == "nonexistent"
        # Cause chain preserved
        assert isinstance(excinfo.value.__cause__, KeyError)

    def test_load_for_phase_general_exception_raises_typed(self) -> None:
        """``load_for_phase`` wraps unexpected exceptions in ``DatasetLoadFailedError``."""
        config = MagicMock()
        config.get_dataset_for_strategy.return_value = _local_dataset_config()
        config.resolve_path = lambda p: Path("valid_source")  # type: ignore[assignment]  # noqa: ARG005

        loader = ConcreteLoader(config)
        phase = SimpleNamespace(strategy_type="sft", dataset="default")

        # Mock validate_source to raise an unexpected exception
        with (
            patch.object(loader, "validate_source", side_effect=RuntimeError("Unexpected error")),
            pytest.raises(DatasetLoadFailedError) as excinfo,
        ):
            loader.load_for_phase(phase)

        assert excinfo.value.code == ErrorCode.DATASET_LOAD_FAILED
        assert "Failed to load dataset" in (excinfo.value.detail or "")
        assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_LOAD_FAILED"
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_load_for_phase_non_local_source_raises_typed(self) -> None:
        """Non-local source kind raises ``DatasetLoadFailedError`` with DATA_LOADER_LOCAL_ONLY legacy code."""
        from ryotenkai_shared.config import DatasetSourceHF

        config = MagicMock()
        dataset_config = SimpleNamespace(
            source=DatasetSourceHF(train_id="some/dataset"),
            max_samples=None,
        )
        config.get_dataset_for_strategy.return_value = dataset_config

        loader = ConcreteLoader(config)
        phase = SimpleNamespace(strategy_type="sft")

        with pytest.raises(DatasetLoadFailedError) as excinfo:
            loader.load_for_phase(phase)

        assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_LOCAL_ONLY"
        assert excinfo.value.context.get("source_kind") == "huggingface"
        assert "MultiSourceDatasetLoader" in (excinfo.value.detail or "")

    def test_load_for_phase_validate_source_false_raises_file_not_found(self) -> None:
        """When ``validate_source`` returns False, raises ``DatasetLoadFailedError``
        with DATA_LOADER_FILE_NOT_FOUND legacy code."""
        config = MagicMock()
        config.get_dataset_for_strategy.return_value = _local_dataset_config(train="data/missing.jsonl")
        # Resolve to a path that ConcreteLoader.validate_source will REJECT
        config.resolve_path = lambda p: Path("invalid_source")  # type: ignore[assignment]  # noqa: ARG005

        loader = ConcreteLoader(config)
        phase = SimpleNamespace(strategy_type="sft")

        with pytest.raises(DatasetLoadFailedError) as excinfo:
            loader.load_for_phase(phase)

        assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_FILE_NOT_FOUND"
        assert excinfo.value.context.get("train_path") == "data/missing.jsonl"
        assert "not found" in (excinfo.value.detail or "")

    def test_load_for_phase_happy_path_returns_dataset(self) -> None:
        """Successful ``load_for_phase`` returns a Dataset directly (no Result wrapper)."""
        config = MagicMock()
        config.get_dataset_for_strategy.return_value = _local_dataset_config()
        config.resolve_path = lambda p: Path("valid_source")  # type: ignore[assignment]  # noqa: ARG005

        loader = ConcreteLoader(config)
        phase = SimpleNamespace(strategy_type="sft")

        dataset = loader.load_for_phase(phase)

        # Dataset directly, not Ok-wrapped
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 1
        assert dataset[0]["text"] == "sample"

    def test_apply_max_samples_no_limit_needed(self) -> None:
        """``_apply_max_samples`` returns dataset as-is if already smaller."""
        config = SimpleNamespace()
        loader = ConcreteLoader(config)

        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(5)]})
        result = loader._apply_max_samples(dataset, max_samples=10)

        assert len(result) == 5
        assert result == dataset

    def test_apply_max_samples_exact_match(self) -> None:
        """``_apply_max_samples`` when dataset size equals max_samples — returns same dataset."""
        config = SimpleNamespace()
        loader = ConcreteLoader(config)

        dataset = Dataset.from_dict({"text": [f"sample_{i}" for i in range(10)]})
        result = loader._apply_max_samples(dataset, max_samples=10)

        assert len(result) == 10
        assert result == dataset

    def test_apply_max_samples_none_returns_unchanged(self) -> None:
        """``_apply_max_samples`` with ``max_samples=None`` returns dataset unchanged."""
        config = SimpleNamespace()
        loader = ConcreteLoader(config)

        dataset = Dataset.from_dict({"text": ["a", "b", "c"]})
        result = loader._apply_max_samples(dataset, max_samples=None)

        assert result is dataset

    def test_apply_max_samples_truncates_when_over(self) -> None:
        """``_apply_max_samples`` actually truncates when dataset exceeds limit."""
        config = SimpleNamespace()
        loader = ConcreteLoader(config)

        dataset = Dataset.from_dict({"text": [f"x{i}" for i in range(20)]})
        result = loader._apply_max_samples(dataset, max_samples=3)

        assert len(result) == 3
        assert [row["text"] for row in result] == ["x0", "x1", "x2"]
