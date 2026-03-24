"""
Unit tests for Dataset Loaders.

Tests:
- IDatasetLoader interface compliance
- JsonDatasetLoader functionality
- HuggingFaceDatasetLoader functionality
- TrainingContainer integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.data.loaders import HuggingFaceDatasetLoader, JsonDatasetLoader, MultiSourceDatasetLoader
from src.training.orchestrator.dataset_loader import DatasetLoader
from src.utils.container import IDatasetLoader, TrainingContainer

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config() -> MagicMock:
    """Create mock PipelineConfig."""
    config = MagicMock()
    config.model.name = "test-model"
    # New strict config helper used by BaseDatasetLoader.load_for_phase()
    config.resolve_path = lambda p: Path(p).expanduser() if p is not None else None  # type: ignore[assignment]

    # Mock dataset config with new fields
    dataset_config = MagicMock()
    dataset_config.source_type = "local"
    dataset_config.source_local = MagicMock()
    dataset_config.source_local.local_paths = MagicMock()
    dataset_config.source_local.local_paths.train = "data/train.jsonl"
    dataset_config.source_local.local_paths.eval = None
    dataset_config.source_local.training_paths = MagicMock()
    dataset_config.source_local.training_paths.train = "data/train.jsonl"
    dataset_config.source_local.training_paths.eval = None
    dataset_config.max_samples = None
    dataset_config.source_hf = None
    dataset_config.get_source_type = MagicMock(return_value="local")
    dataset_config.get_source_uri = MagicMock(return_value="local://data/train.jsonl")
    dataset_config.is_huggingface = MagicMock(return_value=False)
    config.get_dataset_for_strategy = MagicMock(return_value=dataset_config)
    config.datasets = {"default": dataset_config}

    return config


@pytest.fixture
def sample_jsonl_data() -> list[dict]:
    """Sample JSONL data for testing."""
    return [
        {"instruction": "Test 1", "output": "Response 1"},
        {"instruction": "Test 2", "output": "Response 2"},
        {"instruction": "Test 3", "output": "Response 3"},
        {"instruction": "Test 4", "output": "Response 4"},
        {"instruction": "Test 5", "output": "Response 5"},
    ]


@pytest.fixture
def temp_jsonl_file(sample_jsonl_data: list[dict]) -> str:
    """Create temporary JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in sample_jsonl_data:
            f.write(json.dumps(item) + "\n")
        return f.name


@pytest.fixture
def mock_phase() -> MagicMock:
    """Create mock StrategyPhaseConfig."""
    phase = MagicMock()
    phase.strategy_type = "sft"
    phase.dataset = "default"
    phase.hyperparams.epochs = 1
    return phase


# =============================================================================
# IDatasetLoader INTERFACE TESTS
# =============================================================================


class TestIDatasetLoaderInterface:
    """Test IDatasetLoader protocol compliance."""

    def test_json_loader_implements_interface(self, mock_config: MagicMock) -> None:
        """JsonDatasetLoader implements IDatasetLoader."""
        loader = JsonDatasetLoader(config=mock_config)
        assert isinstance(loader, IDatasetLoader)

    def test_hf_loader_implements_interface(self, mock_config: MagicMock) -> None:
        """HuggingFaceDatasetLoader implements IDatasetLoader."""
        loader = HuggingFaceDatasetLoader(config=mock_config)
        assert isinstance(loader, IDatasetLoader)

    def test_interface_methods_exist(self, mock_config: MagicMock) -> None:
        """Interface methods are defined."""
        loader = JsonDatasetLoader(config=mock_config)

        # Check all required methods exist
        assert hasattr(loader, "load")
        assert hasattr(loader, "load_for_phase")
        assert hasattr(loader, "validate_source")

        # Check methods are callable
        assert callable(loader.load)
        assert callable(loader.load_for_phase)
        assert callable(loader.validate_source)


# =============================================================================
# JsonDatasetLoader TESTS
# =============================================================================


class TestJsonDatasetLoader:
    """Test JsonDatasetLoader functionality."""

    def test_init(self, mock_config: MagicMock) -> None:
        """Test initialization."""
        loader = JsonDatasetLoader(config=mock_config)
        assert loader.config == mock_config
        assert loader._log_prefix == "JSON_DL"

    def test_load_jsonl_file(self, mock_config: MagicMock, temp_jsonl_file: str) -> None:
        """Load dataset from JSONL file."""
        loader = JsonDatasetLoader(config=mock_config)
        dataset = loader.load(temp_jsonl_file)

        assert len(dataset) == 5
        assert "instruction" in dataset.column_names
        assert "output" in dataset.column_names

    def test_load_with_max_samples(self, mock_config: MagicMock, temp_jsonl_file: str) -> None:
        """Load dataset with max_samples limit."""
        loader = JsonDatasetLoader(config=mock_config)
        dataset = loader.load(temp_jsonl_file, max_samples=3)

        assert len(dataset) == 3

    def test_load_file_not_found(self, mock_config: MagicMock) -> None:
        """Raise FileNotFoundError for missing file."""
        loader = JsonDatasetLoader(config=mock_config)

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.jsonl")

    def test_validate_source_exists(self, mock_config: MagicMock, temp_jsonl_file: str) -> None:
        """Validate existing file."""
        loader = JsonDatasetLoader(config=mock_config)
        assert loader.validate_source(temp_jsonl_file) is True

    def test_validate_source_not_exists(self, mock_config: MagicMock) -> None:
        """Validate non-existing file."""
        loader = JsonDatasetLoader(config=mock_config)
        assert loader.validate_source("nonexistent.jsonl") is False

    def test_load_for_phase_success(
        self,
        mock_config: MagicMock,
        mock_phase: MagicMock,
        temp_jsonl_file: str,
    ) -> None:
        """Load dataset for training phase."""
        # Configure mock to return temp file path
        dataset_config = MagicMock()
        dataset_config.get_source_type = MagicMock(return_value="local")
        dataset_config.source_local = MagicMock()
        dataset_config.source_local.local_paths = MagicMock()
        dataset_config.source_local.local_paths.train = temp_jsonl_file
        dataset_config.max_samples = None
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        loader = JsonDatasetLoader(config=mock_config)
        result = loader.load_for_phase(mock_phase)

        assert result.is_success()
        dataset = result.unwrap()
        assert len(dataset) == 5

    def test_load_for_phase_file_not_found(
        self,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """Return error for missing dataset."""
        dataset_config = MagicMock()
        dataset_config.get_source_type = MagicMock(return_value="local")
        dataset_config.source_local = MagicMock()
        dataset_config.source_local.local_paths = MagicMock()
        dataset_config.source_local.local_paths.train = "nonexistent.jsonl"
        dataset_config.max_samples = None
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        loader = JsonDatasetLoader(config=mock_config)
        result = loader.load_for_phase(mock_phase)

        assert result.is_failure()
        assert "not found" in str(result.error).lower()

    def test_get_file_info_existing(self, temp_jsonl_file: str) -> None:
        """Get info for existing file."""
        info = JsonDatasetLoader.get_file_info(temp_jsonl_file)

        assert info["exists"] is True
        assert info["size_bytes"] > 0
        assert info["extension"] == ".jsonl"
        assert "lines" in info  # JSONL files should have line count

    def test_get_file_info_not_existing(self) -> None:
        """Get info for non-existing file."""
        info = JsonDatasetLoader.get_file_info("nonexistent.jsonl")

        assert info["exists"] is False
        assert "error" in info

    def test_load_with_validation(
        self,
        mock_config: MagicMock,
        temp_jsonl_file: str,
    ) -> None:
        """Load with field validation."""
        loader = JsonDatasetLoader(config=mock_config)

        # Valid fields
        dataset = loader.load_with_validation(
            temp_jsonl_file,
            required_fields=["instruction", "output"],
        )
        assert len(dataset) == 5

    def test_load_with_validation_missing_field(
        self,
        mock_config: MagicMock,
        temp_jsonl_file: str,
    ) -> None:
        """Raise error for missing required field."""
        loader = JsonDatasetLoader(config=mock_config)

        with pytest.raises(ValueError, match="missing required fields"):
            loader.load_with_validation(
                temp_jsonl_file,
                required_fields=["instruction", "nonexistent_field"],
            )


# =============================================================================
# HuggingFaceDatasetLoader TESTS
# =============================================================================


class TestHuggingFaceDatasetLoader:
    """Test HuggingFaceDatasetLoader functionality."""

    def test_init_no_token(self, mock_config: MagicMock) -> None:
        """Initialize without token."""
        loader = HuggingFaceDatasetLoader(config=mock_config)
        assert loader.config == mock_config
        assert loader._log_prefix == "HF_DL"

    def test_init_with_token(self, mock_config: MagicMock) -> None:
        """Initialize with token."""
        loader = HuggingFaceDatasetLoader(config=mock_config, token="hf_test_token")
        assert loader.token == "hf_test_token"

    def test_init_with_env_token(self, mock_config: MagicMock) -> None:
        """Initialize with token from environment."""
        with patch.dict("os.environ", {"HF_TOKEN": "env_token"}):
            loader = HuggingFaceDatasetLoader(config=mock_config)
            assert loader.token == "env_token"

    def test_is_hf_dataset_id_true(self) -> None:
        """Identify HuggingFace dataset IDs."""
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("tatsu-lab/alpaca") is True
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("databricks/dolly-15k") is True
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("imdb") is True

    def test_is_hf_dataset_id_false(self) -> None:
        """Identify local file paths."""
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("data/train.jsonl") is False
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("data/train.json") is False
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("/absolute/path/data.csv") is False

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_mocked(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Load dataset with mocked HuggingFace."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        loader = HuggingFaceDatasetLoader(config=mock_config)
        dataset = loader.load("test-org/test-dataset")

        mock_load_dataset.assert_called_once()
        assert dataset == mock_dataset

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_with_max_samples(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Load with max_samples limit."""
        # Create mock dataset with select method
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.select.return_value = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        loader = HuggingFaceDatasetLoader(config=mock_config)
        loader.load("test-org/test-dataset", max_samples=50)

        mock_dataset.select.assert_called_once()

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_error(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Handle load error."""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        loader = HuggingFaceDatasetLoader(config=mock_config)

        with pytest.raises(ValueError, match="Failed to load HuggingFace dataset"):
            loader.load("nonexistent/dataset")

    @patch("huggingface_hub.dataset_info")
    def test_validate_source_exists(
        self,
        mock_dataset_info: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Validate existing HuggingFace dataset."""
        mock_dataset_info.return_value = MagicMock()

        loader = HuggingFaceDatasetLoader(config=mock_config)
        assert loader.validate_source("tatsu-lab/alpaca") is True

    @patch("huggingface_hub.dataset_info")
    def test_validate_source_not_exists(
        self,
        mock_dataset_info: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Validate non-existing HuggingFace dataset."""
        mock_dataset_info.side_effect = Exception("Not found")

        loader = HuggingFaceDatasetLoader(config=mock_config)
        assert loader.validate_source("nonexistent/dataset") is False

    @patch("huggingface_hub.dataset_info")
    def test_get_dataset_info(
        self,
        mock_dataset_info: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Get dataset info."""
        mock_info = MagicMock()
        mock_info.id = "test-org/test-dataset"
        mock_info.author = "test-org"
        mock_info.downloads = 1000
        mock_info.likes = 50
        mock_info.tags = ["test"]
        mock_info.card_data = {}
        mock_dataset_info.return_value = mock_info

        loader = HuggingFaceDatasetLoader(config=mock_config)
        info = loader.get_dataset_info("test-org/test-dataset")

        assert info is not None
        assert info["id"] == "test-org/test-dataset"
        assert info["downloads"] == 1000

    @patch("huggingface_hub.dataset_info")
    def test_get_dataset_info_error(
        self,
        mock_dataset_info: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test: get_dataset_info handles errors gracefully."""
        mock_dataset_info.side_effect = Exception("API error")

        loader = HuggingFaceDatasetLoader(config=mock_config)
        info = loader.get_dataset_info("nonexistent/dataset")

        assert info is None

    @patch("datasets.get_dataset_split_names")
    def test_list_splits_error(
        self,
        mock_get_splits: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test: list_splits handles errors gracefully."""
        mock_get_splits.side_effect = Exception("API error")

        loader = HuggingFaceDatasetLoader(config=mock_config)
        splits = loader.list_splits("nonexistent/dataset")

        assert splits == []

    def test_is_hf_dataset_id_absolute_path(self) -> None:
        """Test: Absolute paths are not HF dataset IDs."""
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("/absolute/path/data.json") is False
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("C:\\Windows\\data.json") is False

    def test_is_hf_dataset_id_relative_path(self) -> None:
        """Test: Relative paths are not HF dataset IDs."""
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("./data/train.json") is False
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("../data/train.json") is False

    def test_is_hf_dataset_id_data_directory(self) -> None:
        """Test: Common data directory patterns are not HF IDs."""
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("data/train.jsonl") is False
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("datasets/my_data.json") is False
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("output/result.csv") is False

    def test_is_hf_dataset_id_too_many_slashes(self) -> None:
        """Test: Paths with too many slashes are not HF IDs."""
        assert HuggingFaceDatasetLoader.is_hf_dataset_id("path/to/my/dataset") is False

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_for_phase_success(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """Test: load_for_phase successfully loads HF dataset."""
        # Setup mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        # Setup config
        dataset_config = MagicMock()
        dataset_config.source_hf = MagicMock()
        dataset_config.source_hf.train_id = "test-org/test-dataset"
        dataset_config.max_samples = None
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        with patch.object(HuggingFaceDatasetLoader, "validate_source", return_value=True):
            loader = HuggingFaceDatasetLoader(config=mock_config)
            result = loader.load_for_phase(mock_phase)

        assert result.is_success()
        dataset = result.unwrap()
        assert dataset == mock_dataset

    def test_load_for_phase_missing_dataset_id(
        self,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """Test: load_for_phase fails if source_hf is missing."""
        dataset_config = MagicMock()
        dataset_config.source_hf = None
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        loader = HuggingFaceDatasetLoader(config=mock_config)
        result = loader.load_for_phase(mock_phase)

        assert result.is_failure()
        assert "requires source_hf" in str(result.unwrap_err())

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_for_phase_dataset_not_found(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """Test: load_for_phase fails if dataset doesn't exist."""
        dataset_config = MagicMock()
        dataset_config.source_hf = MagicMock()
        dataset_config.source_hf.train_id = "nonexistent/dataset"
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        with patch.object(HuggingFaceDatasetLoader, "validate_source", return_value=False):
            loader = HuggingFaceDatasetLoader(config=mock_config)
            result = loader.load_for_phase(mock_phase)

        assert result.is_failure()
        assert "not found" in str(result.unwrap_err()).lower()

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_for_phase_key_error(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """Test: load_for_phase handles KeyError gracefully."""
        mock_config.get_dataset_for_strategy.side_effect = KeyError("dataset not in config")

        loader = HuggingFaceDatasetLoader(config=mock_config)
        result = loader.load_for_phase(mock_phase)

        assert result.is_failure()
        assert "not found in config" in str(result.unwrap_err())

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_for_phase_general_exception(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """Test: load_for_phase handles general exceptions."""
        mock_load_dataset.side_effect = Exception("Unexpected error")

        dataset_config = MagicMock()
        dataset_config.source_hf = MagicMock()
        dataset_config.source_hf.train_id = "test/dataset"
        dataset_config.max_samples = None
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        with patch.object(HuggingFaceDatasetLoader, "validate_source", return_value=True):
            loader = HuggingFaceDatasetLoader(config=mock_config)
            result = loader.load_for_phase(mock_phase)

        assert result.is_failure()
        assert "Failed to load" in str(result.unwrap_err())


# =============================================================================
# MultiSourceDatasetLoader TESTS
# =============================================================================


class TestMultiSourceDatasetLoader:
    """Test MultiSourceDatasetLoader routing logic."""

    def test_routes_local_phase_to_json_loader(
        self,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """Local dataset phases should be routed to JsonDatasetLoader."""
        from src.utils.result import Ok

        # Ensure config reports local
        dataset_config = MagicMock()
        dataset_config.get_source_type = MagicMock(return_value="local")
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        mock_json_loader = MagicMock()
        mock_json_loader.load_for_phase.return_value = Ok([{"x": 1}])

        with patch("src.data.loaders.json_loader.JsonDatasetLoader", return_value=mock_json_loader) as mock_cls:
            loader = MultiSourceDatasetLoader(config=mock_config)
            result = loader.load_for_phase(mock_phase)

            assert result.is_success()
            mock_cls.assert_called_once_with(mock_config)
            mock_json_loader.load_for_phase.assert_called_once_with(mock_phase)

    def test_routes_hf_phase_to_hf_loader(
        self,
        mock_config: MagicMock,
        mock_phase: MagicMock,
    ) -> None:
        """HuggingFace dataset phases should be routed to HuggingFaceDatasetLoader."""
        from src.utils.result import Ok

        dataset_config = MagicMock()
        dataset_config.get_source_type = MagicMock(return_value="huggingface")
        mock_config.get_dataset_for_strategy.return_value = dataset_config

        mock_hf_loader = MagicMock()
        mock_hf_loader.load_for_phase.return_value = Ok([{"instruction": "x", "output": "y"}])

        with patch("src.data.loaders.hf_loader.HuggingFaceDatasetLoader", return_value=mock_hf_loader) as mock_cls:
            loader = MultiSourceDatasetLoader(config=mock_config)
            result = loader.load_for_phase(mock_phase)

            assert result.is_success()
            mock_cls.assert_called_once_with(mock_config)
            mock_hf_loader.load_for_phase.assert_called_once_with(mock_phase)

    def test_load_local_file(
        self,
        mock_config: MagicMock,
        tmp_path,
    ) -> None:
        """Test: load() with local file path."""
        # Create temporary file
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"text": "test"}\n')

        loader = MultiSourceDatasetLoader(config=mock_config)
        dataset = loader.load(str(dataset_file))

        assert len(dataset) == 1

    @patch("src.data.loaders.hf_loader.load_dataset")
    def test_load_hf_dataset(
        self,
        mock_load_dataset: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test: load() with HF dataset ID."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        loader = MultiSourceDatasetLoader(config=mock_config)
        dataset = loader.load("org/dataset")

        assert dataset == mock_dataset
        mock_load_dataset.assert_called_once()

    def test_validate_source_local(
        self,
        mock_config: MagicMock,
        tmp_path,
    ) -> None:
        """Test: validate_source for local file."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"text": "test"}\n')

        loader = MultiSourceDatasetLoader(config=mock_config)
        assert loader.validate_source(str(dataset_file)) is True

    def test_validate_source_local_not_exists(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test: validate_source for non-existent local file."""
        loader = MultiSourceDatasetLoader(config=mock_config)
        assert loader.validate_source("nonexistent.jsonl") is False

    @patch("huggingface_hub.dataset_info")
    def test_validate_source_hf(
        self,
        mock_dataset_info: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test: validate_source for HF dataset."""
        mock_dataset_info.return_value = MagicMock()

        loader = MultiSourceDatasetLoader(config=mock_config)
        assert loader.validate_source("org/dataset") is True


# =============================================================================
# TrainingContainer INTEGRATION TESTS
# =============================================================================


class TestTrainingContainerDatasetLoader:
    """Test TrainingContainer DatasetLoader integration."""

    def test_container_has_dataset_loader_property(self, mock_config: MagicMock) -> None:
        """Container has dataset_loader property."""
        container = TrainingContainer(config=mock_config)
        assert hasattr(container, "dataset_loader")

    def test_container_returns_training_dataset_loader_by_default(self, mock_config: MagicMock) -> None:
        """Container returns training DatasetLoader by default."""
        container = TrainingContainer(config=mock_config)
        loader = container.dataset_loader

        assert isinstance(loader, DatasetLoader)

    def test_container_accepts_injected_loader(self, mock_config: MagicMock) -> None:
        """Container accepts injected dataset loader."""
        mock_loader = MagicMock(spec=IDatasetLoader)

        container = TrainingContainer(
            config=mock_config,
            _dataset_loader=mock_loader,
        )

        assert container.dataset_loader is mock_loader

    def test_container_for_testing_accepts_loader(self, mock_config: MagicMock) -> None:
        """for_testing() accepts dataset loader."""
        mock_loader = MagicMock(spec=IDatasetLoader)

        container = TrainingContainer.for_testing(
            config=mock_config,
            dataset_loader=mock_loader,
        )

        assert container.dataset_loader is mock_loader

    def test_container_override_loader(self, mock_config: MagicMock) -> None:
        """override() works with dataset loader."""
        mock_loader = MagicMock(spec=IDatasetLoader)
        container = TrainingContainer(config=mock_config)

        new_container = container.override(dataset_loader=mock_loader)

        assert new_container.dataset_loader is mock_loader
        # Original container unchanged
        assert isinstance(container.dataset_loader, DatasetLoader)


# =============================================================================
# MOCK DATASET LOADER FOR TESTING
# =============================================================================


class MockDatasetLoader:
    """Mock DatasetLoader for testing."""

    def __init__(self, mock_dataset: Any = None):
        self.mock_dataset = mock_dataset or []
        self.load_calls: list[tuple] = []
        self.validate_calls: list[str] = []

    def load(
        self,
        source: str,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Any:
        self.load_calls.append((source, split, max_samples))
        return self.mock_dataset

    def load_for_phase(self, _phase: Any) -> Any:
        from src.utils.result import Ok

        return Ok(self.mock_dataset)

    def validate_source(self, source: str) -> bool:
        self.validate_calls.append(source)
        return True


class TestMockDatasetLoader:
    """Test MockDatasetLoader for testing."""

    def test_mock_implements_interface(self) -> None:
        """Mock implements IDatasetLoader interface."""
        mock = MockDatasetLoader()
        assert isinstance(mock, IDatasetLoader)

    def test_mock_tracks_calls(self) -> None:
        """Mock tracks method calls."""
        mock = MockDatasetLoader(mock_dataset=[1, 2, 3])

        mock.load("data/train.jsonl", max_samples=10)
        mock.validate_source("data/train.jsonl")

        assert len(mock.load_calls) == 1
        assert mock.load_calls[0] == ("data/train.jsonl", "train", 10)
        assert mock.validate_calls == ["data/train.jsonl"]

    def test_container_accepts_mock(self, mock_config: MagicMock) -> None:
        """Container accepts MockDatasetLoader."""
        mock = MockDatasetLoader()

        container = TrainingContainer.for_testing(
            config=mock_config,
            dataset_loader=mock,  # type: ignore
        )

        assert container.dataset_loader is mock


# =============================================================================
# DatasetLoaderFactory TESTS
# =============================================================================


class TestDatasetLoaderFactory:
    """Test DatasetLoaderFactory functionality."""

    def test_create_for_dataset_local(self, mock_config: MagicMock) -> None:
        """Test: Create loader for local dataset."""
        from src.data.loaders.factory import DatasetLoaderFactory

        dataset_config = MagicMock()
        dataset_config.get_source_type = MagicMock(return_value="local")

        factory = DatasetLoaderFactory(mock_config)
        loader = factory.create_for_dataset(dataset_config)

        assert isinstance(loader, JsonDatasetLoader)

    def test_create_for_dataset_huggingface(self, mock_config: MagicMock) -> None:
        """Test: Create loader for HuggingFace dataset."""
        from src.data.loaders.factory import DatasetLoaderFactory

        dataset_config = MagicMock()
        dataset_config.get_source_type = MagicMock(return_value="huggingface")

        factory = DatasetLoaderFactory(mock_config)
        loader = factory.create_for_dataset(dataset_config)

        assert isinstance(loader, HuggingFaceDatasetLoader)

    def test_create_for_source_type_local(self, mock_config: MagicMock) -> None:
        """Test: Create loader by source type string (local)."""
        from src.data.loaders.factory import DatasetLoaderFactory

        factory = DatasetLoaderFactory(mock_config)
        loader = factory.create_for_source_type("local")

        assert isinstance(loader, JsonDatasetLoader)

    def test_create_for_source_type_huggingface(self, mock_config: MagicMock) -> None:
        """Test: Create loader by source type string (huggingface)."""
        from src.data.loaders.factory import DatasetLoaderFactory

        factory = DatasetLoaderFactory(mock_config)
        loader = factory.create_for_source_type("huggingface")

        assert isinstance(loader, HuggingFaceDatasetLoader)

    def test_create_default(self, mock_config: MagicMock) -> None:
        """Test: Create default loader (JsonDatasetLoader)."""
        from src.data.loaders.factory import DatasetLoaderFactory

        factory = DatasetLoaderFactory(mock_config)
        loader = factory.create_default()

        assert isinstance(loader, JsonDatasetLoader)


# =============================================================================
# CLEANUP
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_temp_files(temp_jsonl_file: str) -> None:
    """Clean up temporary files after tests."""
    yield
    # Cleanup happens after test
    path = Path(temp_jsonl_file)
    if path.exists():
        path.unlink()
