"""
Unit tests for orchestrator components.

Tests:
- DatasetLoader: Dataset loading and validation
- MetricsCollector: Metrics extraction
- ResumeManager: Resume state management
- ChainRunner: Chain execution
- PhaseExecutor: Phase execution (integration)
"""

from unittest.mock import MagicMock

import pytest

from src.training.orchestrator import (
    ChainRunner,
    DatasetLoader,
    MetricsCollector,
    ResumeManager,
    StrategyOrchestrator,
)
from src.utils.config import StrategyPhaseConfig


class TestDatasetLoader:
    """Tests for DatasetLoader component."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        dataset_config = MagicMock()
        dataset_config.train_path = "data/datasets/train.jsonl"
        dataset_config.max_samples = None
        config.get_dataset_for_strategy.return_value = dataset_config
        return config

    def test_init(self, mock_config):
        """Test DatasetLoader initialization."""
        loader = DatasetLoader(mock_config)
        assert loader.config == mock_config

    def test_validate_exists_true(self, mock_config, tmp_path):
        """Test validate_exists with existing file."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"test": 1}')

        loader = DatasetLoader(mock_config)
        assert loader.validate_exists(str(test_file)) is True

    def test_validate_exists_false(self, mock_config):
        """Test validate_exists with non-existing file."""
        loader = DatasetLoader(mock_config)
        assert loader.validate_exists("/nonexistent/path.jsonl") is False

    def test_load_for_phase_file_not_found(self, mock_config):
        """Test load_for_phase with non-existing file."""
        mock_config.get_dataset_for_strategy.return_value.train_path = "/nonexistent.jsonl"

        loader = DatasetLoader(mock_config)
        phase = StrategyPhaseConfig(strategy_type="sft", dataset="default")

        result = loader.load_for_phase(phase)
        assert result.is_failure()
        assert "not found" in str(result.error).lower()


class TestMetricsCollector:
    """Tests for MetricsCollector component."""

    def test_init(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert collector is not None

    def test_extract_from_trainer_with_state(self):
        """Test metrics extraction from trainer with state."""
        trainer = MagicMock()
        trainer.state.log_history = [{"loss": 0.5, "train_runtime": 100.0, "learning_rate": 0.0001}]
        trainer.state.global_step = 100
        trainer.state.epoch = 1.0

        collector = MetricsCollector()
        metrics = collector.extract_from_trainer(trainer)

        assert metrics["train_loss"] == 0.5
        assert metrics["train_runtime"] == 100.0
        assert metrics["global_step"] == 100
        assert metrics["epoch"] == 1.0

    def test_extract_from_trainer_no_state(self):
        """Test metrics extraction from trainer without state."""
        trainer = MagicMock()
        trainer.state = None

        collector = MetricsCollector()
        metrics = collector.extract_from_trainer(trainer)

        assert metrics == {}

    def test_extract_from_trainer_loss_not_in_last_log(self):
        """Test metrics extraction when last log doesn't contain loss.

        FIX: MetricsCollector now searches log_history in reverse order
        to find the most recent value for each metric.
        """
        trainer = MagicMock()
        # Simulate: loss was logged earlier, but last entry is a save/eval log without loss
        trainer.state.log_history = [
            {"loss": 0.8, "learning_rate": 0.0002},  # Step 50
            {"loss": 0.5, "learning_rate": 0.0001},  # Step 100 - actual last loss
            {"train_runtime": 120.0, "train_samples_per_second": 10.5},  # Final log (no loss!)
        ]
        trainer.state.global_step = 150
        trainer.state.epoch = 1.0

        collector = MetricsCollector()
        metrics = collector.extract_from_trainer(trainer)

        # Should find loss=0.5 from second-to-last entry
        assert metrics["train_loss"] == 0.5
        # Should find learning_rate=0.0001 from second-to-last entry
        assert metrics["learning_rate"] == 0.0001
        # Should find train_runtime from last entry
        assert metrics["train_runtime"] == 120.0
        # Should find throughput metrics
        assert metrics["train_samples_per_second"] == 10.5
        assert metrics["global_step"] == 150
        assert metrics["epoch"] == 1.0

    def test_extract_from_trainer_all_metrics_in_different_logs(self):
        """Test extraction when metrics are spread across different logs."""
        trainer = MagicMock()
        trainer.state.log_history = [
            {"loss": 0.9},  # Only loss
            {"eval_loss": 0.7},  # Only eval_loss
            {"learning_rate": 0.0001},  # Only lr
            {"train_runtime": 60.0},  # Only runtime
        ]
        trainer.state.global_step = 100
        trainer.state.epoch = 1.0

        collector = MetricsCollector()
        metrics = collector.extract_from_trainer(trainer)

        assert metrics["train_loss"] == 0.9
        assert metrics["eval_loss"] == 0.7
        assert metrics["learning_rate"] == 0.0001
        assert metrics["train_runtime"] == 60.0

    def test_aggregate_phases_empty(self):
        """Test aggregation with empty list."""
        collector = MetricsCollector()
        summary = collector.aggregate_phases([])
        assert summary == {}

    def test_aggregate_phases(self):
        """Test aggregation of multiple phases."""
        collector = MetricsCollector()
        phase_metrics = [
            {"global_step": 100, "train_runtime": 50.0, "train_loss": 0.8},
            {"global_step": 200, "train_runtime": 60.0, "train_loss": 0.5},
        ]

        summary = collector.aggregate_phases(phase_metrics)

        assert summary["total_phases"] == 2
        assert summary["total_steps"] == 300
        assert summary["total_runtime_seconds"] == 110.0
        assert summary["final_loss"] == 0.5


class TestResumeManager:
    """Tests for ResumeManager component."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.model.name = "test-model"
        return config

    def test_init(self, mock_config):
        """Test ResumeManager initialization."""
        manager = ResumeManager(mock_config)
        assert manager.config == mock_config

    def test_can_resume_no_buffer(self, mock_config):
        """Test can_resume with no buffer."""
        manager = ResumeManager(mock_config)
        assert manager.can_resume(None) is False

    def test_can_resume_with_buffer(self, mock_config):
        """Test can_resume with buffer."""
        buffer = MagicMock()
        buffer.can_resume.return_value = True

        manager = ResumeManager(mock_config)
        assert manager.can_resume(buffer) is True

    def test_is_all_complete(self, mock_config):
        """Test is_all_complete check."""
        buffer = MagicMock()
        buffer.get_resume_phase.return_value = None  # All complete

        manager = ResumeManager(mock_config)
        assert manager.is_all_complete(buffer) is True

    def test_get_checkpoint_path_phase_zero(self, mock_config):
        """Test checkpoint path for phase 0 returns None."""
        buffer = MagicMock()

        manager = ResumeManager(mock_config)
        path = manager.get_checkpoint_path_for_phase(buffer, 0)

        assert path is None  # Phase 0 uses base model


class TestChainRunner:
    """Tests for ChainRunner component."""

    @pytest.fixture
    def mock_executor(self):
        """Create mock PhaseExecutor."""
        executor = MagicMock()
        return executor

    def test_init(self, mock_executor):
        """Test ChainRunner initialization."""
        runner = ChainRunner(mock_executor)
        assert runner.phase_executor == mock_executor

    def test_get_remaining_phases(self, mock_executor):
        """Test get_remaining_phases."""
        runner = ChainRunner(mock_executor)
        strategies = [
            StrategyPhaseConfig(strategy_type="sft", dataset="d"),
            StrategyPhaseConfig(strategy_type="cot", dataset="d"),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d"),
        ]

        remaining = runner.get_remaining_phases(strategies, start_phase=1)

        assert len(remaining) == 2
        assert remaining[0].strategy_type == "cot"
        assert remaining[1].strategy_type == "dpo"


class TestStrategyOrchestratorFacade:
    """Tests for StrategyOrchestrator Facade."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return MagicMock()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.training.get_strategy_chain.return_value = []
        config.model.name = "test-model"
        return config

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        mm = MagicMock()
        mm.safe_operation.return_value.__enter__ = MagicMock()
        mm.safe_operation.return_value.__exit__ = MagicMock(return_value=False)
        return mm

    def test_init(self, mock_model, mock_tokenizer, mock_config, mock_memory_manager):
        """Test StrategyOrchestrator initialization."""
        orchestrator = StrategyOrchestrator(
            mock_model,
            mock_tokenizer,
            mock_config,
            memory_manager=mock_memory_manager,
        )

        assert orchestrator.model == mock_model
        assert orchestrator.tokenizer == mock_tokenizer
        assert orchestrator.config == mock_config
        assert orchestrator.buffer is None

    def test_run_chain_no_strategies(self, mock_model, mock_tokenizer, mock_config, mock_memory_manager):
        """Test run_chain with no strategies returns error."""
        mock_config.training.get_strategy_chain.return_value = []

        orchestrator = StrategyOrchestrator(
            mock_model,
            mock_tokenizer,
            mock_config,
            memory_manager=mock_memory_manager,
        )

        result = orchestrator.run_chain()
        assert result.is_failure()
        assert "No strategies configured" in str(result.error)

    def test_repr_not_initialized(self, mock_model, mock_tokenizer, mock_config, mock_memory_manager):
        """Test __repr__ when not initialized."""
        orchestrator = StrategyOrchestrator(
            mock_model,
            mock_tokenizer,
            mock_config,
            memory_manager=mock_memory_manager,
        )

        assert "not_initialized" in repr(orchestrator)

    def test_get_summary_not_initialized(self, mock_model, mock_tokenizer, mock_config, mock_memory_manager):
        """Test get_summary when not initialized."""
        orchestrator = StrategyOrchestrator(
            mock_model,
            mock_tokenizer,
            mock_config,
            memory_manager=mock_memory_manager,
        )

        summary = orchestrator.get_summary()
        assert summary == {"status": "not_initialized"}

    def test_can_resume_not_initialized(self, mock_model, mock_tokenizer, mock_config, mock_memory_manager):
        """Test can_resume when not initialized."""
        orchestrator = StrategyOrchestrator(
            mock_model,
            mock_tokenizer,
            mock_config,
            memory_manager=mock_memory_manager,
        )

        assert orchestrator.can_resume() is False

    def test_get_completed_phases_not_initialized(self, mock_model, mock_tokenizer, mock_config, mock_memory_manager):
        """Test get_completed_phases when not initialized."""
        orchestrator = StrategyOrchestrator(
            mock_model,
            mock_tokenizer,
            mock_config,
            memory_manager=mock_memory_manager,
        )

        assert orchestrator.get_completed_phases() == []


class TestBackwardCompatibility:
    """Tests for backward compatibility of imports."""

    def test_import_from_training(self):
        """Test import from src.training still works."""
        from src.training import StrategyOrchestrator

        assert StrategyOrchestrator is not None

    def test_import_from_orchestrator_module(self):
        """Test import from src.training.orchestrator module works."""
        from src.training.orchestrator import StrategyOrchestrator

        assert StrategyOrchestrator is not None

    def test_both_imports_same_class(self):
        """Test both imports point to same class."""
        from src.training import StrategyOrchestrator as SO1
        from src.training.orchestrator import StrategyOrchestrator as SO2

        assert SO1 is SO2
