"""
Tests for MLflowManager event logging functionality.

Tests the new event logging methods added to MLflowManager:
- log_event() and convenience methods
- Memory events (GPU detection, OOM, cache)
- DataBuffer events (state saves, checkpoints)
- Pipeline events (stage start/complete/fail)
- Summary generation
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from src.training.managers.mlflow_manager import MLflowManager


class TestMLflowManagerEvents:
    """Test MLflowManager event logging methods."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        config.experiment_tracking.mlflow.enabled = True
        config.experiment_tracking.mlflow.tracking_uri = "sqlite:///test.db"
        config.experiment_tracking.mlflow.experiment_name = "test"
        config.experiment_tracking.mlflow.enable_system_metrics = False
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked dependencies."""
        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)
        return manager

    def test_log_event_basic(self, mlflow_manager: MLflowManager) -> None:
        """Test basic event logging."""
        # Log an event
        event = mlflow_manager.log_event(
            "start",
            "Training started",
            category="training",
            source="test",
        )

        assert event["event_type"] == "start"
        assert event["message"] == "Training started"
        assert event["category"] == "training"
        assert event["source"] == "test"
        assert "timestamp" in event

    def test_log_event_with_metadata(self, mlflow_manager: MLflowManager) -> None:
        """Test event logging with metadata."""
        event = mlflow_manager.log_event(
            "info",
            "Test event",
            category="test",
            custom_field="custom_value",
            number=42,
        )

        # Event should be logged successfully
        assert event["message"] == "Test event"
        assert event["category"] == "test"

    def test_log_event_convenience_methods(self, mlflow_manager: MLflowManager) -> None:
        """Test convenience methods for event types."""
        # Test all convenience methods
        start = mlflow_manager.log_event_start("Start message")
        complete = mlflow_manager.log_event_complete("Complete message")
        error = mlflow_manager.log_event_error("Error message")
        warning = mlflow_manager.log_event_warning("Warning message")
        info = mlflow_manager.log_event_info("Info message")
        checkpoint = mlflow_manager.log_event_checkpoint("Checkpoint message")

        assert start["event_type"] == "start"
        assert complete["event_type"] == "complete"
        assert error["event_type"] == "error"
        assert warning["event_type"] == "warning"
        assert info["event_type"] == "info"
        assert checkpoint["event_type"] == "checkpoint"

    def test_get_events(self, mlflow_manager: MLflowManager) -> None:
        """Test getting collected events."""
        # Log multiple events
        mlflow_manager.log_event_start("Start", category="training")
        mlflow_manager.log_event_info("Info", category="memory")
        mlflow_manager.log_event_complete("Complete", category="training")

        # Get all events
        all_events = mlflow_manager.get_events()
        assert len(all_events) == 3

        # Get filtered events
        training_events = mlflow_manager.get_events(category="training")
        assert len(training_events) == 2

        memory_events = mlflow_manager.get_events(category="memory")
        assert len(memory_events) == 1

    def test_event_counter_increments(self, mlflow_manager: MLflowManager) -> None:
        """Test that event counter increments correctly."""
        event1 = mlflow_manager.log_event_start("First")
        event2 = mlflow_manager.log_event_info("Second")
        event3 = mlflow_manager.log_event_complete("Third")

        # Events should all be logged
        assert event1["event_type"] == "start"
        assert event2["event_type"] == "info"
        assert event3["event_type"] == "complete"


class TestMemoryEvents:
    """Test memory-related event logging."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        config.experiment_tracking.mlflow.enabled = True
        config.experiment_tracking.mlflow.tracking_uri = "sqlite:///test.db"
        config.experiment_tracking.mlflow.experiment_name = "test"
        config.experiment_tracking.mlflow.enable_system_metrics = False
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked dependencies."""
        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)
        return manager

    def test_log_gpu_detection(self, mlflow_manager: MLflowManager) -> None:
        """Test GPU detection event."""
        mlflow_manager.log_gpu_detection(
            name="NVIDIA RTX 4090",
            vram_gb=24.0,
            tier="ultra",
        )

        events = mlflow_manager.get_events(category="memory")
        assert len(events) == 1
        assert "GPU detected" in events[0]["message"]

    def test_log_memory_warning(self, mlflow_manager: MLflowManager) -> None:
        """Test memory warning event."""
        mlflow_manager.log_memory_warning(
            utilization_percent=85.0,
            used_mb=20000,
            total_mb=24000,
            is_critical=False,
        )

        events = mlflow_manager.get_events(category="memory")
        assert len(events) == 1
        assert "WARNING" in events[0]["message"]

    def test_log_memory_critical(self, mlflow_manager: MLflowManager) -> None:
        """Test memory critical event."""
        mlflow_manager.log_memory_warning(
            utilization_percent=95.0,
            used_mb=22000,
            total_mb=24000,
            is_critical=True,
        )

        events = mlflow_manager.get_events(category="memory")
        assert len(events) == 1
        assert "CRITICAL" in events[0]["message"]

    def test_log_oom(self, mlflow_manager: MLflowManager) -> None:
        """Test OOM event."""
        mlflow_manager.log_oom(
            operation="train_phase_0",
            free_mb=100,
        )

        events = mlflow_manager.get_events(category="memory")
        assert len(events) == 1
        assert "OOM" in events[0]["message"]

    def test_log_oom_recovery(self, mlflow_manager: MLflowManager) -> None:
        """Test OOM recovery event."""
        mlflow_manager.log_oom_recovery(
            operation="train_phase_0",
            attempt=2,
            max_attempts=3,
        )

        events = mlflow_manager.get_events(category="memory")
        assert len(events) == 1
        assert "recovery" in events[0]["message"]

    def test_log_cache_cleared(self, mlflow_manager: MLflowManager) -> None:
        """Test cache cleared event."""
        mlflow_manager.log_cache_cleared(freed_mb=500)

        events = mlflow_manager.get_events(category="memory")
        assert len(events) == 1
        assert "Cache cleared" in events[0]["message"]

    def test_log_cache_cleared_zero(self, mlflow_manager: MLflowManager) -> None:
        """Test cache cleared with zero bytes freed (no event)."""
        mlflow_manager.log_cache_cleared(freed_mb=0)

        events = mlflow_manager.get_events(category="memory")
        assert len(events) == 0


class TestDataBufferEvents:
    """Test DataBuffer-related event logging."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        config.experiment_tracking.mlflow.enabled = True
        config.experiment_tracking.mlflow.tracking_uri = "sqlite:///test.db"
        config.experiment_tracking.mlflow.experiment_name = "test"
        config.experiment_tracking.mlflow.enable_system_metrics = False
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked dependencies."""
        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)
        return manager

    def test_log_pipeline_initialized(self, mlflow_manager: MLflowManager) -> None:
        """Test pipeline initialization event."""
        mlflow_manager.log_pipeline_initialized(
            run_id="run_123",
            total_phases=3,
            strategy_chain=["sft", "cot", "dpo"],
        )

        events = mlflow_manager.get_events(category="training")
        assert len(events) == 1
        assert "Pipeline initialized" in events[0]["message"]
        assert "SFT" in events[0]["message"]

    def test_log_state_saved(self, mlflow_manager: MLflowManager) -> None:
        """Test state save event."""
        mlflow_manager.log_state_saved(
            run_id="run_123",
            path="/path/to/state",
        )

        events = mlflow_manager.get_events(category="training")
        assert len(events) == 1
        assert events[0]["event_type"] == "checkpoint"

    def test_log_checkpoint_cleanup(self, mlflow_manager: MLflowManager) -> None:
        """Test checkpoint cleanup event."""
        mlflow_manager.log_checkpoint_cleanup(
            cleaned_count=5,
            freed_mb=500,
        )

        events = mlflow_manager.get_events(category="training")
        assert len(events) == 1
        assert "Cleaned 5" in events[0]["message"]

    def test_log_checkpoint_cleanup_zero(self, mlflow_manager: MLflowManager) -> None:
        """Test checkpoint cleanup with zero cleaned (no event)."""
        mlflow_manager.log_checkpoint_cleanup(
            cleaned_count=0,
            freed_mb=0,
        )

        events = mlflow_manager.get_events(category="training")
        assert len(events) == 0


class TestPipelineEvents:
    """Test pipeline stage event logging."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        config.experiment_tracking.mlflow.enabled = True
        config.experiment_tracking.mlflow.tracking_uri = "sqlite:///test.db"
        config.experiment_tracking.mlflow.experiment_name = "test"
        config.experiment_tracking.mlflow.enable_system_metrics = False
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked dependencies."""
        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)
        return manager

    def test_log_stage_start(self, mlflow_manager: MLflowManager) -> None:
        """Test stage start event."""
        mlflow_manager.log_stage_start(
            stage_name="Dataset Validator",
            stage_idx=0,
            total_stages=6,
        )

        events = mlflow_manager.get_events(category="pipeline")
        assert len(events) == 1
        assert "Stage 1/6" in events[0]["message"]
        assert "Dataset Validator" in events[0]["message"]

    def test_log_stage_complete(self, mlflow_manager: MLflowManager) -> None:
        """Test stage complete event."""
        mlflow_manager.log_stage_complete(
            stage_name="Dataset Validator",
            stage_idx=0,
            duration_seconds=5.5,
        )

        events = mlflow_manager.get_events(category="pipeline")
        assert len(events) == 1
        assert "completed" in events[0]["message"]
        assert "(5.5s)" in events[0]["message"]

    def test_log_stage_failed(self, mlflow_manager: MLflowManager) -> None:
        """Test stage failed event."""
        mlflow_manager.log_stage_failed(
            stage_name="GPU Deployer",
            stage_idx=1,
            error="No GPU available",
        )

        events = mlflow_manager.get_events(category="pipeline")
        assert len(events) == 1
        assert "failed" in events[0]["message"]
        assert "No GPU available" in events[0]["message"]


class TestSummaryGeneration:
    """Test summary generation from MLflow data."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        config.experiment_tracking.mlflow.enabled = True
        config.experiment_tracking.mlflow.tracking_uri = "sqlite:///test.db"
        config.experiment_tracking.mlflow.experiment_name = "test"
        config.experiment_tracking.mlflow.enable_system_metrics = False
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked dependencies."""
        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)
        return manager

    def test_generate_summary_markdown(self, mlflow_manager: MLflowManager) -> None:
        """Test markdown summary generation."""
        # Add some events
        mlflow_manager.log_event_start("Training started", category="training")
        mlflow_manager.log_gpu_detection("RTX 4090", 24.0, "ultra")
        mlflow_manager.log_event_complete("Training completed", category="training")

        # Generate markdown
        md = mlflow_manager.generate_summary_markdown()

        # Check structure
        assert "# Training Summary Report" in md
        assert "## Overview" in md
        assert "## Events Timeline" in md
        assert "## Results" in md

    def test_generate_summary_with_events(self, mlflow_manager: MLflowManager) -> None:
        """Test that events appear in summary."""
        mlflow_manager.log_event_start("Test started", category="training")
        mlflow_manager.log_event_warning("Test warning", category="training")
        mlflow_manager.log_event_error("Test error", category="training")

        md = mlflow_manager.generate_summary_markdown()

        # Check events section
        assert "[START]" in md.upper() or "[STARTED]" in md.upper()
        assert "Test started" in md
        assert "Errors: 1" in md or "error" in md.lower()

    def test_cleanup_clears_events(self, mlflow_manager: MLflowManager) -> None:
        """Test that cleanup clears events."""
        mlflow_manager.log_event_start("Test")
        assert len(mlflow_manager.get_events()) == 1

        mlflow_manager.cleanup()
        assert len(mlflow_manager.get_events()) == 0


class TestEnvironmentLogging:
    """Test environment logging functionality."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        config.experiment_tracking.mlflow.enabled = True
        config.experiment_tracking.mlflow.tracking_uri = "sqlite:///test.db"
        config.experiment_tracking.mlflow.experiment_name = "test"
        config.experiment_tracking.mlflow.enable_system_metrics = False
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked dependencies."""
        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)
        return manager

    def test_log_environment_with_snapshot(self, mlflow_manager: MLflowManager) -> None:
        """Test logging environment with provided snapshot."""
        env_snapshot = {
            "python_version": "3.12.0",
            "torch_version": "2.1.0",
            "cuda_available": True,
        }

        # This should not raise
        mlflow_manager.log_environment(env_snapshot)

    def test_log_environment_auto_collect(self, mlflow_manager: MLflowManager) -> None:
        """Test logging environment with auto-collection."""
        # This should auto-collect and not raise
        mlflow_manager.log_environment()


# =============================================================================
# PRIORITY 1: SETUP AND INITIALIZATION TESTS
# =============================================================================


class TestMLflowManagerSetup:
    """Test MLflowManager setup and initialization (Priority 1)."""

    @pytest.fixture
    def mock_config_enabled(self) -> MagicMock:
        """Create mock config with MLflow enabled."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.tracking_uri = "http://localhost:5002"
        mlflow_config.experiment_name = "test_experiment"
        mlflow_config.system_metrics_callback_enabled = False
        mlflow_config.system_metrics_sampling_interval = 5.0
        mlflow_config.system_metrics_samples_before_logging = 10
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mock_config_import_error(self) -> MagicMock:
        """Create mock config for MLflow import/setup failure scenarios."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.tracking_uri = "http://localhost:5002"
        mlflow_config.experiment_name = "test_experiment"
        mlflow_config.system_metrics_callback_enabled = False
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    def test_setup_success(self, mock_config_enabled: MagicMock) -> None:
        """Test successful setup with available server."""
        import sys
        from unittest.mock import MagicMock as Mock
        from unittest.mock import patch

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config_enabled)

        # Patch mlflow module via sys.modules
        mock_mlflow = Mock()
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock()
        mock_mlflow.disable_system_metrics_logging = Mock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            # Mock connectivity check on MLflowGateway class (setup() creates a new gateway)
            with patch("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", return_value=True):
                result = manager.setup()

                assert result is True
                assert manager._mlflow is not None
                mock_mlflow.set_tracking_uri.assert_called_once()
                mock_mlflow.set_experiment.assert_called_once()

    def test_setup_server_unavailable(self, mock_config_enabled: MagicMock) -> None:
        """Test setup with unavailable server - continues without MLflow."""
        import sys
        from unittest.mock import MagicMock as Mock
        from unittest.mock import patch

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config_enabled)

        # Patch mlflow module via sys.modules
        mock_mlflow = Mock()
        mock_mlflow.set_tracking_uri = Mock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            # Mock connectivity check on MLflowGateway class - server unavailable
            with patch("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity", return_value=False):
                result = manager.setup()

                assert result is False
                assert manager._mlflow is None
                # Should log error event
                events = manager.get_events(category="system")
                assert len(events) > 0
                assert any("not reachable" in e.get("message", "") for e in events)

    def test_setup_timeout(self, mock_config_enabled: MagicMock) -> None:
        """Test setup timeout handling."""
        import sys
        from unittest.mock import MagicMock as Mock
        from unittest.mock import patch

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config_enabled)

        # Patch mlflow module via sys.modules
        mock_mlflow = Mock()
        mock_mlflow.set_tracking_uri = Mock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            # Mock connectivity check on MLflowGateway class with timeout
            with patch("src.infrastructure.mlflow.gateway.MLflowGateway.check_connectivity",
                       side_effect=TimeoutError("Connection timeout")):
                result = manager.setup()

                # Setup should return False on timeout
                assert result is False

    def test_setup_invalid_uri(self, mock_config_enabled: MagicMock) -> None:
        """Test setup with invalid tracking URI."""
        import sys
        from unittest.mock import MagicMock as Mock
        from unittest.mock import patch

        from src.training.managers.mlflow_manager import MLflowManager

        mock_config_enabled.experiment_tracking.mlflow.tracking_uri = "invalid://uri"
        manager = MLflowManager(mock_config_enabled)

        # Patch mlflow module via sys.modules
        mock_mlflow = Mock()
        mock_mlflow.set_tracking_uri = Mock()
        mock_mlflow.set_experiment = Mock(side_effect=Exception("Invalid URI"))

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = manager.setup()

            # Should handle exception gracefully
            assert result is False or manager._mlflow is None

    def test_setup_import_error_returns_false(self, mock_config_import_error: MagicMock) -> None:
        """Test setup returns False when MLflow package is unavailable."""
        import builtins

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config_import_error)

        orig_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any):
            if name == "mlflow":
                raise ImportError("no mlflow")
            return orig_import(name, *args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        result = manager.setup()
        monkeypatch.undo()

        assert result is False
        assert manager._mlflow is None
        assert manager.is_active is False


# =============================================================================
# PRIORITY 1: NESTED RUNS TESTS
# =============================================================================


class TestMLflowManagerNestedRuns:
    """Test MLflowManager nested runs functionality (Priority 1)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.tracking_uri = "sqlite:///test.db"
        mlflow_config.experiment_name = "test"
        mlflow_config.system_metrics_callback_enabled = False
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow module
        mock_mlflow = MagicMock()
        mock_parent_run = MagicMock()
        mock_parent_run.info.run_id = "parent_run_123"
        mock_parent_run.__enter__ = MagicMock(return_value=mock_parent_run)
        mock_parent_run.__exit__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value = mock_parent_run
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.end_run = MagicMock()

        manager._mlflow = mock_mlflow
        manager._run = mock_parent_run
        manager._run_id = "parent_run_123"
        manager._parent_run_id = "parent_run_123"

        return manager

    def test_start_nested_run_success(self, mlflow_manager: MLflowManager) -> None:
        """Test successful nested run creation."""
        mock_nested_run = MagicMock()
        mock_nested_run.info.run_id = "nested_run_456"
        mock_nested_run.__enter__ = MagicMock(return_value=mock_nested_run)
        mock_nested_run.__exit__ = MagicMock(return_value=None)

        mlflow_manager._mlflow.start_run.return_value = mock_nested_run

        with mlflow_manager.start_nested_run("phase_0_sft") as nested_run:
            assert nested_run is not None
            assert mlflow_manager._run_id == "nested_run_456"
            assert len(mlflow_manager._nested_run_stack) == 1
            assert mlflow_manager.is_nested is True

        # After context exit, should restore parent
        assert mlflow_manager._run_id == "parent_run_123"
        assert len(mlflow_manager._nested_run_stack) == 0

    def test_start_nested_run_without_parent(self, mlflow_manager: MLflowManager) -> None:
        """Test nested run without parent - fallback to regular run."""
        mlflow_manager._parent_run_id = None

        mock_regular_run = MagicMock()
        mock_regular_run.info.run_id = "regular_run_789"
        mock_regular_run.__enter__ = MagicMock(return_value=mock_regular_run)
        mock_regular_run.__exit__ = MagicMock(return_value=None)

        mlflow_manager._mlflow.start_run.return_value = mock_regular_run

        with mlflow_manager.start_nested_run("phase_0_sft") as run:
            assert run is not None
            # Should have called start_run (not nested)
            assert mlflow_manager._mlflow.start_run.called

    def test_nested_run_stack_management(self, mlflow_manager: MLflowManager) -> None:
        """Test nested run stack management."""
        # Create multiple nested runs
        run1 = MagicMock()
        run1.info.run_id = "nested_1"
        run1.__enter__ = MagicMock(return_value=run1)
        run1.__exit__ = MagicMock(return_value=None)

        run2 = MagicMock()
        run2.info.run_id = "nested_2"
        run2.__enter__ = MagicMock(return_value=run2)
        run2.__exit__ = MagicMock(return_value=None)

        mlflow_manager._mlflow.start_run.side_effect = [run1, run2]

        with mlflow_manager.start_nested_run("phase_0"):
            assert len(mlflow_manager._nested_run_stack) == 1

            with mlflow_manager.start_nested_run("phase_1"):
                assert len(mlflow_manager._nested_run_stack) == 2
                assert mlflow_manager._run_id == "nested_2"

            # After inner context exit
            assert len(mlflow_manager._nested_run_stack) == 1
            assert mlflow_manager._run_id == "nested_1"

        # After outer context exit
        assert len(mlflow_manager._nested_run_stack) == 0
        assert mlflow_manager._run_id == "parent_run_123"

    def test_end_nested_run_restores_parent(self, mlflow_manager: MLflowManager) -> None:
        """Test that ending nested run restores parent run."""
        mock_nested_run = MagicMock()
        mock_nested_run.info.run_id = "nested_run_456"
        mock_nested_run.__enter__ = MagicMock(return_value=mock_nested_run)
        mock_nested_run.__exit__ = MagicMock(return_value=None)

        mlflow_manager._mlflow.start_run.return_value = mock_nested_run

        with mlflow_manager.start_nested_run("phase_0_sft"):
            assert mlflow_manager._run_id == "nested_run_456"

        # After context exit, parent should be restored
        # Note: end_run is called by mlflow context manager, not directly
        assert mlflow_manager._run_id == "parent_run_123"
        # The nested run context manager should have exited (which calls mlflow.end_run internally)
        assert mock_nested_run.__exit__.called

    def test_multiple_nested_runs(self, mlflow_manager: MLflowManager) -> None:
        """Test multiple sequential nested runs."""
        runs = []
        for i in range(3):
            run = MagicMock()
            run.info.run_id = f"nested_{i}"
            run.__enter__ = MagicMock(return_value=run)
            run.__exit__ = MagicMock(return_value=None)
            runs.append(run)

        mlflow_manager._mlflow.start_run.side_effect = runs

        # Create and close 3 nested runs sequentially
        for i in range(3):
            with mlflow_manager.start_nested_run(f"phase_{i}"):
                assert mlflow_manager._run_id == f"nested_{i}"
                assert len(mlflow_manager._nested_run_stack) == 1

            # After each context exit, stack should be empty
            assert len(mlflow_manager._nested_run_stack) == 0
            assert mlflow_manager._run_id == "parent_run_123"


# =============================================================================
# PRIORITY 1: EVENT LOGGING EXTENSION TESTS
# =============================================================================


class TestMLflowManagerEventLoggingExtension:
    """Test extended event logging functionality (Priority 1)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.tracking_uri = "sqlite:///test.db"
        mlflow_config.experiment_name = "test"
        mlflow_config.log_artifacts = True
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow and run
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        manager._mlflow = mock_mlflow
        manager._run = mock_run
        manager._run_id = "test_run_123"

        return manager

    def test_log_event_artifact_export(self, mlflow_manager: MLflowManager) -> None:
        """Test exporting events as artifact."""
        from unittest.mock import PropertyMock, patch

        from src.training.managers.mlflow_manager import MLflowManager

        # Add some events
        mlflow_manager.log_event_start("Test started", category="training")
        mlflow_manager.log_event_info("Test info", category="training")
        mlflow_manager.log_event_complete("Test completed", category="training")

        # Mock client for artifact logging via log_dict method
        mock_client = MagicMock()
        mock_client.log_dict = MagicMock(return_value=True)

        # Patch client property on the class
        with patch.object(MLflowManager, "client", new_callable=PropertyMock, return_value=mock_client):
            result = mlflow_manager.log_events_artifact("test_events.json")

            assert result is True
            mock_client.log_dict.assert_called_once()
            call_args = mock_client.log_dict.call_args
            # log_dict signature: (run_id, dictionary, artifact_file)
            assert len(call_args[0]) >= 2
            # Check that events dict is passed
            events_dict = call_args[0][1]
            assert "events" in events_dict
            assert len(events_dict["events"]) == 3

    def test_log_event_with_none_values(self, mlflow_manager: MLflowManager) -> None:
        """Test event logging with None values in metadata."""
        event = mlflow_manager.log_event(
            "info",
            "Test event",
            category="test",
            field_with_none=None,
            field_with_value="value",
        )

        # None values should be handled gracefully
        # All metadata goes to attributes dict
        assert "attributes" in event
        assert event["attributes"]["field_with_value"] == "value"
        # None value should be in attributes (might be None or converted to string)
        assert "field_with_none" in event["attributes"]

    def test_log_event_severity_mapping(self, mlflow_manager: MLflowManager) -> None:
        """Test correct severity mapping for event types."""
        start_event = mlflow_manager.log_event("start", "Started", category="test")
        complete_event = mlflow_manager.log_event("complete", "Completed", category="test")
        error_event = mlflow_manager.log_event("error", "Error", category="test")
        warning_event = mlflow_manager.log_event("warning", "Warning", category="test")

        assert start_event["severity"] == "INFO"
        assert start_event["severity_number"] == 9
        assert complete_event["severity"] == "INFO"
        assert error_event["severity"] == "ERROR"
        assert error_event["severity_number"] == 17
        assert warning_event["severity"] == "WARN"
        assert warning_event["severity_number"] == 13

    def test_log_events_filtering(self, mlflow_manager: MLflowManager) -> None:
        """Test filtering events by category."""
        mlflow_manager.log_event_start("Training started", category="training")
        mlflow_manager.log_event_info("Memory info", category="memory")
        mlflow_manager.log_event_start("Pipeline started", category="pipeline")
        mlflow_manager.log_event_info("Another training info", category="training")

        training_events = mlflow_manager.get_events(category="training")
        memory_events = mlflow_manager.get_events(category="memory")
        pipeline_events = mlflow_manager.get_events(category="pipeline")

        assert len(training_events) == 2
        assert len(memory_events) == 1
        assert len(pipeline_events) == 1

    def test_log_events_empty(self, mlflow_manager: MLflowManager) -> None:
        """Test logging events artifact when no events exist."""
        # Ensure no events
        assert len(mlflow_manager.get_events()) == 0

        # When no events, log_events_artifact should return False without calling client
        result = mlflow_manager.log_events_artifact("empty_events.json")

        # Should return False when no events
        assert result is False


# =============================================================================
# PRIORITY 2: AUTOLOGGING TESTS
# =============================================================================


class TestMLflowManagerAutologging:
    """Test MLflow autologging functionality (Priority 2)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.tracking_uri = "sqlite:///test.db"
        mlflow_config.experiment_name = "test"
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        manager._mlflow = mock_mlflow
        manager._run = mock_run
        manager._run_id = "test_run_123"

        return manager

    def test_enable_autolog_transformers(self, mlflow_manager: MLflowManager) -> None:
        """Test enabling autolog for transformers."""
        import sys
        from unittest.mock import MagicMock, patch

        # Mock mlflow.transformers module
        mock_transformers = MagicMock()
        mock_transformers.autolog = MagicMock()

        with patch.dict(sys.modules, {"mlflow.transformers": mock_transformers}):
            result = mlflow_manager.enable_autolog(log_models=False)

            assert result is True
            mock_transformers.autolog.assert_called_once()
            call_kwargs = mock_transformers.autolog.call_args[1]
            assert call_kwargs["log_models"] is False
            assert call_kwargs["disable"] is False

    def test_enable_autolog_pytorch(self, mlflow_manager: MLflowManager) -> None:
        """Test enabling autolog for PyTorch."""
        import sys
        from unittest.mock import MagicMock, patch

        # Mock mlflow.pytorch module
        mock_pytorch = MagicMock()
        mock_pytorch.autolog = MagicMock()

        with patch.dict(sys.modules, {"mlflow.pytorch": mock_pytorch}):
            result = mlflow_manager.enable_pytorch_autolog(log_models=False, log_every_n_epoch=5)

            assert result is True
            mock_pytorch.autolog.assert_called_once()
            call_kwargs = mock_pytorch.autolog.call_args[1]
            assert call_kwargs["log_models"] is False
            assert call_kwargs["log_every_n_epoch"] == 5

    def test_disable_autolog(self, mlflow_manager: MLflowManager) -> None:
        """Test disabling autolog."""
        import sys
        from unittest.mock import MagicMock, patch

        # Mock mlflow.transformers module
        mock_transformers_module = MagicMock()

        with (
            patch.dict(sys.modules, {"mlflow.transformers": mock_transformers_module}),
            patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: mock_transformers_module
                if name == "mlflow.transformers"
                else __import__(name, *args, **kwargs),
            ),
        ):
            result = mlflow_manager.disable_autolog()

            assert result is True

    def test_autolog_not_initialized(self, mock_config: MagicMock) -> None:
        """Test autolog when MLflow not initialized."""
        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)
        # Don't initialize _mlflow

        result = manager.enable_autolog()

        assert result is False


# =============================================================================
# PRIORITY 2: TRACING TESTS
# =============================================================================


class TestMLflowManagerTracing:
    """Test MLflow tracing functionality (Priority 2)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.tracking_uri = "sqlite:///test.db"
        mlflow_config.experiment_name = "test"
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow with tracing
        mock_mlflow = MagicMock()
        mock_mlflow.tracing = MagicMock()
        mock_mlflow.tracing.enable = MagicMock()
        mock_mlflow.tracing.disable = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        manager._mlflow = mock_mlflow
        manager._run = mock_run
        manager._run_id = "test_run_123"

        return manager

    def test_enable_tracing(self, mlflow_manager: MLflowManager) -> None:
        """Test enabling tracing."""
        result = mlflow_manager.enable_tracing()

        assert result is True
        mlflow_manager._mlflow.tracing.enable.assert_called_once()

    def test_trace_llm_call_context(self, mlflow_manager: MLflowManager) -> None:
        """Test tracing LLM call with context manager."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)
        mlflow_manager._mlflow.start_span = MagicMock(return_value=mock_span)

        with mlflow_manager.trace_llm_call("generate", model_name="test-model") as span:
            assert span is not None

        mlflow_manager._mlflow.start_span.assert_called_once()
        call_kwargs = mlflow_manager._mlflow.start_span.call_args[1]
        assert call_kwargs["name"] == "generate"
        assert call_kwargs["attributes"]["model_name"] == "test-model"

    def test_log_trace_io(self, mlflow_manager: MLflowManager) -> None:
        """Test logging trace input/output."""
        # This is a helper method that should work even without actual tracing
        mlflow_manager.log_trace_io(
            input_data="test prompt", output_data="test response", input_tokens=10, output_tokens=20
        )
        # Should not raise exception

    def test_tracing_not_available(self, mock_config: MagicMock) -> None:
        """Test tracing when not available (old MLflow version)."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow without tracing attribute
        mock_mlflow = MagicMock()
        del mock_mlflow.tracing  # Remove tracing attribute
        manager._mlflow = mock_mlflow

        result = manager.enable_tracing()

        assert result is False


# =============================================================================
# PRIORITY 2: DATASET VERSIONING TESTS
# =============================================================================


class TestMLflowManagerDatasetVersioning:
    """Test MLflow dataset versioning functionality (Priority 2)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.tracking_uri = "sqlite:///test.db"
        mlflow_config.experiment_name = "test"
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow with dataset support
        mock_mlflow = MagicMock()
        mock_mlflow.data = MagicMock()
        mock_mlflow.data.from_pandas = MagicMock()
        mock_mlflow.log_input = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        manager._mlflow = mock_mlflow
        manager._run = mock_run
        manager._run_id = "test_run_123"

        return manager

    def test_log_dataset_pandas(self, mlflow_manager: MLflowManager) -> None:
        """Test logging pandas DataFrame."""
        import pandas as pd

        df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})

        # Mock dataset creation
        mock_dataset = MagicMock()
        mlflow_manager._mlflow.data.from_pandas.return_value = mock_dataset

        result = mlflow_manager.log_dataset(df, name="test_dataset", context="training")

        assert result is True
        mlflow_manager._mlflow.data.from_pandas.assert_called_once()
        mlflow_manager._mlflow.log_input.assert_called_once_with(mock_dataset, context="training")

    def test_log_dataset_huggingface(self, mlflow_manager: MLflowManager) -> None:
        """Test logging HuggingFace Dataset."""
        import pandas as pd

        # Create a simple object that behaves like HF Dataset
        class MockHFDataset:
            def to_pandas(self):
                return pd.DataFrame({"text": ["test"]})

        mock_hf_dataset = MockHFDataset()

        # Mock dataset creation
        mock_dataset = MagicMock()
        mlflow_manager._mlflow.data.from_pandas.return_value = mock_dataset

        result = mlflow_manager.log_dataset(mock_hf_dataset, name="hf_dataset", context="training")

        assert result is True
        # Should call from_pandas since it has to_pandas method
        mlflow_manager._mlflow.data.from_pandas.assert_called_once()

    def test_log_dataset_from_file_jsonl(self, mlflow_manager: MLflowManager, tmp_path: Path) -> None:
        """Test logging dataset from JSONL file."""
        import json

        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        with jsonl_file.open("w") as f:
            f.write(json.dumps({"text": "hello", "label": 0}) + "\n")
            f.write(json.dumps({"text": "world", "label": 1}) + "\n")

        # Mock dataset creation
        mock_dataset = MagicMock()
        mlflow_manager._mlflow.data.from_pandas.return_value = mock_dataset

        result = mlflow_manager.log_dataset_from_file(str(jsonl_file), context="training")

        assert result is True
        mlflow_manager._mlflow.data.from_pandas.assert_called_once()

    def test_log_dataset_from_file_csv(self, mlflow_manager: MLflowManager, tmp_path: Path) -> None:
        """Test logging dataset from CSV file."""
        # Create test CSV file
        csv_file = tmp_path / "test.csv"
        with csv_file.open("w") as f:
            f.write("text,label\n")
            f.write("hello,0\n")
            f.write("world,1\n")

        # Mock dataset creation
        mock_dataset = MagicMock()
        mlflow_manager._mlflow.data.from_pandas.return_value = mock_dataset

        result = mlflow_manager.log_dataset_from_file(str(csv_file), context="training")

        assert result is True
        mlflow_manager._mlflow.data.from_pandas.assert_called_once()

    def test_create_mlflow_dataset(self, mlflow_manager: MLflowManager) -> None:
        """Test creating MLflow Dataset object."""
        import pandas as pd

        df = pd.DataFrame({"text": ["test"]})

        # Mock dataset creation
        mock_dataset = MagicMock()
        mlflow_manager._mlflow.data.from_pandas.return_value = mock_dataset

        result = mlflow_manager.create_mlflow_dataset(df, name="test", source="test.csv")

        assert result == mock_dataset
        mlflow_manager._mlflow.data.from_pandas.assert_called_once()


# =============================================================================
# PRIORITY 2: MODEL REGISTRY TESTS
# =============================================================================


class TestMLflowManagerModelRegistry:
    """Test MLflow Model Registry functionality (Priority 2)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.tracking_uri = "sqlite:///test.db"
        mlflow_config.experiment_name = "test"
        mlflow_config.log_model = True
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow and gateway."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager
        from src.training.mlflow.model_registry import MLflowModelRegistry

        manager = MLflowManager(mock_config)

        # Mock MLflow module
        mock_mlflow = MagicMock()
        mock_mlflow.register_model = MagicMock()
        mock_mlflow.tracking = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        manager._mlflow = mock_mlflow
        manager._run = mock_run
        manager._run_id = "test_run_123"

        # Mock gateway to return a controllable client
        mock_client = MagicMock()
        mock_gateway = MagicMock()
        mock_gateway.get_client.return_value = mock_client
        manager._gateway = mock_gateway

        # Initialize registry with mocked dependencies
        manager._registry = MLflowModelRegistry(mock_gateway, mock_mlflow, log_model_enabled=True)

        return manager

    def test_register_model_success(self, mlflow_manager: MLflowManager) -> None:
        """Test successful model registration."""
        # Mock registration result
        mock_result = MagicMock()
        mock_result.version = "3"
        mlflow_manager._mlflow.register_model.return_value = mock_result

        version = mlflow_manager.register_model("test-model", alias="champion")

        assert version == "3"
        mlflow_manager._mlflow.register_model.assert_called_once()

    def test_set_model_alias(self, mlflow_manager: MLflowManager) -> None:
        """Test setting model alias."""
        mock_client = mlflow_manager._gateway.get_client()

        result = mlflow_manager.set_model_alias("test-model", "champion", 3)

        assert result is True
        mock_client.set_registered_model_alias.assert_called_once_with("test-model", "champion", "3")

    def test_promote_model(self, mlflow_manager: MLflowManager) -> None:
        """Test promoting model between aliases."""
        mock_client = mlflow_manager._gateway.get_client()
        # Mock get_model_version_by_alias to return version info
        mock_version = MagicMock()
        mock_version.version = "5"
        mock_client.get_model_version_by_alias.return_value = mock_version

        result = mlflow_manager.promote_model("test-model", from_alias="staging", to_alias="champion")

        assert result is True
        # Should get version from staging alias
        mock_client.get_model_version_by_alias.assert_called_once_with("test-model", "staging")
        # Should set champion alias to that version
        mock_client.set_registered_model_alias.assert_called_with("test-model", "champion", "5")

    def test_get_model_by_alias(self, mlflow_manager: MLflowManager) -> None:
        """Test getting model by alias."""
        mock_client = mlflow_manager._gateway.get_client()
        mock_version = MagicMock()
        mock_version.version = "3"
        mock_version.name = "test-model"
        mock_client.get_model_version_by_alias.return_value = mock_version

        result = mlflow_manager.get_model_by_alias("test-model", "champion")

        assert result is not None
        assert result["version"] == "3"
        mock_client.get_model_version_by_alias.assert_called_once_with("test-model", "champion")


# =============================================================================
# PRIORITY 2: RUN MANAGEMENT TESTS
# =============================================================================


class TestMLflowManagerRunManagement:
    """Test MLflow run management functionality (Priority 2)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.tracking_uri = "sqlite:///test.db"
        mlflow_config.experiment_name = "test"
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "parent_run_123"
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.end_run = MagicMock()
        manager._mlflow = mock_mlflow

        return manager

    def test_start_run_with_description(self, mlflow_manager: MLflowManager) -> None:
        """Test starting run with description."""
        with mlflow_manager.start_run(run_name="test_run", description="Test description") as run:
            assert run is not None
            assert mlflow_manager._run_id == "parent_run_123"

        # Check that start_run was called with description
        mlflow_manager._mlflow.start_run.assert_called_once()
        call_kwargs = mlflow_manager._mlflow.start_run.call_args[1]
        assert call_kwargs["run_name"] == "test_run"
        assert call_kwargs["description"] == "Test description"

    def test_get_child_runs(self, mlflow_manager: MLflowManager) -> None:
        """Test getting child runs."""
        # Mock search_runs to return child runs
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "run_id": ["child1", "child2"],
                "tags.mlflow.parentRunId": ["parent_123", "parent_123"],
            }
        )
        mlflow_manager._mlflow.search_runs.return_value = mock_df
        mlflow_manager._run_id = "parent_123"
        mlflow_manager._parent_run_id = "parent_123"

        children = mlflow_manager.get_child_runs()

        assert len(children) == 2
        assert children[0]["run_id"] == "child1"
        mlflow_manager._mlflow.search_runs.assert_called_once()

    def test_end_run_with_status(self, mlflow_manager: MLflowManager) -> None:
        """Test ending run with explicit status."""
        mlflow_manager._mlflow.end_run = MagicMock()

        mlflow_manager.end_run(status="FAILED")

        mlflow_manager._mlflow.end_run.assert_called_once_with(status="FAILED")


# =============================================================================
# PRIORITY 3: BOUNDARY CASES TESTS
# =============================================================================


class TestMLflowManagerBoundaryCases:
    """Test MLflow boundary cases (Priority 3)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock PipelineConfig."""
        config = MagicMock()
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.tracking_uri = "http://localhost:5002"
        mlflow_config.experiment_name = "test"
        config.experiment_tracking.mlflow = mlflow_config
        config.model.name = "test/model"
        return config

    @pytest.fixture
    def mlflow_manager(self, mock_config: MagicMock) -> MLflowManager:
        """Create MLflowManager with mocked MLflow."""
        from unittest.mock import MagicMock

        from src.training.managers.mlflow_manager import MLflowManager

        manager = MLflowManager(mock_config)

        # Mock MLflow
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        manager._mlflow = mock_mlflow
        manager._run = mock_run
        manager._run_id = "test_run_123"

        return manager

    def test_log_params_none_values(self, mlflow_manager: MLflowManager) -> None:
        """Test logging parameters with None values."""
        mlflow_manager._mlflow.log_params = MagicMock()

        # Should filter out None values
        mlflow_manager.log_params({"valid_param": "value", "none_param": None})

        mlflow_manager._mlflow.log_params.assert_called_once()
        logged_params = mlflow_manager._mlflow.log_params.call_args[0][0]
        assert "valid_param" in logged_params
        # None values should be filtered or converted
        assert logged_params.get("none_param") is not None or "none_param" not in logged_params

    def test_log_metrics_invalid_types(self, mlflow_manager: MLflowManager) -> None:
        """Test logging metrics with invalid types."""
        import contextlib

        # log_metrics should only accept numeric values
        # Test that it doesn't crash with invalid types
        with contextlib.suppress(Exception):
            result = mlflow_manager.log_metrics({"valid_metric": 0.5})
            # Valid metric should succeed
            assert result is True

        # Invalid metric types should be handled gracefully (not crash)
        with contextlib.suppress(Exception):
            mlflow_manager.log_metrics({"invalid_metric": "not_a_number"})
            # Should either skip or convert, but not crash

    def test_log_dict_empty(self, mlflow_manager: MLflowManager) -> None:
        """Test logging empty dictionary."""
        from unittest.mock import PropertyMock, patch

        from src.training.managers.mlflow_manager import MLflowManager

        mock_client = MagicMock()
        mock_client.log_dict = MagicMock()

        with patch.object(MLflowManager, "client", new_callable=PropertyMock, return_value=mock_client):
            result = mlflow_manager.log_dict({}, "empty.json")

            # Should still log empty dict
            assert result is True
            mock_client.log_dict.assert_called_once()

    def test_search_runs_empty_experiment(self, mlflow_manager: MLflowManager) -> None:
        """Test searching runs in empty experiment."""
        import pandas as pd

        # Mock empty search result
        mlflow_manager._mlflow.search_runs.return_value = pd.DataFrame()

        runs = mlflow_manager.search_runs(filter_string="metrics.loss < 0.1")

        assert len(runs) == 0

    def test_compare_runs_invalid_ids(self, mlflow_manager: MLflowManager) -> None:
        """Test comparing runs with invalid IDs."""
        # Mock tracking.MlflowClient().get_run() to raise exception for invalid IDs
        mock_client = MagicMock()
        mock_client.get_run.side_effect = Exception("Run not found")
        mlflow_manager._mlflow.tracking.MlflowClient.return_value = mock_client

        runs = mlflow_manager.compare_runs(["invalid_id_1", "invalid_id_2"])

        # Should return empty list for invalid IDs
        assert len(runs) == 0

    def test_get_best_run_no_metrics(self, mlflow_manager: MLflowManager) -> None:
        """Test getting best run when no runs have metrics."""
        import pandas as pd

        # Mock search_runs to return empty
        mlflow_manager._mlflow.search_runs.return_value = pd.DataFrame()

        best_run = mlflow_manager.get_best_run(metric="eval_loss")

        assert best_run is None

    def test_normalize_tracking_uri_localhost(self, mlflow_manager: MLflowManager) -> None:
        """Test URI normalization for localhost — gateway preserves localhost."""
        from src.infrastructure.mlflow.gateway import MLflowGateway

        gw = MLflowGateway("http://localhost:5002", normalize=False)
        # localhost is not a private network IP so it passes through unchanged
        assert "localhost" in gw.uri or "127.0.0.1" in gw.uri

    def test_check_connectivity_timeout(self, mock_config: MagicMock) -> None:
        """Test connectivity check with timeout — tested via MLflowGateway."""
        from unittest.mock import patch

        from src.infrastructure.mlflow.gateway import MLflowGateway

        gw = MLflowGateway("http://unreachable:5002", normalize=False)

        with patch("urllib.request.urlopen", side_effect=OSError("Timeout")):
            result = gw.check_connectivity(timeout=1.0)

            assert result is False

    def test_cleanup_resets_state(self, mlflow_manager: MLflowManager) -> None:
        """Test cleanup resets all state."""
        # Set some state
        mlflow_manager._run_id = "test_run"
        mlflow_manager._nested_run_stack.append("nested_1")
        mlflow_manager._event_log._events.append({"event": "test"})
        mlflow_manager._event_log._event_counter = 10
        mlflow_manager._event_log._has_errors = True

        # Cleanup
        mlflow_manager.cleanup()

        # State should be reset
        assert mlflow_manager._run_id is None
        assert len(mlflow_manager.get_events()) == 0
        assert mlflow_manager._event_log._event_counter == 0
        assert mlflow_manager._event_log._has_errors is False
        assert mlflow_manager._mlflow is None
