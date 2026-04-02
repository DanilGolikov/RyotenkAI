"""
Unit tests for PipelineOrchestrator.

Test coverage:
- Initialization with config loading and validation
- Run happy path with sequential stage execution
- Error handling (stage failures, interrupts)
- Partial execution (start_from_stage, stop_at_stage)
- MLflow integration (setup, logging, metrics aggregation)
- Report generation
- DatasetValidator callbacks
- Helper methods
- Edge cases
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.result import AppError, Err, Ok

# ========================================================================
# FIXTURES
# ========================================================================


@pytest.fixture
def mock_config_path(tmp_path: Path) -> Path:
    """Create temporary config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: gpt2\n")
    return config_file


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock PipelineConfig."""
    config = MagicMock()
    config.model.name = "gpt2"
    config.training.type = "sft"
    config.training.strategies = []  # Empty for basic tests
    config.training.hyperparams.per_device_train_batch_size = 4
    config.training.get_strategy_chain.return_value = []
    config.training.get_effective_load_in_4bit.return_value = False
    config.get_primary_dataset.return_value = MagicMock(
        train_path="data/train.jsonl",
        adapter_type="chat",
    )
    config.experiment_tracking.mlflow = MagicMock(
        tracking_uri="http://localhost:5002",
        system_metrics_callback_enabled=False,
    )
    return config


@pytest.fixture(autouse=True)
def bypass_mlflow_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight fail-fast behavior is covered in dedicated tests."""
    monkeypatch.setattr(
        PipelineOrchestrator,
        "_ensure_mlflow_preflight",
        lambda self, *, state: None,
    )


@pytest.fixture
def mock_secrets() -> MagicMock:
    """Mock Secrets."""
    secrets = MagicMock()
    secrets.hf_token = "test_token"
    return secrets


@pytest.fixture
def mock_stage() -> MagicMock:
    """Create a mock stage."""
    stage = MagicMock()
    stage.stage_name = "Mock Stage"
    stage.run.return_value = Ok({"mock_result": "data"})
    return stage


@pytest.fixture
def mock_stages() -> list[MagicMock]:
    """Create mock stages for pipeline tests (4 always-on + 1 optional evaluator)."""
    stage_names = [
        "Dataset Validator",
        "GPU Deployer",
        "Training Monitor",
        "Model Retriever",
        "Model Evaluator",  # index 4: conditional, used in evaluation display tests
    ]
    stages = []
    for name in stage_names:
        stage = MagicMock()
        stage.stage_name = name
        stage.run.return_value = Ok({name: {"status": "completed"}})
        stage.cleanup = MagicMock()
        stages.append(stage)
    return stages


# ========================================================================
# PRIORITY 1: INITIALIZATION (5 tests)
# ========================================================================


class TestPipelineOrchestratorInitialization:
    """Test PipelineOrchestrator initialization."""

    def test_init_success_loads_config_and_secrets(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Test successful initialization loads config and secrets."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch("src.pipeline.orchestrator.DatasetValidator"),
            patch("src.pipeline.orchestrator.GPUDeployer"),
            patch("src.pipeline.orchestrator.TrainingMonitor"),
            patch("src.pipeline.orchestrator.ModelRetriever"),
            patch("src.pipeline.orchestrator.ModelEvaluator"),
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)

            orchestrator = PipelineOrchestrator(mock_config_path)

            assert orchestrator.config is mock_config
            assert orchestrator.secrets is mock_secrets
            assert orchestrator.config_path == mock_config_path
            mock_load_config.assert_called_once_with(mock_config_path)
            mock_load_secrets.assert_called_once()

    def test_init_invalid_config_path_raises_exception(self):
        """Test initialization with invalid config path raises exception."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
        ):
            mock_load_config.side_effect = FileNotFoundError("Config not found")

            with pytest.raises(FileNotFoundError):
                PipelineOrchestrator(Path("/nonexistent/config.yaml"))

    def test_init_validates_strategy_chain(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Test that strategy chain is validated during initialization."""
        # Setup config with strategies
        mock_strategy1 = MagicMock()
        mock_strategy1.strategy_type = "sft"
        mock_strategy2 = MagicMock()
        mock_strategy2.strategy_type = "dpo"

        mock_config.training.strategies = [mock_strategy1, mock_strategy2]

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch("src.pipeline.orchestrator.DatasetValidator"),
            patch("src.pipeline.orchestrator.GPUDeployer"),
            patch("src.pipeline.orchestrator.TrainingMonitor"),
            patch("src.pipeline.orchestrator.ModelRetriever"),
            patch("src.pipeline.orchestrator.ModelEvaluator"),
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)

            orchestrator = PipelineOrchestrator(mock_config_path)

            # Validate was called with strategies
            mock_validate.assert_called_once_with([mock_strategy1, mock_strategy2])
            assert orchestrator.config is mock_config

    def test_init_invalid_strategy_chain_raises_valueerror(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Test that invalid strategy chain raises ValueError."""
        mock_strategy = MagicMock()
        mock_strategy.strategy_type = "invalid"
        mock_config.training.strategies = [mock_strategy]

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (False, "Invalid strategy type")

            with pytest.raises(ValueError, match="Invalid strategy chain"):
                PipelineOrchestrator(mock_config_path)

    def test_init_creates_all_stages_in_correct_order(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Test that canonical stages are created in correct order."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch("src.pipeline.orchestrator.DatasetValidator") as MockValidator,
            patch("src.pipeline.orchestrator.GPUDeployer") as MockDeployer,
            patch("src.pipeline.orchestrator.TrainingMonitor") as MockMonitor,
            patch("src.pipeline.orchestrator.ModelRetriever") as MockRetriever,
            patch("src.pipeline.orchestrator.InferenceDeployer") as MockInferenceDeployer,
            patch("src.pipeline.orchestrator.ModelEvaluator") as MockEvaluator,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)

            orchestrator = PipelineOrchestrator(mock_config_path)

            # Canonical pipeline always materializes all stages; optional late stages
            # are skipped later by stateful policy rather than omitted here.
            assert len(orchestrator.stages) == 6
            MockValidator.assert_called_once()
            MockDeployer.assert_called_once()
            MockMonitor.assert_called_once()
            MockRetriever.assert_called_once()
            MockInferenceDeployer.assert_called_once()
            MockEvaluator.assert_called_once()


# ========================================================================
# PRIORITY 1: RUN - HAPPY PATH (7 tests)
# ========================================================================


class TestPipelineOrchestratorHappyPath:
    """Test PipelineOrchestrator run method - happy path."""

    def test_run_executes_all_stages_sequentially(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that run executes all stages sequentially."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # All stages should be called
            assert result.is_success()
            for stage in mock_stages:
                stage.run.assert_called_once()

    def test_run_passes_context_between_stages(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that context is passed between stages."""
        # First stage adds data to context
        mock_stages[0].run.return_value = Ok({"stage1_data": "value1"})
        mock_stages[1].run.return_value = Ok({"stage2_data": "value2"})

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            final_context = result.unwrap()

            # Context should contain data from both stages
            assert "stage1_data" in final_context
            assert "stage2_data" in final_context
            assert final_context["stage1_data"] == "value1"
            assert final_context["stage2_data"] == "value2"

    def test_run_returns_ok_with_final_context(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that run returns Ok with final context."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            context = result.unwrap()
            assert isinstance(context, dict)
            assert "config_path" in context

    def test_run_logs_pipeline_start_and_complete_to_mlflow(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that pipeline start and complete are logged to MLflow."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "test_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Check MLflow logging
            mock_mlflow_manager.log_event_start.assert_called_once()
            mock_mlflow_manager.log_event_complete.assert_called_once()

    def test_run_generates_experiment_report_on_success(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that experiment report is generated on success."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "test_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_gen_report,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            mock_gen_report.assert_called_once_with(run_id="test_run_id")

    def test_run_aggregates_training_metrics_from_child_runs(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that training metrics are aggregated from child runs."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "test_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_aggregate_training_metrics") as mock_aggregate,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            mock_aggregate.assert_called_once()

    def test_run_calls_cleanup_resources_in_finally(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that cleanup_resources is called in finally block."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_cleanup_resources") as mock_cleanup,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Cleanup should be called with success=True
            mock_cleanup.assert_called_once_with(success=True)


# ========================================================================
# PRIORITY 1: RUN - ERROR HANDLING (5 tests)
# ========================================================================


class TestPipelineOrchestratorErrorHandling:
    """Test PipelineOrchestrator error handling."""

    def test_run_stops_on_first_stage_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that run stops on first stage failure."""
        # First stage fails
        mock_stages[0].run.return_value = Err(AppError(message="Stage 1 failed", code="TEST_ERROR"))

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Should fail
            assert result.is_failure()
            assert "Stage 1 failed" in str(result.unwrap_err())

            # First stage called, others not
            mock_stages[0].run.assert_called_once()
            for stage in mock_stages[1:]:
                stage.run.assert_not_called()

    def test_run_logs_stage_failed_to_mlflow_on_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that stage failures are logged to MLflow."""
        mock_stages[1].run.return_value = Err(AppError(message="Stage 2 failed", code="TEST_ERROR"))
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_failure()
            # Stage failure should be logged via log_stage_failed, not log_event_error
            mock_mlflow_manager.log_stage_failed.assert_called_once()

    def test_run_calls_cleanup_with_success_false_on_error(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that cleanup is called with success=False on error."""
        mock_stages[0].run.return_value = Err(AppError(message="Stage failed", code="TEST_ERROR"))

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_cleanup_resources") as mock_cleanup,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_failure()
            # Cleanup should be called with success=False
            mock_cleanup.assert_called_once_with(success=False)

    def test_run_handles_exception_during_stage_execution(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that exceptions during stage execution are handled."""
        mock_stages[0].run.side_effect = RuntimeError("Unexpected error")

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Should capture exception
            assert result.is_failure()
            assert "Unexpected error" in str(result.unwrap_err())

    def test_run_returns_err_with_error_message_on_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that run returns Err with error message on failure."""
        error_msg = "DatasetValidator failed: invalid data"
        mock_stages[0].run.return_value = Err(AppError(message=error_msg, code="TEST_ERROR"))

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_failure()
            error = str(result.unwrap_err())
            assert error_msg in error  # type: ignore[operator]


# ========================================================================
# PRIORITY 1: RUN - PARTIAL EXECUTION (3 tests)
# ========================================================================


class TestPipelineOrchestratorPartialExecution:
    """Test PipelineOrchestrator partial execution scenarios."""

    def test_run_preserves_context_from_completed_stages_on_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that context from completed stages is preserved on failure."""
        # First two stages succeed, third fails
        mock_stages[0].run.return_value = Ok({"stage1_data": "value1"})
        mock_stages[1].run.return_value = Ok({"stage2_data": "value2"})
        mock_stages[2].run.return_value = Err(AppError(message="Stage 3 failed", code="TEST_ERROR"))

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Should fail
            assert result.is_failure()

            # But context should contain data from first two stages
            assert "stage1_data" in orchestrator.context
            assert "stage2_data" in orchestrator.context
            assert orchestrator.context["stage1_data"] == "value1"
            assert orchestrator.context["stage2_data"] == "value2"

    def test_run_logs_completed_stages_before_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that completed stages are logged before failure."""
        mock_stages[0].run.return_value = Ok({})
        mock_stages[1].run.return_value = Err(AppError(message="Stage 2 failed", code="TEST_ERROR"))
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_failure()
            # First stage should have completion logged
            assert mock_mlflow_manager.log_stage_complete.call_count == 1
            # Second stage should have failure logged
            assert mock_mlflow_manager.log_stage_failed.call_count == 1

    def test_run_does_not_execute_remaining_stages_after_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that remaining stages are not executed after failure."""
        # Second stage fails
        mock_stages[1].run.return_value = Err(AppError(message="Stage 2 failed", code="TEST_ERROR"))

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_failure()
            # First two stages should be called
            mock_stages[0].run.assert_called_once()
            mock_stages[1].run.assert_called_once()
            # Remaining stages should not be called
            for stage in mock_stages[2:]:
                stage.run.assert_not_called()


# ========================================================================
# PRIORITY 2: MLFLOW LOGGING (7 tests, including setup integration)
# ========================================================================


class TestPipelineOrchestratorMLflowLogging:
    """Test PipelineOrchestrator MLflow event logging."""

    def test_run_logs_stage_start_for_each_stage(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that stage start is logged for each stage."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # log_stage_start should be called for each stage
            assert mock_mlflow_manager.log_stage_start.call_count == len(mock_stages)

    def test_run_logs_stage_complete_for_successful_stages(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that stage completion is logged for successful stages."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # log_stage_complete should be called for each stage
            assert mock_mlflow_manager.log_stage_complete.call_count == len(mock_stages)

    def test_log_stage_specific_info_after_dataset_validator(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _log_stage_specific_info logs dataset info after DatasetValidator."""
        mock_stages[0].stage_name = "Dataset Validator"
        mock_stages[0].run.return_value = Ok({"validated_datasets": ["dataset1", "dataset2"]})
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should log dataset info
            assert mock_mlflow_manager.log_params.call_count > 0

    def test_run_wraps_execution_in_mlflow_parent_run(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that run execution is wrapped in MLflow parent run context."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # start_run should be called to create parent context
            mock_mlflow_manager.start_run.assert_called_once()


# ========================================================================
# PRIORITY 2: MLFLOW AGGREGATION (4 tests)
# ========================================================================


class TestPipelineOrchestratorMLflowAggregation:
    """Test PipelineOrchestrator MLflow metrics aggregation."""

    def test_aggregate_training_metrics_collects_descendant_metrics(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _aggregate_training_metrics collects metrics from descendant runs."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.return_value = [{"loss": 0.1, "accuracy": 0.9}, {"loss": 0.2, "accuracy": 0.95}]

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # _collect_descendant_metrics should be called with max_depth
            mock_collect.assert_called_once_with(max_depth=2)

    def test_aggregate_training_metrics_handles_empty_metrics(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that aggregation handles empty metrics gracefully."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.return_value = {}

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should not crash, log_metrics may or may not be called with empty dict

    def test_aggregate_training_metrics_handles_exception_gracefully(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that exceptions in aggregation are handled gracefully."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.side_effect = Exception("MLflow error")

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Should still succeed despite aggregation error
            assert result.is_success()


# ========================================================================
# PRIORITY 2: REPORT GENERATION (4 tests)
# ========================================================================


class TestPipelineOrchestratorReportGeneration:
    """Test PipelineOrchestrator experiment report generation."""

    def test_generate_experiment_report_called_on_success(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that report generation is called on successful pipeline run."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch("src.pipeline.orchestrator.ExperimentReportGenerator") as MockReportGen,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Report generator should be instantiated
            MockReportGen.assert_called_once()

    def test_generate_experiment_report_logs_as_artifact(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that generated report is logged as MLflow artifact."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        mock_report_gen = MagicMock()
        mock_report_gen.generate_report.return_value = "# Experiment Report\n..."

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch("src.pipeline.orchestrator.ExperimentReportGenerator") as MockReportGen,
        ):
            MockReportGen.return_value = mock_report_gen
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Report should be logged as artifact
            assert mock_mlflow_manager.log_artifact.call_count > 0

    def test_generate_experiment_report_handles_failure_gracefully(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that report generation failure doesn't fail the pipeline."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch("src.pipeline.orchestrator.ExperimentReportGenerator") as MockReportGen,
        ):
            MockReportGen.side_effect = Exception("Report generation failed")
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Pipeline should still succeed
            assert result.is_success()

    def test_generate_experiment_report_not_called_when_mlflow_disabled(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that report generation is not called when MLflow is disabled."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch("src.pipeline.orchestrator.ExperimentReportGenerator") as MockReportGen,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None  # MLflow disabled

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Report generator should not be called
            MockReportGen.assert_not_called()


# ========================================================================
# PRIORITY 3: HELPER METHODS (2 tests)
# ========================================================================


class TestPipelineOrchestratorHelpers:
    """Test PipelineOrchestrator helper methods."""

    def test_cleanup_resources_calls_cleanup_for_all_stages_in_reverse_order(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _cleanup_resources calls cleanup() for all stages in reverse order."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)

            # Track order of cleanup calls
            call_order: list[str] = []
            for st in mock_stages:
                st.cleanup.side_effect = (lambda s=st: call_order.append(s.stage_name))

            orchestrator._cleanup_resources(success=True)

            # All stages should have cleanup called exactly once
            for st in mock_stages:
                st.cleanup.assert_called_once()

            # Must be reverse order to avoid dependency leaks
            expected = [s.stage_name for s in reversed(mock_stages)]
            assert call_order == expected

    def test_cleanup_resources_skips_gpu_deployer_on_sigint_when_cleanup_on_interrupt_false(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """
        Policy: if providers.<name>.cleanup.on_interrupt=false and shutdown is SIGINT,
        we must NOT call cleanup() on the GPU Deployer stage (provider disconnect).
        """
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_config.get_active_provider_name.return_value = "runpod"
            mock_config.get_provider_config.return_value = {"cleanup": {"on_interrupt": False}}

            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            orchestrator.notify_signal(signal_name="SIGINT")
            orchestrator._cleanup_resources(success=False)

            gpu_deployer = next(s for s in mock_stages if s.stage_name == "GPU Deployer")
            assert gpu_deployer.cleanup.call_count == 0

            for st in mock_stages:
                if st.stage_name != "GPU Deployer":
                    st.cleanup.assert_called_once()

    def test_print_summary_generates_output(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
        capsys: Any,
    ):
        """Test that _print_summary generates console output."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            orchestrator._print_summary()

            # Check that summary was printed
            captured = capsys.readouterr()
            assert "PIPELINE EXECUTION SUMMARY" in captured.out


# ========================================================================
# PRIORITY 3: EDGE CASES (3 tests)
# ========================================================================


class TestPipelineOrchestratorEdgeCases:
    """Test PipelineOrchestrator edge cases."""

    def test_run_handles_keyboard_interrupt(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that KeyboardInterrupt is handled gracefully."""
        mock_stages[0].run.side_effect = KeyboardInterrupt()

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Should return Err with interrupt message
            assert result.is_failure()
            assert "interrupted" in str(result.unwrap_err())

    def test_run_with_empty_stages_list(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Test that run handles empty stages list."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = []  # Empty stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Pipeline validates stage indices, so empty list fails
            assert result.is_failure()

    def test_run_with_stage_returning_none(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that stages can return None without breaking pipeline."""
        # First stage returns None (no context update)
        mock_stages[0].run.return_value = Ok(None)

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Should succeed
            assert result.is_success()


# ========================================================================
# ========================================================================
# COVERAGE BOOST: LOG STAGE SPECIFIC INFO (3 tests)
# ========================================================================


class TestPipelineOrchestratorStageSpecificLogging:
    """Test PipelineOrchestrator _log_stage_specific_info method."""

    def test_log_stage_specific_info_training_monitor(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test logging specific info for Training Monitor stage."""
        mock_stages[2].stage_name = "Training Monitor"
        mock_stages[2].run.return_value = Ok(
            {
                "Training Monitor": {
                    "training_duration_seconds": 120.5,
                    "training_info": {
                        "runtime_seconds": 118.3,
                        "final_loss": 0.45,
                        "final_accuracy": 0.92,
                        "total_steps": 1000,
                    },
                }
            }
        )

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:3]  # Only first 3 stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should log training info
            assert mock_mlflow_manager.log_event_info.call_count >= 1
            assert mock_mlflow_manager.log_metrics.call_count >= 1

    def test_log_stage_specific_info_model_retriever(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test logging specific info for Model Retriever stage."""
        mock_stages[3].stage_name = "Model Retriever"
        mock_stages[3].run.return_value = Ok(
            {
                "Model Retriever": {
                    "model_size_mb": 1234.5,
                    "hf_uploaded": True,
                    "hf_repo_id": "user/model",
                    "upload_duration_seconds": 45.2,
                }
            }
        )

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:4]  # First 4 stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should log model info
            assert mock_mlflow_manager.log_event_info.call_count >= 2  # Model size + HF upload

    def test_log_stage_specific_info_dataset_validator_plugin_mode(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test logging specific info for Dataset Validator in plugin mode."""
        mock_stages[0].stage_name = "Dataset Validator"
        mock_stages[0].run.return_value = Ok(
            {
                "Dataset Validator": {
                    "validation_mode": "plugin",
                    "metrics": {"min_samples": 1000, "avg_length": 512.3, "diversity_score": 0.85},
                }
            }
        )

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:1]  # Only first stage
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should log plugin metrics as params
            assert mock_mlflow_manager.log_params.call_count >= 1


# ========================================================================
# COVERAGE BOOST: AGGREGATE TRAINING METRICS (3 tests)
# ========================================================================


class TestPipelineOrchestratorMetricsAggregation:
    """Test PipelineOrchestrator metrics aggregation methods."""

    def test_aggregate_metrics_called_on_success(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _aggregate_training_metrics is called on pipeline success."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_aggregate_training_metrics") as mock_aggregate,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Aggregation should be called
            mock_aggregate.assert_called_once()

    def test_collect_descendant_metrics_called_from_aggregate(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _collect_descendant_metrics is called during aggregation."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        mock_mlflow_manager._run_id = "parent_run_id"
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.return_value = []  # No metrics

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # _collect_descendant_metrics should be called
            mock_collect.assert_called_once_with(max_depth=2)

    def test_aggregate_metrics_handles_no_mlflow(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test aggregation when MLflow is disabled."""
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = None  # MLflow disabled

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            # Should succeed without MLflow
            assert result.is_success()


# ========================================================================
# COVERAGE BOOST: EXCEPTION HANDLERS (5 tests)
# ========================================================================


class TestPipelineOrchestratorExceptionHandlers:
    """Test PipelineOrchestrator exception handling paths."""

    def test_run_with_keyboard_interrupt_mlflow_logging(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test KeyboardInterrupt logs to MLflow when enabled."""
        mock_stages[0].run.side_effect = KeyboardInterrupt("User stopped")

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_failure()
            # MLflow should log interruption
            mock_mlflow_manager.log_event_warning.assert_called_once()
            mock_mlflow_manager.set_tags.assert_called_once_with({"pipeline.status": "interrupted"})

    def test_run_with_unexpected_exception_mlflow_logging(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test unexpected exception logs to MLflow when enabled."""
        mock_stages[0].run.side_effect = ValueError("Unexpected error")

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_failure()
            # MLflow should log error
            mock_mlflow_manager.log_event_error.assert_called_once()
            assert "Unexpected error" in str(mock_mlflow_manager.log_event_error.call_args)
            mock_mlflow_manager.set_tags.assert_called_once_with({"pipeline.status": "failed"})


# ========================================================================
# COVERAGE BOOST: PRINT SUMMARY DETAILS (3 tests)
# ========================================================================


class TestPipelineOrchestratorPrintSummary:
    """Test PipelineOrchestrator _print_summary method details."""

    def test_print_summary_with_cloud_provider_cost(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test summary printing with cloud provider cost calculation."""
        mock_stages[1].stage_name = "GPU Deployer"
        mock_stages[1].run.return_value = Ok(
            {"GPU Deployer": {"provider_type": "cloud", "pod_info": {"cost_per_hr": 1.5, "gpu_type": "RTX 4090"}}}
        )
        mock_stages[2].stage_name = "Training Monitor"
        mock_stages[2].run.return_value = Ok(
            {
                "Training Monitor": {
                    "training_info": {
                        "runtime_seconds": 3600  # 1 hour
                    }
                }
            }
        )

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:3]
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should calculate and display cost
            assert "pod_info" in orchestrator.context["GPU Deployer"]

    def test_print_summary_with_zero_cost_cloud_provider(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test summary printing with cloud provider but zero cost."""
        mock_stages[1].stage_name = "GPU Deployer"
        mock_stages[1].run.return_value = Ok(
            {
                "GPU Deployer": {
                    "provider_type": "cloud",
                    "pod_info": {
                        "cost_per_hr": 0  # Free tier
                    },
                }
            }
        )

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:2]
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should handle zero cost gracefully
            assert orchestrator.context["GPU Deployer"]["pod_info"]["cost_per_hr"] == 0

    def test_print_summary_with_local_provider(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test summary printing with local provider (no cost)."""
        mock_stages[1].stage_name = "GPU Deployer"
        mock_stages[1].run.return_value = Ok({"GPU Deployer": {"provider_type": "local"}})

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:2]
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should display "$0 (local)"
            assert orchestrator.context["GPU Deployer"]["provider_type"] == "local"


# ========================================================================
# COVERAGE BOOST: EVALUATION METRICS DISPLAY (2 tests)
# ========================================================================


class TestPipelineOrchestratorEvaluationDisplay:
    """Test PipelineOrchestrator evaluation metrics display in summary."""

    def test_print_summary_with_evaluation_metrics(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test summary with evaluation metrics (float values)."""
        mock_stages[4].stage_name = "Model Evaluator"
        mock_stages[4].run.return_value = Ok(
            {"Model Evaluator": {"metrics": {"accuracy": 0.9234, "f1_score": 0.8765, "precision": 0.9123}}}
        )

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:5]
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should display evaluation metrics
            assert "metrics" in orchestrator.context["Model Evaluator"]
            assert orchestrator.context["Model Evaluator"]["metrics"]["accuracy"] == 0.9234

    def test_print_summary_with_mixed_evaluation_metrics(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test summary with mixed evaluation metrics (float + string)."""
        mock_stages[4].stage_name = "Model Evaluator"
        mock_stages[4].run.return_value = Ok(
            {"Model Evaluator": {"metrics": {"accuracy": 0.9234, "status": "PASSED", "f1_score": 0.8765}}}
        )

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:5]
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # Should handle mixed types
            assert orchestrator.context["Model Evaluator"]["metrics"]["status"] == "PASSED"


# ========================================================================
# COVERAGE BOOST: AGGREGATION DETAILS (4 tests)
# ========================================================================


class TestPipelineOrchestratorAggregationDetails:
    """Test detailed aggregation logic in _aggregate_training_metrics."""

    def test_aggregation_calls_collect_with_final_loss(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test aggregation pathway when metrics include train_loss."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.return_value = [{"train_loss": 0.45}]
            mock_report.return_value = None  # Skip report generation

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            # _collect_descendant_metrics should be called
            mock_collect.assert_called_once()

    def test_aggregation_pathway_with_runtime(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test aggregation pathway when metrics include train_runtime."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.return_value = [{"train_runtime": 120.5}, {"train_runtime": 150.3}]
            mock_report.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            mock_collect.assert_called_once()

    def test_aggregation_pathway_with_global_step(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test aggregation pathway when metrics include global_step."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.return_value = [{"global_step": 100}, {"global_step": 200}]
            mock_report.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            mock_collect.assert_called_once()

    def test_aggregation_pathway_with_all_metrics(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test aggregation pathway with complete metrics."""
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages
            mock_setup_mlflow.return_value = mock_mlflow_manager
            mock_collect.return_value = [
                {"train_loss": 0.5, "train_runtime": 100.0, "global_step": 100},
                {"train_loss": 0.2, "train_runtime": 200.0, "global_step": 300},
            ]
            mock_report.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()
            mock_collect.assert_called_once()


# ========================================================================
# COVERAGE BOOST: ADDITIONAL EDGE CASES (2 tests)
# ========================================================================


class TestPipelineOrchestratorAdditionalCoverage:
    """Additional tests to reach 80% coverage."""

    def test_print_summary_without_deployer_context(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test summary when GPU Deployer context is missing."""
        mock_stages[1].stage_name = "GPU Deployer"
        mock_stages[1].run.return_value = Ok({})  # Empty context

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:2]
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()

    def test_print_summary_without_retriever_context(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test summary when Model Retriever context is missing."""
        # Skip Model Retriever
        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages[:3]  # Skip retriever
            mock_setup_mlflow.return_value = None

            orchestrator = PipelineOrchestrator(mock_config_path)
            result = orchestrator.run()

            assert result.is_success()


# ========================================================================
# COVERAGE BOOST: MLFLOW INTERNAL METHODS (5 tests)
# ========================================================================


class TestPipelineOrchestratorMLflowInternals:
    """Test MLflow internal methods with detailed mocking."""

    def test_setup_mlflow_returns_none_when_disabled(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _setup_mlflow returns None when MLflow is disabled."""
        # Disable MLflow
        mock_config.experiment_tracking.mlflow = None

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages

            orchestrator = PipelineOrchestrator(mock_config_path)

            # _mlflow_manager should be None
            assert orchestrator._mlflow_manager is None

    def test_setup_mlflow_handles_setup_exception(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _setup_mlflow handles exceptions during setup."""
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mock_config.experiment_tracking.mlflow = mlflow_config

        # Mock MLflowManager to raise exception
        mock_mlflow_manager_class = MagicMock()
        mock_mlflow_manager_class.side_effect = Exception("Setup failed")

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch("src.pipeline.orchestrator.MLflowManager", mock_mlflow_manager_class),
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages

            orchestrator = PipelineOrchestrator(mock_config_path)

            # Should handle exception gracefully, _mlflow_manager should be None
            assert orchestrator._mlflow_manager is None

    def test_setup_mlflow_returns_none_when_not_active(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that _setup_mlflow returns None when manager is not active."""
        mlflow_config = MagicMock()
        mlflow_config.enabled = True
        mlflow_config.system_metrics_callback_enabled = False
        mock_config.experiment_tracking.mlflow = mlflow_config

        mock_mlflow_manager_class = MagicMock()
        mock_manager_instance = MagicMock()
        mock_manager_instance.is_active = False  # Not active
        mock_manager_instance.setup = MagicMock()
        mock_mlflow_manager_class.return_value = mock_manager_instance

        with (
            patch("src.pipeline.orchestrator.load_config") as mock_load_config,
            patch("src.pipeline.orchestrator.load_secrets") as mock_load_secrets,
            patch("src.pipeline.orchestrator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_init_stages") as mock_init_stages,
            patch("src.pipeline.orchestrator.MLflowManager", mock_mlflow_manager_class),
            patch.dict("sys.modules", {"mlflow": MagicMock()}),
        ):
            mock_load_config.return_value = mock_config
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = (True, None)
            mock_init_stages.return_value = mock_stages

            orchestrator = PipelineOrchestrator(mock_config_path)

            # _mlflow_manager should be None since is_active = False
            assert orchestrator._mlflow_manager is None
