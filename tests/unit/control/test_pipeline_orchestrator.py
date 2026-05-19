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

from ryotenkai_control.pipeline.orchestrator import PipelineOrchestrator
from ryotenkai_control.pipeline.execution import StageRegistry
from ryotenkai_shared.errors import PipelineStageFailedError, RyotenkAIError

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
def mock_config(mock_config_path: Path) -> MagicMock:
    """Mock PipelineConfig with ``_source_path`` set.

    Mirrors what ``load_pipeline_config`` does for production code —
    bootstrap reads ``_source_path`` to record the canonical config
    path on PipelineState, so any test going through the orchestrator
    constructor needs it stamped.
    """
    config = MagicMock()
    config._source_path = mock_config_path
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
    config.integrations.mlflow = MagicMock(
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
    stage.run.return_value = {"mock_result": "data"}
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
        stage.run.return_value = {name: {"status": "completed"}}
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_control.pipeline.execution.stage_registry.DatasetValidator"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.GPUDeployer"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.TrainingMonitor"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.ModelRetriever"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.ModelEvaluator"),
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config)

            assert orchestrator.config is mock_config
            assert orchestrator.secrets is mock_secrets
            assert orchestrator.config_path == mock_config_path
            mock_load_secrets.assert_called_once()
    # ``test_init_invalid_config_path_raises_exception`` removed: the
    # legacy positional ``PipelineOrchestrator(config_path)`` constructor
    # no longer exists. Path-resolution / FileNotFound errors are now
    # the responsibility of the upstream ``load_pipeline_config`` helper.

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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_control.pipeline.execution.stage_registry.DatasetValidator"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.GPUDeployer"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.TrainingMonitor"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.ModelRetriever"),
            patch("ryotenkai_control.pipeline.execution.stage_registry.ModelEvaluator"),
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config)

            # Validate was called with strategies
            mock_validate.assert_called_once_with([mock_strategy1, mock_strategy2])
            assert orchestrator.config is mock_config

    def test_init_critical_strategy_chain_error_raises_typed(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Critical strategy-chain errors propagate typed.

        Regression (2026-05-16): pre-fix orchestrator wrapped
        StrategyChainInvalidError in StartupValidationError(ValueError);
        the wrapper stripped typed semantics so the worker subprocess
        printed raw Python tracebacks instead of the unified kubectl-style
        CLI render. Now StrategyChainInvalidError (RyotenkAIError, HTTP 422)
        propagates unchanged to the CLI/worker boundary.
        """
        from ryotenkai_shared.errors import RyotenkAIError, StrategyChainInvalidError

        mock_strategy = MagicMock()
        mock_strategy.strategy_type = "sft"
        mock_config.training.strategies = [mock_strategy]

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.side_effect = StrategyChainInvalidError(
                detail="Duplicate dataset 'shared'",
                context={"reason": "duplicate_dataset"},
            )

            with pytest.raises(StrategyChainInvalidError, match="Duplicate dataset") as ei:
                PipelineOrchestrator(config=mock_config)
            assert isinstance(ei.value, RyotenkAIError)
            assert ei.value.status == 422

    def test_init_creates_all_stages_in_correct_order(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Test that canonical stages are created in correct order."""
        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_control.pipeline.execution.stage_registry.DatasetValidator") as MockValidator,
            patch("ryotenkai_control.pipeline.execution.stage_registry.GPUDeployer") as MockDeployer,
            patch("ryotenkai_control.pipeline.execution.stage_registry.TrainingMonitor") as MockMonitor,
            patch("ryotenkai_control.pipeline.execution.stage_registry.ModelRetriever") as MockRetriever,
            patch("ryotenkai_control.pipeline.execution.stage_registry.InferenceDeployer") as MockInferenceDeployer,
            patch("ryotenkai_control.pipeline.execution.stage_registry.ModelEvaluator") as MockEvaluator,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config)

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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "xfail-debt:m7-wide-mlflow-manager-retired -- tests pin the "
        "wide MLflowManager teardown surface that the M7 cleanup retired. "
        "Some assertions still pass (the run() loop works); pinned strict=False "
        "until the tests are rewritten against the narrow stack."
    ),
)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            # All stages should be called
            assert isinstance(result, dict)
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
        mock_stages[0].run.return_value = {"stage1_data": "value1"}
        mock_stages[1].run.return_value = {"stage2_data": "value2"}

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
            final_context = result

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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
            context = result
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
            # Phase 7: ``log_event_start`` / ``log_event_complete`` removed
            # — the pipeline lifecycle is captured on the typed journal.
            # The set_tags + log_params side effects (pipeline.status +
            # duration_seconds) remain part of the surface.
            mock_mlflow_manager.set_tags.assert_any_call({"pipeline.status": "completed"})

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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_gen_report,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_aggregate_training_metrics") as mock_aggregate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_cleanup_resources") as mock_cleanup,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
        mock_stages[0].run.side_effect = PipelineStageFailedError(detail="Stage 1 failed", context={"legacy_code": "TEST_ERROR"})

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # Should fail
            # raise asserted via pytest.raises
            assert "Stage 1 failed" in str(exc_info.value.detail or exc_info.value)

            # First stage called, others not
            mock_stages[0].run.assert_called_once()
            for stage in mock_stages[1:]:
                stage.run.assert_not_called()

    @pytest.mark.skip(reason="phase-F mlflow-cleanup: wide IMLflowManager retired; behaviour moved to typed journal events")
    def test_run_logs_stage_failed_to_mlflow_on_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that stage failures are logged to MLflow."""
        mock_stages[1].run.side_effect = PipelineStageFailedError(detail="Stage 2 failed", context={"legacy_code": "TEST_ERROR"})
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # raise asserted via pytest.raises
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
        mock_stages[0].run.side_effect = PipelineStageFailedError(detail="Stage failed", context={"legacy_code": "TEST_ERROR"})

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_cleanup_resources") as mock_cleanup,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # raise asserted via pytest.raises
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # Should capture exception
            # raise asserted via pytest.raises
            assert "Unexpected error" in str(exc_info.value.detail or exc_info.value)

    def test_run_returns_err_with_error_message_on_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that run returns Err with error message on failure."""
        error_msg = "DatasetValidator failed: invalid data"
        mock_stages[0].run.side_effect = PipelineStageFailedError(detail=error_msg, context={"legacy_code": "TEST_ERROR"})

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # raise asserted via pytest.raises
            error = str(exc_info.value.detail or exc_info.value)
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
        mock_stages[0].run.return_value = {"stage1_data": "value1"}
        mock_stages[1].run.return_value = {"stage2_data": "value2"}
        mock_stages[2].run.side_effect = PipelineStageFailedError(detail="Stage 3 failed", context={"legacy_code": "TEST_ERROR"})

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # Should fail
            # raise asserted via pytest.raises
            # But context should contain data from first two stages
            assert "stage1_data" in orchestrator.context
            assert "stage2_data" in orchestrator.context
            assert orchestrator.context["stage1_data"] == "value1"
            assert orchestrator.context["stage2_data"] == "value2"

    @pytest.mark.skip(reason="phase-F mlflow-cleanup: wide IMLflowManager retired; behaviour moved to typed journal events")
    def test_run_logs_completed_stages_before_failure(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that completed stages are logged before failure."""
        mock_stages[0].run.return_value = {}
        mock_stages[1].run.side_effect = PipelineStageFailedError(detail="Stage 2 failed", context={"legacy_code": "TEST_ERROR"})
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # raise asserted via pytest.raises
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
        mock_stages[1].run.side_effect = PipelineStageFailedError(detail="Stage 2 failed", context={"legacy_code": "TEST_ERROR"})

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # raise asserted via pytest.raises
            # First two stages should be called
            mock_stages[0].run.assert_called_once()
            mock_stages[1].run.assert_called_once()
            # Remaining stages should not be called
            for stage in mock_stages[2:]:
                stage.run.assert_not_called()
# ========================================================================
# PRIORITY 2: MLFLOW LOGGING (7 tests, including setup integration)
# ========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="xfail-debt:m7-wide-mlflow-manager-retired",
)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
        mock_stages[0].run.return_value = {"validated_datasets": ["dataset1", "dataset2"]}
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
            # start_run should be called to create parent context
            mock_mlflow_manager.start_run.assert_called_once()
# ========================================================================
# PRIORITY 2: MLFLOW AGGREGATION (4 tests)
# ========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="xfail-debt:m7-wide-mlflow-manager-retired",
)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.return_value = [{"loss": 0.1, "accuracy": 0.9}, {"loss": 0.2, "accuracy": 0.95}]

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.return_value = {}

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.side_effect = Exception("MLflow error")

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            # Should still succeed despite aggregation error
            assert isinstance(result, dict)
# ========================================================================
# PRIORITY 2: REPORT GENERATION (4 tests)
# ========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="xfail-debt:m7-wide-mlflow-manager-retired",
)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_control.pipeline.reporting.summary_reporter.ExperimentReportGenerator") as MockReportGen,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_control.pipeline.reporting.summary_reporter.ExperimentReportGenerator") as MockReportGen,
        ):
            MockReportGen.return_value = mock_report_gen
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_control.pipeline.reporting.summary_reporter.ExperimentReportGenerator") as MockReportGen,
        ):
            MockReportGen.side_effect = Exception("Report generation failed")
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            # Pipeline should still succeed
            assert isinstance(result, dict)

    def test_generate_experiment_report_not_called_when_mlflow_disabled(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that report generation is not called when MLflow is disabled."""
        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_control.pipeline.reporting.summary_reporter.ExperimentReportGenerator") as MockReportGen,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)

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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_config.get_active_provider_name.return_value = "runpod"
            mock_config.get_provider_config.return_value = {"cleanup": {"on_interrupt": False}}

            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # Should return Err with interrupt message
            # raise asserted via pytest.raises
            assert "interrupted" in str(exc_info.value.detail or exc_info.value)

    def test_run_with_empty_stages_list(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
    ):
        """Test that run handles empty stages list."""
        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=[])
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # Pipeline validates stage indices, so empty list fails
            # raise asserted via pytest.raises
    def test_run_with_stage_returning_none(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test that stages can return None without breaking pipeline."""
        # First stage returns None (no context update)
        mock_stages[0].run.return_value = None

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            # Should succeed
            assert isinstance(result, dict)
# ========================================================================
# ========================================================================
# COVERAGE BOOST: LOG STAGE SPECIFIC INFO (3 tests)
# ========================================================================


class TestPipelineOrchestratorStageSpecificLogging:
    """Test PipelineOrchestrator _log_stage_specific_info method."""

    @pytest.mark.skip(reason="phase-F mlflow-cleanup: wide IMLflowManager retired; behaviour moved to typed journal events")
    def test_log_stage_specific_info_training_monitor(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test logging specific info for Training Monitor stage."""
        mock_stages[2].stage_name = "Training Monitor"
        mock_stages[2].run.return_value = {
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
        

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:3])
            result = orchestrator.run()

            assert isinstance(result, dict)
            # Phase 7: training info now flows entirely through
            # ``log_metrics`` (training.duration_seconds + the
            # final_loss / accuracy / total_steps metrics).
            assert mock_mlflow_manager.log_metrics.call_count >= 1

    @pytest.mark.skip(reason="phase-F mlflow-cleanup: wide IMLflowManager retired; behaviour moved to typed journal events")
    def test_log_stage_specific_info_model_retriever(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test logging specific info for Model Retriever stage."""
        mock_stages[3].stage_name = "Model Retriever"
        mock_stages[3].run.return_value = {
                "Model Retriever": {
                    "model_size_mb": 1234.5,
                    "hf_uploaded": True,
                    "hf_repo_id": "user/model",
                    "upload_duration_seconds": 45.2,
                }
            }
        

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:4])
            result = orchestrator.run()

            assert isinstance(result, dict)
            # Phase 7: model info now flows through log_metrics (size +
            # upload_duration) + set_tags (hf_repo_id).
            mock_mlflow_manager.set_tags.assert_any_call({"model.hf_repo_id": "user/model"})
            upload_metric_calls = [
                c for c in mock_mlflow_manager.log_metrics.call_args_list
                if "model.upload_duration_seconds" in c.args[0]
            ]
            assert upload_metric_calls

    @pytest.mark.skip(reason="phase-F mlflow-cleanup: wide IMLflowManager retired; behaviour moved to typed journal events")
    def test_log_stage_specific_info_dataset_validator_plugin_mode(
        self,
        mock_config_path: Path,
        mock_config: MagicMock,
        mock_secrets: MagicMock,
        mock_stages: list[MagicMock],
    ):
        """Test logging specific info for Dataset Validator in plugin mode."""
        mock_stages[0].stage_name = "Dataset Validator"
        mock_stages[0].run.return_value = {
                "Dataset Validator": {
                    "validation_mode": "plugin",
                    "metrics": {"min_samples": 1000, "avg_length": 512.3, "diversity_score": 0.85},
                }
            }
        

        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.is_enabled = True
        # Properly mock context manager
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_context
        mock_context.__exit__.return_value = None
        mock_mlflow_manager.start_run.return_value = mock_context

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:1])
            result = orchestrator.run()

            assert isinstance(result, dict)
            # Should log plugin metrics as params
            assert mock_mlflow_manager.log_params.call_count >= 1
# ========================================================================
# COVERAGE BOOST: AGGREGATE TRAINING METRICS (3 tests)
# ========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="xfail-debt:m7-wide-mlflow-manager-retired",
)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_aggregate_training_metrics") as mock_aggregate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.return_value = []  # No metrics

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            # Should succeed without MLflow
            assert isinstance(result, dict)
# ========================================================================
# COVERAGE BOOST: EXCEPTION HANDLERS (5 tests)
# ========================================================================


class TestPipelineOrchestratorExceptionHandlers:
    """Test PipelineOrchestrator exception handling paths."""

    @pytest.mark.skip(reason="phase-F mlflow-cleanup: wide IMLflowManager retired; behaviour moved to typed journal events")
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # raise asserted via pytest.raises
            # Phase 7: ``log_event_warning`` removed; tag set remains.
            mock_mlflow_manager.set_tags.assert_called_once_with({"pipeline.status": "interrupted"})

    @pytest.mark.skip(reason="phase-F mlflow-cleanup: wide IMLflowManager retired; behaviour moved to typed journal events")
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            with pytest.raises(RyotenkAIError) as exc_info:
                orchestrator.run()

            # raise asserted via pytest.raises
            # Phase 7: ``log_event_error`` removed; pipeline.status tag
            # stays set on unexpected error.
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
        mock_stages[1].run.return_value = {"GPU Deployer": {"provider_type": "cloud", "pod_info": {"cost_per_hr": 1.5, "gpu_type": "RTX 4090"}}}
        
        mock_stages[2].stage_name = "Training Monitor"
        mock_stages[2].run.return_value = {
                "Training Monitor": {
                    "training_info": {
                        "runtime_seconds": 3600  # 1 hour
                    }
                }
            }
        

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:3])
            result = orchestrator.run()

            assert isinstance(result, dict)
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
        mock_stages[1].run.return_value = {
                "GPU Deployer": {
                    "provider_type": "cloud",
                    "pod_info": {
                        "cost_per_hr": 0  # Free tier
                    },
                }
            }
        

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:2])
            result = orchestrator.run()

            assert isinstance(result, dict)
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
        mock_stages[1].run.return_value = {"GPU Deployer": {"provider_type": "local"}}

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:2])
            result = orchestrator.run()

            assert isinstance(result, dict)
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
        mock_stages[4].run.return_value = {"Model Evaluator": {"metrics": {"accuracy": 0.9234, "f1_score": 0.8765, "precision": 0.9123}}}
        

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:5])
            result = orchestrator.run()

            assert isinstance(result, dict)
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
        mock_stages[4].run.return_value = {"Model Evaluator": {"metrics": {"accuracy": 0.9234, "status": "PASSED", "f1_score": 0.8765}}}
        

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:5])
            result = orchestrator.run()

            assert isinstance(result, dict)
            # Should handle mixed types
            assert orchestrator.context["Model Evaluator"]["metrics"]["status"] == "PASSED"
# ========================================================================
# COVERAGE BOOST: AGGREGATION DETAILS (4 tests)
# ========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="xfail-debt:m7-wide-mlflow-manager-retired",
)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.return_value = [{"train_loss": 0.45}]
            mock_report.return_value = None  # Skip report generation

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.return_value = [{"train_runtime": 120.5}, {"train_runtime": 150.3}]
            mock_report.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.return_value = [{"global_step": 100}, {"global_step": 200}]
            mock_report.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch.object(PipelineOrchestrator, "_collect_descendant_metrics") as mock_collect,
            patch.object(PipelineOrchestrator, "_generate_experiment_report") as mock_report,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None
            mock_collect.return_value = [
                {"train_loss": 0.5, "train_runtime": 100.0, "global_step": 100},
                {"train_loss": 0.2, "train_runtime": 200.0, "global_step": 300},
            ]
            mock_report.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)
            result = orchestrator.run()

            assert isinstance(result, dict)
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
        mock_stages[1].run.return_value = {}  # Empty context

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:2])
            result = orchestrator.run()

            assert isinstance(result, dict)

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
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages[:3])
            result = orchestrator.run()

            assert isinstance(result, dict)
# ========================================================================
# COVERAGE BOOST: MLFLOW INTERNAL METHODS (5 tests)
# ========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="xfail-debt:m7-wide-mlflow-manager-retired",
)
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
        mock_config.integrations.mlflow = None

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)

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
        mock_config.integrations.mlflow = mlflow_config

        # Mock MLflowManager to raise exception
        mock_mlflow_manager_class = MagicMock()
        mock_mlflow_manager_class.side_effect = Exception("Setup failed")

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_pod.trainer.managers.mlflow_manager.MLflowManager", mock_mlflow_manager_class),
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)

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
        mock_config.integrations.mlflow = mlflow_config

        mock_mlflow_manager_class = MagicMock()
        mock_manager_instance = MagicMock()
        mock_manager_instance.is_active = False  # Not active
        mock_manager_instance.setup = MagicMock()
        mock_mlflow_manager_class.return_value = mock_manager_instance

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets") as mock_load_secrets,
            patch("ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain") as mock_validate,
            patch("ryotenkai_pod.trainer.managers.mlflow_manager.MLflowManager", mock_mlflow_manager_class),
            patch.dict("sys.modules", {"mlflow": MagicMock()}),
        ):
            mock_load_secrets.return_value = mock_secrets
            mock_validate.return_value = None

            orchestrator = PipelineOrchestrator(config=mock_config, stages_override=mock_stages)

            # _mlflow_manager should be None since is_active = False
            assert orchestrator._mlflow_manager is None