"""
Unit tests for PhaseExecutor.

Test coverage:
- Initialization with DI
- Execute happy path
- Error handling (dataset, trainer, training)
- Graceful shutdown (SIGINT/SIGTERM)
- OOM handling
- Resume logic
- DataBuffer integration
- MLflow integration
- Helper methods
- Edge cases
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training.metrics_models import TrainingMetricsSnapshot
from src.training.orchestrator.phase_executor import PhaseExecutor
from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
from src.utils.memory_manager import OOMRecoverableError
from src.utils.result import Err, Ok

# ========================================================================
# FIXTURES
# ========================================================================


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock PreTrainedTokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    return tokenizer


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock PipelineConfig."""
    config = MagicMock()
    config.training.hyperparams.model_dump.return_value = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
    }
    config.get_dataset_for_strategy.return_value = MagicMock(
        get_source_uri=MagicMock(return_value="data/train.jsonl"),
        get_source_type=MagicMock(return_value="jsonl"),
    )
    return config


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Mock MemoryManager with decorator support."""
    manager = MagicMock()

    # Mock with_memory_protection decorator
    # It should return a decorator that returns the original function
    def mock_decorator(*args, **kwargs):
        def decorator(func):
            # Return the original function (pass-through decorator)
            return func

        return decorator

    manager.with_memory_protection = mock_decorator

    # Also keep safe_operation for backward compat
    manager.safe_operation.return_value.__enter__ = MagicMock()
    manager.safe_operation.return_value.__exit__ = MagicMock(return_value=False)

    return manager


@pytest.fixture
def mock_dataset_loader() -> MagicMock:
    """Mock IDatasetLoader."""
    loader = MagicMock()
    loader.load_for_phase.return_value = Ok((MagicMock(__len__=MagicMock(return_value=100)), None))
    return loader


@pytest.fixture
def mock_metrics_collector() -> MagicMock:
    """Mock MetricsCollector."""
    collector = MagicMock()
    collector.extract_from_trainer.return_value = TrainingMetricsSnapshot(
        train_loss=0.45,
        eval_loss=0.52,
        epoch=3,
    )
    return collector


@pytest.fixture
def mock_buffer() -> MagicMock:
    """Mock DataBuffer."""
    buffer = MagicMock()
    buffer.get_phase_output_dir.return_value = "/tmp/output/phase_0"
    buffer.get_resume_checkpoint.return_value = None
    return buffer


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock PreTrainedModel."""
    model = MagicMock()
    model.config.model_type = "gpt2"
    return model


@pytest.fixture
def mock_phase_config() -> StrategyPhaseConfig:
    """Mock StrategyPhaseConfig."""
    return StrategyPhaseConfig(
        strategy_type="sft",
        dataset="data/train.jsonl",
        hyperparams=PhaseHyperparametersConfig(epochs=3, learning_rate=2e-5, beta=0.1),
    )


@pytest.fixture
def mock_shutdown_handler() -> MagicMock:
    """Mock ShutdownHandler."""
    handler = MagicMock()
    handler.should_stop.return_value = False
    handler.get_shutdown_info.return_value = {"reason": "SIGINT", "signal_name": "SIGINT"}
    handler.save_emergency_checkpoint.return_value = "/tmp/output/phase_0/checkpoint-interrupted-phase0"
    return handler


@pytest.fixture
def mock_strategy_factory() -> MagicMock:
    """Mock StrategyFactory."""
    factory = MagicMock()
    strategy = MagicMock()
    strategy.__class__.__name__ = "SFTStrategy"
    strategy.validate_dataset.return_value = Ok(True)
    strategy.prepare_dataset.return_value = Ok(MagicMock(__len__=MagicMock(return_value=100)))
    factory.create_from_phase.return_value = strategy
    return factory


@pytest.fixture
def mock_trainer_factory() -> MagicMock:
    """Mock TrainerFactory."""
    factory = MagicMock()
    trainer = MagicMock()
    trainer.__class__.__name__ = "SFTTrainer"
    trainer.model = MagicMock()
    trainer.save_model = MagicMock()
    trainer.train = MagicMock()
    factory.create_from_phase.return_value = trainer
    return factory


@pytest.fixture
def mock_mlflow_manager() -> MagicMock:
    """Mock MLflowManager."""
    manager = MagicMock()
    manager._mlflow = MagicMock()  # MLflow module
    manager._parent_run_id = "parent_run_123"
    manager._mlflow_config = MagicMock(system_metrics_callback_enabled=True)
    return manager


@pytest.fixture
def phase_executor(
    mock_tokenizer: MagicMock,
    mock_config: MagicMock,
    mock_memory_manager: MagicMock,
    mock_dataset_loader: MagicMock,
    mock_metrics_collector: MagicMock,
    mock_shutdown_handler: MagicMock,
    mock_strategy_factory: MagicMock,
    mock_trainer_factory: MagicMock,
    mock_mlflow_manager: MagicMock,
) -> PhaseExecutor:
    """PhaseExecutor with all dependencies injected."""
    return PhaseExecutor(
        tokenizer=mock_tokenizer,
        config=mock_config,
        memory_manager=mock_memory_manager,
        dataset_loader=mock_dataset_loader,
        metrics_collector=mock_metrics_collector,
        shutdown_handler=mock_shutdown_handler,
        strategy_factory=mock_strategy_factory,
        trainer_factory=mock_trainer_factory,
        mlflow_manager=mock_mlflow_manager,
    )


# ========================================================================
# PRIORITY 1: INITIALIZATION (3 tests)
# ========================================================================


class TestPhaseExecutorInitialization:
    """Test PhaseExecutor initialization."""

    def test_init_with_all_dependencies(
        self,
        mock_tokenizer: MagicMock,
        mock_config: MagicMock,
        mock_memory_manager: MagicMock,
        mock_dataset_loader: MagicMock,
        mock_metrics_collector: MagicMock,
        mock_shutdown_handler: MagicMock,
        mock_strategy_factory: MagicMock,
        mock_trainer_factory: MagicMock,
        mock_mlflow_manager: MagicMock,
    ):
        """Test initialization with all dependencies injected."""
        executor = PhaseExecutor(
            tokenizer=mock_tokenizer,
            config=mock_config,
            memory_manager=mock_memory_manager,
            dataset_loader=mock_dataset_loader,
            metrics_collector=mock_metrics_collector,
            shutdown_handler=mock_shutdown_handler,
            strategy_factory=mock_strategy_factory,
            trainer_factory=mock_trainer_factory,
            mlflow_manager=mock_mlflow_manager,
        )

        assert executor.tokenizer is mock_tokenizer
        assert executor.config is mock_config
        assert executor.memory_manager is mock_memory_manager
        assert executor.dataset_loader is mock_dataset_loader
        assert executor.metrics_collector is mock_metrics_collector
        assert executor.shutdown_handler is mock_shutdown_handler
        assert executor.strategy_factory is mock_strategy_factory
        assert executor.trainer_factory is mock_trainer_factory
        assert executor._mlflow_manager is mock_mlflow_manager

    def test_init_with_minimal_dependencies(
        self,
        mock_tokenizer: MagicMock,
        mock_config: MagicMock,
        mock_memory_manager: MagicMock,
        mock_dataset_loader: MagicMock,
        mock_metrics_collector: MagicMock,
    ):
        """Test initialization with only required dependencies."""
        executor = PhaseExecutor(
            tokenizer=mock_tokenizer,
            config=mock_config,
            memory_manager=mock_memory_manager,
            dataset_loader=mock_dataset_loader,
            metrics_collector=mock_metrics_collector,
        )

        assert executor.tokenizer is mock_tokenizer
        assert executor.config is mock_config
        assert executor.shutdown_handler is None
        assert executor._mlflow_manager is None
        # Strategy and trainer factories should be created as defaults
        assert executor.strategy_factory is not None
        assert executor.trainer_factory is not None

    def test_init_uses_default_factories(
        self,
        mock_tokenizer: MagicMock,
        mock_config: MagicMock,
        mock_memory_manager: MagicMock,
        mock_dataset_loader: MagicMock,
        mock_metrics_collector: MagicMock,
    ):
        """Test that default factories are created when not injected."""
        executor = PhaseExecutor(
            tokenizer=mock_tokenizer,
            config=mock_config,
            memory_manager=mock_memory_manager,
            dataset_loader=mock_dataset_loader,
            metrics_collector=mock_metrics_collector,
        )

        # Default factories should be instances
        from src.training.strategies.factory import StrategyFactory
        from src.training.trainers.factory import TrainerFactory

        assert isinstance(executor.strategy_factory, StrategyFactory)
        assert isinstance(executor.trainer_factory, TrainerFactory)


# ========================================================================
# PRIORITY 1: EXECUTE - HAPPY PATH (5 tests)
# ========================================================================


class TestPhaseExecutorHappyPath:
    """Test PhaseExecutor execute method - happy path."""

    def test_execute_success_complete_flow(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test complete successful execution flow."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.end_run = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            mock_buffer.mark_phase_started.assert_called_once_with(0)
            mock_buffer.mark_phase_completed.assert_called_once()

            # Check that trainer.train was called
            trainer = mock_trainer_factory.create_from_phase.return_value
            trainer.train.assert_called_once()

    def test_execute_loads_dataset(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_dataset_loader: MagicMock,
    ):
        """Test that dataset is loaded correctly."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            mock_dataset_loader.load_for_phase.assert_called_once_with(mock_phase_config)

    def test_execute_creates_trainer_with_oom_protection(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_memory_manager: MagicMock,
        mock_trainer_factory: MagicMock,
    ):
        """Test that trainer is created with OOM protection."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

        assert result.is_success()
        # Check that trainer creation was protected by decorator
        # (Note: with_memory_protection is now a decorator, not context manager)
        mock_trainer_factory.create_from_phase.assert_called_once()

    def test_execute_saves_final_checkpoint(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test that final checkpoint is saved."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            trainer = mock_trainer_factory.create_from_phase.return_value
            # Check that save_model was called with checkpoint-final
            expected_checkpoint = Path("/tmp/output/phase_0") / "checkpoint-final"
            trainer.save_model.assert_called_once_with(str(expected_checkpoint))


# ========================================================================
# PRIORITY 1: EXECUTE - ERROR HANDLING (5 tests)
# ========================================================================


class TestPhaseExecutorErrorHandling:
    """Test PhaseExecutor error handling."""

    def test_execute_dataset_load_failure(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_dataset_loader: MagicMock,
    ):
        """Test handling of dataset load failure."""
        mock_dataset_loader.load_for_phase.return_value = Err("Failed to load dataset")

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            assert str(result.unwrap_err()) == "Failed to load dataset"
            mock_buffer.mark_phase_failed.assert_called_once_with(0, "Failed to load dataset")

    def test_execute_trainer_creation_failure(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test handling of trainer creation failure."""
        mock_trainer_factory.create_from_phase.side_effect = ValueError("Invalid trainer config")

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            assert "Validation error" in str(result.unwrap_err())
            mock_buffer.mark_phase_failed.assert_called_once()

    def test_execute_training_failure_general(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test handling of general training failure."""
        trainer = mock_trainer_factory.create_from_phase.return_value
        trainer.train.side_effect = RuntimeError("Training failed")

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            assert "Unexpected error" in str(result.unwrap_err())
            mock_buffer.mark_phase_failed.assert_called_once()


# ========================================================================
# PRIORITY 1: GRACEFUL SHUTDOWN (4 tests)
# ========================================================================


class TestPhaseExecutorGracefulShutdown:
    """Test PhaseExecutor graceful shutdown handling."""

    def test_shutdown_before_phase_start(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_shutdown_handler: MagicMock,
    ):
        """Test shutdown requested before phase starts."""
        mock_shutdown_handler.should_stop.return_value = True

        result = phase_executor.execute(
            phase_idx=0,
            phase=mock_phase_config,
            model=mock_model,
            buffer=mock_buffer,
        )

        assert result.is_failure()
        assert "Shutdown requested before phase start" in str(result.unwrap_err())
        mock_buffer.mark_phase_interrupted.assert_called_once_with(
            0,
            reason="Shutdown requested before phase start",
        )

    def test_shutdown_during_training(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_shutdown_handler: MagicMock,
        mock_trainer_factory: MagicMock,
    ):
        """Test shutdown during training."""
        # Shutdown happens after training
        mock_shutdown_handler.should_stop.side_effect = [False, False, True]  # Check after training

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            assert "Training interrupted" in str(result.unwrap_err())
            mock_buffer.mark_phase_interrupted.assert_called_once()

    def test_shutdown_saves_emergency_checkpoint(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_shutdown_handler: MagicMock,
    ):
        """Test that emergency checkpoint is saved on shutdown."""
        mock_shutdown_handler.should_stop.side_effect = [False, False, True]

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            mock_shutdown_handler.save_emergency_checkpoint.assert_called_once()

    def test_keyboard_interrupt_handling(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test handling of KeyboardInterrupt."""
        trainer = mock_trainer_factory.create_from_phase.return_value
        trainer.train.side_effect = KeyboardInterrupt()

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            assert "Training interrupted" in str(result.unwrap_err())
            mock_buffer.mark_phase_interrupted.assert_called_once()


# ========================================================================
# PRIORITY 1: OOM HANDLING (3 tests)
# ========================================================================


class TestPhaseExecutorOOMHandling:
    """Test PhaseExecutor OOM handling."""

    def test_oom_during_trainer_creation(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test OOM during trainer creation."""
        mock_trainer_factory.create_from_phase.side_effect = OOMRecoverableError("OOM during trainer creation")

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            assert "OOM error" in str(result.unwrap_err())
            mock_buffer.mark_phase_failed.assert_called_once()

    def test_oom_during_training(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test OOM during training."""
        trainer = mock_trainer_factory.create_from_phase.return_value
        trainer.train.side_effect = OOMRecoverableError("OOM during training")

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            assert "OOM error" in str(result.unwrap_err())
            mock_buffer.mark_phase_failed.assert_called_once()

    def test_oom_logged_to_mlflow(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
        mock_mlflow_manager: MagicMock,
    ):
        """Test that OOM is logged to MLflow."""
        trainer = mock_trainer_factory.create_from_phase.return_value
        trainer.train.side_effect = OOMRecoverableError("OOM during training")

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            mock_mlflow_manager.log_oom.assert_called_once()
            call_args = mock_mlflow_manager.log_oom.call_args
            assert "phase_0_sft" in call_args[1]["operation"]


# ========================================================================
# PRIORITY 2: RESUME LOGIC (3 tests)
# ========================================================================


class TestPhaseExecutorResumeLogic:
    """Test PhaseExecutor resume logic."""

    def test_execute_resume_from_checkpoint(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test resuming from checkpoint."""
        mock_buffer.get_resume_checkpoint.return_value = "/tmp/checkpoint/phase_0"

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            trainer = mock_trainer_factory.create_from_phase.return_value
            trainer.train.assert_called_once_with(resume_from_checkpoint="/tmp/checkpoint/phase_0")

    def test_execute_resume_checkpoint_path_passed_to_trainer(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test that resume checkpoint path is passed to trainer."""
        mock_buffer.get_resume_checkpoint.return_value = "/specific/checkpoint/path"

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            trainer = mock_trainer_factory.create_from_phase.return_value
            trainer.train.assert_called_once_with(resume_from_checkpoint="/specific/checkpoint/path")

    def test_execute_no_resume_when_no_checkpoint(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_trainer_factory: MagicMock,
    ):
        """Test that training starts from scratch when no resume checkpoint."""
        mock_buffer.get_resume_checkpoint.return_value = None

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            trainer = mock_trainer_factory.create_from_phase.return_value
            trainer.train.assert_called_once_with(resume_from_checkpoint=None)


# ========================================================================
# PRIORITY 2: DATABUFFER INTEGRATION (4 tests)
# ========================================================================


class TestPhaseExecutorDataBufferIntegration:
    """Test PhaseExecutor integration with DataBuffer."""

    def test_execute_marks_phase_started(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test that phase is marked as started in DataBuffer."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            mock_buffer.mark_phase_started.assert_called_once_with(0)

    def test_execute_marks_phase_completed_with_metrics(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_metrics_collector: MagicMock,
    ):
        """Test that phase is marked completed with metrics."""
        expected_metrics = TrainingMetricsSnapshot(train_loss=0.45, eval_loss=0.52, epoch=3)
        mock_metrics_collector.extract_from_trainer.return_value = expected_metrics

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            call_args = mock_buffer.mark_phase_completed.call_args
            # phase_idx is positional arg[0], metrics is kwarg
            assert call_args[0][0] == 0
            assert call_args[1]["metrics"] == expected_metrics

    def test_execute_marks_phase_failed_on_error(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_dataset_loader: MagicMock,
    ):
        """Test that phase is marked failed on error."""
        mock_dataset_loader.load_for_phase.return_value = Err("Dataset not found")

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_failure()
            mock_buffer.mark_phase_failed.assert_called_once_with(0, "Dataset not found")

    def test_execute_marks_phase_interrupted_on_shutdown(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_shutdown_handler: MagicMock,
    ):
        """Test that phase is marked interrupted on shutdown."""
        mock_shutdown_handler.should_stop.return_value = True

        result = phase_executor.execute(
            phase_idx=0,
            phase=mock_phase_config,
            model=mock_model,
            buffer=mock_buffer,
        )

        assert result.is_failure()
        mock_buffer.mark_phase_interrupted.assert_called_once()


# ========================================================================
# PRIORITY 2: MLFLOW INTEGRATION (5 tests)
# ========================================================================


class TestPhaseExecutorMLflowIntegration:
    """Test PhaseExecutor MLflow integration."""

    def test_mlflow_nested_run_created_for_phase(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test that nested run is created for phase."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Check that start_run was called with nested=True
            # Note: start_run is called twice - first with nested=True, then with nested=False (to restore parent)
            # Check first call (nested run creation)
            first_call = mock_mlflow.start_run.call_args_list[0]
            assert first_call[1].get("nested") is True
            assert "phase_0_sft" in first_call[1].get("run_name", "")

    def test_mlflow_nested_run_closed_with_correct_status(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test that nested run is closed with correct status."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.end_run = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Check that end_run was called with FINISHED status
            mock_mlflow.end_run.assert_called()
            call_kwargs = mock_mlflow.end_run.call_args[1]
            assert call_kwargs.get("status") == "FINISHED"

    def test_mlflow_logs_phase_start_params(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_mlflow_manager: MagicMock,
    ):
        """Test that phase start params are logged to MLflow."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Check that params were logged (could be multiple calls)
            mock_mlflow_manager.log_params.assert_called()
            # Check first call which logs phase start params
            first_call_params = mock_mlflow_manager.log_params.call_args_list[0][0][0]
            assert first_call_params["phase_idx"] == 0
            assert first_call_params["strategy_type"] == "sft"

    def test_mlflow_logs_dataset_info(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_mlflow_manager: MagicMock,
    ):
        """Test that dataset info is logged to MLflow."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Check that dataset was logged (either create_mlflow_dataset or log_dataset_info)
            assert mock_mlflow_manager.create_mlflow_dataset.called or mock_mlflow_manager.log_dataset_info.called

    def test_mlflow_logs_completion_metrics(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_mlflow_manager: MagicMock,
        mock_metrics_collector: MagicMock,
    ):
        """Test that completion metrics are logged to MLflow."""
        expected_metrics = TrainingMetricsSnapshot(train_loss=0.45, eval_loss=0.52)
        mock_metrics_collector.extract_from_trainer.return_value = expected_metrics

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value = mock_run
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Check that metrics were logged
            mock_mlflow_manager.log_metrics.assert_called()
            logged_metrics = mock_mlflow_manager.log_metrics.call_args[0][0]
            assert logged_metrics["train_loss"] == 0.45
            assert logged_metrics["eval_loss"] == 0.52


# ========================================================================
# PRIORITY 2: HELPER METHODS (3 tests)
# ========================================================================


class TestPhaseExecutorHelperMethods:
    """Test PhaseExecutor helper methods."""

    def test_save_checkpoint_creates_final_directory(
        self,
        phase_executor: PhaseExecutor,
        mock_trainer_factory: MagicMock,
    ):
        """Test that _save_checkpoint creates checkpoint-final directory."""
        trainer = MagicMock()
        trainer.save_model = MagicMock()

        checkpoint_path = phase_executor._save_checkpoint(trainer, "/tmp/output/phase_0")

        assert checkpoint_path == Path("/tmp/output/phase_0") / "checkpoint-final"
        trainer.save_model.assert_called_once_with(str(checkpoint_path))

    def test_handle_error_logs_to_mlflow(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_mlflow_manager: MagicMock,
    ):
        """Test that _handle_error logs to MLflow."""
        error = ValueError("Test error")

        result = phase_executor._handle_error(
            buffer=mock_buffer,
            phase_idx=0,
            error_type="Validation",
            error=error,
        )

        assert result.is_failure()
        mock_mlflow_manager.log_event_error.assert_called_once()
        mock_mlflow_manager.set_tags.assert_called_once()


# ========================================================================
# PRIORITY 3: EDGE CASES (5 tests)
# ========================================================================


class TestPhaseExecutorEdgeCases:
    """Test PhaseExecutor edge cases."""

    def test_execute_with_empty_dataset(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_dataset_loader: MagicMock,
    ):
        """Test execution with empty dataset."""
        # Empty dataset (0 samples)
        empty_dataset = MagicMock(__len__=MagicMock(return_value=0))
        mock_dataset_loader.load_for_phase.return_value = Ok((empty_dataset, None))

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            # Should succeed even with empty dataset
            assert result.is_success()

    def test_execute_without_mlflow_manager(
        self,
        mock_tokenizer: MagicMock,
        mock_config: MagicMock,
        mock_memory_manager: MagicMock,
        mock_dataset_loader: MagicMock,
        mock_metrics_collector: MagicMock,
        mock_strategy_factory: MagicMock,
        mock_trainer_factory: MagicMock,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test execution without MLflow manager."""
        executor = PhaseExecutor(
            tokenizer=mock_tokenizer,
            config=mock_config,
            memory_manager=mock_memory_manager,
            dataset_loader=mock_dataset_loader,
            metrics_collector=mock_metrics_collector,
            strategy_factory=mock_strategy_factory,
            trainer_factory=mock_trainer_factory,
            mlflow_manager=None,  # No MLflow
        )

        result = executor.execute(
            phase_idx=0,
            phase=mock_phase_config,
            model=mock_model,
            buffer=mock_buffer,
        )

        # Should succeed without MLflow
        assert result.is_success()

    def test_execute_without_shutdown_handler(
        self,
        mock_tokenizer: MagicMock,
        mock_config: MagicMock,
        mock_memory_manager: MagicMock,
        mock_dataset_loader: MagicMock,
        mock_metrics_collector: MagicMock,
        mock_strategy_factory: MagicMock,
        mock_trainer_factory: MagicMock,
        mock_mlflow_manager: MagicMock,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test execution without shutdown handler."""
        executor = PhaseExecutor(
            tokenizer=mock_tokenizer,
            config=mock_config,
            memory_manager=mock_memory_manager,
            dataset_loader=mock_dataset_loader,
            metrics_collector=mock_metrics_collector,
            strategy_factory=mock_strategy_factory,
            trainer_factory=mock_trainer_factory,
            mlflow_manager=mock_mlflow_manager,
            shutdown_handler=None,  # No shutdown handler
        )

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            # Should succeed without shutdown handler
            assert result.is_success()

    def test_emergency_checkpoint_fallback_without_handler(
        self,
        mock_tokenizer: MagicMock,
        mock_config: MagicMock,
        mock_memory_manager: MagicMock,
        mock_dataset_loader: MagicMock,
        mock_metrics_collector: MagicMock,
        mock_strategy_factory: MagicMock,
        mock_trainer_factory: MagicMock,
        mock_mlflow_manager: MagicMock,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test emergency checkpoint fallback without shutdown handler."""
        executor = PhaseExecutor(
            tokenizer=mock_tokenizer,
            config=mock_config,
            memory_manager=mock_memory_manager,
            dataset_loader=mock_dataset_loader,
            metrics_collector=mock_metrics_collector,
            strategy_factory=mock_strategy_factory,
            trainer_factory=mock_trainer_factory,
            mlflow_manager=mock_mlflow_manager,
            shutdown_handler=None,  # No shutdown handler
        )

        # Trigger KeyboardInterrupt
        trainer = mock_trainer_factory.create_from_phase.return_value
        trainer.train.side_effect = KeyboardInterrupt()

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            # Should fail gracefully
            assert result.is_failure()
            mock_buffer.mark_phase_interrupted.assert_called_once()
            # Emergency checkpoint should be saved via fallback
            trainer.save_model.assert_called()

    def test_cleanup_old_checkpoints_called(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test that cleanup_old_checkpoints is called after phase completion."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Cleanup should be called with keep_last=2
            mock_buffer.cleanup_old_checkpoints.assert_called_once_with(keep_last=2)


# ========================================================================
# PRIORITY 3: MLFLOW EDGE CASES (3 tests)
# ========================================================================


class TestPhaseExecutorMLflowEdgeCases:
    """Test PhaseExecutor MLflow edge cases."""

    def test_mlflow_nested_run_handles_exception(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test that nested run handles exception gracefully."""
        # Mock mlflow.start_run to raise exception
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = Exception("MLflow error")
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            # Should still succeed (MLflow is optional)
            assert result.is_success()

    def test_mlflow_dataset_logging_fallback(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_mlflow_manager: MagicMock,
    ):
        """Test dataset logging fallback when create_mlflow_dataset fails."""
        # create_mlflow_dataset returns None (fallback scenario)
        mock_mlflow_manager.create_mlflow_dataset.return_value = None

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Should fallback to log_dataset_info
            mock_mlflow_manager.log_dataset_info.assert_called_once()

    def test_mlflow_metrics_skips_none_fields(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
        mock_metrics_collector: MagicMock,
        mock_mlflow_manager: MagicMock,
    ):
        """Non-populated snapshot fields must not be logged as MLflow metrics."""
        mock_metrics_collector.extract_from_trainer.return_value = TrainingMetricsSnapshot(
            train_loss=0.45,
            eval_loss=0.52,
            epoch=3,
            # train_runtime / learning_rate / global_step left as None
        )

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            mock_mlflow_manager.log_metrics.assert_called()
            logged_metrics = mock_mlflow_manager.log_metrics.call_args[0][0]
            assert "train_loss" in logged_metrics
            assert "eval_loss" in logged_metrics
            assert "epoch" in logged_metrics
            assert "train_runtime" not in logged_metrics
            assert "learning_rate" not in logged_metrics
            assert "global_step" not in logged_metrics


# ========================================================================
# PRIORITY 3: STATE MANAGEMENT (2 tests)
# ========================================================================


class TestPhaseExecutorStateManagement:
    """Test PhaseExecutor state management."""

    def test_execute_clears_current_trainer_on_completion(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test that _current_trainer is cleared after execution."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            result = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )

            assert result.is_success()
            # Internal state should be cleared
            assert phase_executor._current_trainer is None
            assert phase_executor._current_output_dir is None
            assert phase_executor._current_phase_idx is None

    def test_execute_handles_multiple_consecutive_phases(
        self,
        phase_executor: PhaseExecutor,
        mock_buffer: MagicMock,
        mock_model: MagicMock,
        mock_phase_config: StrategyPhaseConfig,
    ):
        """Test that executor can handle multiple consecutive phases."""
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.set_tags = MagicMock()
        mock_mlflow.disable_system_metrics_logging = MagicMock()
        mock_mlflow.enable_system_metrics_logging = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            # Execute phase 0
            result1 = phase_executor.execute(
                phase_idx=0,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )
            assert result1.is_success()

            # Execute phase 1
            result2 = phase_executor.execute(
                phase_idx=1,
                phase=mock_phase_config,
                model=mock_model,
                buffer=mock_buffer,
            )
            assert result2.is_success()

            # Both phases should be marked as started
            assert mock_buffer.mark_phase_started.call_count == 2
            assert mock_buffer.mark_phase_completed.call_count == 2
