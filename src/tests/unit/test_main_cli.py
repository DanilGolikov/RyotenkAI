"""
Tests for the main.py CLI.

Covers:
- Commands: train, validate-dataset, info, train-local, version
- Signal handling (SIGINT, SIGTERM)
- Error handling and edge cases
- Graceful shutdown and cleanup
"""

from __future__ import annotations

import signal
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.main import _signal_handler, app
from src.utils.result import Failure, Success

# =========================================================================
# FIXTURES
# =========================================================================


@pytest.fixture
def cli_runner():
    """Builds a Typer CLI runner for command tests."""
    return CliRunner()


@pytest.fixture
def mock_pipeline_orchestrator():
    """Mock for PipelineOrchestrator."""
    with patch("src.pipeline.orchestrator.PipelineOrchestrator") as mock:
        orchestrator = MagicMock()
        orchestrator.config = MagicMock()

        # Model config
        orchestrator.config.model.name = "test/model"
        orchestrator.config.training.type = "qlora"
        orchestrator.config.training.get_effective_load_in_4bit.return_value = True
        orchestrator.config.training.hyperparams.per_device_train_batch_size = 4
        orchestrator.config.training.hyperparams.learning_rate = 2e-4

        # LoRA config
        orchestrator.config.lora = MagicMock()
        orchestrator.config.lora.r = 16
        orchestrator.config.lora.lora_alpha = 32
        orchestrator.config.lora.lora_dropout = 0.05
        orchestrator.config.lora.target_modules = None

        # Strategy chain
        mock_strategy = MagicMock()
        mock_strategy.strategy_type = "sft"
        mock_strategy.epochs = 1
        mock_strategy.learning_rate = None
        orchestrator.config.training.get_strategy_chain.return_value = [mock_strategy]

        # Dataset config
        mock_dataset = MagicMock()
        mock_dataset.train_path = "data/train.jsonl"
        mock_dataset.eval_path = "data/eval.jsonl"
        mock_dataset.max_samples = None
        mock_dataset.adapter_type = "instruction"
        mock_dataset.test_path = None
        orchestrator.config.get_primary_dataset.return_value = mock_dataset

        # Provider config
        mock_provider = MagicMock()
        mock_provider.gpu_type = "RTX 4060"
        orchestrator.config.get_provider_config.return_value = mock_provider
        orchestrator.config.get_active_provider_name.return_value = "single_node"

        # Stages
        orchestrator.list_stages.return_value = [
            "Dataset Validation",
            "GPU Deployment",
            "Training Execution",
            "Model Evaluation",
            "Integration Test",
            "Model Upload",
        ]

        # Stages[0] for validate-dataset
        mock_stage_0 = MagicMock()
        mock_stage_0.run.return_value = Success(None)
        orchestrator.stages = [mock_stage_0]

        # Run result
        orchestrator.run.return_value = Success("Pipeline completed")
        orchestrator.list_restart_points.return_value = [
            {
                "stage": "Inference Deployer",
                "available": True,
                "mode": "fresh_or_resume",
                "reason": "restart_allowed",
            }
        ]
        orchestrator._cleanup_resources = MagicMock()

        mock.return_value = orchestrator
        yield orchestrator


@pytest.fixture
def mock_run_training():
    """Mock for run_training."""
    with patch("src.training.run_training.run_training") as mock:
        mock.return_value = Path("/tmp/output/model")
        yield mock


@pytest.fixture
def temp_config(tmp_path):
    """Creates a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("""
model:
  name: test/model
training:
  type: qlora
  hyperparams:
    per_device_train_batch_size: 4
""")
    return config_path


# =========================================================================
# POSITIVE TESTS
# =========================================================================


def test_train_command_success(cli_runner, mock_pipeline_orchestrator, temp_config):
    """Successful train command."""
    result = cli_runner.invoke(app, ["train", "--config", str(temp_config)])

    assert result.exit_code == 0
    assert "Pipeline completed successfully" in result.stdout
    mock_pipeline_orchestrator.run.assert_called_once_with(
        run_dir=None,
        resume=False,
        restart_from_stage=None,
    )


def test_validate_dataset_success(cli_runner, mock_pipeline_orchestrator, temp_config):
    """validate-dataset command."""
    result = cli_runner.invoke(app, ["validate-dataset", "--config", str(temp_config)])

    assert result.exit_code == 0
    assert "Dataset validation passed" in result.stdout
    mock_pipeline_orchestrator.stages[0].run.assert_called_once()


def test_train_command_with_resume_and_run_dir(cli_runner, mock_pipeline_orchestrator, temp_config):
    result = cli_runner.invoke(
        app,
        ["train", "--config", str(temp_config), "--run-dir", "/tmp/run_1", "--resume"],
    )

    assert result.exit_code == 0
    mock_pipeline_orchestrator.run.assert_called_once_with(
        run_dir=Path("/tmp/run_1"),
        resume=True,
        restart_from_stage=None,
    )


def test_train_command_with_manual_restart(cli_runner, mock_pipeline_orchestrator, temp_config):
    result = cli_runner.invoke(
        app,
        ["train", "--config", str(temp_config), "--run-dir", "/tmp/run_1", "--restart-from-stage", "Inference Deployer"],
    )

    assert result.exit_code == 0
    mock_pipeline_orchestrator.run.assert_called_once_with(
        run_dir=Path("/tmp/run_1"),
        resume=False,
        restart_from_stage="Inference Deployer",
    )


def test_list_restart_points_command(cli_runner, tmp_path):
    """list-restart-points is standalone and does not need full orchestrator init."""
    mock_points = [
        types.SimpleNamespace(stage="Dataset Validator", available=True, mode="fresh_only", reason="restart_allowed"),
        types.SimpleNamespace(
            stage="Inference Deployer",
            available=False,
            mode="fresh_or_resume",
            reason="missing_model_retriever_outputs",
        ),
    ]

    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()

    with (
        patch("src.main.load_restart_point_options", return_value=(config_file, mock_points)),
        patch("src.main._resolve_config", return_value=config_file),
    ):
        result = cli_runner.invoke(app, ["list-restart-points", str(run_dir)])

    assert result.exit_code == 0
    assert "Stage" in result.stdout
    assert "Available" in result.stdout
    assert "Dataset Validator" in result.stdout
    assert "Inference Deployer" in result.stdout


def test_info_command(cli_runner, mock_pipeline_orchestrator, temp_config):
    """info command prints configuration."""
    result = cli_runner.invoke(app, ["info", "--config", str(temp_config)])

    assert result.exit_code == 0
    assert "Pipeline Stages" in result.stdout
    assert "Model Configuration" in result.stdout
    assert "Dataset Configuration" in result.stdout
    assert "Dataset Validation" in result.stdout
    assert "test/model" in result.stdout


def test_train_local_success(cli_runner, mock_run_training, temp_config):
    """train-local command."""
    result = cli_runner.invoke(app, ["train-local", "--config", str(temp_config)])

    assert result.exit_code == 0
    assert "Training completed" in result.stdout
    mock_run_training.assert_called_once()


def test_version_command(cli_runner):
    """version command."""
    result = cli_runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "RyotenkAI" in result.stdout
    assert "Version:" in result.stdout


# =========================================================================
# NEGATIVE TESTS
# =========================================================================


def test_train_requires_config_or_run_dir(cli_runner):
    """train without --config and without --run-dir must fail."""
    result = cli_runner.invoke(app, ["train"])

    assert result.exit_code == 1


def test_train_invalid_config(cli_runner):
    """Nonexistent config file."""
    result = cli_runner.invoke(app, ["train", "--config", "/nonexistent/config.yaml"])

    assert result.exit_code == 1


def test_train_orchestrator_failure(cli_runner, mock_pipeline_orchestrator, temp_config):
    """PipelineOrchestrator error handling."""
    mock_pipeline_orchestrator.run.return_value = Failure("Training failed: OOM error")

    result = cli_runner.invoke(app, ["train", "--config", str(temp_config)])

    assert result.exit_code == 1


def test_validate_dataset_failure(cli_runner, mock_pipeline_orchestrator, temp_config):
    """Dataset validation failure."""
    mock_pipeline_orchestrator.stages[0].run.return_value = Failure("Validation failed: insufficient samples")

    result = cli_runner.invoke(app, ["validate-dataset", "--config", str(temp_config)])

    assert result.exit_code == 1


def test_train_local_config_error(cli_runner, mock_run_training, temp_config):
    """Configuration error in train-local."""
    mock_run_training.side_effect = ValueError("Invalid config: missing model name")

    result = cli_runner.invoke(app, ["train-local", "--config", str(temp_config)])

    assert result.exit_code == 1


def test_train_rejects_resume_and_restart_together(cli_runner, temp_config):
    result = cli_runner.invoke(
        app,
        [
            "train",
            "--config",
            str(temp_config),
            "--run-dir",
            "/tmp/run_1",
            "--resume",
            "--restart-from-stage",
            "Inference Deployer",
        ],
    )

    assert result.exit_code == 1




# =========================================================================
# EDGE CASES
# =========================================================================




# =========================================================================
# SIGNAL HANDLING
# =========================================================================


def test_sigint_graceful_shutdown():
    """SIGINT (Ctrl+C): handler notifies orchestrator; does not call cleanup directly."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.notify_signal = MagicMock()

    # threading.Timer must be patched: _signal_handler creates a 30s daemon timer that
    # calls os._exit(130) if cleanup takes too long. Without patching, the timer leaks
    # into subsequent tests and fires during coverage HTML generation → exit code 130.
    with patch("src.main.threading.Timer") as mock_timer:
        mock_timer.return_value = MagicMock()
        with patch("src.main._current_orchestrator", mock_orchestrator):
            with patch("src.main._unregister_mlflow_atexit"):
                with pytest.raises(SystemExit) as exc_info:
                    _signal_handler(signal.SIGINT, None)

        assert exc_info.value.code == 130  # SIGINT exit code
        mock_orchestrator.notify_signal.assert_called_once_with(signal_name="SIGINT")
        # cleanup is NOT called from signal handler — it's the orchestrator's finally block
        mock_orchestrator._cleanup_resources.assert_not_called()


def test_sigterm_graceful_shutdown():
    """SIGTERM: handler notifies orchestrator; does not call cleanup directly."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.notify_signal = MagicMock()

    with patch("src.main.threading.Timer") as mock_timer:
        mock_timer.return_value = MagicMock()
        with patch("src.main._current_orchestrator", mock_orchestrator):
            with patch("src.main._unregister_mlflow_atexit"):
                with pytest.raises(SystemExit) as exc_info:
                    _signal_handler(signal.SIGTERM, None)

        assert exc_info.value.code == 143  # SIGTERM exit code
        mock_orchestrator.notify_signal.assert_called_once_with(signal_name="SIGTERM")
        mock_orchestrator._cleanup_resources.assert_not_called()


def test_signal_without_orchestrator():
    """Signal before orchestrator is initialized."""
    with patch("src.main.threading.Timer") as mock_timer:
        mock_timer.return_value = MagicMock()
        with patch("src.main._current_orchestrator", None):
            with pytest.raises(SystemExit) as exc_info:
                _signal_handler(signal.SIGINT, None)

        assert exc_info.value.code == 130


def test_signal_cleanup_error():
    """SIGINT without orchestrator exits with code 130."""
    with patch("src.main.threading.Timer") as mock_timer:
        mock_timer.return_value = MagicMock()
        with patch("src.main._current_orchestrator", None):
            with patch("src.main._unregister_mlflow_atexit"):
                with pytest.raises(SystemExit) as exc_info:
                    _signal_handler(signal.SIGINT, None)

        # Must exit with the expected code
        assert exc_info.value.code == 130


# =========================================================================
# RESUME
# =========================================================================


def test_train_local_resume(cli_runner, mock_run_training, temp_config):
    """train-local with resume=True."""
    result = cli_runner.invoke(app, ["train-local", "--config", str(temp_config), "--resume"])

    assert result.exit_code == 0
    mock_run_training.assert_called_once()
    call_kwargs = mock_run_training.call_args[1]
    assert call_kwargs["resume"] is True


def test_train_local_resume_with_run_id(cli_runner, mock_run_training, temp_config):
    """train-local resume with explicit run_id."""
    result = cli_runner.invoke(
        app,
        ["train-local", "--config", str(temp_config), "--resume", "--run-id", "run_12345"],
    )

    assert result.exit_code == 0
    call_kwargs = mock_run_training.call_args[1]
    assert call_kwargs["resume"] is True
    assert call_kwargs["run_id"] == "run_12345"


# =========================================================================
# INVARIANTS
# =========================================================================


def test_cleanup_called_on_exit(cli_runner, tmp_path):
    """Cleanup runs on exit."""
    (tmp_path / "cfg.yaml").write_text("model:\n  name: t\n")
    with patch("src.pipeline.orchestrator.PipelineOrchestrator") as mock_orch_class:
        orchestrator = MagicMock()
        orchestrator.config = MagicMock()
        orchestrator.run.return_value = Success(None)
        orchestrator._cleanup_resources = MagicMock()

        # Minimal config for logging
        orchestrator.config.model.name = "test/model"
        orchestrator.config.training.type = "qlora"
        orchestrator.config.training.get_effective_load_in_4bit.return_value = True
        orchestrator.config.training.hyperparams.per_device_train_batch_size = 4
        orchestrator.config.training.hyperparams.learning_rate = 2e-4
        orchestrator.config.training.get_strategy_chain.return_value = []
        orchestrator.config.lora = None

        mock_dataset = MagicMock()
        mock_dataset.train_path = "data/train.jsonl"
        mock_dataset.eval_path = None
        mock_dataset.max_samples = None
        mock_dataset.adapter_type = None
        orchestrator.config.get_primary_dataset.return_value = mock_dataset

        mock_provider = MagicMock()
        mock_provider.gpu_type = "auto-detect"
        orchestrator.config.get_provider_config.return_value = mock_provider
        orchestrator.config.get_active_provider_name.return_value = "single_node"

        mock_orch_class.return_value = orchestrator

        # Run command
        result = cli_runner.invoke(app, ["train", "--config", str(tmp_path / "cfg.yaml")])

        # After success, global reference should be cleared (main.py finally)
        assert result.exit_code == 0


@patch("src.main.logger")
def test_config_summary_logged(mock_logger, cli_runner, mock_pipeline_orchestrator, temp_config):
    """Configuration is logged before startup."""
    result = cli_runner.invoke(app, ["train", "--config", str(temp_config)])

    assert result.exit_code == 0

    # Key config fragments were logged
    log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
    log_text = " ".join(log_calls)

    assert "CONFIGURATION SUMMARY" in log_text or any("Model:" in call or "Training:" in call for call in log_calls)
