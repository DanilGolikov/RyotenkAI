"""
Run Training Script Tests.

Tests for run_training.py which runs on GPU servers.

Tests cover:
- Configuration loading and validation
- CLI entry point
- Backward compatibility with train_v2 alias
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pydantic
import pytest
import yaml

from src.utils.result import Err, Ok

# =============================================================================
# TEST CLASS: Configuration Loading
# =============================================================================


class TestConfigurationLoading:
    """Tests for configuration loading in run_training."""

    def test_nonexistent_config_raises_error(self):
        """Loading nonexistent config should raise FileNotFoundError."""
        from src.training.run_training import run_training

        with pytest.raises(FileNotFoundError):
            run_training("/nonexistent/path/config.yaml")

    def test_invalid_yaml_syntax_raises_error(self, tmp_path):
        """Invalid YAML syntax should raise error."""
        from src.training.run_training import run_training

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("model: {name: [unclosed")

        with pytest.raises((yaml.YAMLError, SystemExit)):
            run_training(str(config_file))

    def test_config_missing_required_field_raises_error(self, tmp_path):
        """Config missing required field should raise ValidationError."""
        from src.training.run_training import run_training

        config_file = tmp_path / "incomplete.yaml"
        config_file.write_text("""
training:
  type: qlora
""")

        with pytest.raises((pydantic.ValidationError, SystemExit)):
            run_training(str(config_file))


# =============================================================================
# TEST CLASS: CLI Entry Point
# =============================================================================


class TestCLIEntryPoint:
    """Tests for CLI main() function."""

    @patch("src.training.run_training.run_training")
    def test_main_returns_0_on_success(self, mock_run_training):
        """main() should return 0 on success."""
        import sys

        from src.training.run_training import main

        mock_run_training.return_value = Path("/tmp/output")

        with patch.object(sys, "argv", ["run_training", "--config", "test.yaml"]):
            result = main()

        assert result == 0

    @patch("src.training.run_training.run_training")
    def test_main_returns_1_on_error(self, mock_run_training):
        """main() should return 1 on error."""
        import sys

        from src.training.run_training import main

        mock_run_training.side_effect = RuntimeError("Test error")

        with patch.object(sys, "argv", ["run_training", "--config", "test.yaml"]):
            result = main()

        assert result == 1

    @patch("src.training.run_training.run_training")
    def test_main_returns_130_on_keyboard_interrupt(self, mock_run_training):
        """main() should return 130 on KeyboardInterrupt."""
        import sys

        from src.training.run_training import main

        mock_run_training.side_effect = KeyboardInterrupt()

        with patch.object(sys, "argv", ["run_training", "--config", "test.yaml"]):
            result = main()

        assert result == 130

    @patch("src.training.run_training.run_training")
    def test_main_passes_resume_flag(self, mock_run_training):
        """main() should pass --resume flag to run_training."""
        import sys

        from src.training.run_training import main

        mock_run_training.return_value = Path("/tmp/output")

        with patch.object(sys, "argv", ["run_training", "--config", "test.yaml", "--resume"]):
            main()

        mock_run_training.assert_called_once()
        call_kwargs = mock_run_training.call_args[1]
        assert call_kwargs["resume"] is True

    @patch("src.training.run_training.run_training")
    def test_main_passes_run_id(self, mock_run_training):
        """main() should pass --run-id to run_training."""
        import sys

        from src.training.run_training import main

        mock_run_training.return_value = Path("/tmp/output")

        with patch.object(sys, "argv", ["run_training", "--config", "test.yaml", "--run-id", "my_run_123"]):
            main()

        call_kwargs = mock_run_training.call_args[1]
        assert call_kwargs["run_id"] == "my_run_123"


# =============================================================================
# TEST CLASS: Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with train_v2."""

    def test_train_v2_alias_exists(self):
        """train_v2 should be an alias for run_training."""
        from src.training.run_training import run_training, train_v2

        assert train_v2 is run_training

    def test_import_train_v2_from_training_module(self):
        """train_v2 should be importable from src.training."""
        from src.training import train_v2

        assert train_v2 is not None


# =============================================================================
# TEST CLASS: Core run_training() flow (DI + MLflow + notifier)
# =============================================================================


class _Param:
    def __init__(self, n: int, requires_grad: bool):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        """Return number of parameters."""
        return self._n


class _ModelStub:
    def __init__(self):
        self._params = [_Param(10, True), _Param(90, False)]

    def parameters(self):
        return list(self._params)


class _RunCtx:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class TestRunTrainingFlow:
    def test_extract_model_size(self):
        from src.training.run_training import _extract_model_size

        assert _extract_model_size("Qwen/Qwen2.5-0.5B-Instruct") == "0.5B"
        assert _extract_model_size("meta-llama/Llama-3.2-7B") == "7B"
        assert _extract_model_size("no-size-here") == "unknown"

    def test_run_training_success_with_mlflow_and_container_di(self, monkeypatch, tmp_path):
        import importlib

        rt = importlib.import_module("src.training.run_training")

        # Fake config
        strategies = [MagicMock(strategy_type="sft"), MagicMock(strategy_type="dpo")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = True

        # MLflow stub
        mlflow_mgr = MagicMock()
        mlflow_mgr.is_active = True
        mlflow_mgr._mlflow_config = MagicMock(system_metrics_callback_enabled=True)
        run_ctx = _RunCtx()
        mlflow_mgr.start_run.return_value = run_ctx

        # Environment reporter stub
        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        # Memory manager stub
        gpu = MagicMock()
        gpu.name = "GPU"
        gpu.tier.value = "TIER"
        gpu.total_memory_gb = 8.0
        mem_stats = MagicMock(used_mb=100, free_mb=7000, total_mb=8000, utilization_percent=1.2)
        memory_manager = MagicMock()
        memory_manager.gpu_info = gpu
        memory_manager.preset = None
        memory_manager.get_memory_stats.return_value = mem_stats

        # Orchestrator stub
        buffer = MagicMock()
        buffer.get_phase_output_dir.return_value = str(tmp_path / "phase_last")
        buffer.run_id = "run_1"
        orchestrator = MagicMock()
        orchestrator.buffer = buffer
        orchestrator.run_chain.return_value = Ok(None)

        # Container stub (Phase 6.3b: notifier abstraction removed —
        # trainer events flow via RunnerEventCallback now).
        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.return_value = (_ModelStub(), object())
        container.create_orchestrator.return_value = orchestrator

        # Patch module functions
        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(rt, "_setup_mlflow", lambda c: mlflow_mgr)
        monkeypatch.setattr(rt.EnvironmentReporter, "collect", classmethod(lambda cls=None: env_reporter))
        monkeypatch.setenv("MLFLOW_PARENT_RUN_ID", "parent_123")
        monkeypatch.setenv("DOCKER_IMAGE_SHA", "deadbeef" * 8)

        out = rt.run_training(str(tmp_path / "cfg.yaml"), resume=False, run_id="run_x", container=container)

        assert out.name == "checkpoint-final"
        # Phase 6.3b: notifier.notify_complete assertion removed —
        # the abstraction was deleted alongside the marker-file IPC.
        mlflow_mgr.enable_autolog.assert_called_once_with(log_models=False)
        mlflow_mgr.end_run.assert_called_once_with(status="FINISHED")
        assert run_ctx.entered is True
        assert run_ctx.exited is True

    def test_run_training_failure_notifies_once(self, monkeypatch, tmp_path):
        import importlib

        rt = importlib.import_module("src.training.run_training")

        strategies = [MagicMock(strategy_type="sft")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = False

        mlflow_mgr = MagicMock()
        mlflow_mgr.is_active = True
        mlflow_mgr._mlflow_config = MagicMock(system_metrics_callback_enabled=True)
        run_ctx = _RunCtx()
        mlflow_mgr.start_run.return_value = run_ctx

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        memory_manager = MagicMock()
        memory_manager.gpu_info = None
        memory_manager.preset = None
        memory_manager.get_memory_stats.return_value = None

        orchestrator = MagicMock()
        orchestrator.buffer = None
        orchestrator.run_chain.return_value = Err("boom")

        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.return_value = (_ModelStub(), object())
        container.create_orchestrator.return_value = orchestrator

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(rt, "_setup_mlflow", lambda c: mlflow_mgr)
        monkeypatch.setattr(rt.EnvironmentReporter, "collect", classmethod(lambda cls=None: env_reporter))

        with pytest.raises(RuntimeError, match="Training failed: boom"):
            _ = rt.run_training(str(tmp_path / "cfg.yaml"), resume=False, run_id="run_x", container=container)

        # Phase 6.3b: notify_failed was called via the (now-deleted)
        # ICompletionNotifier abstraction; the regression "called
        # twice" is no longer reachable. MLflow is still expected to
        # finalise with FAILED.
        mlflow_mgr.end_run.assert_called_once_with(status="FAILED")

    def test_run_training_continues_when_mlflow_setup_returns_none(self, monkeypatch, tmp_path):
        import importlib

        rt = importlib.import_module("src.training.run_training")

        strategies = [MagicMock(strategy_type="sft")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = False

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        memory_manager = MagicMock()
        memory_manager.gpu_info = None
        memory_manager.preset = None
        memory_manager.get_memory_stats.return_value = None

        buffer = MagicMock()
        buffer.get_phase_output_dir.return_value = str(tmp_path / "phase_last")
        buffer.run_id = "run_1"
        orchestrator = MagicMock()
        orchestrator.buffer = buffer
        orchestrator.run_chain.return_value = Ok(None)

        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.return_value = (_ModelStub(), object())
        container.create_orchestrator.return_value = orchestrator

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(rt, "_setup_mlflow", lambda c: None)
        monkeypatch.setattr(rt.EnvironmentReporter, "collect", classmethod(lambda cls=None: env_reporter))

        out = rt.run_training(str(tmp_path / "cfg.yaml"), resume=False, run_id="run_x", container=container)

        assert out.name == "checkpoint-final"
        container.create_memory_manager_with_callbacks.assert_called_once_with(None)
        container.create_orchestrator.assert_called_once()
        kwargs = container.create_orchestrator.call_args.kwargs
        assert kwargs["mlflow_manager"] is None
        # Phase 6.3b: notify_complete assertion removed.


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ---------------------------------------------------------------------------
# training.log FileHandler installation
# ---------------------------------------------------------------------------


class TestTrainingFileHandler:
    """``_install_training_file_handler`` attaches a FileHandler so the
    pipeline-pulled ``runs/<id>/attempts/<n>/logs/training.log`` is non-empty.
    """

    def _detach_training_handlers(self) -> None:
        """Test isolation: drop any FileHandlers we attached so a re-run
        starts from a clean root logger."""
        import logging
        root = logging.getLogger("ryotenkai")
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()

    def teardown_method(self) -> None:
        self._detach_training_handlers()

    def test_attaches_filehandler_at_default_path(
        self, tmp_path, monkeypatch,
    ) -> None:
        from src.training.run_training import _install_training_file_handler
        import logging

        log_path = tmp_path / "workspace" / "training.log"
        monkeypatch.setenv("RYOTENKAI_TRAINING_LOG_PATH", str(log_path))

        _install_training_file_handler()

        # Emit something via the trainer logger; it must end up in the file.
        from src.utils.logger import logger
        logger.info("hello from trainer")

        # Force flush and read.
        for h in logging.getLogger("ryotenkai").handlers:
            h.flush()

        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "hello from trainer" in content

    def test_idempotent_double_install(
        self, tmp_path, monkeypatch,
    ) -> None:
        from src.training.run_training import _install_training_file_handler
        import logging

        log_path = tmp_path / "training.log"
        monkeypatch.setenv("RYOTENKAI_TRAINING_LOG_PATH", str(log_path))

        _install_training_file_handler()
        _install_training_file_handler()

        # Only one matching FileHandler; a second install is a no-op.
        root = logging.getLogger("ryotenkai")
        matching = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == str(log_path.resolve())
        ]
        assert len(matching) == 1

    def test_unwritable_path_does_not_raise(
        self, tmp_path, monkeypatch,
    ) -> None:
        from src.training.run_training import _install_training_file_handler

        # /dev/null/foo always fails on POSIX since /dev/null is a char device.
        monkeypatch.setenv("RYOTENKAI_TRAINING_LOG_PATH", "/dev/null/cannot-mkdir")
        # Must not raise — observability is best-effort.
        _install_training_file_handler()

    def test_default_path_when_env_missing(
        self, monkeypatch,
    ) -> None:
        # ``src.training.run_training`` is shadowed by a same-named function
        # re-exported from the package's ``__init__``; reach the module
        # via importlib so we can introspect its constants.
        import importlib
        run_training_module = importlib.import_module("src.training.run_training")

        # Env var unset → falls back to /workspace/training.log default.
        monkeypatch.delenv("RYOTENKAI_TRAINING_LOG_PATH", raising=False)
        # On a non-pod machine /workspace might not be writable; we only
        # assert the function does not raise.
        run_training_module._install_training_file_handler()
        assert run_training_module._DEFAULT_TRAINING_LOG_PATH == "/workspace/training.log"
