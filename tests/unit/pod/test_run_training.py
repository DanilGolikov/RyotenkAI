"""
Run Training Script Tests (Phase M4 post-rewire).

Tests for run_training.py which runs on GPU servers. Under Pattern A
the trainer subprocess no longer opens MLflow runs itself -- the
control plane owns the parent run and exports MLFLOW_RUN_ID +
MLFLOW_NESTED_RUN=TRUE; the HF MLflowCallback adopts those via env.

Tests cover:
- Configuration loading and validation
- CLI entry point
- Backward compatibility with train_v2 alias
- Pattern A env-driven control flow (no top-level MLflow run open)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pydantic
import pytest
import yaml

from ryotenkai_shared.errors import TrainingFailedError


# =============================================================================
# TEST CLASS: Configuration Loading
# =============================================================================


class TestConfigurationLoading:
    """Tests for configuration loading in run_training."""

    def test_nonexistent_config_raises_error(self):
        """Loading nonexistent config should raise FileNotFoundError."""
        from ryotenkai_pod.trainer.run_training import run_training

        with pytest.raises(FileNotFoundError):
            run_training("/nonexistent/path/config.yaml")

    def test_invalid_yaml_syntax_raises_error(self, tmp_path):
        """Invalid YAML syntax should raise error."""
        from ryotenkai_pod.trainer.run_training import run_training

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("model: {name: [unclosed")

        with pytest.raises((yaml.YAMLError, SystemExit)):
            run_training(str(config_file))

    def test_config_missing_required_field_raises_error(self, tmp_path):
        """Config missing required field should raise ValidationError."""
        from ryotenkai_pod.trainer.run_training import run_training

        config_file = tmp_path / "incomplete.yaml"
        config_file.write_text("""
training:
  type: qlora
""")

        with pytest.raises((pydantic.ValidationError, SystemExit)):
            run_training(str(config_file))


# =============================================================================
# Helpers
# =============================================================================


class _Param:
    def __init__(self, numel: int, requires_grad: bool):
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._numel


class _ModelStub:
    def __init__(self):
        self._params = [_Param(10, True), _Param(90, False)]

    def parameters(self):
        return list(self._params)


# =============================================================================
# TEST CLASS: Pattern A Flow
# =============================================================================


class TestRunTrainingFlow:
    """Pattern A flow: trainer never opens its own MLflow run.

    Tests focus on the control-flow contract that ``run_training``
    plumbs ``mlflow_manager=None`` through container hooks, that
    Pattern A env is validated when present, and that typed
    failure events are emitted exactly once.
    """

    def test_run_training_success_passes_none_mlflow_manager(
        self, monkeypatch, tmp_path,
    ):
        """Container hooks receive ``mlflow_manager=None`` under Pattern A."""
        import importlib

        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        strategies = [MagicMock(strategy_type="sft"), MagicMock(strategy_type="dpo")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = True
        cfg.training.adapter.kind = "qlora"

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        gpu = MagicMock()
        gpu.name = "GPU"
        gpu.tier.value = "TIER"
        gpu.total_memory_gb = 8.0
        memory_manager = MagicMock()
        memory_manager.gpu_info = gpu
        memory_manager.preset = None

        buffer = MagicMock()
        buffer.get_phase_output_dir.return_value = str(tmp_path / "phase_last")
        buffer.run_id = "run_1"
        orchestrator = MagicMock()
        orchestrator.buffer = buffer
        orchestrator.run_chain.return_value = MagicMock(name="trained_model")

        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.return_value = (_ModelStub(), object())
        container.create_orchestrator.return_value = orchestrator

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(
            rt.EnvironmentReporter,
            "collect",
            classmethod(lambda cls=None: env_reporter),
        )
        # Standalone trainer: no MLFLOW_RUN_ID -> pattern_a_active=False
        # (validate_env not called).
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

        out = rt.run_training(
            str(tmp_path / "cfg.yaml"),
            resume=False,
            run_id="run_x",
            container=container,
        )

        assert out.name == "checkpoint-final"
        # Pattern A: memory manager receives no mlflow manager.
        container.create_memory_manager_with_callbacks.assert_called_once_with(None)
        # Orchestrator receives no mlflow manager.
        kwargs = container.create_orchestrator.call_args.kwargs
        assert kwargs["mlflow_manager"] is None

    def test_run_training_failure_emits_typed_training_failed_event(
        self, monkeypatch, tmp_path,
    ):
        """RyotenkAIError failure path must emit a typed TrainingFailedEvent."""
        import importlib

        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        strategies = [MagicMock(strategy_type="sft")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.adapter.kind = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = False

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        memory_manager = MagicMock()
        memory_manager.gpu_info = None
        memory_manager.preset = None

        orchestrator = MagicMock()
        orchestrator.buffer = None
        orchestrator.run_chain.side_effect = TrainingFailedError(detail="kaboom")

        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.return_value = (_ModelStub(), object())
        container.create_orchestrator.return_value = orchestrator

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(
            rt.EnvironmentReporter,
            "collect",
            classmethod(lambda cls=None: env_reporter),
        )
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

        emit_spy = MagicMock()
        monkeypatch.setattr(rt, "_emit_training_failed", emit_spy)

        with pytest.raises(RuntimeError, match="Training failed: kaboom"):
            _ = rt.run_training(
                str(tmp_path / "cfg.yaml"),
                resume=False,
                run_id="run_x",
                container=container,
            )

        # Typed event emitted exactly once with the original typed exception.
        assert emit_spy.call_count == 1
        call_kwargs = emit_spy.call_args.kwargs
        assert isinstance(call_kwargs["exc"], TrainingFailedError)

    def test_run_training_unexpected_exception_emits_event(
        self, monkeypatch, tmp_path,
    ):
        """Unexpected (non-typed) exception path also emits the typed event."""
        import importlib

        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        strategies = [MagicMock(strategy_type="sft")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.adapter.kind = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = False

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        memory_manager = MagicMock()
        memory_manager.gpu_info = None
        memory_manager.preset = None

        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.side_effect = RuntimeError(
            "model load failed",
        )

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(
            rt.EnvironmentReporter,
            "collect",
            classmethod(lambda cls=None: env_reporter),
        )
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

        emit_spy = MagicMock()
        monkeypatch.setattr(rt, "_emit_training_failed", emit_spy)

        with pytest.raises(RuntimeError, match="model load failed"):
            _ = rt.run_training(
                str(tmp_path / "cfg.yaml"),
                resume=False,
                run_id="run_x",
                container=container,
            )

        # Exactly one typed emission.
        assert emit_spy.call_count == 1
        emit_kwargs = emit_spy.call_args.kwargs
        assert isinstance(emit_kwargs["exc"], RuntimeError)
        assert "model load failed" in str(emit_kwargs["exc"])

    def test_run_training_invokes_publish_when_pattern_a_active(
        self, monkeypatch, tmp_path,
    ):
        """Phase M5: successful training under Pattern A calls ``_publish_trained_model``."""
        import importlib

        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        strategies = [MagicMock(strategy_type="sft")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.adapter.kind = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = False

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        memory_manager = MagicMock()
        memory_manager.gpu_info = None
        memory_manager.preset = None

        buffer = MagicMock()
        buffer.get_phase_output_dir.return_value = str(tmp_path / "phase_last")
        buffer.run_id = "run_1"
        orchestrator = MagicMock()
        orchestrator.buffer = buffer
        trained_model = MagicMock(name="trained_model")
        orchestrator.run_chain.return_value = trained_model

        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.return_value = (_ModelStub(), object())
        container.create_orchestrator.return_value = orchestrator

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(
            rt.EnvironmentReporter,
            "collect",
            classmethod(lambda cls=None: env_reporter),
        )
        # Pattern A active: every required env var present.
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent_123")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_NESTED_RUN", "TRUE")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "dev__alignment__sft_smoke")

        publish_spy = MagicMock()
        monkeypatch.setattr(rt, "_publish_trained_model", publish_spy)

        out = rt.run_training(
            str(tmp_path / "cfg.yaml"),
            resume=False,
            run_id="run_x",
            container=container,
        )
        assert out.name == "checkpoint-final"
        # Publish was called exactly once with the trained model.
        publish_spy.assert_called_once()
        call_kwargs = publish_spy.call_args.kwargs
        assert call_kwargs["trainer_model"] is trained_model
        assert call_kwargs["config"] is cfg

    def test_run_training_skips_publish_when_pattern_a_inactive(
        self, monkeypatch, tmp_path,
    ):
        """Standalone (no MLFLOW_RUN_ID) trainer runs never call publish."""
        import importlib

        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        strategies = [MagicMock(strategy_type="sft")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.adapter.kind = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = False

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        memory_manager = MagicMock()
        memory_manager.gpu_info = None
        memory_manager.preset = None

        buffer = MagicMock()
        buffer.get_phase_output_dir.return_value = str(tmp_path / "phase_last")
        buffer.run_id = "run_1"
        orchestrator = MagicMock()
        orchestrator.buffer = buffer
        orchestrator.run_chain.return_value = MagicMock(name="trained_model")

        container = MagicMock()
        container.create_memory_manager_with_callbacks.return_value = memory_manager
        container.load_model_and_tokenizer.return_value = (_ModelStub(), object())
        container.create_orchestrator.return_value = orchestrator

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(
            rt.EnvironmentReporter,
            "collect",
            classmethod(lambda cls=None: env_reporter),
        )
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

        publish_spy = MagicMock()
        monkeypatch.setattr(rt, "_publish_trained_model", publish_spy)

        _ = rt.run_training(
            str(tmp_path / "cfg.yaml"),
            resume=False,
            run_id="run_x",
            container=container,
        )
        publish_spy.assert_not_called()

    def test_run_training_validates_env_when_pattern_a_active(
        self, monkeypatch, tmp_path,
    ):
        """When MLFLOW_RUN_ID is set, validate_env is invoked fail-fast."""
        import importlib

        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        strategies = [MagicMock(strategy_type="sft")]
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.training.type = "qlora"
        cfg.training.adapter.kind = "qlora"
        cfg.training.get_strategy_chain.return_value = strategies
        cfg.training.get_effective_load_in_4bit.return_value = True
        cfg.training.is_multi_phase.return_value = False

        env_reporter = MagicMock()
        env_reporter.snapshot.to_dict.return_value = {"os": "x"}

        monkeypatch.setattr(rt, "load_pipeline_config", lambda p: cfg)
        monkeypatch.setattr(
            rt.EnvironmentReporter,
            "collect",
            classmethod(lambda cls=None: env_reporter),
        )

        # Only MLFLOW_RUN_ID set → other required vars missing → validate_env raises.
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent_123")
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_NESTED_RUN", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

        # ConfigInvalidError bubbles up through run_training's outer except,
        # which re-raises (after emitting failed event).
        with pytest.raises(Exception):
            _ = rt.run_training(
                str(tmp_path / "cfg.yaml"),
                resume=False,
                run_id="run_x",
                container=MagicMock(),
            )


# =============================================================================
# TEST CLASS: _derive_model_family helper
# =============================================================================


class TestDeriveModelFamily:
    """Phase M5 ``_derive_model_family`` helper — template substitution."""

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct"),
            ("meta-llama/Llama-3.2-1B", "llama-3.2-1b"),
            ("bare-name-no-slash", "bare-name-no-slash"),
            ("", "model"),
            ("Org/Name with spaces!", "name-with-spaces"),
            ("Org/UPPER", "upper"),
        ],
    )
    def test_derive_model_family(self, input_name: str, expected: str) -> None:
        from ryotenkai_pod.trainer.run_training import _derive_model_family

        assert _derive_model_family(input_name) == expected


# =============================================================================
# TEST CLASS: _publish_trained_model helper
# =============================================================================


class TestPublishTrainedModel:
    """Phase M5 ``_publish_trained_model`` helper — registry wiring."""

    def _make_cfg(self) -> MagicMock:
        cfg = MagicMock()
        cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.integrations.mlflow.model_registry_name_template = (
            "ryotenkai/{experiment}/{model_family}"
        )
        cfg.integrations.mlflow.experiment_name = "dev__alignment__sft"
        cfg.integrations.mlflow.alias_on_success = "challenger"
        return cfg

    def test_publish_skips_when_no_active_run(
        self, monkeypatch, tmp_path,
    ):
        """If ``mlflow.active_run()`` returns None and env is empty, skip cleanly."""
        import importlib
        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        # Stub the mlflow module so we never hit real I/O.
        stub_mlflow = MagicMock()
        stub_mlflow.active_run.return_value = None
        monkeypatch.setitem(
            __import__("sys").modules, "mlflow", stub_mlflow,
        )
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

        # No ImportError, no publisher invocation expected.
        rt._publish_trained_model(
            config=self._make_cfg(),
            trainer_model=MagicMock(),
            tokenizer=MagicMock(),
            output_path=tmp_path,
        )
        stub_mlflow.transformers.log_model.assert_not_called()

    def test_publish_logs_model_and_calls_publisher(
        self, monkeypatch, tmp_path,
    ):
        """Happy path: log_model + publisher.publish are both invoked."""
        import importlib
        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        # Capture transformers.log_model + tracking_uri.
        stub_mlflow = MagicMock()
        stub_mlflow.active_run.return_value = MagicMock(
            info=MagicMock(run_id="run-xyz"),
        )
        stub_mlflow.get_tracking_uri.return_value = "http://mlflow:5000"
        monkeypatch.setitem(__import__("sys").modules, "mlflow", stub_mlflow)

        stub_transformers = MagicMock()
        monkeypatch.setitem(
            __import__("sys").modules,
            "mlflow.transformers",
            stub_transformers,
        )
        # Also expose as attribute on stub_mlflow so the
        # ``from mlflow import transformers as mlflow_transformers``
        # path resolves.
        stub_mlflow.transformers = stub_transformers

        # Patch the publisher + registry constructors to spies.
        publisher_spy = MagicMock()
        publisher_spy.publish.return_value = MagicMock(version="7")
        monkeypatch.setattr(
            "ryotenkai_pod.trainer.mlflow.model_publisher.ModelPublisher",
            MagicMock(return_value=publisher_spy),
        )
        registry_cls_spy = MagicMock()
        monkeypatch.setattr(
            "ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry",
            registry_cls_spy,
        )

        cfg = self._make_cfg()
        rt._publish_trained_model(
            config=cfg,
            trainer_model=MagicMock(name="trained"),
            tokenizer=MagicMock(name="tok"),
            output_path=tmp_path,
        )

        # log_model called with save_pretrained=True (R-21).
        stub_transformers.log_model.assert_called_once()
        log_kwargs = stub_transformers.log_model.call_args.kwargs
        assert log_kwargs["save_pretrained"] is True
        assert log_kwargs["artifact_path"] == "model"

        # Registry constructed against the active tracking URI.
        registry_cls_spy.assert_called_once_with(tracking_uri="http://mlflow:5000")

        # Publisher.publish called with templated name + active run id.
        publisher_spy.publish.assert_called_once()
        pub_kwargs = publisher_spy.publish.call_args.kwargs
        assert pub_kwargs["run_id"] == "run-xyz"
        assert pub_kwargs["artifact_path"] == "model"
        assert pub_kwargs["registered_name"] == (
            "ryotenkai/dev__alignment__sft/qwen2.5-0.5b-instruct"
        )
        assert pub_kwargs["alias_on_success"] == "challenger"

    def test_publish_alias_env_override(self, monkeypatch, tmp_path):
        """``MLFLOW_ALIAS_ON_SUCCESS`` env var overrides config default."""
        import importlib
        rt = importlib.import_module("ryotenkai_pod.trainer.run_training")

        stub_mlflow = MagicMock()
        stub_mlflow.active_run.return_value = MagicMock(
            info=MagicMock(run_id="run-xyz"),
        )
        stub_mlflow.get_tracking_uri.return_value = "http://mlflow:5000"
        monkeypatch.setitem(__import__("sys").modules, "mlflow", stub_mlflow)
        stub_transformers = MagicMock()
        stub_mlflow.transformers = stub_transformers
        monkeypatch.setitem(
            __import__("sys").modules, "mlflow.transformers", stub_transformers,
        )

        publisher_spy = MagicMock()
        publisher_spy.publish.return_value = MagicMock(version="1")
        monkeypatch.setattr(
            "ryotenkai_pod.trainer.mlflow.model_publisher.ModelPublisher",
            MagicMock(return_value=publisher_spy),
        )
        monkeypatch.setattr(
            "ryotenkai_shared.infrastructure.mlflow.registry.MlflowModelRegistry",
            MagicMock(),
        )

        monkeypatch.setenv("MLFLOW_ALIAS_ON_SUCCESS", "shadow")
        rt._publish_trained_model(
            config=self._make_cfg(),
            trainer_model=MagicMock(),
            tokenizer=MagicMock(),
            output_path=tmp_path,
        )
        pub_kwargs = publisher_spy.publish.call_args.kwargs
        assert pub_kwargs["alias_on_success"] == "shadow"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
