from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.utils.container import TrainingContainer, _create_noop_memory_manager, _create_noop_notifier


def _mk_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "m"
    cfg.training.type = "qlora"
    cfg.training.get_effective_load_in_4bit.return_value = False
    cfg.datasets = {}
    cfg.experiment_tracking.mlflow = SimpleNamespace(
        tracking_uri="http://localhost:5002",
        system_metrics_callback_enabled=False,
    )
    return cfg


class TestMemoryManagerProperty:
    def test_returns_injected_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        injected = MagicMock()
        container = TrainingContainer(cfg, _memory_manager=injected)

        auto = MagicMock()
        monkeypatch.setattr("src.utils.memory_manager.MemoryManager.auto_configure", auto)

        assert container.memory_manager is injected
        auto.assert_not_called()

    def test_lazy_initializes_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        container = TrainingContainer(cfg)

        mm = MagicMock()
        auto = MagicMock(return_value=mm)
        monkeypatch.setattr("src.utils.memory_manager.MemoryManager.auto_configure", auto)

        assert container.memory_manager is mm
        assert container.memory_manager is mm
        auto.assert_called_once_with()


class TestMemoryManagerWithCallbacks:
    def test_returns_regular_memory_manager_when_mlflow_is_none(self) -> None:
        cfg = _mk_cfg()
        injected = MagicMock()
        container = TrainingContainer(cfg, _memory_manager=injected)
        assert container.create_memory_manager_with_callbacks(mlflow_manager=None) is injected

    def test_builds_callbacks_and_wires_to_mlflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        container = TrainingContainer(cfg, _memory_manager=MagicMock())

        mlflow = MagicMock()
        auto = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("src.utils.memory_manager.MemoryManager.auto_configure", auto)

        _ = container.create_memory_manager_with_callbacks(mlflow_manager=mlflow)
        callbacks = auto.call_args.kwargs["callbacks"]

        callbacks.on_gpu_detected("GPU", 10.0, "TIER")
        callbacks.on_cache_cleared(123)
        callbacks.on_memory_warning(80.0, 100, 200, True)
        callbacks.on_oom("op", 42)
        callbacks.on_oom_retry("op", 1, 3)

        mlflow.log_gpu_detection.assert_called_once_with("GPU", 10.0, "TIER")
        mlflow.log_cache_cleared.assert_called_once_with(123)
        mlflow.log_memory_warning.assert_called_once_with(80.0, 100, 200, is_critical=True)
        mlflow.log_oom.assert_called_once_with("op", 42)
        mlflow.log_oom_recovery.assert_called_once_with("op", 1, 3)


class TestDatasetLoaderFactory:
    def test_get_loader_for_dataset_falls_back_to_default_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        injected_loader = MagicMock()
        container = TrainingContainer(cfg, _dataset_loader=injected_loader)

        factory_cls = MagicMock()
        monkeypatch.setattr("src.data.loaders.DatasetLoaderFactory", factory_cls)

        loader = container.get_loader_for_dataset("missing")
        assert loader is injected_loader
        factory_cls.assert_not_called()

    def test_get_loader_for_dataset_uses_factory_when_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        cfg.datasets = {"x": SimpleNamespace(get_source_type=lambda: "local")}
        container = TrainingContainer(cfg, _dataset_loader=MagicMock())

        factory = MagicMock()
        out_loader = MagicMock()
        factory.create_for_dataset.return_value = out_loader
        factory_cls = MagicMock(return_value=factory)
        monkeypatch.setattr("src.data.loaders.DatasetLoaderFactory", factory_cls)

        loader = container.get_loader_for_dataset("x")
        assert loader is out_loader
        factory_cls.assert_called_once()
        factory.create_for_dataset.assert_called_once()


class TestMLflowManagerProperty:
    def test_returns_injected_instance(self) -> None:
        cfg = _mk_cfg()
        injected = MagicMock()
        container = TrainingContainer(cfg, _mlflow_manager=injected)
        assert container.mlflow_manager is injected

    def test_lazy_initializes_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        container = TrainingContainer(cfg)

        mgr = MagicMock()
        cls = MagicMock(return_value=mgr)
        monkeypatch.setattr("src.training.managers.mlflow_manager.MLflowManager", cls)

        assert container.mlflow_manager is mgr
        assert container.mlflow_manager is mgr
        cls.assert_called_once_with(cfg)


class TestCompletionNotifierProperty:
    def test_returns_injected_notifier(self) -> None:
        cfg = _mk_cfg()
        injected = MagicMock()
        container = TrainingContainer(cfg, _completion_notifier=injected)
        assert container.completion_notifier is injected

    def test_uses_marker_path_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        cfg.training.marker_path = "/cfg"

        notifier = MagicMock()
        cls = MagicMock(return_value=notifier)
        monkeypatch.setattr("src.training.notifiers.marker_file.MarkerFileNotifier", cls)

        container = TrainingContainer(cfg)
        assert container.completion_notifier is notifier
        cls.assert_called_once_with(base_path="/cfg")

    def test_uses_helix_workspace_env_when_no_config_marker_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        cfg.training.marker_path = None
        monkeypatch.setenv("HELIX_WORKSPACE", "/env")

        notifier = MagicMock()
        cls = MagicMock(return_value=notifier)
        monkeypatch.setattr("src.training.notifiers.marker_file.MarkerFileNotifier", cls)

        container = TrainingContainer(cfg)
        assert container.completion_notifier is notifier
        cls.assert_called_once_with(base_path="/env")

    def test_uses_cwd_when_no_config_and_no_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cfg = _mk_cfg()
        cfg.training.marker_path = None
        monkeypatch.delenv("HELIX_WORKSPACE", raising=False)
        monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

        notifier = MagicMock()
        cls = MagicMock(return_value=notifier)
        monkeypatch.setattr("src.training.notifiers.marker_file.MarkerFileNotifier", cls)

        container = TrainingContainer(cfg)
        assert container.completion_notifier is notifier
        cls.assert_called_once_with(base_path=str(tmp_path))


class TestOrchestratorAndModelLoading:
    def test_create_orchestrator_injects_dependencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        mm = MagicMock()
        sf = MagicMock()
        tf = MagicMock()
        dl = MagicMock()
        mlflow = MagicMock()
        container = TrainingContainer(
            cfg,
            _memory_manager=mm,
            _strategy_factory=sf,
            _trainer_factory=tf,
            _dataset_loader=dl,
            _mlflow_manager=mlflow,
        )

        orchestrator = MagicMock()
        orch_cls = MagicMock(return_value=orchestrator)
        monkeypatch.setattr("src.training.orchestrator.StrategyOrchestrator", orch_cls)

        model = MagicMock()
        tok = MagicMock()
        out = container.create_orchestrator(model, tok)
        assert out is orchestrator

        orch_cls.assert_called_once()
        kwargs = orch_cls.call_args.kwargs
        assert kwargs["model"] is model
        assert kwargs["tokenizer"] is tok
        assert kwargs["config"] is cfg
        assert kwargs["memory_manager"] is mm
        assert kwargs["strategy_factory"] is sf
        assert kwargs["trainer_factory"] is tf
        assert kwargs["dataset_loader"] is dl
        assert kwargs["mlflow_manager"] is mlflow

    def test_load_model_and_tokenizer_uses_safe_operation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        cfg.model.name = "base"

        mm = MagicMock()
        mm.safe_operation.return_value = nullcontext()
        container = TrainingContainer(cfg, _memory_manager=mm)

        model = MagicMock()
        tok = MagicMock()
        loader = MagicMock(return_value=(model, tok))
        monkeypatch.setattr("src.training.models.loader.load_model_and_tokenizer", loader)

        out_model, out_tok = container.load_model_and_tokenizer()
        assert out_model is model
        assert out_tok is tok
        mm.safe_operation.assert_called_once_with("model_loading")
        loader.assert_called_once_with(config=cfg)


class TestOverrideAndNoopFactories:
    def test_override_is_non_destructive(self) -> None:
        cfg = _mk_cfg()
        orig_mm = MagicMock()
        container = TrainingContainer(cfg, _memory_manager=orig_mm)

        new_mm = MagicMock()
        overridden = container.override(memory_manager=new_mm)

        assert overridden is not container
        assert overridden.config is container.config
        assert overridden.memory_manager is new_mm
        assert container.memory_manager is orig_mm

    def test_noop_memory_manager_and_notifier_are_callable(self) -> None:
        mm = _create_noop_memory_manager()
        assert mm.gpu_info is None
        assert mm.get_memory_stats() is None
        assert mm.is_memory_critical() is False
        assert mm.clear_cache() == 0
        assert mm.aggressive_cleanup() == 0
        with mm.safe_operation("x"):
            pass

        notifier = _create_noop_notifier()
        notifier.notify_complete({"output_path": "x"})
        notifier.notify_failed("err", {"output_path": "x"})
