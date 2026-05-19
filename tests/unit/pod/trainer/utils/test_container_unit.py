from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ryotenkai_pod.trainer.container import TrainingContainer, _create_noop_memory_manager


def _mk_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "m"
    cfg.training.type = "qlora"
    cfg.training.get_effective_load_in_4bit.return_value = False
    cfg.datasets = {}
    cfg.integrations.mlflow = SimpleNamespace(
        tracking_uri="http://localhost:5002",
        system_metrics_callback_enabled=False,
    )
    return cfg


class TestMemoryManagerProperty:
    def test_returns_injected_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        injected = SimpleNamespace()
        container = TrainingContainer(cfg, _memory_manager=injected)

        auto = MagicMock()
        monkeypatch.setattr("ryotenkai_pod.trainer.memory_manager.MemoryManager.auto_configure", auto)

        assert container.memory_manager is injected
        auto.assert_not_called()

    def test_lazy_initializes_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        container = TrainingContainer(cfg)

        mm = SimpleNamespace()
        auto = MagicMock(return_value=mm)
        monkeypatch.setattr("ryotenkai_pod.trainer.memory_manager.MemoryManager.auto_configure", auto)

        assert container.memory_manager is mm
        assert container.memory_manager is mm
        auto.assert_called_once_with()


class TestMemoryManagerWithCallbacks:
    """M7 cleanup: ``create_memory_manager_with_callbacks`` no longer accepts
    a ``mlflow_manager`` argument and does not wire MLflow callbacks — memory
    events flow through typed journal events emitted by downstream callbacks
    (e.g. :class:`RunnerEventCallback`). The method is a thin wrapper around
    :attr:`memory_manager` preserved for callers using the factory entrypoint.
    """

    def test_returns_memory_manager_without_callbacks_after_retirement(self) -> None:
        cfg = _mk_cfg()
        injected = SimpleNamespace()
        container = TrainingContainer(cfg, _memory_manager=injected)
        assert container.create_memory_manager_with_callbacks() is injected


# DEAD: `ryotenkai_control.data.loaders.DatasetLoaderFactory` was removed
# during Phase B packagization. The dataset-loader injection path is
# exercised through the container's direct `_dataset_loader` argument in
# the tests that survived above.


@pytest.mark.xfail(
    strict=False,
    reason="xfail-debt:m7-wide-mlflow-manager-retired",
)
class TestMLflowManagerProperty:
    def test_returns_injected_instance(self) -> None:
        cfg = _mk_cfg()
        injected = SimpleNamespace()
        container = TrainingContainer(cfg, _mlflow_manager=injected)
        assert container.mlflow_manager is injected

    def test_lazy_initializes_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        container = TrainingContainer(cfg)

        mgr = SimpleNamespace()
        cls = MagicMock(return_value=mgr)
        monkeypatch.setattr("ryotenkai_pod.trainer.managers.mlflow_manager.MLflowManager", cls)

        assert container.mlflow_manager is mgr
        assert container.mlflow_manager is mgr
        cls.assert_called_once_with(cfg)


# Phase 6.3b: ``TestCompletionNotifierProperty`` removed along with
# the ``ICompletionNotifier`` protocol and the ``completion_notifier``
# container property. Trainer-side completion signalling now flows
# through :class:`RunnerEventCallback` (separate test suite).


class TestOrchestratorAndModelLoading:
    def test_create_orchestrator_injects_dependencies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """M7 cleanup: ``create_orchestrator`` no longer wires a
        ``mlflow_manager`` — MLflow nesting flows from env vars consumed by
        HF MLflowCallback (Pattern A). ``StrategyOrchestrator`` receives the
        remaining DI dependencies as-is.
        """
        cfg = _mk_cfg()
        mm = SimpleNamespace()
        sf = SimpleNamespace()
        tf = SimpleNamespace()
        dl = SimpleNamespace()
        container = TrainingContainer(
            cfg,
            _memory_manager=mm,
            _strategy_factory=sf,
            _trainer_factory=tf,
            _dataset_loader=dl,
        )

        orchestrator = SimpleNamespace()
        orch_cls = MagicMock(return_value=orchestrator)
        monkeypatch.setattr("ryotenkai_pod.trainer.orchestrator.StrategyOrchestrator", orch_cls)

        model = SimpleNamespace()
        tok = SimpleNamespace()
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
        assert "mlflow_manager" not in kwargs

    def test_load_model_and_tokenizer_uses_safe_operation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        cfg.model.name = "base"

        mm = MagicMock()
        mm.safe_operation.return_value = nullcontext()
        container = TrainingContainer(cfg, _memory_manager=mm)

        model = SimpleNamespace()
        tok = SimpleNamespace()
        loader = MagicMock(return_value=(model, tok))
        monkeypatch.setattr("ryotenkai_pod.trainer.models.loader.load_model_and_tokenizer", loader)

        out_model, out_tok = container.load_model_and_tokenizer()
        assert out_model is model
        assert out_tok is tok
        mm.safe_operation.assert_called_once_with("model_loading")
        loader.assert_called_once_with(config=cfg)


class TestOverrideAndNoopFactories:
    def test_override_is_non_destructive(self) -> None:
        cfg = _mk_cfg()
        orig_mm = SimpleNamespace()
        container = TrainingContainer(cfg, _memory_manager=orig_mm)

        new_mm = SimpleNamespace()
        overridden = container.override(memory_manager=new_mm)

        assert overridden is not container
        assert overridden.config is container.config
        assert overridden.memory_manager is new_mm
        assert container.memory_manager is orig_mm

    def test_noop_memory_manager_is_callable(self) -> None:
        mm = _create_noop_memory_manager()
        assert mm.gpu_info is None
        assert mm.get_memory_stats() is None
        assert mm.is_memory_critical() is False
        assert mm.clear_cache() == 0
        assert mm.aggressive_cleanup() == 0
        with mm.safe_operation("x"):
            pass
        # Phase 6.3b: ``_create_noop_notifier`` removed along with the
        # ICompletionNotifier protocol.
