from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.training.managers.data_buffer import PhaseStatus
from src.training.orchestrator.strategy_orchestrator import StrategyOrchestrator
from src.utils.result import Err, Ok


def _mk_cfg(*, strategies: list[Any] | None = None) -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "base-model"
    cfg.training.hyperparams = MagicMock()
    cfg.training.get_strategy_chain.return_value = strategies or []
    return cfg


def _mk_orchestrator(
    *,
    cfg: MagicMock,
    mlflow_manager: Any | None = None,
    shutdown_handler: Any | None = None,
    graceful_shutdown: bool = False,
) -> StrategyOrchestrator:
    return StrategyOrchestrator(
        model=MagicMock(name="model"),
        tokenizer=MagicMock(name="tokenizer"),
        config=cfg,
        memory_manager=MagicMock(name="mm"),
        dataset_loader=MagicMock(name="dl"),
        strategy_factory=MagicMock(name="sf"),
        trainer_factory=MagicMock(name="tf"),
        mlflow_manager=mlflow_manager,
        shutdown_handler=shutdown_handler,
        graceful_shutdown=graceful_shutdown,
    )


class TestDataBufferCallbacks:
    def test_callbacks_are_none_without_mlflow(self) -> None:
        orch = _mk_orchestrator(cfg=_mk_cfg(strategies=[MagicMock()]), mlflow_manager=None)
        assert orch._create_data_buffer_callbacks() is None

    def test_callbacks_call_mlflow_manager(self) -> None:
        mlflow = MagicMock()
        orch = _mk_orchestrator(cfg=_mk_cfg(strategies=[MagicMock()]), mlflow_manager=mlflow)
        cbs = orch._create_data_buffer_callbacks()
        assert cbs is not None

        cbs.on_pipeline_initialized("rid", 2, ["sft", "dpo"])
        mlflow.log_pipeline_initialized.assert_called_once_with("rid", 2, ["sft", "dpo"])

        cbs.on_state_saved("rid", "/state.json")
        mlflow.log_state_saved.assert_called_once_with("rid", "/state.json")

        cbs.on_phase_started(1, "sft")
        mlflow.log_event_start.assert_called()

        cbs.on_phase_completed(1, "sft", "completed")
        mlflow.log_event_complete.assert_called()

        cbs.on_checkpoint_cleanup(3, 100)
        mlflow.log_checkpoint_cleanup.assert_called_once_with(3, 100)


class TestRunChain:
    def test_returns_err_when_no_strategies(self) -> None:
        orch = _mk_orchestrator(cfg=_mk_cfg(strategies=[]))
        res = cast("Any", orch.run_chain(strategies=None))
        assert res.is_failure()
        assert "No strategies configured" in str(res.unwrap_err())

    def test_resume_all_complete_returns_base_model(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)

        buf = MagicMock()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=True)
        orch._resume_manager.get_interrupt_info = MagicMock(return_value={"phase_idx": 1, "reason": "sig"})
        orch._resume_manager.is_all_complete = MagicMock(return_value=True)

        orch._chain_runner.run = MagicMock(return_value=Ok(MagicMock()))
        res = cast("Any", orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=True, run_id="rid"))
        assert res.is_success()
        assert res.unwrap() is orch.model
        orch._chain_runner.run.assert_not_called()

    def test_load_checkpoint_failure_propagates(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        buf = MagicMock()

        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 1, True))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)
        orch._resume_manager.get_checkpoint_path_for_phase = MagicMock(return_value="/ckpt")
        orch._resume_manager.load_model_from_checkpoint = MagicMock(return_value=Err("boom"))

        res = cast("Any", orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=True, run_id="rid"))
        assert res.is_failure()
        assert "boom" in str(res.unwrap_err())

    def test_loaded_model_none_returns_err(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        buf = MagicMock()

        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 1, True))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)
        orch._resume_manager.get_checkpoint_path_for_phase = MagicMock(return_value="/ckpt")
        orch._resume_manager.load_model_from_checkpoint = MagicMock(return_value=Ok(None))

        res = cast("Any", orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=True, run_id="rid"))
        assert res.is_failure()
        assert "Model is None after checkpoint loading" in str(res.unwrap_err())

    def test_registers_and_unregisters_shutdown_handler(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        sh = MagicMock()
        sh.should_stop.return_value = True
        orch = _mk_orchestrator(cfg=cfg, shutdown_handler=sh)

        buf = MagicMock()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)

        final_model = MagicMock()
        orch._chain_runner.run = MagicMock(return_value=Ok(final_model))

        res = cast("Any", orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=False, run_id="rid"))
        assert res.is_success()
        assert res.unwrap() is final_model

        sh.register.assert_called_once()
        sh.unregister.assert_called_once()


class TestSinglePhaseAndUtilities:
    def test_run_single_phase_initializes_buffer_if_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        orch.buffer = None

        created_buf = MagicMock()
        monkeypatch.setattr("src.training.orchestrator.strategy_orchestrator.DataBuffer", lambda **kw: created_buf)

        out_model = MagicMock()
        orch._phase_executor.execute = MagicMock(return_value=Ok(out_model))

        phase = MagicMock(strategy_type="sft")
        res = cast("Any", orch.run_single_phase(phase_idx=0, phase=phase, model=None))
        assert res.is_success()
        assert res.unwrap() is out_model
        created_buf.init_pipeline.assert_called_once()

    def test_get_summary_and_phase_lists(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)

        assert orch.get_summary()["status"] == "not_initialized"
        assert orch.get_completed_phases() == []
        assert orch.get_interrupted_phases() == []

        buf = MagicMock()
        buf._state = object()
        buf.get_summary.return_value = {"ok": True}
        buf.state.phases = [
            SimpleNamespace(phase_idx=0, status=PhaseStatus.COMPLETED),
            SimpleNamespace(phase_idx=1, status=PhaseStatus.INTERRUPTED),
        ]
        orch.buffer = buf

        assert orch.get_summary() == {"ok": True}
        assert orch.get_completed_phases() == [0]
        assert orch.get_interrupted_phases() == [1]

    def test_was_interrupted_uses_shutdown_handler(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        sh = MagicMock()
        sh.should_stop.return_value = True
        orch = _mk_orchestrator(cfg=cfg, shutdown_handler=sh)
        assert orch.was_interrupted() is True
