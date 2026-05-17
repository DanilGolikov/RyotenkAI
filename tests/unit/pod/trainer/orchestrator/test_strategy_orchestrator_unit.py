from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus
from ryotenkai_pod.trainer.orchestrator.strategy_orchestrator import StrategyOrchestrator
from ryotenkai_shared.errors import (
    RyotenkAIError,
    StrategyChainInvalidError,
    TrainingFailedError,
)


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

        # Phase 7: per-phase ``log_event_start`` / ``log_event_complete``
        # were removed — the callbacks are now no-ops. They still must
        # accept the call and not raise.
        cbs.on_phase_started(1, "sft")
        cbs.on_phase_completed(1, "sft", "completed")

        cbs.on_checkpoint_cleanup(3, 100)
        mlflow.log_checkpoint_cleanup.assert_called_once_with(3, 100)


class TestRunChain:
    def test_raises_when_no_strategies(self) -> None:
        orch = _mk_orchestrator(cfg=_mk_cfg(strategies=[]))
        with pytest.raises(StrategyChainInvalidError) as exc_info:
            orch.run_chain(strategies=None)
        assert "No strategies configured" in (exc_info.value.detail or "")
        assert exc_info.value.context.get("legacy_code") == "TRAINING_NO_STRATEGIES"

    def test_resume_all_complete_returns_base_model(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)

        buf = SimpleNamespace()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=True)
        orch._resume_manager.get_interrupt_info = MagicMock(return_value={"phase_idx": 1, "reason": "sig"})
        orch._resume_manager.is_all_complete = MagicMock(return_value=True)

        orch._chain_runner.run = MagicMock(return_value=MagicMock(name="should_not_be_used"))
        out = orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=True, run_id="rid")
        assert out is orch.model
        orch._chain_runner.run.assert_not_called()

    def test_load_checkpoint_failure_propagates(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        buf = SimpleNamespace()

        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 1, True))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)
        orch._resume_manager.get_checkpoint_path_for_phase = MagicMock(return_value="/ckpt")
        orch._resume_manager.load_model_from_checkpoint = MagicMock(
            side_effect=TrainingFailedError(detail="boom"),
        )

        with pytest.raises(TrainingFailedError) as exc_info:
            orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=True, run_id="rid")
        assert "boom" in (exc_info.value.detail or "")

    def test_registers_and_unregisters_shutdown_handler(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        sh = MagicMock()
        sh.should_stop.return_value = True
        orch = _mk_orchestrator(cfg=cfg, shutdown_handler=sh)

        buf = SimpleNamespace()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)

        final_model = SimpleNamespace()
        orch._chain_runner.run = MagicMock(return_value=final_model)

        out = orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=False, run_id="rid")
        assert out is final_model

        sh.register.assert_called_once()
        sh.unregister.assert_called_once()

    def test_shutdown_handler_unregistered_on_failure(self) -> None:
        """Phase A2 Batch 14 invariant: signal handlers are released
        even when chain_runner raises — the finally block must run.
        """
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        sh = MagicMock()
        sh.should_stop.return_value = False
        orch = _mk_orchestrator(cfg=cfg, shutdown_handler=sh)

        buf = SimpleNamespace()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)

        orch._chain_runner.run = MagicMock(side_effect=TrainingFailedError(detail="kaboom"))

        with pytest.raises(TrainingFailedError):
            orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=False, run_id="rid")
        sh.register.assert_called_once()
        sh.unregister.assert_called_once()


class TestSinglePhaseAndUtilities:
    def test_run_single_phase_initializes_buffer_if_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        orch.buffer = None

        created_buf = MagicMock()
        monkeypatch.setattr("ryotenkai_pod.trainer.orchestrator.strategy_orchestrator.DataBuffer", lambda **kw: created_buf)

        out_model = SimpleNamespace()
        orch._phase_executor.execute = MagicMock(return_value=out_model)

        phase = SimpleNamespace(strategy_type="sft")
        out = orch.run_single_phase(phase_idx=0, phase=phase, model=None)
        assert out is out_model
        created_buf.init_pipeline.assert_called_once()

    def test_run_single_phase_propagates_failure(self) -> None:
        """``run_single_phase`` no longer wraps in Result; it propagates."""
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        orch.buffer = MagicMock()
        orch.buffer.is_initialized = True

        orch._phase_executor.execute = MagicMock(
            side_effect=TrainingFailedError(detail="boom in phase"),
        )

        phase = SimpleNamespace(strategy_type="sft")
        with pytest.raises(TrainingFailedError):
            orch.run_single_phase(phase_idx=0, phase=phase, model=None)

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


class TestRaiseContract:
    """7-class coverage for the new raise-based public surface.

    Pins the contract described in :class:`StrategyOrchestrator` docstring:
    no strategies → ``StrategyChainInvalidError``; downstream errors
    propagate untouched; signal handlers are always unregistered.
    """

    def test_positive_success_returns_model(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        buf = SimpleNamespace()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)
        final = SimpleNamespace()
        orch._chain_runner.run = MagicMock(return_value=final)
        assert orch.run_chain(strategies=[MagicMock(strategy_type="sft")]) is final

    def test_negative_no_strategies_raises(self) -> None:
        orch = _mk_orchestrator(cfg=_mk_cfg(strategies=[]))
        with pytest.raises(StrategyChainInvalidError):
            orch.run_chain(strategies=None)

    def test_boundary_empty_strategies_arg_falls_back_to_config(self) -> None:
        # When ``strategies=None`` we read from config; if config is also
        # empty we raise.
        orch = _mk_orchestrator(cfg=_mk_cfg(strategies=[]))
        with pytest.raises(StrategyChainInvalidError):
            orch.run_chain(strategies=[])

    def test_invariant_legacy_code_preserved(self) -> None:
        orch = _mk_orchestrator(cfg=_mk_cfg(strategies=[]))
        try:
            orch.run_chain(strategies=None)
        except StrategyChainInvalidError as exc:
            assert exc.context["legacy_code"] == "TRAINING_NO_STRATEGIES"

    def test_dependency_error_chain_runner_failure_propagates(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        buf = SimpleNamespace()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)
        sentinel = TrainingFailedError(detail="downstream")
        orch._chain_runner.run = MagicMock(side_effect=sentinel)
        with pytest.raises(TrainingFailedError) as exc:
            orch.run_chain(strategies=[MagicMock(strategy_type="sft")])
        assert exc.value is sentinel

    def test_regression_run_chain_returns_model_not_result(self) -> None:
        """Pin: ``run_chain`` does not return a ``Result`` wrapper post-Batch-14."""
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        buf = SimpleNamespace()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=False)
        model_sentinel = SimpleNamespace(name="final")
        orch._chain_runner.run = MagicMock(return_value=model_sentinel)
        out = orch.run_chain(strategies=[MagicMock(strategy_type="sft")])
        # If we accidentally re-introduced Result wrapping, this assertion would fail.
        assert not hasattr(out, "is_failure")
        assert not hasattr(out, "unwrap")
        assert out is model_sentinel

    def test_combinatorial_resume_complete_skips_chain_runner(self) -> None:
        cfg = _mk_cfg(strategies=[MagicMock(strategy_type="sft")])
        orch = _mk_orchestrator(cfg=cfg)
        buf = SimpleNamespace()
        orch._resume_manager.setup_buffer = MagicMock(return_value=(buf, 0, False))
        orch._resume_manager.was_interrupted = MagicMock(return_value=False)
        orch._resume_manager.is_all_complete = MagicMock(return_value=True)
        orch._chain_runner.run = MagicMock()
        out = orch.run_chain(strategies=[MagicMock(strategy_type="sft")], resume=True)
        assert out is orch.model
        orch._chain_runner.run.assert_not_called()
