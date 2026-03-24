"""
Integration tests: StrategyOrchestrator + ResumeManager + DataBuffer.

Goal:
- Cover the real resume decision path:
  StrategyOrchestrator.run_chain(resume=True) -> ResumeManager.setup_buffer() -> DataBuffer.load_state/get_resume_phase
  -> (optional) ResumeManager.load_model_from_checkpoint() -> ChainRunner.run(start_phase=...)

We mock heavy parts (actual training), but keep DataBuffer persistence real.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.training.managers.data_buffer import DataBuffer
from src.training.orchestrator.resume_manager import ResumeManager
from src.training.orchestrator.strategy_orchestrator import StrategyOrchestrator
from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
from src.utils.result import Ok


def test_resume_manager_fresh_init_pipeline_applies_global_hyperparams_to_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Start flow:
    - ResumeManager.setup_buffer(resume=False) must init DataBuffer with global_hyperparams
    - DataBuffer stores epochs fallback from global hyperparams when phase hyperparams.epochs is not set
      (used for summary/debug, strict config still uses phase.hyperparams for overrides elsewhere).
    """
    monkeypatch.chdir(tmp_path)

    # Phase does NOT specify epochs -> should fall back to global_hyperparams.epochs
    strategies = [
        StrategyPhaseConfig(
            strategy_type="sft",
            dataset="default",
            hyperparams=PhaseHyperparametersConfig(learning_rate=1e-5),
        )
    ]

    config = MagicMock()
    config.model.name = "base-model"
    config.training.hyperparams = PhaseHyperparametersConfig(epochs=7, learning_rate=2e-4)

    rm = ResumeManager(config)
    buffer, start_phase, should_load = rm.setup_buffer(strategies, resume=False, run_id="run_start_flow")

    assert start_phase == 0
    assert should_load is False
    assert buffer.state.phases[0].epochs == 7  # fallback from global hyperparams


def test_strategy_orchestrator_run_chain_resume_calls_chain_runner_with_loaded_model_and_start_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Restart flow:
    - There is an existing run with phase0 completed and phase1 failed
    - run_chain(resume=True) must start from phase1 and request checkpoint loading
    - ChainRunner.run must receive start_phase=1 and the loaded model
    """
    run_id = "run_resume_flow"
    run_workspace = tmp_path / run_id
    run_workspace.mkdir(parents=True, exist_ok=True)

    # 1) Prepare a real previous run state on disk
    strategies = [
        StrategyPhaseConfig(
            strategy_type="sft",
            dataset="default",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4),
        ),
        StrategyPhaseConfig(
            strategy_type="dpo",
            dataset="default",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=5e-6, beta=0.1),
        ),
    ]

    buffer = DataBuffer(base_output_dir=run_workspace / "output", base_model_path="base-model", run_id=run_id)
    buffer.init_pipeline(strategies, force=True)

    # Simulate phase0 completed with some checkpoint path
    buffer.mark_phase_started(0)
    buffer.mark_phase_completed(0, checkpoint_path="/tmp/checkpoint-phase0")

    # Simulate phase1 failed -> should resume from phase1
    buffer.mark_phase_started(1)
    buffer.mark_phase_failed(1, "simulated failure")

    # 2) Build orchestrator with mocked heavy deps
    config = MagicMock()
    config.model.name = "base-model"
    config.training.hyperparams = PhaseHyperparametersConfig(epochs=3, learning_rate=2e-4)

    base_model = MagicMock(name="base_model")
    tokenizer = MagicMock(name="tokenizer")

    orchestrator = StrategyOrchestrator(
        model=base_model,
        tokenizer=tokenizer,
        config=config,
        memory_manager=MagicMock(),
        dataset_loader=MagicMock(),
        strategy_factory=MagicMock(),
        trainer_factory=MagicMock(),
        mlflow_manager=None,
        graceful_shutdown=False,
    )

    # Patch the heavy parts of resume + chain execution
    resumed_model = MagicMock(name="resumed_model")
    final_model = MagicMock(name="final_model")

    orchestrator._resume_manager.load_model_from_checkpoint = MagicMock(return_value=Ok(resumed_model))
    orchestrator._chain_runner.run = MagicMock(return_value=Ok(final_model))

    # 3) Run chain in resume mode
    monkeypatch.chdir(run_workspace)
    result = orchestrator.run_chain(strategies=strategies, resume=True, run_id=run_id)

    assert result.is_ok()
    assert result.unwrap() is final_model

    # Ensure checkpoint loading was requested (start_phase > 0)
    orchestrator._resume_manager.load_model_from_checkpoint.assert_called_once()

    # ChainRunner must start from the resume phase and use loaded model
    orchestrator._chain_runner.run.assert_called_once()
    _args, kwargs = orchestrator._chain_runner.run.call_args
    assert kwargs["start_phase"] == 1
    assert kwargs["model"] is resumed_model
    assert kwargs["buffer"] is not None


def test_strategy_orchestrator_resume_with_corrupted_state_file_starts_fresh_and_runs_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Fault-tolerance / restart flow:
    - There is an existing run with a corrupted pipeline_state.json
    - run_chain(resume=True) should NOT crash; it should start fresh (overwrite state) and continue
    """
    run_id = "run_corrupted_state"
    run_workspace = tmp_path / run_id
    run_workspace.mkdir(parents=True, exist_ok=True)

    # Prepare corrupted state file on disk
    output_dir = run_workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / DataBuffer.STATE_FILENAME
    state_file.write_text("{not-json", encoding="utf-8")

    strategies = [
        StrategyPhaseConfig(
            strategy_type="sft",
            dataset="default",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4),
        )
    ]

    config = MagicMock()
    config.model.name = "base-model"
    config.training.hyperparams = PhaseHyperparametersConfig(epochs=3, learning_rate=2e-4)

    base_model = MagicMock(name="base_model")
    tokenizer = MagicMock(name="tokenizer")

    orchestrator = StrategyOrchestrator(
        model=base_model,
        tokenizer=tokenizer,
        config=config,
        memory_manager=MagicMock(),
        dataset_loader=MagicMock(),
        strategy_factory=MagicMock(),
        trainer_factory=MagicMock(),
        mlflow_manager=None,
        graceful_shutdown=False,
    )

    # Track that we don't attempt checkpoint loading for phase0 restart
    orchestrator._resume_manager.load_model_from_checkpoint = MagicMock()

    final_model = MagicMock(name="final_model")
    orchestrator._chain_runner.run = MagicMock(return_value=Ok(final_model))

    monkeypatch.chdir(run_workspace)
    res = orchestrator.run_chain(strategies=strategies, resume=True, run_id=run_id)
    assert res.is_ok()
    assert res.unwrap() is final_model

    orchestrator._resume_manager.load_model_from_checkpoint.assert_not_called()

    orchestrator._chain_runner.run.assert_called_once()
    _args, kwargs = orchestrator._chain_runner.run.call_args
    assert kwargs["start_phase"] == 0
    assert kwargs["model"] is base_model

    # Corrupted state should be overwritten with valid JSON (dict)
    loaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    assert loaded.get("run_id") == run_id

