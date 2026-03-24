"""
Integration tests: PhaseExecutor + DataBuffer + MemoryManager (training start/restart points).

Goal:
- Cover the real runtime handshake between DataBuffer resume checkpoint discovery and PhaseExecutor.train(...).
- Verify MemoryManager.safe_operation is used around create_trainer and train.

We keep everything heavy mocked (datasets/model/trainer), but use a *real* DataBuffer on disk.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.training.managers.data_buffer import DataBuffer, PhaseStatus
from src.training.orchestrator.phase_executor import PhaseExecutor
from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
from src.utils.result import Ok


def test_phase_executor_passes_latest_resume_checkpoint_from_data_buffer(tmp_path: Path) -> None:
    """
    Restart flow (mid-phase):
    - DataBuffer finds latest checkpoint-N inside phase output dir
    - PhaseExecutor passes it to trainer.train(resume_from_checkpoint=...)
    - MemoryManager.safe_operation wraps create_trainer and train
    """
    # 1) Real DataBuffer state
    phase = StrategyPhaseConfig(
        strategy_type="sft",
        dataset="default",
        hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=1e-5),
    )
    buffer = DataBuffer(
        base_output_dir=tmp_path / "checkpoints",
        base_model_path="base-model",
        run_id="run_test_phase_executor",
    )
    buffer.init_pipeline([phase], force=True)

    phase_out = Path(buffer.get_phase_output_dir(0))
    # Simulate interrupted training checkpoints (no checkpoint-final yet)
    (phase_out / "checkpoint-2").mkdir(parents=True, exist_ok=True)
    (phase_out / "checkpoint-10").mkdir(parents=True, exist_ok=True)

    expected_resume = str(phase_out / "checkpoint-10")

    # 2) MemoryManager spy (record operations)
    entered_ops: list[str] = []

    memory_manager = MagicMock()
    # PhaseExecutor uses with_memory_protection decorators; make them pass-through for this test
    # while still recording the operation names.
    def _with_memory_protection(name: str, **_kwargs):
        entered_ops.append(name)

        def decorator(func):
            return func

        return decorator

    memory_manager.with_memory_protection = _with_memory_protection

    # 3) Training dependencies (mocked)
    dataset_loader = MagicMock()
    dataset_loader.load_for_phase.return_value = Ok((MagicMock(), None))

    strategy_factory = MagicMock()
    strategy = MagicMock()
    strategy.validate_dataset.return_value = Ok(True)
    prepared = MagicMock()
    prepared.__len__.return_value = 5
    strategy.prepare_dataset.return_value = Ok(prepared)
    strategy_factory.create_from_phase.return_value = strategy

    trainer_factory = MagicMock()
    trainer = MagicMock()
    trainer.model = MagicMock(name="trained_model")
    trainer.train = MagicMock()
    trainer.save_model = MagicMock()
    trainer_factory.create_from_phase.return_value = trainer

    metrics_collector = MagicMock()
    metrics_collector.extract_from_trainer.return_value = {"train_loss": 0.123}

    executor = PhaseExecutor(
        tokenizer=MagicMock(),
        config=MagicMock(),
        memory_manager=memory_manager,
        dataset_loader=dataset_loader,
        metrics_collector=metrics_collector,
        strategy_factory=strategy_factory,
        trainer_factory=trainer_factory,
        shutdown_handler=None,
        mlflow_manager=None,
    )

    # 4) Execute phase (should pick resume checkpoint and pass it to trainer.train)
    result = executor.execute(
        phase_idx=0,
        phase=phase,
        model=MagicMock(),
        buffer=buffer,
    )

    assert result.is_ok()

    # TrainerFactory invoked with per-phase output_dir (set by DataBuffer/PhaseExecutor)
    trainer_factory.create_from_phase.assert_called_once()
    assert trainer_factory.create_from_phase.call_args.kwargs.get("output_dir") == str(phase_out)

    # Resume checkpoint must be passed
    trainer.train.assert_called_once()
    _args, kwargs = trainer.train.call_args
    assert kwargs["resume_from_checkpoint"] == expected_resume

    # Memory protection wraps create_trainer and train
    assert "create_trainer_phase_0" in entered_ops
    assert "train_phase_0" in entered_ops

    # Final checkpoint is saved, and DataBuffer state is updated
    trainer.save_model.assert_called_once()
    save_target = trainer.save_model.call_args[0][0]
    assert save_target.endswith("checkpoint-final")

    assert buffer.state.phases[0].status == PhaseStatus.COMPLETED


