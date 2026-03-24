"""
ChainRunner - Run sequence of training phases.

Single Responsibility: Iterate over phases and coordinate execution.
Integrates with MLflow for experiment tracking (parent run for entire chain).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.logger import logger
from src.utils.result import Ok, Result, TrainingError

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from src.training.managers.data_buffer import DataBuffer
    from src.training.orchestrator.phase_executor import PhaseExecutor
    from src.utils.config import StrategyPhaseConfig
    from src.utils.container import IMLflowManager


class ChainRunner:
    """
    Runs a sequence of training phases.

    Responsibilities:
    - Iterate over phases from start_phase to end
    - Call PhaseExecutor for each phase
    - Pass model from one phase to next
    - Log progress and handle errors
    - MLflow integration: parent run for entire chain

    Example:
        runner = ChainRunner(phase_executor, mlflow_manager=mlflow_manager)
        result = runner.run(
            strategies=strategies,
            model=model,
            buffer=buffer,
            start_phase=0,
        )
    """

    def __init__(
        self,
        phase_executor: PhaseExecutor,
        *,
        mlflow_manager: IMLflowManager | None = None,
    ):
        """
        Initialize ChainRunner.

        Args:
            phase_executor: PhaseExecutor for running individual phases
            mlflow_manager: Optional MLflowManager for experiment tracking
        """
        self.phase_executor = phase_executor
        self._mlflow_manager = mlflow_manager
        logger.debug(f"[CR:INIT] ChainRunner initialized (mlflow={mlflow_manager is not None})")

    def run(
        self,
        strategies: list[StrategyPhaseConfig],
        model: PreTrainedModel,
        buffer: DataBuffer,
        start_phase: int = 0,
    ) -> Result[PreTrainedModel, TrainingError]:
        """
        Run the training chain from start_phase to completion.

        MLflow parent run should be created by caller (train_v2.py).
        ChainRunner logs chain-level params, PhaseExecutor creates nested runs.

        Args:
            strategies: List of strategy phases to execute
            model: Initial model to train
            buffer: DataBuffer for state management
            start_phase: Phase index to start from (for resume)

        Returns:
            Result[PreTrainedModel, TrainingError]: Final trained model or error
        """
        total_phases = len(strategies)
        chain_str = " → ".join(s.strategy_type.upper() for s in strategies)

        # Log chain-level params to existing MLflow run (created by train_v2.py)
        if self._mlflow_manager is not None and self._mlflow_manager.is_enabled:
            self._mlflow_manager.log_params(
                {
                    "chain": chain_str,
                    "total_phases": total_phases,
                    "start_phase": start_phase,
                    "run_id": buffer.run_id,
                }
            )
            self._mlflow_manager.set_tags(
                {
                    "chain_type": chain_str,
                    "phases_count": str(total_phases),
                }
            )

        return self._run_phases(strategies, model, buffer, start_phase, total_phases)

    def _run_phases(
        self,
        strategies: list[StrategyPhaseConfig],
        model: PreTrainedModel,
        buffer: DataBuffer,
        start_phase: int,
        total_phases: int,
    ) -> Result[PreTrainedModel, TrainingError]:
        """
        Execute phases sequentially.

        Extracted to allow clean MLflow context management.
        """
        current_model = model
        chain_str = " → ".join(s.strategy_type.upper() for s in strategies)

        logger.debug(f"[CR:RUN_START] chain={chain_str}, start={start_phase}, total={total_phases}")

        for idx in range(start_phase, total_phases):
            phase = strategies[idx]

            # Log phase header
            self._log_phase_header(idx, total_phases, phase)

            # Execute phase
            logger.debug(f"[CR:PHASE_EXECUTING] idx={idx}")
            result = self.phase_executor.execute(
                phase_idx=idx,
                phase=phase,
                model=current_model,
                buffer=buffer,
            )

            # Handle result
            if result.is_failure():
                logger.debug(f"[CR:PHASE_FAILED] idx={idx}, error={result.error}")  # type: ignore[union-attr]
                logger.error(f"❌ Phase {idx} ({phase.strategy_type}) failed")
                # Log failure to MLflow if enabled
                if self._mlflow_manager is not None:
                    self._mlflow_manager.set_tags({"status": "failed", "failed_phase": str(idx)})
                return result

            current_model = result.unwrap()
            logger.debug(f"[CR:PHASE_SUCCESS] idx={idx}")
            logger.info(f"✅ Phase {idx} ({phase.strategy_type}) completed\n")

        logger.debug(f"[CR:RUN_COMPLETE] run_id={buffer.run_id}, phases={total_phases}")
        logger.info(f"Training chain completed: {buffer.run_id}")

        # Log event: chain completed via MLflow
        if self._mlflow_manager is not None:
            self._mlflow_manager.log_event_complete(
                f"Training chain completed: {chain_str}",
                category="training",
                source="ChainRunner",
                run_id=buffer.run_id,
                total_phases=total_phases,
            )
            self._mlflow_manager.set_tags({"status": "completed"})

        return Ok(current_model)

    @staticmethod
    def _log_phase_header(
        idx: int,
        total: int,
        phase: StrategyPhaseConfig,
    ) -> None:
        """Log phase start header."""
        epochs = phase.hyperparams.epochs
        lr = phase.hyperparams.learning_rate
        epochs_str = str(epochs) if epochs is not None else "default"
        lr_str = str(lr) if lr is not None else "default"

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📍 Phase {idx + 1}/{total}: {phase.strategy_type.upper()}")
        logger.info(f"   Epochs: {epochs_str}, LR: {lr_str}")
        logger.info(f"{'=' * 60}")

    @staticmethod
    def get_remaining_phases(
        strategies: list[StrategyPhaseConfig],
        start_phase: int,
    ) -> list[StrategyPhaseConfig]:
        """
        Get list of remaining phases to execute.

        Args:
            strategies: Full list of strategies
            start_phase: Starting phase index

        Returns:
            List of remaining strategy phases
        """
        remaining = strategies[start_phase:]
        logger.debug(f"[CR:REMAINING] start={start_phase}, remaining={len(remaining)}, total={len(strategies)}")
        return remaining


__all__ = ["ChainRunner"]
