"""
ChainRunner - Run sequence of training phases.

Single Responsibility: Iterate over phases and coordinate execution.
Integrates with MLflow for experiment tracking (parent run for entire chain).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.errors import RyotenkAIError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ryotenkai_pod.trainer.managers.data_buffer import DataBuffer
    from ryotenkai_pod.trainer.orchestrator.phase_executor import PhaseExecutor
    from ryotenkai_shared.config import StrategyPhaseConfig


class ChainRunner:
    """
    Runs a sequence of training phases.

    Responsibilities:
    - Iterate over phases from start_phase to end
    - Call PhaseExecutor for each phase
    - Pass model from one phase to next
    - Log progress and surface errors via raised exceptions
    - MLflow integration: parent run for entire chain

    Raise contract (post Phase A2 Batch 14):
    - Success: returns final ``PreTrainedModel``.
    - Failure: propagates the typed :class:`RyotenkAIError` raised by the
      underlying :class:`PhaseExecutor`. The MLflow "failed" tags are
      logged on the way out before the exception is re-raised.

    Example:
        runner = ChainRunner(phase_executor)
        try:
            model = runner.run(strategies=strategies, model=model, buffer=buffer)
        except TrainingFailedError as exc:
            ...
    """

    def __init__(self, phase_executor: PhaseExecutor):
        """Initialize :class:`ChainRunner`.

        :param phase_executor: :class:`PhaseExecutor` for running
            individual phases.
        """
        self.phase_executor = phase_executor
        logger.debug("[CR:INIT] ChainRunner initialized")

    def run(
        self,
        strategies: list[StrategyPhaseConfig],
        model: PreTrainedModel,
        buffer: DataBuffer,
        start_phase: int = 0,
    ) -> PreTrainedModel:
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
            Final trained ``PreTrainedModel``.

        Raises:
            RyotenkAIError: surfaced from the underlying ``PhaseExecutor``.
        """
        total_phases = len(strategies)

        # Chain-level params/tags were historically logged to the wide
        # ``IMLflowManager`` here. The wide manager has been retired;
        # chain identity is now carried by :class:`TrainingStartedEvent`
        # on the typed journal and by the parent attempt run tags set
        # in :func:`open_attempt_with_coord`.

        return self._run_phases(strategies, model, buffer, start_phase, total_phases)

    def _run_phases(
        self,
        strategies: list[StrategyPhaseConfig],
        model: PreTrainedModel,
        buffer: DataBuffer,
        start_phase: int,
        total_phases: int,
    ) -> PreTrainedModel:
        """
        Execute phases sequentially.

        Extracted to allow clean MLflow context management.
        """
        current_model = model
        chain_str = " → ".join(s.strategy_type.upper() for s in strategies)

        logger.debug(f"[CR:RUN_START] chain={chain_str}, start={start_phase}, total={total_phases}")

        upstream_retrained = False

        for idx in range(start_phase, total_phases):
            phase = strategies[idx]

            # Log phase header
            self._log_phase_header(idx, total_phases, phase)

            # Execute phase (pass upstream_retrained for cascade cache invalidation)
            logger.debug(f"[CR:PHASE_EXECUTING] idx={idx}, upstream_retrained={upstream_retrained}")
            try:
                current_model = self.phase_executor.execute(
                    phase_idx=idx,
                    phase=phase,
                    model=current_model,
                    buffer=buffer,
                    upstream_retrained=upstream_retrained,
                )
            except RyotenkAIError as exc:
                logger.debug(f"[CR:PHASE_FAILED] idx={idx}, error={exc}")
                logger.error(f"Phase {idx} ({phase.strategy_type}) failed")
                # Failure tag was historically logged via the wide
                # ``IMLflowManager``. After retirement, the failure is
                # captured by ``TrainingFailedEvent`` on the typed
                # journal and the orchestrator's coord finalize.
                raise

            # Update cascade flag: if phase was actually trained (not skipped), mark upstream as retrained
            from ryotenkai_pod.trainer.managers.data_buffer import PhaseStatus

            phase_status = buffer.state.phases[idx].status
            if phase_status != PhaseStatus.SKIPPED:
                upstream_retrained = True

            logger.debug(f"[CR:PHASE_SUCCESS] idx={idx}, upstream_retrained_now={upstream_retrained}")
            logger.info(f"✅ Phase {idx} ({phase.strategy_type}) completed\n")

        logger.debug(f"[CR:RUN_COMPLETE] run_id={buffer.run_id}, phases={total_phases}")
        logger.info(f"Training chain completed: {buffer.run_id}")

        # Chain-complete signal flows via :class:`TrainingCompletedEvent`
        # on the typed journal; the wide manager's status tag was the
        # legacy carrier and is now redundant.

        return current_model

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
