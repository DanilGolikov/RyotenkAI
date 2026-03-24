"""
ResumeManager - Handle checkpoint resume logic for multi-phase training.

Single Responsibility: Resume state management and model loading from checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from src.training.managers.data_buffer import DataBuffer, DataBufferEventCallbacks
from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, TrainingError

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from src.utils.config import PipelineConfig, StrategyPhaseConfig


class ResumeManager:
    """
    Manages resume logic for interrupted training runs.

    Responsibilities:
    - Determine if a previous run can be resumed
    - Find the correct phase to resume from
    - Load model from checkpoint
    - Initialize or restore DataBuffer state

    Example:
        manager = ResumeManager(config)
        resume_info = manager.get_resume_info(strategies)
        if resume_info.can_resume:
            model = manager.load_checkpoint_model(resume_info.checkpoint_path)
    """

    def __init__(
        self,
        config: PipelineConfig,
        data_buffer_callbacks: DataBufferEventCallbacks | None = None,
    ):
        """
        Initialize ResumeManager.

        Args:
            config: Pipeline configuration
            data_buffer_callbacks: Optional callbacks for DataBuffer events
        """
        self.config = config
        self._data_buffer_callbacks = data_buffer_callbacks
        logger.debug("[RM:INIT] ResumeManager initialized")

    def setup_buffer(
        self,
        strategies: list[StrategyPhaseConfig],
        *,
        resume: bool = False,
        run_id: str | None = None,
    ) -> tuple[DataBuffer, int, bool]:
        """
        Setup DataBuffer and determine resume state.

        Args:
            strategies: List of strategy phases
            resume: Whether to attempt resume
            run_id: Optional run ID for reproducibility

        Returns:
            Tuple of (buffer, start_phase_idx, should_load_checkpoint)
        """
        buffer = DataBuffer(
            base_output_dir="output",
            base_model_path=self.config.model.name,
            run_id=run_id,
            callbacks=self._data_buffer_callbacks,
        )

        start_phase = 0
        should_load_checkpoint = False

        if resume:
            state_loaded = buffer.load_state()
            if state_loaded:
                resume_phase = buffer.get_resume_phase()
                if resume_phase is not None:
                    start_phase = resume_phase
                    should_load_checkpoint = start_phase > 0
                    logger.info(f"📂 Resuming from phase {start_phase}")
                    logger.debug(f"[RM:RESUME] phase={start_phase}, load_checkpoint={should_load_checkpoint}")
                else:
                    logger.info("✅ All phases already completed")
                    logger.debug("[RM:RESUME] All phases complete, nothing to resume")
            else:
                # State file not found or corrupted — start fresh (overwrite state if needed)
                logger.warning("State file missing or corrupted, starting fresh")
                logger.debug("[RM:RESUME] State load failed, initializing fresh pipeline (force=True)")
                buffer.init_pipeline(
                    strategies,
                    global_hyperparams=self.config.training.hyperparams,
                    force=True,
                )
        else:
            buffer.init_pipeline(strategies, global_hyperparams=self.config.training.hyperparams, force=True)
            logger.debug("[RM:INIT_PIPELINE] Fresh pipeline initialized")

        return buffer, start_phase, should_load_checkpoint

    @staticmethod
    def get_checkpoint_path_for_phase(
        buffer: DataBuffer,
        phase_idx: int,
    ) -> str | None:
        """
        Get checkpoint path for loading model before a phase.

        For phase 0, returns None (use base model).
        For phase N > 0, returns checkpoint from phase N-1.

        Args:
            buffer: DataBuffer with state
            phase_idx: Phase index to get checkpoint for

        Returns:
            Checkpoint path or None if using base model
        """
        if phase_idx == 0:
            return None

        # Get checkpoint from previous phase
        checkpoint_path = buffer.get_model_path_for_phase(phase_idx)
        logger.debug(f"[RM:CHECKPOINT_PATH] phase={phase_idx}, path={checkpoint_path}")
        return checkpoint_path

    @staticmethod
    def load_model_from_checkpoint(
        checkpoint_path: str,
        base_model: PreTrainedModel,
    ) -> Result[PreTrainedModel, TrainingError]:
        """
        Load model from checkpoint for resume.

        Handles both PEFT adapters and full model checkpoints.

        Args:
            checkpoint_path: Path to checkpoint directory
            base_model: Base model to load adapters onto

        Returns:
            Result with loaded model or error
        """
        try:
            logger.info(f"   Loading model from: {checkpoint_path}")
            logger.debug(f"[RM:LOAD_CHECKPOINT] path={checkpoint_path}")

            checkpoint_path_obj = Path(checkpoint_path)

            # Check if checkpoint exists
            if not checkpoint_path_obj.exists():
                error_msg = f"Checkpoint not found: {checkpoint_path}"
                logger.error(f"[RM:ERROR] {error_msg}")
                return Err(TrainingError(message=error_msg, code="TRAINING_CHECKPOINT_NOT_FOUND"))

            # Check if it's a PEFT adapter
            if (checkpoint_path_obj / "adapter_config.json").exists():
                from peft import PeftModel

                model = PeftModel.from_pretrained(base_model, checkpoint_path)
                logger.info("   Loaded PEFT adapter from checkpoint")
                logger.debug("[RM:LOADED] type=PEFT_adapter")
                # NOTE: `PeftModel` is not a subclass of `PreTrainedModel`, but it is a valid model for training/inference.
                # We keep the public API typed as `PreTrainedModel` for the orchestrator, and intentionally erase here.
                return Ok(cast("Any", model))

            # For full model checkpoints, we need to reload entirely
            # This is handled by ModelFactory in the orchestrator
            logger.debug("[RM:LOADED] type=full_model (use base)")
            return Ok(base_model)

        except Exception as e:
            error_msg = f"Failed to load model from checkpoint: {e}"
            logger.error(f"[RM:ERROR] {error_msg}")
            return Err(TrainingError(message=error_msg, code="TRAINING_CHECKPOINT_LOAD_FAILED"))

    @staticmethod
    def can_resume(buffer: DataBuffer | None) -> bool:
        """
        Check if there's a previous run that can be resumed.

        Args:
            buffer: DataBuffer to check

        Returns:
            True if resume is possible
        """
        if buffer is None:
            return False
        can = buffer.can_resume()
        logger.debug(f"[RM:CAN_RESUME] result={can}")
        return can

    @staticmethod
    def is_all_complete(buffer: DataBuffer) -> bool:
        """
        Check if all phases are already completed.

        Args:
            buffer: DataBuffer with state

        Returns:
            True if all phases are complete
        """
        resume_phase = buffer.get_resume_phase()
        is_complete = resume_phase is None
        logger.debug(f"[RM:ALL_COMPLETE] result={is_complete}")
        return is_complete

    @staticmethod
    def was_interrupted(buffer: DataBuffer) -> bool:
        """
        Check if previous run was interrupted (SIGINT/SIGTERM).

        Args:
            buffer: DataBuffer with state

        Returns:
            True if previous run was interrupted
        """
        from src.training.managers.data_buffer import PhaseStatus

        # FIX BUG-004: Check if buffer is initialized before accessing state
        if not buffer.is_initialized:
            logger.debug("[RM:WAS_INTERRUPTED] Buffer not initialized, returning False")
            return False

        if buffer.state.status == "interrupted":
            return True

        for phase in buffer.state.phases:
            if phase.status == PhaseStatus.INTERRUPTED:
                logger.debug(f"[RM:WAS_INTERRUPTED] phase={phase.phase_idx}")
                return True

        return False

    @staticmethod
    def get_interrupt_info(buffer: DataBuffer) -> dict[str, Any] | None:
        """
        Get information about the interrupted phase.

        Args:
            buffer: DataBuffer with state

        Returns:
            Dict with interrupt info or None if not interrupted
        """
        from src.training.managers.data_buffer import PhaseStatus

        for phase in buffer.state.phases:
            if phase.status == PhaseStatus.INTERRUPTED:
                return {
                    "phase_idx": phase.phase_idx,
                    "strategy_type": phase.strategy_type,
                    "reason": phase.error_message,
                    "checkpoint_path": phase.checkpoint_path,
                    "interrupted_at": phase.completed_at.isoformat() if phase.completed_at else None,
                }

        return None


__all__ = ["ResumeManager"]
