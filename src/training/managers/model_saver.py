"""
Model Saver Manager - Single Responsibility: Model saving and checkpointing

Handles all model persistence operations:
- Saving trained models
- Saving tokenizers
- Checkpoint management
- Model artifacts

Follows Single Responsibility Principle (SOLID).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, TrainingError

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedModel, PreTrainedTokenizer


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


@dataclass
class ModelSaverEventCallbacks:
    """Callbacks for ModelSaver events."""

    on_model_saved: Callable[[str], None] | None = None
    # Args: output_path

    on_checkpoint_saved: Callable[[str, int | None, int | None], None] | None = None
    # Args: path, step, epoch

    on_cleanup_completed: Callable[[int], None] | None = None
    # Args: deleted_count


class ModelSaverManager:
    """
    Manager for model saving and checkpointing.

    Single Responsibility: Handle all model persistence operations.

    Example:
        saver = ModelSaverManager()
        result = saver.save_model(model, tokenizer, "output/model")
        if result.is_success():
            path = result.unwrap()
            print(f"Model saved to {path}")
    """

    def __init__(
        self,
        create_dirs: bool = True,
        callbacks: ModelSaverEventCallbacks | None = None,
    ):
        """
        Initialize model saver manager.

        Args:
            create_dirs: Whether to create output directories automatically
            callbacks: Optional event callbacks
        """
        self.create_dirs = create_dirs
        self._callbacks = callbacks or ModelSaverEventCallbacks()

    def save_model(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str, save_tokenizer: bool = True
    ) -> Result[str, TrainingError]:
        """
        Save model and tokenizer.

        Args:
            model: Trained model to save
            tokenizer: Tokenizer to save
            output_dir: Output directory path
            save_tokenizer: Whether to save tokenizer

        Returns:
            Result[str, TrainingError]: Output path or error
        """
        try:
            output_path = Path(output_dir)

            # Create directory if needed
            if self.create_dirs:
                output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving model to {output_path}...")

            # Save model
            model.save_pretrained(output_path)
            logger.info("Model saved")

            # Save tokenizer
            if save_tokenizer:
                tokenizer.save_pretrained(output_path)
                logger.info("Tokenizer saved")

            logger.info(f"Complete! Model saved to {output_path}")

            # Fire callback
            if self._callbacks.on_model_saved:
                self._callbacks.on_model_saved(str(output_path))

            return Ok(str(output_path))

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return Err(TrainingError(message=f"Model save failed: {e!s}", code="MODEL_SAVE_FAILED"))

    def save_checkpoint(
        self, model: PreTrainedModel, checkpoint_dir: str, step: int | None = None, epoch: int | None = None
    ) -> Result[str, TrainingError]:
        """
        Save a training checkpoint.

        Args:
            model: Model to checkpoint
            checkpoint_dir: Base checkpoint directory
            step: Optional training step
            epoch: Optional epoch number

        Returns:
            Result[str, TrainingError]: Checkpoint path or error
        """
        try:
            # Generate checkpoint name
            if step is not None:
                checkpoint_name = f"checkpoint-step-{step}"
            elif epoch is not None:
                checkpoint_name = f"checkpoint-epoch-{epoch}"
            else:
                checkpoint_name = "checkpoint"

            checkpoint_path = Path(checkpoint_dir) / checkpoint_name

            # Create directory
            if self.create_dirs:
                checkpoint_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving checkpoint to {checkpoint_path}...")

            # Save model
            model.save_pretrained(checkpoint_path)

            logger.info(f"Checkpoint saved to {checkpoint_path}")

            # Fire callback
            if self._callbacks.on_checkpoint_saved:
                self._callbacks.on_checkpoint_saved(str(checkpoint_path), step, epoch)

            return Ok(str(checkpoint_path))

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return Err(TrainingError(message=f"Checkpoint save failed: {e!s}", code="CHECKPOINT_SAVE_FAILED"))

    def cleanup_checkpoints(self, checkpoint_dir: str, keep_last_n: int = 3) -> Result[int, TrainingError]:
        """
        Clean up old checkpoints, keeping only the last N.

        Args:
            checkpoint_dir: Checkpoint directory
            keep_last_n: Number of checkpoints to keep

        Returns:
            Result[int, str]: Number of checkpoints deleted or error
        """
        try:
            checkpoint_path = Path(checkpoint_dir)

            if not checkpoint_path.exists():
                return Ok(0)

            # Find all checkpoint directories
            checkpoints = sorted(
                [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint")],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            # Delete old checkpoints
            deleted = 0
            for checkpoint in checkpoints[keep_last_n:]:
                try:
                    import shutil

                    shutil.rmtree(checkpoint)
                    deleted += 1
                    logger.info(f"Deleted old checkpoint: {checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {checkpoint}: {e}")

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old checkpoints")
                # Fire callback
                if self._callbacks.on_cleanup_completed:
                    self._callbacks.on_cleanup_completed(deleted)

            return Ok(deleted)

        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return Err(TrainingError(message=f"Checkpoint cleanup failed: {e!s}", code="CHECKPOINT_CLEANUP_FAILED"))


__all__ = ["ModelSaverManager"]
