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

from ryotenkai_shared.errors import ModelLoadFailedError
from ryotenkai_shared.utils.logger import logger

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
        path = saver.save_model(model, tokenizer, "output/model")
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
    ) -> str:
        """
        Save model and tokenizer.

        Args:
            model: Trained model to save
            tokenizer: Tokenizer to save
            output_dir: Output directory path
            save_tokenizer: Whether to save tokenizer

        Returns:
            Output path where the model was written.

        Raises:
            ModelLoadFailedError: model/tokenizer artefact write failed
                (disk full, permission denied, transformers/save bug).
                Legacy error codes preserved on ``context["legacy_code"]``.
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

            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelLoadFailedError(
                detail=f"Model save failed: {e!s}",
                context={
                    "legacy_code": "MODEL_SAVE_FAILED",
                    "output_dir": output_dir,
                },
                cause=e,
            )

    def save_checkpoint(
        self, model: PreTrainedModel, checkpoint_dir: str, step: int | None = None, epoch: int | None = None
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            model: Model to checkpoint
            checkpoint_dir: Base checkpoint directory
            step: Optional training step
            epoch: Optional epoch number

        Returns:
            Filesystem path to the checkpoint directory.

        Raises:
            ModelLoadFailedError: checkpoint write failed. Legacy error
                codes preserved on ``context["legacy_code"]``.
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

            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise ModelLoadFailedError(
                detail=f"Checkpoint save failed: {e!s}",
                context={
                    "legacy_code": "CHECKPOINT_SAVE_FAILED",
                    "checkpoint_dir": checkpoint_dir,
                    "step": step,
                    "epoch": epoch,
                },
                cause=e,
            )

    def cleanup_checkpoints(self, checkpoint_dir: str, keep_last_n: int = 3) -> int:
        """
        Clean up old checkpoints, keeping only the last N.

        Args:
            checkpoint_dir: Checkpoint directory
            keep_last_n: Number of checkpoints to keep

        Returns:
            Number of checkpoints deleted (``0`` if the directory does
            not exist).

        Raises:
            ModelLoadFailedError: directory listing failed. Per-checkpoint
                ``rmtree`` failures are logged at WARNING and do NOT
                escalate (best-effort cleanup, matches pre-migration
                semantics).
        """
        try:
            checkpoint_path = Path(checkpoint_dir)

            if not checkpoint_path.exists():
                return 0

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

            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            raise ModelLoadFailedError(
                detail=f"Checkpoint cleanup failed: {e!s}",
                context={
                    "legacy_code": "CHECKPOINT_CLEANUP_FAILED",
                    "checkpoint_dir": checkpoint_dir,
                },
                cause=e,
            )


__all__ = ["ModelSaverManager"]
