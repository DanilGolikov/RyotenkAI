"""
Training Events Callback for HuggingFace Trainer.

Logs epoch-level events and checkpoint events to MLflow.

Usage:
    from src.training.callbacks.training_events_callback import TrainingEventsCallback

    callback = TrainingEventsCallback(mlflow_manager)
    trainer = SFTTrainer(..., callbacks=[callback])
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from transformers import TrainerCallback

from src.training.constants import (
    MLFLOW_CATEGORY_TRAINING,
    MLFLOW_SOURCE_TRAINING_EVENTS,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

    from src.utils.container import IMLflowManager

logger = get_logger(__name__)


class TrainingEventsCallback(TrainerCallback):
    """
    Callback for logging training events to MLflow.

    Events logged:
    - Epoch started (with epoch number)
    - Epoch completed (with duration)
    - Checkpoint saved (intermediate)
    - Training started/completed
    """

    def __init__(self, mlflow_manager: IMLflowManager | None = None):
        """
        Args:
            mlflow_manager: MLflowManager for event logging
        """
        self._mlflow = mlflow_manager
        self._epoch_start_time: float | None = None
        self._training_start_time: float | None = None
        self._current_epoch: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Log training start event."""
        _ = control, kwargs
        self._training_start_time = time.time()

        if self._mlflow and self._mlflow.is_enabled:
            self._mlflow.log_event_start(
                "Training loop started",
                category=MLFLOW_CATEGORY_TRAINING,
                source=MLFLOW_SOURCE_TRAINING_EVENTS,
                total_steps=state.max_steps,
                num_epochs=args.num_train_epochs,
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Log training end event."""
        _ = args, control, kwargs
        if self._training_start_time:
            total_duration = time.time() - self._training_start_time

            if self._mlflow and self._mlflow.is_enabled:
                self._mlflow.log_event_complete(
                    f"Training loop completed ({total_duration:.1f}s)",
                    category=MLFLOW_CATEGORY_TRAINING,
                    source=MLFLOW_SOURCE_TRAINING_EVENTS,
                    total_duration_seconds=total_duration,
                    final_step=state.global_step,
                    final_epoch=state.epoch,
                )

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Log epoch start event."""
        _ = control, kwargs
        self._epoch_start_time = time.time()
        epoch_raw = state.epoch
        epoch_num = int(epoch_raw) if epoch_raw is not None else 0
        self._current_epoch = epoch_num + 1  # 1-indexed for display

        if self._mlflow and self._mlflow.is_enabled:
            self._mlflow.log_event_start(
                f"Epoch {self._current_epoch}/{int(args.num_train_epochs)} started",
                category=MLFLOW_CATEGORY_TRAINING,
                source=MLFLOW_SOURCE_TRAINING_EVENTS,
                epoch=self._current_epoch,
                total_epochs=int(args.num_train_epochs),
            )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Log epoch end event with duration."""
        _ = args, control, kwargs
        epoch_duration = 0.0
        if self._epoch_start_time:
            epoch_duration = time.time() - self._epoch_start_time

        if self._mlflow and self._mlflow.is_enabled:
            self._mlflow.log_event_complete(
                f"Epoch {self._current_epoch} completed ({epoch_duration:.1f}s)",
                category=MLFLOW_CATEGORY_TRAINING,
                source=MLFLOW_SOURCE_TRAINING_EVENTS,
                epoch=self._current_epoch,
                epoch_duration_seconds=epoch_duration,
                global_step=state.global_step,
            )

            # Log epoch duration as metric for graph
            self._mlflow.log_metrics(
                {"epoch_duration_seconds": epoch_duration},
                step=self._current_epoch,
            )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Log checkpoint saved event."""
        _ = args, control, kwargs
        if self._mlflow and self._mlflow.is_enabled:
            self._mlflow.log_event_checkpoint(
                f"Checkpoint saved at step {state.global_step}",
                category=MLFLOW_CATEGORY_TRAINING,
                source=MLFLOW_SOURCE_TRAINING_EVENTS,
                step=state.global_step,
                epoch=state.epoch,
            )


__all__ = ["TrainingEventsCallback"]
