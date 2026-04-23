"""
MetricsCollector - Extract and aggregate training metrics.

Single Responsibility: Metrics extraction from trainers after training.
"""

from __future__ import annotations

from typing import Any

from src.training.constants import (
    KEY_EVAL_LOSS,
    KEY_LEARNING_RATE,
    KEY_TRAIN_LOSS,
    KEY_TRAIN_RUNTIME,
    KEY_TRAIN_SAMPLES_PER_SECOND,
    KEY_TRAIN_STEPS_PER_SECOND,
)
from src.training.metrics_models import PhasesMetricsAggregate, TrainingMetricsSnapshot
from src.utils.logger import logger


def _as_float(value: Any) -> float | None:
    """Coerce trainer log value to a plain float. Handles numpy/torch scalars."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MetricsCollector:
    """
    Extracts training metrics from TRL trainers.

    Responsibilities:
    - Extract loss, runtime, steps from trainer state
    - Format metrics for storage in DataBuffer
    - Aggregate metrics across phases (optional)

    Example:
        collector = MetricsCollector()
        snapshot = collector.extract_from_trainer(trainer)
        print(snapshot.train_loss, snapshot.global_step)
    """

    def __init__(self) -> None:
        """Initialize MetricsCollector."""
        logger.debug("[MC:INIT] MetricsCollector initialized")

    def extract_from_trainer(self, trainer: Any) -> TrainingMetricsSnapshot:
        """
        Extract training metrics from trainer state.

        Searches log_history in reverse order to find the most recent value
        for each metric. This is important because the last log entry may not
        contain all metrics (e.g., it could be a save or eval log).

        Args:
            trainer: TRL trainer after training

        Returns:
            Populated ``TrainingMetricsSnapshot``. Empty snapshot if trainer
            has no state.
        """
        if not hasattr(trainer, "state") or trainer.state is None:
            logger.debug("[MC:NO_STATE] Trainer has no state")
            return TrainingMetricsSnapshot()

        state = trainer.state

        # Pre-fill with fields that come directly from state — more reliable
        snapshot = TrainingMetricsSnapshot(
            global_step=getattr(state, "global_step", None),
            epoch=_as_float(getattr(state, "epoch", None)),
            peak_memory_gb=self._get_peak_memory_gb(),
        )

        # Search log_history in reverse to find most recent values.
        # HuggingFace Trainer logs:
        #   - "loss" = per-step loss (every logging_steps)
        #   - "train_loss" = average loss (in final summary entry)
        log_history = getattr(state, "log_history", None) or []
        for log_entry in reversed(log_history):
            # Priority 1: train_loss from final summary (most reliable)
            if snapshot.train_loss is None and KEY_TRAIN_LOSS in log_entry:
                snapshot.train_loss = _as_float(log_entry[KEY_TRAIN_LOSS])

            # Priority 2: loss from per-step logging (fallback)
            if snapshot.train_loss is None and "loss" in log_entry:
                snapshot.train_loss = _as_float(log_entry["loss"])

            if snapshot.train_runtime is None and KEY_TRAIN_RUNTIME in log_entry:
                snapshot.train_runtime = _as_float(log_entry[KEY_TRAIN_RUNTIME])

            if snapshot.learning_rate is None and KEY_LEARNING_RATE in log_entry:
                snapshot.learning_rate = _as_float(log_entry[KEY_LEARNING_RATE])

            if snapshot.eval_loss is None and KEY_EVAL_LOSS in log_entry:
                snapshot.eval_loss = _as_float(log_entry[KEY_EVAL_LOSS])

            if snapshot.train_samples_per_second is None and KEY_TRAIN_SAMPLES_PER_SECOND in log_entry:
                snapshot.train_samples_per_second = _as_float(log_entry[KEY_TRAIN_SAMPLES_PER_SECOND])

            if snapshot.train_steps_per_second is None and KEY_TRAIN_STEPS_PER_SECOND in log_entry:
                snapshot.train_steps_per_second = _as_float(log_entry[KEY_TRAIN_STEPS_PER_SECOND])

            # Early exit when the core trio has been recovered.
            if (
                snapshot.train_loss is not None
                and snapshot.train_runtime is not None
                and snapshot.learning_rate is not None
            ):
                break

        logger.debug(
            f"[MC:EXTRACTED] loss={snapshot.train_loss}, "
            f"steps={snapshot.global_step}, epoch={snapshot.epoch}, "
            f"peak_mem={snapshot.peak_memory_gb}"
        )

        return snapshot

    @staticmethod
    def _get_peak_memory_gb() -> float | None:
        """
        Get peak GPU memory usage in GB.

        Returns:
            Peak memory in GB or None if CUDA not available
        """
        try:
            import torch as torch_module

            torch: Any = torch_module

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_gb = peak_bytes / (1024**3)
                # Reset peak stats for next phase
                torch.cuda.reset_peak_memory_stats()
                return round(peak_gb, 2)
        except Exception as e:
            logger.debug(f"[MC:PEAK_MEM_ERROR] {e}")
        return None

    @staticmethod
    def aggregate_phases(
        phase_metrics: list[TrainingMetricsSnapshot],
    ) -> PhasesMetricsAggregate:
        """
        Aggregate metrics across multiple phases.

        Args:
            phase_metrics: List of per-phase metric snapshots.

        Returns:
            ``PhasesMetricsAggregate`` with summary fields and the original
            per-phase snapshots. Returns an empty aggregate when the input is
            empty.
        """
        if not phase_metrics:
            return PhasesMetricsAggregate()

        total_steps = sum(m.global_step or 0 for m in phase_metrics)
        total_runtime = sum(m.train_runtime or 0.0 for m in phase_metrics)
        final_loss = phase_metrics[-1].train_loss

        aggregate = PhasesMetricsAggregate(
            total_phases=len(phase_metrics),
            total_steps=total_steps,
            total_runtime_seconds=total_runtime,
            final_loss=final_loss,
            per_phase=list(phase_metrics),
        )

        logger.debug(
            f"[MC:AGGREGATED] phases={aggregate.total_phases}, "
            f"total_steps={aggregate.total_steps}, final_loss={aggregate.final_loss}"
        )

        return aggregate


__all__ = ["MetricsCollector"]
