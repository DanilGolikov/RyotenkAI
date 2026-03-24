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
from src.utils.logger import logger


class MetricsCollector:
    """
    Extracts training metrics from TRL trainers.

    Responsibilities:
    - Extract loss, runtime, steps from trainer state
    - Format metrics for storage in DataBuffer
    - Aggregate metrics across phases (optional)

    Example:
        collector = MetricsCollector()
        metrics = collector.extract_from_trainer(trainer)
        print(metrics)
        # {'train_loss': 0.5, 'global_step': 100, 'epoch': 1.0}
    """

    def __init__(self) -> None:
        """Initialize MetricsCollector."""
        logger.debug("[MC:INIT] MetricsCollector initialized")

    def extract_from_trainer(self, trainer: Any) -> dict[str, Any]:
        """
        Extract training metrics from trainer state.

        Searches log_history in reverse order to find the most recent value
        for each metric. This is important because the last log entry may not
        contain all metrics (e.g., it could be a save or eval log).

        Args:
            trainer: TRL trainer after training

        Returns:
            Dict with metrics (train_loss, train_runtime, global_step, epoch, peak_memory_gb)
        """
        metrics: dict[str, Any] = {}

        if not hasattr(trainer, "state") or trainer.state is None:
            logger.debug("[MC:NO_STATE] Trainer has no state")
            return metrics

        state = trainer.state

        # Search log_history in reverse to find most recent values
        # HuggingFace Trainer logs:
        #   - "loss" = per-step loss (every logging_steps)
        #   - "train_loss" = average loss (in final summary entry)
        if state.log_history:
            for log_entry in reversed(state.log_history):
                # Priority 1: train_loss from final summary (most reliable)
                if KEY_TRAIN_LOSS in log_entry and metrics.get(KEY_TRAIN_LOSS) is None:
                    metrics[KEY_TRAIN_LOSS] = log_entry[KEY_TRAIN_LOSS]

                # Priority 2: loss from per-step logging (fallback)
                if "loss" in log_entry and metrics.get(KEY_TRAIN_LOSS) is None:
                    metrics[KEY_TRAIN_LOSS] = log_entry["loss"]

                # Get train_runtime (logged at end of training)
                if KEY_TRAIN_RUNTIME in log_entry and metrics.get(KEY_TRAIN_RUNTIME) is None:
                    metrics[KEY_TRAIN_RUNTIME] = log_entry[KEY_TRAIN_RUNTIME]

                # Get learning_rate (logged per step)
                if KEY_LEARNING_RATE in log_entry and metrics.get(KEY_LEARNING_RATE) is None:
                    metrics[KEY_LEARNING_RATE] = log_entry[KEY_LEARNING_RATE]

                # Get eval_loss if available
                if KEY_EVAL_LOSS in log_entry and metrics.get(KEY_EVAL_LOSS) is None:
                    metrics[KEY_EVAL_LOSS] = log_entry[KEY_EVAL_LOSS]

                # Get throughput metrics if available
                if KEY_TRAIN_SAMPLES_PER_SECOND in log_entry and metrics.get(KEY_TRAIN_SAMPLES_PER_SECOND) is None:
                    metrics[KEY_TRAIN_SAMPLES_PER_SECOND] = log_entry[KEY_TRAIN_SAMPLES_PER_SECOND]

                if KEY_TRAIN_STEPS_PER_SECOND in log_entry and metrics.get(KEY_TRAIN_STEPS_PER_SECOND) is None:
                    metrics[KEY_TRAIN_STEPS_PER_SECOND] = log_entry[KEY_TRAIN_STEPS_PER_SECOND]

                # Early exit if we found all important metrics
                if all(metrics.get(k) is not None for k in [KEY_TRAIN_LOSS, KEY_TRAIN_RUNTIME, KEY_LEARNING_RATE]):
                    break

        # Always get these from state (more reliable)
        metrics["global_step"] = state.global_step
        metrics["epoch"] = state.epoch

        # Get peak GPU memory if available
        peak_memory_gb = self._get_peak_memory_gb()
        if peak_memory_gb is not None:
            metrics["peak_memory_gb"] = peak_memory_gb

        logger.debug(
            f"[MC:EXTRACTED] loss={metrics.get('train_loss')}, "
            f"steps={metrics.get('global_step')}, epoch={metrics.get('epoch')}, "
            f"peak_mem={metrics.get('peak_memory_gb')}"
        )

        return metrics

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
    def aggregate_phases(phase_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate metrics across multiple phases.

        Args:
            phase_metrics: List of metrics dicts from each phase

        Returns:
            Aggregated metrics summary
        """
        if not phase_metrics:
            return {}

        total_steps = sum(m.get("global_step", 0) for m in phase_metrics)
        total_runtime = sum(m.get("train_runtime", 0) or 0 for m in phase_metrics)
        final_loss = phase_metrics[-1].get("train_loss")

        summary = {
            "total_phases": len(phase_metrics),
            "total_steps": total_steps,
            "total_runtime_seconds": total_runtime,
            "final_loss": final_loss,
            "per_phase": phase_metrics,
        }

        logger.debug(f"[MC:AGGREGATED] phases={len(phase_metrics)}, total_steps={total_steps}, final_loss={final_loss}")

        return summary


__all__ = ["MetricsCollector"]
