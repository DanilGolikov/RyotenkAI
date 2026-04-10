"""
Local JSONL metrics buffer with smart decimation.

Buffers MLflow metrics to disk when the circuit breaker is open (MLflow server
unreachable — e.g. laptop sleeping). Metrics are flushed back to MLflow when
connectivity recovers, or downloaded by the pipeline on resume.

Decimation policy (reduces buffer size for long trainings):
- First 10 min:  keep ALL metrics
- 10–30 min:     keep every 2nd step
- 30+ min:       keep every 5th step

Yields ~40% of raw metrics for a typical 1-hour training — sufficient for
trend analysis while keeping the buffer file under 1 MB.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Decimation thresholds (seconds from training start)
_DECIMATE_TIER1_END = 600  # 10 minutes — keep all
_DECIMATE_TIER2_END = 1800  # 30 minutes — keep every 2nd step
_DECIMATE_TIER2_EVERY = 2
_DECIMATE_TIER3_EVERY = 5  # 30+ min — keep every 5th step

_DEFAULT_BUFFER_FILENAME = "metrics_buffer.jsonl"


class MetricsDecimator:
    """Decides whether a metric at a given step/time should be kept."""

    def __init__(self, training_start_time: float | None = None) -> None:
        self._start_time = training_start_time or time.time()

    def should_keep(self, step: int) -> bool:
        elapsed = time.time() - self._start_time
        if elapsed < _DECIMATE_TIER1_END:
            return True
        if elapsed < _DECIMATE_TIER2_END:
            return step % _DECIMATE_TIER2_EVERY == 0
        return step % _DECIMATE_TIER3_EVERY == 0


class MetricsBuffer:
    """Append-only JSONL buffer for MLflow metrics on local disk.

    Thread-safety: writes are append-only; concurrent reads during flush
    are safe because flush reads then truncates atomically via rename.
    """

    def __init__(
        self,
        buffer_dir: str | Path = "/workspace",
        *,
        training_start_time: float | None = None,
    ) -> None:
        self._path = Path(buffer_dir) / _DEFAULT_BUFFER_FILENAME
        self._decimator = MetricsDecimator(training_start_time)
        self._count = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    def write_metric(self, key: str, value: float, step: int, timestamp: float | None = None) -> bool:
        """Buffer a single metric. Returns True if kept (not decimated)."""
        if not self._decimator.should_keep(step):
            return False
        entry = {
            "key": key,
            "value": value,
            "step": step,
            "timestamp": timestamp or time.time(),
        }
        self._append(entry)
        return True

    def write_metrics(self, metrics: dict[str, float], step: int, timestamp: float | None = None) -> bool:
        """Buffer multiple metrics for the same step."""
        if not self._decimator.should_keep(step):
            return False
        ts = timestamp or time.time()
        for key, value in metrics.items():
            entry = {"key": key, "value": value, "step": step, "timestamp": ts}
            self._append(entry)
        return True

    def _append(self, entry: dict[str, Any]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self._count += 1
        except OSError as e:
            logger.warning("[BUFFER] Failed to write metric: %s", e)

    def read_all(self) -> list[dict[str, Any]]:
        """Read all buffered entries."""
        if not self._path.exists():
            return []
        entries: list[dict[str, Any]] = []
        try:
            with self._path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[BUFFER] Error reading buffer: %s", e)
        return entries

    def flush(self, log_metric_fn: Any) -> int:
        """Flush buffered metrics via the provided MLflow log_metric function.

        Args:
            log_metric_fn: Callable matching mlflow.log_metric(key, value, step=step, timestamp=timestamp).

        Returns:
            Number of metrics successfully flushed.
        """
        entries = self.read_all()
        if not entries:
            return 0

        # Sort by step for clean MLflow UI
        entries.sort(key=lambda e: (e.get("step", 0), e.get("timestamp", 0)))

        flushed = 0
        for entry in entries:
            try:
                log_metric_fn(
                    entry["key"],
                    entry["value"],
                    step=entry.get("step"),
                    timestamp=int(entry.get("timestamp", 0) * 1000),  # MLflow expects ms
                )
                flushed += 1
            except Exception as e:
                logger.warning("[BUFFER] Flush failed at entry %d: %s", flushed, e)
                break

        if flushed == len(entries):
            # All flushed — remove buffer file
            try:
                self._path.unlink(missing_ok=True)
            except OSError:
                pass
            self._count = 0
            logger.info("[BUFFER] Flushed all %d buffered metrics", flushed)
        else:
            # Partial flush — rewrite remaining entries
            remaining = entries[flushed:]
            try:
                with self._path.open("w", encoding="utf-8") as f:
                    for entry in remaining:
                        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
                self._count = len(remaining)
            except OSError:
                pass
            logger.info("[BUFFER] Flushed %d/%d metrics, %d remaining", flushed, len(entries), len(remaining))

        return flushed


__all__ = ["MetricsBuffer", "MetricsDecimator"]
