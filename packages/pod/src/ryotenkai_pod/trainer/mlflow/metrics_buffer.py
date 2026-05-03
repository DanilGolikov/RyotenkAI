"""
Local JSONL metrics buffer with config-driven decimation.

Buffers MLflow metrics to disk when the circuit breaker is open (MLflow server
unreachable — e.g. laptop sleeping). Metrics are flushed back to MLflow when
connectivity recovers, or downloaded by the pipeline on resume (Phase 12.A.1).

Decimation policy
-----------------
Phase 12.A.2 made decimation **config-driven**. By default every metric is
preserved losslessly (``keep_all=true``); set ``training.metrics_buffer.keep_all=false``
to enable adaptive decimation. The legacy three-tier defaults
(``1 / 2 / 5`` keep_every per ``10 / 30 / late`` minute window) ARE preserved
inside :class:`DecimationWindowConfig` — flipping ``keep_all=false`` without
tuning the windows reproduces the pre-12.A.2 behaviour.

Yields ~40% of raw metrics for a typical 1-hour training when configured for
decimation — sufficient for trend analysis while keeping the buffer file under
1 MB. With the new lossless default, expect proportionally larger files
(typical ~2–10 MB on a 1h run) — well within the 100 MiB Mac-side retrieval
cap (Phase 12.A.1 :class:`MetricsBufferRetriever`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ryotenkai_shared.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BUFFER_FILENAME = "metrics_buffer.jsonl"


class MetricsDecimator:
    """Decides whether a metric at a given step/time should be kept.

    Phase 12.A.2: optionally accepts a
    :class:`~src.config.training.metrics_buffer.MetricsBufferConfig`.
    Construction without one keeps the legacy default (``keep_all=True``) so
    existing call sites that didn't propagate the config (tests, ad-hoc
    instantiation) get the **safer, more permissive** lossless behaviour.
    """

    # Defaults retained as module-level constants so tests / fallbacks
    # don't have to spin up a Pydantic model just to query them.
    DEFAULT_WINDOW_FIRST_S: float = 600.0   # 10 min
    DEFAULT_WINDOW_FIRST_KEEP_EVERY: int = 1
    DEFAULT_WINDOW_MID_S: float = 1800.0    # 30 min — boundary between mid/late
    DEFAULT_WINDOW_MID_KEEP_EVERY: int = 2
    DEFAULT_WINDOW_LATE_KEEP_EVERY: int = 5

    def __init__(
        self,
        training_start_time: float | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        # Phase 12.A.2 — explicit `is None` check so a caller that
        # legitimately wants ``training_start_time=0.0`` (tests anchoring
        # to a synthetic clock) doesn't fall through to ``time.time()``.
        # The legacy ``or time.time()`` was subtly buggy because 0.0 is
        # falsy in Python.
        self._start_time = (
            training_start_time
            if training_start_time is not None
            else time.time()
        )
        self._keep_all, self._windows = self._extract_decimation(config)

    @classmethod
    def _extract_decimation(
        cls, config: Any | None,
    ) -> tuple[bool, tuple[float, int, float, int, int]]:
        """Pull the five window numbers out of a
        :class:`MetricsBufferConfig` (Phase 12.A.2).

        Returns the legacy defaults when ``config is None`` so existing
        call sites still work. Read defensively — production code passes
        a Pydantic model, but tests can pass a SimpleNamespace stand-in
        and we don't want to import the Pydantic class here (keeps the
        trainer-side trampoline light).
        """
        # No config → lossless (keep_all=True).
        if config is None:
            return True, (
                cls.DEFAULT_WINDOW_FIRST_S,
                cls.DEFAULT_WINDOW_FIRST_KEEP_EVERY,
                cls.DEFAULT_WINDOW_MID_S,
                cls.DEFAULT_WINDOW_MID_KEEP_EVERY,
                cls.DEFAULT_WINDOW_LATE_KEEP_EVERY,
            )

        keep_all = bool(getattr(config, "keep_all", True))
        decim = getattr(config, "decimation", None)

        first_s = float(
            getattr(decim, "window_first_minutes", 10) * 60.0
            if decim is not None
            else cls.DEFAULT_WINDOW_FIRST_S
        )
        first_keep = int(
            getattr(decim, "window_first_keep_every", cls.DEFAULT_WINDOW_FIRST_KEEP_EVERY)
            if decim is not None
            else cls.DEFAULT_WINDOW_FIRST_KEEP_EVERY
        )
        mid_minutes = (
            float(getattr(decim, "window_mid_minutes", 30))
            if decim is not None
            else 30.0
        )
        # Boundary between mid and late = first + mid (in seconds).
        mid_boundary_s = first_s + mid_minutes * 60.0
        mid_keep = int(
            getattr(decim, "window_mid_keep_every", cls.DEFAULT_WINDOW_MID_KEEP_EVERY)
            if decim is not None
            else cls.DEFAULT_WINDOW_MID_KEEP_EVERY
        )
        late_keep = int(
            getattr(decim, "window_late_keep_every", cls.DEFAULT_WINDOW_LATE_KEEP_EVERY)
            if decim is not None
            else cls.DEFAULT_WINDOW_LATE_KEEP_EVERY
        )

        return keep_all, (first_s, first_keep, mid_boundary_s, mid_keep, late_keep)

    def should_keep(self, step: int) -> bool:
        # Phase 12.A.2 lossless default.
        if self._keep_all:
            return True

        first_s, first_keep, mid_boundary_s, mid_keep, late_keep = self._windows
        elapsed = time.time() - self._start_time

        if elapsed < first_s:
            return step % first_keep == 0
        if elapsed < mid_boundary_s:
            return step % mid_keep == 0
        return step % late_keep == 0


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
        config: Any | None = None,
    ) -> None:
        """
        Args:
            buffer_dir:           Directory holding ``metrics_buffer.jsonl``.
                                  Default ``/workspace`` matches the trainer-
                                  side runtime layout.
            training_start_time:  Wall-clock timestamp anchoring the
                                  decimation windows. ``None`` → ``time.time()``.
            config:               Phase 12.A.2 — optional
                                  :class:`MetricsBufferConfig`. ``None`` =
                                  lossless default (``keep_all=True``);
                                  drives :class:`MetricsDecimator`.
        """
        self._path = Path(buffer_dir) / _DEFAULT_BUFFER_FILENAME
        self._decimator = MetricsDecimator(training_start_time, config=config)
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
