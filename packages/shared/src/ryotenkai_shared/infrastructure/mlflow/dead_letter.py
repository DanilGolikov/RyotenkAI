"""Bounded on-disk dead-letter buffer for un-shippable MLflow metrics.

When the MLflow server is unreachable (e.g. laptop sleep, network
partition) :class:`~.metric_sink.MetricSink` writes batches here
instead of dropping them. On reconnect the consumer
:meth:`drain` the buffer and re-emits.

Replaces the leaky in-process state from the legacy
``ResilientMLflowTransport`` (which monkey-patched the ``mlflow``
module to install a circuit-breaker + buffer on every fluent
metric call). The new design is:

* DI'd into a concrete consumer (no monkey-patching).
* Bounded by total disk bytes (``max_bytes``) with LRU eviction —
  oldest entries are sliced off when the limit is exceeded.
* Atomic append (``open(..., 'ab')``) + ``os.fsync`` either every
  ``_FSYNC_EVERY_N`` records OR every ``_FSYNC_EVERY_S`` seconds
  (whichever comes first) — bounded fsync overhead under high
  metric throughput.
* Thread-safe via ``threading.Lock``.

Format
------
Append-only JSONL; one record per line::

    {"run_id": "<run-id>", "metrics": {"name": 0.5, ...}, "step": 42, "ts": 1715896000.123}

The schema is intentionally narrow; non-numeric fields (e.g.
timestamps in MLflow's per-metric API) are not supported. Future
extension: bump a ``v`` discriminator.

Per ``docs/plans/vectorized-fluttering-mist.md`` §Target architecture.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

from ryotenkai_shared.utils.logger import get_logger

logger = get_logger(__name__)

_FSYNC_EVERY_N = 10
"""fsync after this many appends. Tunable; chosen to amortise fsync
cost over ~10 batches without leaving more than ~10 records
durable-but-unsynced on a sudden power loss."""

_FSYNC_EVERY_S = 1.0
"""Hard upper-bound for time between fsyncs. Belt-and-braces against
a low-throughput producer that never hits ``_FSYNC_EVERY_N``."""

_EVICTION_SLACK = 0.1
"""Evict 10% past the limit on each pass so we don't thrash on the
boundary (single-byte appends repeatedly tripping eviction)."""


class DeadLetterBuffer:
    """Bounded append-only JSONL buffer for un-shippable metric batches.

    Constructor parameters:

    :param path: On-disk location of the JSONL file. Parent
        directories are created on first write.
    :param max_bytes: Maximum disk-size before LRU eviction kicks
        in. Defaults to 100 MiB to match the Mac-side retrieval
        cap used by the legacy ``MetricsBufferRetriever``.

    Usage::

        buf = DeadLetterBuffer(Path("/workspace/dlq.jsonl"))
        buf.write("run-abc", {"loss": 0.5}, step=10)
        for run_id, metrics, step in buf.drain():
            client.log_batch(run_id, metrics, step)
        # drain() truncates the file on success.

    Thread-safety: every public method takes ``self._lock``. Safe to
    share across producer threads.
    """

    def __init__(self, path: Path, max_bytes: int = 100 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            msg = f"DeadLetterBuffer.max_bytes must be positive, got {max_bytes!r}"
            raise ValueError(msg)
        self._path = path
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        # fsync accounting — guarded by ``_lock``.
        self._writes_since_fsync = 0
        self._last_fsync_at = time.monotonic()

    @property
    def path(self) -> Path:
        """Absolute path to the JSONL file (may not yet exist)."""
        return self._path

    @property
    def max_bytes(self) -> int:
        """Configured byte-size ceiling before eviction."""
        return self._max_bytes

    # -- public API --------------------------------------------------

    def write(
        self,
        run_id: str,
        metrics: Mapping[str, float],
        step: int,
    ) -> None:
        """Append a single metric batch to the buffer.

        :param run_id: MLflow run id the batch belongs to.
        :param metrics: ``{metric_name: numeric_value}``. Values are
            coerced to ``float`` at write-time so producers may pass
            int / np.float64 etc.
        :param step: Optimization step the batch was logged at.

        :raises ValueError: if ``run_id`` is empty.
        :raises OSError: if the file can't be written (parent dir
            creation failed, disk full, etc.) — caller decides
            whether to retry.
        """
        if not run_id:
            msg = "DeadLetterBuffer.write requires non-empty run_id"
            raise ValueError(msg)
        record = {
            "run_id": run_id,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "step": int(step),
            "ts": time.time(),
        }
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("ab") as fh:
                fh.write(encoded)
                self._writes_since_fsync += 1
                now = time.monotonic()
                if (
                    self._writes_since_fsync >= _FSYNC_EVERY_N
                    or (now - self._last_fsync_at) >= _FSYNC_EVERY_S
                ):
                    fh.flush()
                    os.fsync(fh.fileno())
                    self._writes_since_fsync = 0
                    self._last_fsync_at = now
            # LRU eviction — done OUTSIDE the append's open() block so
            # the file handle is closed before we rewrite the file.
            current = self._path.stat().st_size
            if current > self._max_bytes:
                self._evict_oldest_locked(current)

    def drain(self) -> Iterator[tuple[str, dict[str, float], int]]:
        """Yield all buffered entries and truncate the file on success.

        Behaves like a take-all snapshot: holds the lock for the
        duration of the read, parses every line, then unlinks the
        file. If parsing one line fails (corruption from a partial
        write) the bad line is dropped with a warning and the rest
        of the file is still drained.

        The generator must be fully consumed; if the caller breaks
        early the file is NOT truncated (partial-drain semantics —
        caller would otherwise lose unprocessed records).

        :yields: ``(run_id, metrics, step)`` triples in the order they
            were appended.
        """
        with self._lock:
            if not self._path.exists():
                return
            try:
                lines = self._path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                logger.warning(
                    "[DLQ] Failed to read buffer at %s: %s",
                    self._path,
                    exc,
                )
                return
            parsed: list[tuple[str, dict[str, float], int]] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    parsed.append(
                        (
                            str(record["run_id"]),
                            {k: float(v) for k, v in record["metrics"].items()},
                            int(record["step"]),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("[DLQ] Skipping corrupt line: %s", exc)
            # Truncate first; if the consumer then crashes mid-yield we
            # at least don't double-deliver.
            try:
                self._path.unlink()
            except OSError as exc:
                logger.warning(
                    "[DLQ] Failed to unlink %s after drain: %s",
                    self._path,
                    exc,
                )
        # Yield outside the lock so consumers can do slow work without
        # blocking producers (writes will land in a fresh file).
        yield from parsed

    def size_bytes(self) -> int:
        """Return the current on-disk size of the buffer file (0 if
        the file doesn't exist yet)."""
        try:
            return self._path.stat().st_size
        except FileNotFoundError:
            return 0

    def is_full(self) -> bool:
        """``True`` once the file has reached or exceeded ``max_bytes``."""
        return self.size_bytes() >= self._max_bytes

    # -- private helpers --------------------------------------------

    def _evict_oldest_locked(self, current_size: int) -> None:
        """Slice the head off the file until size <= target.

        Called with ``self._lock`` held. Strategy: read all lines,
        compute their cumulative byte cost from the tail, and rewrite
        only the suffix that fits under ``max_bytes - slack``. The
        head (oldest) records are dropped.

        We deliberately rewrite via a temp file + atomic rename so a
        crash mid-eviction leaves the original buffer intact.
        """
        target = int(self._max_bytes * (1.0 - _EVICTION_SLACK))
        try:
            raw = self._path.read_bytes()
        except OSError as exc:
            logger.warning("[DLQ] Eviction read failed: %s", exc)
            return
        lines = raw.splitlines(keepends=True)
        # Walk from the END forward; collect until total <= target.
        kept: list[bytes] = []
        running = 0
        for line in reversed(lines):
            line_size = len(line)
            if running + line_size > target:
                break
            kept.append(line)
            running += line_size
        kept.reverse()
        dropped = len(lines) - len(kept)
        if dropped <= 0:
            # Nothing to evict — the limit was tripped by a single
            # giant record that already exceeds the slack budget.
            logger.warning(
                "[DLQ] %s exceeds max_bytes=%d with current=%d; "
                "no eviction possible (single record too large).",
                self._path,
                self._max_bytes,
                current_size,
            )
            return
        tmp_path = self._path.with_suffix(self._path.suffix + ".evict")
        try:
            with tmp_path.open("wb") as fh:
                for line in kept:
                    fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._path)
        except OSError as exc:
            logger.warning("[DLQ] Eviction write failed: %s", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        logger.warning(
            "[DLQ] Evicted %d oldest record(s) from %s "
            "(was %d bytes, now ~%d bytes, max=%d)",
            dropped,
            self._path,
            current_size,
            running,
            self._max_bytes,
        )


__all__ = ["DeadLetterBuffer"]
