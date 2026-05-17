"""Append-only JSONL writer for control-side events (Phase 3).

Single file at ``<run_directory>/events.jsonl`` — control-side journals
do NOT rotate the way pod-side journals do because:

* Control-plane volume is much lower than pod-plane (run lifecycle +
  stage transitions, not per-step training telemetry).
* A single file keeps offset/source bookkeeping trivial during resume
  and replay; rotation can be added later if a run ever overflows.

Wire format: one length-prefixed line per envelope via
:func:`ryotenkai_shared.events.to_jsonl` —
``<utf8_byte_length>\\t<envelope_json>\\n``. Torn writes (kill -9 mid-
``write()``) are detected on resume by :class:`JournalReader`.

Crash-safety contract:

* The file is opened in append mode with line buffering so each
  ``write()`` becomes a kernel-level append. We rely on the OS to keep
  the bytes in the right order; partial writes are tolerated by the
  reader's truncate-torn-tail pass on next open.
* :meth:`append` schedules an fsync inline when ANY of these holds:

    1. severity >= ``error`` (immediate fsync, hot-path);
    2. batch size reached (default 50 events);
    3. interval elapsed (default 1 s since last fsync).

  Otherwise the kernel-buffered bytes wait for the next batch trigger
  or for :meth:`close`.

* Thread-safety: a single :class:`threading.Lock` guards write + fsync.
  Concurrent producers (orchestrator main thread, stage callbacks
  running on the trainer pump executor) cannot interleave bytes.

* :meth:`close` performs a final flush + fsync. Idempotent.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import IO

from ryotenkai_shared.events import (
    SEVERITY_ORDER,
    BaseEvent,
    to_jsonl,
)
from ryotenkai_shared.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FSYNC_BATCH_SIZE = 50
DEFAULT_FSYNC_INTERVAL_S = 1.0

# Severities at or above this threshold flush immediately, ignoring batch.
_IMMEDIATE_FSYNC_THRESHOLD = SEVERITY_ORDER["error"]


__all__ = [
    "DEFAULT_FSYNC_BATCH_SIZE",
    "DEFAULT_FSYNC_INTERVAL_S",
    "JournalWriter",
]


class JournalWriter:
    """Append-only JSONL writer with batched fsync.

    See module docstring for the wire format and crash-safety contract.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        fsync_batch_size: int = DEFAULT_FSYNC_BATCH_SIZE,
        fsync_interval_s: float = DEFAULT_FSYNC_INTERVAL_S,
    ) -> None:
        if fsync_batch_size < 1:
            raise ValueError("fsync_batch_size must be >= 1")
        if fsync_interval_s < 0:
            raise ValueError("fsync_interval_s must be >= 0")

        self._path = Path(path)
        self._batch_size = fsync_batch_size
        self._interval_s = fsync_interval_s

        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Binary append — to_jsonl produces a UTF-8 string that we encode
        # at write time. Binary mode avoids platform-specific newline
        # translation that would corrupt the length prefix.
        self._fh: IO[bytes] | None = self._path.open("ab")

        self._lock = threading.Lock()
        self._unflushed = 0
        self._last_fsync_at = time.monotonic()
        self._closed = False

        # Metrics — Phase 8 health endpoint scrapes these. Exposed as
        # attributes so tests can assert on them directly and the
        # metrics aggregator can snapshot them without locking.
        self.events_appended = 0
        self.fsyncs_total = 0
        self.write_failures_total = 0
        # Phase 8 — separated from ``write_failures_total`` because an
        # fsync failure (disk-full at flush time) is operationally
        # different from a write failure (transient EIO). The two are
        # surfaced separately in the health snapshot.
        self.fsync_failed_total = 0
        # Bytes written since the writer was opened. ``len(line)`` is
        # incremented BEFORE the ``write()`` call returns — partial
        # writes that raise ``OSError`` don't count.
        self.total_bytes_written = 0
        # Monotonic timestamp of the last successful ``os.fsync``.
        # ``None`` until the first fsync happens. The health endpoint
        # converts to ``last_fsync_age_seconds = now - this``.
        self.last_fsync_at: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_closed(self) -> bool:
        return self._closed

    def append(self, event: BaseEvent) -> None:
        """Write one envelope as a length-prefixed JSONL line.

        Never raises to the caller — write/fsync failures are logged and
        counted on ``write_failures_total``. The caller's never-raises
        contract on :meth:`IEventEmitter.emit` propagates through here.
        """
        if self._closed:
            logger.warning(
                "[JournalWriter] append after close ignored: kind=%s",
                event.kind,
            )
            return

        line = to_jsonl(event).encode("utf-8")
        immediate = SEVERITY_ORDER.get(event.severity, 0) >= _IMMEDIATE_FSYNC_THRESHOLD

        with self._lock:
            if self._fh is None:  # pragma: no cover — defensive
                return
            try:
                self._fh.write(line)
                self._fh.flush()
            except OSError as exc:
                self.write_failures_total += 1
                logger.warning(
                    "[JournalWriter] write failed (event dropped from journal): %s: %s",
                    type(exc).__name__,
                    exc,
                )
                return

            self.events_appended += 1
            self.total_bytes_written += len(line)
            self._unflushed += 1

            now = time.monotonic()
            if (
                immediate
                or self._unflushed >= self._batch_size
                or (now - self._last_fsync_at) >= self._interval_s
            ):
                self._fsync_locked(now=now)

    def fsync_now(self) -> None:
        """Force an immediate fsync (best-effort, silent on failure).

        Used by tests and by :meth:`close`. Idempotent — calling twice
        in a row just costs a second syscall.
        """
        with self._lock:
            if self._closed or self._fh is None:
                return
            self._fsync_locked(now=time.monotonic())

    def close(self) -> None:
        """Final flush + fsync + close. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            fh = self._fh
            self._fh = None
            if fh is None:
                return
            try:
                fh.flush()
                os.fsync(fh.fileno())
                self.fsyncs_total += 1
                self.last_fsync_at = time.monotonic()
            except OSError as exc:
                self.fsync_failed_total += 1
                logger.debug(
                    "[JournalWriter] final fsync failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            with contextlib.suppress(OSError):
                fh.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fsync_locked(self, *, now: float) -> None:
        """Caller MUST hold ``self._lock``. Best-effort; resets bookkeeping."""
        if self._fh is None:  # pragma: no cover — defensive
            return
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self.fsyncs_total += 1
            # Phase 8 — public ``last_fsync_at`` mirrors the private
            # bookkeeping value but is exposed via the metrics
            # aggregator. Updated only on success so a long stretch of
            # fsync failures is visible as a growing "age".
            self.last_fsync_at = now
        except OSError as exc:
            self.fsync_failed_total += 1
            logger.debug(
                "[JournalWriter] fsync failed: %s: %s",
                type(exc).__name__,
                exc,
            )
        self._unflushed = 0
        self._last_fsync_at = now
