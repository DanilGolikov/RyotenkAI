"""Mac-side mirror of the pod runner's event journal.

The pod runner persists every event it publishes into
``/workspace/.runner/events/events.<seq>.jsonl`` (see
:mod:`src.runner.event_journal`). The Mac control plane subscribes to
this stream over a WebSocket via :class:`JobClient.subscribe_events`
and feeds it into :class:`TrainingMonitor`.

This module is the **mirror writer** — it appends every event the
monitor consumes to a Mac-local file
``runs/<id>/attempts/<n>/events/events_mirror.jsonl``. Three reasons:

1. **Cold replay.** Frontend opens ``/runs/<id>/live`` after the run
   ended. The pod is gone, but ``events_mirror.jsonl`` is still on
   the Mac — the API can serve historical events without an SSH
   tunnel.

2. **Reconnect catch-up.** Frontend reconnects with
   ``since=last_offset+1``. The Mac WS endpoint reads the mirror
   from that offset forward and only switches to the live JobClient
   subscription once the file is exhausted — no events lost across
   gaps.

3. **Tooling.** ``curl /api/v1/runs/<id>/.../events`` returns events
   even hours after the run finished — for scripts, dashboards,
   ad-hoc analysis.

The mirror format is **identical** to the pod-side journal record
(``{v, offset, ts, kind, payload}``) so downstream consumers can use
the same parsing code.

Usage::

    with EventMirrorWriter(attempt_dir) as mirror:
        async for event in client.subscribe_events(job_id, since=0):
            mirror.write(event)

The writer is **synchronous** — it's safe to call from inside an
asyncio coroutine because the file IO is small (one short JSON line
per event). Periodic ``os.fsync`` runs every ``fsync_every_n``
writes so a crash mid-run loses at most that many trailing events.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, ClassVar

from src.utils.logger import logger

__all__ = ["EventMirrorWriter"]


class EventMirrorWriter:
    """Append-only writer for ``events_mirror.jsonl``.

    Threading: writes acquire an internal lock so concurrent calls
    from different threads don't interleave bytes mid-line. The
    expected caller is a single asyncio loop, but a stray thread
    (e.g. a logging handler dispatched off-loop) won't corrupt the
    JSONL stream.

    Lifecycle: use as a context manager (``with EventMirrorWriter(...)
    as m: ...``). The constructor opens the file lazily on first
    ``write()`` to keep the no-write case (e.g. monitor exits before
    any event arrives) free of empty-file artefacts.
    """

    EVENTS_DIR_NAME: ClassVar[str] = "events"
    MIRROR_FILE_NAME: ClassVar[str] = "events_mirror.jsonl"
    DEFAULT_FSYNC_EVERY_N: ClassVar[int] = 50

    def __init__(
        self,
        attempt_dir: Path,
        *,
        fsync_every_n: int = DEFAULT_FSYNC_EVERY_N,
    ) -> None:
        if fsync_every_n < 0:
            msg = f"fsync_every_n must be >= 0, got {fsync_every_n}"
            raise ValueError(msg)
        self._attempt_dir = Path(attempt_dir)
        self._fsync_every_n = fsync_every_n
        self._mirror_path = (
            self._attempt_dir / self.EVENTS_DIR_NAME / self.MIRROR_FILE_NAME
        )
        self._fp: Any = None  # opened on first write
        self._write_count = 0
        self._lock = threading.Lock()
        self._closed = False

    # ----- public API -----------------------------------------------------

    @property
    def path(self) -> Path:
        """Path the mirror writes to (whether or not the file yet exists)."""
        return self._mirror_path

    def write(self, event: dict[str, Any]) -> None:
        """Append ``event`` as a single JSON line.

        ``event`` shape matches the runner's :class:`JournalRecord`:
        ``{"v": int, "offset": int, "ts": str, "kind": str, "payload":
        dict}``. We don't validate — the upstream
        :class:`JobClient.subscribe_events` already returns parsed
        :class:`EventResponse` instances; we json-serialize them back
        to the on-disk format the journal uses.
        """
        if self._closed:
            msg = "EventMirrorWriter is closed; cannot write"
            raise RuntimeError(msg)

        # Compact separators keep the line short; one record per line
        # is the JSONL contract. ``ensure_ascii=False`` so non-ASCII
        # payloads (e.g. Russian log lines, emojis) round-trip
        # readably.
        line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)

        with self._lock:
            if self._fp is None:
                self._open_for_append()
            self._fp.write(line)
            self._fp.write("\n")
            self._write_count += 1
            if (
                self._fsync_every_n > 0
                and self._write_count % self._fsync_every_n == 0
            ):
                self._fp.flush()
                os.fsync(self._fp.fileno())

    def close(self) -> None:
        """Flush, fsync, and close the underlying file. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._fp is not None:
                try:
                    self._fp.flush()
                    os.fsync(self._fp.fileno())
                except OSError as exc:  # pragma: no cover — best-effort
                    logger.debug(
                        f"[EventMirrorWriter] fsync on close failed: {exc}",
                    )
                self._fp.close()
                self._fp = None

    # ----- context manager ------------------------------------------------

    def __enter__(self) -> EventMirrorWriter:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # ----- internals ------------------------------------------------------

    def _open_for_append(self) -> None:
        """Create the events dir and open the mirror file in append mode.

        Append mode preserves any prior content (e.g. resumed
        attempt — same dir reused). Line-buffered so each ``write()``
        without an explicit fsync still leaves the line on disk for
        ``tail -f`` even before fsync interval elapses.
        """
        self._mirror_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._mirror_path.open(
            "a", encoding="utf-8", buffering=1,  # line-buffered
        )
