"""
Phase 12.B — durable on-disk JSONL journal for the EventBus.

Background
----------
:class:`~src.runner.event_bus.EventBus` keeps the last ~10k events in a
RAM ring buffer (capacity tunable via ``RYOTENKAI_EVENT_BUFFER_SIZE``).
At a typical 30 events/sec the ring fills in ~5.5 minutes, so a Mac
sleeping longer than that hits a :class:`BufferTruncatedError` on
WS reconnect and the UI sees a gap covering the entire sleep window.

Phase 12.B closes the gap by **persisting every published event to
disk** in a rotated JSONL file under ``<workspace>/.runner/events/``.
The WS handler transparently replays from disk when the requested
``since`` cursor is older than the ring's oldest offset but still
present on disk; only when the offset is older than the journal's
oldest persisted record does the handler return :class:`DiskJournalExhausted`
(close code 4410).

Design constraints
------------------
* **Single-writer** — :class:`EventBus.publish` is sync and called
  only from the FastAPI event loop (cross-thread publishes go via
  ``loop.call_soon_threadsafe``). No ``fcntl`` lock needed — contention
  is impossible by construction.
* **Append-only inside a file** — never rewrite or seek backwards.
  Failed writes are logged but never raise into the publisher
  (events_disk_pressure event emitted, rate-limited).
* **fsync batched** — every 50 events OR 1 s, whichever first. fsync
  per-write would cost ~5 ms × 30 events/s = 150 ms/s wasted CPU+IO
  for a 1 s gap-on-crash that's already << the 5+ minute gap from
  ring overflow today.
* **Rotation by file size** — current file caps at 100 MiB then
  rotates to ``events.NNN+1.jsonl``. Total file count caps at 5
  → ~500 MiB worst-case footprint, deterministic for operator
  provisioning. On rotation past the cap, the oldest file is
  deleted (drop-oldest policy).
* **Schema versioned per record** — ``{"v": 1, "offset": N, ...}``.
  Readers ignore unknown keys; ``v > MAX_SUPPORTED_VERSION`` rejects
  with a logged warning. Per-record versioning beats a file header
  for append-safety + concat-safety.
* **UTF-8 + non-serialisable payloads** — ``json.dumps`` is called
  with ``ensure_ascii=False`` (compact UTF-8 for Cyrillic / non-ASCII
  user messages) and ``default=str`` (datetime / Path / Enum coerce
  to string instead of raising).
* **Crash recovery on init** — walks the directory for existing
  ``events.NNN.jsonl`` files, picks ``max(seq)`` + 1 as the next
  rotation index, and resumes appending. Truncated trailing lines
  are detected on read (``iter_records`` skips malformed JSON).

Storage layout
--------------
::

    <workspace>/events/
    ├── events.000.jsonl
    ├── events.001.jsonl
    ├── events.002.jsonl
    ├── events.003.jsonl
    └── events.004.jsonl  (current — being appended)

After rotation past the 5-file cap, the oldest is deleted:
opening events.005.jsonl deletes events.000.jsonl first.

The directory location is supplied by the caller (the runner's
lifespan reads ``pod_layout.events_dir``). The journal itself is
layout-agnostic — it only knows the absolute ``root_dir`` it was
constructed with.

Thread / async safety
---------------------
NOT thread-safe. The bus enforces single-writer-from-event-loop.
Multiple WebSocket subscribers reading via :meth:`iter_records` use
their own file handles and never collide with the writer (each open
is a separate kernel-level file descriptor; appends are atomic at
the syscall level for line-sized writes < 4 KiB).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


__all__ = [
    "DEFAULT_FILE_SIZE_CAP",
    "DEFAULT_FSYNC_BATCH",
    "DEFAULT_FSYNC_INTERVAL_MS",
    "DEFAULT_MAX_FILES",
    "EVENTS_DIR_REL",
    "EVENTS_FILE_FMT",
    "MAX_SUPPORTED_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "EventJournal",
    "JournalRecord",
]


# -- Constants ----------------------------------------------------------------

# Legacy constant — pre-PodLayout layout used a ``.runner/events``
# subdirectory under workspace. PodLayout migrated this to a flat
# ``events/`` directory rooted at the per-run workspace. The constant
# is kept ONLY for tests that build paths manually; production code
# constructs the directory via ``PodLayout.events_dir`` and passes
# the absolute path through ``EventJournal(root_dir=...)``.
EVENTS_DIR_REL = "events"
EVENTS_FILE_FMT = "events.{seq:03d}.jsonl"
_FILE_NAME_RE = re.compile(r"^events\.(\d{3,})\.jsonl$")

DEFAULT_FILE_SIZE_CAP = 100 * 1024 * 1024  # 100 MiB
DEFAULT_MAX_FILES = 5
DEFAULT_FSYNC_BATCH = 50
DEFAULT_FSYNC_INTERVAL_MS = 1000

SCHEMA_VERSION = 1
MAX_SUPPORTED_SCHEMA_VERSION = 1


# -- Record dataclass --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JournalRecord:
    """Reader-side projection of a single JSONL line.

    Mirrors :class:`src.runner.event_bus.Event` but lives in the
    journal module to keep the directional dependency one-way (bus
    knows about journal; journal does NOT know about bus).
    """

    v: int
    offset: int
    ts: str
    kind: str
    payload: dict[str, Any]


# -- Configuration validation (Phase 14.E V6) ---------------------------------


def validate_journal_config(
    *,
    file_size_cap: int,
    max_files: int,
    fsync_batch: int,
    fsync_interval_ms: int,
) -> None:
    """Phase 14.E (V6) — pure-function parameter validation.

    Extracted from :class:`EventJournal.__init__` so callers can
    fail-fast on bad config at bootstrap time without paying for
    filesystem state initialization. Raises :class:`ValueError`
    with a descriptive message — same exceptions the constructor
    used to throw, so existing callers' error handling is
    unchanged.
    """
    if file_size_cap <= 0:
        raise ValueError("file_size_cap must be positive")
    if max_files < 1:
        raise ValueError("max_files must be >= 1")
    if fsync_batch < 1:
        raise ValueError("fsync_batch must be >= 1")
    if fsync_interval_ms < 0:
        raise ValueError("fsync_interval_ms must be >= 0")


# -- Journal -----------------------------------------------------------------


class EventJournal:
    """Append-only rotating JSONL journal.

    Construct once in :func:`_lifespan`; pass to :class:`EventBus` so
    every :meth:`EventBus.publish` also persists the record.

    Lifecycle:
        ``__init__`` walks the directory and resumes from
        ``max(seq)``. ``append(...)`` writes a line. ``close()`` flushes
        and shuts the current file handle. Idempotent close().

    Read access:
        :meth:`iter_records(since=N)` yields every record with
        ``offset >= N`` in ascending order across all files. Used by
        the WS handler when the ring buffer truncated past ``since``.
    """

    def __init__(
        self,
        *,
        root_dir: Path | str,
        file_size_cap: int = DEFAULT_FILE_SIZE_CAP,
        max_files: int = DEFAULT_MAX_FILES,
        fsync_batch: int = DEFAULT_FSYNC_BATCH,
        fsync_interval_ms: int = DEFAULT_FSYNC_INTERVAL_MS,
        on_rotate: Any = None,
    ) -> None:
        """
        Args:
            on_rotate: Phase 12.C — optional callback fired after each
                       rotation. Signature
                       ``(from_seq, to_seq, file_size_bytes, oldest_remaining_seq) -> None``.
                       Phase 14.E added :meth:`set_rotation_callback`
                       as a post-construction binding alternative —
                       the lifespan now uses that to avoid the
                       circular-binding-closure pattern. ``on_rotate``
                       remains supported for direct test wiring.
                       Failure-tolerant: exceptions in the callback
                       are swallowed (don't block journal progress).
        """
        # Phase 14.E (V6) — extracted parameter validation. Pure
        # function, no filesystem touch; callable from bootstrap
        # to fail-fast on bad config without paying for state init.
        validate_journal_config(
            file_size_cap=file_size_cap,
            max_files=max_files,
            fsync_batch=fsync_batch,
            fsync_interval_ms=fsync_interval_ms,
        )

        self._root_dir = Path(root_dir)
        self._file_size_cap = file_size_cap
        self._max_files = max_files
        self._fsync_batch = fsync_batch
        self._fsync_interval_ms = fsync_interval_ms
        self._on_rotate = on_rotate

        # Discovered + active state populated by _initialize().
        self._current_fh: IO[bytes] | None = None
        self._current_seq: int = 0
        self._current_size: int = 0
        self._unflushed_count: int = 0
        self._last_fsync_ms: int = 0
        self._closed: bool = False

        self._initialize()

    def set_rotation_callback(self, callback: Any) -> None:
        """Phase 14.E (V1) — post-construction binding.

        The lifespan uses this to register :class:`EventBus` as the
        rotation observer AFTER both objects exist, avoiding the
        pre-14.E circular-binding-closure pattern (mutable dict cell
        carrying a future ``bus.publish`` reference). The bus's
        :meth:`attach_journal_rotation_listener` calls this method.

        Idempotent — calling twice replaces the previous callback.
        Pass ``None`` to detach.
        """
        self._on_rotate = callback

    # ------------------------------------------------------------------
    # Construction-time directory walk
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Scan ``root_dir`` for existing files; resume or start fresh.

        Crash recovery: if a previous process died mid-rotation (file
        N+1 partially created with size 0, file N still under cap), we
        resume on file N to keep records contiguous. The "size > cap"
        case forces a rotate-on-next-append so we never grow a file
        past the cap by more than one record.
        """
        self._root_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

        existing = self._list_existing_seqs()
        if not existing:
            self._current_seq = 0
        else:
            self._current_seq = existing[-1]

        current_path = self._file_for(self._current_seq)
        # File may not exist yet if directory is brand new.
        size = current_path.stat().st_size if current_path.exists() else 0
        self._current_size = size

        # If a partial rotate-and-die left this file already past cap,
        # rotate on the FIRST append rather than letting it grow more.
        # We achieve this by leaving size as-is; ``append`` checks
        # ``current_size + line > cap`` and rotates.
        self._current_fh = current_path.open("ab")
        self._last_fsync_ms = self._now_ms()

    def _list_existing_seqs(self) -> list[int]:
        """Return seq numbers found in ``root_dir`` in ascending order."""
        seqs: list[int] = []
        if not self._root_dir.is_dir():
            return seqs
        for entry in self._root_dir.iterdir():
            if not entry.is_file():
                continue
            m = _FILE_NAME_RE.match(entry.name)
            if m is None:
                continue
            seqs.append(int(m.group(1)))
        seqs.sort()
        return seqs

    def _file_for(self, seq: int) -> Path:
        return self._root_dir / EVENTS_FILE_FMT.format(seq=seq)

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    # ------------------------------------------------------------------
    # Public API — write side
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        offset: int,
        ts: str,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        """Append a single event as one JSONL line.

        Schema:
            ``{"v":1,"offset":N,"ts":"...","kind":"...","payload":{...}}``

        Best-effort. OSError on write is logged + raised so
        :class:`EventBus.publish` can emit ``events_disk_pressure``;
        catastrophic write failures (disk full) bubble up.
        """
        if self._closed:
            raise RuntimeError("event journal is closed")
        if self._current_fh is None:  # pragma: no cover — defensive
            raise RuntimeError("event journal not initialized")

        record = {
            "v": SCHEMA_VERSION,
            "offset": int(offset),
            "ts": str(ts),
            "kind": str(kind),
            "payload": payload,
        }
        line = (
            json.dumps(
                record,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        line_bytes = line.encode("utf-8")

        # Rotate BEFORE writing if this would put us past the cap.
        # Empty file edge case: a brand-new file where the very first
        # record happens to exceed the cap is unusual but legal —
        # we'd rotate-then-write into a fresh file, leaving the empty
        # one behind. Cleaner to write into the empty file in that
        # case (don't rotate when current_size == 0). Hence the
        # extra ``> 0`` guard.
        if self._current_size > 0 and self._current_size + len(line_bytes) > self._file_size_cap:
            self._rotate()

        self._current_fh.write(line_bytes)
        # Always flush() to push the Python-level buffer down to the
        # OS — cheap (~µs), and makes the just-written record visible
        # to concurrent readers (other open file descriptors). Without
        # this, reader fds opened via `path.open("rb")` would see
        # only what was previously fsync'd, defeating the whole disk-
        # replay path. Durability is enforced separately via batched
        # ``fsync()`` below.
        try:
            self._current_fh.flush()
        except OSError as exc:
            logger.debug("[JOURNAL] flush failed: %s", exc)
        self._current_size += len(line_bytes)
        self._unflushed_count += 1

        now = self._now_ms()
        if self._unflushed_count >= self._fsync_batch or (now - self._last_fsync_ms) >= self._fsync_interval_ms:
            self._fsync_now_locked()

    def fsync_now(self) -> None:
        """Force fsync immediately. Best-effort — silent on failure."""
        if self._closed or self._current_fh is None:
            return
        self._fsync_now_locked()

    def _fsync_now_locked(self) -> None:
        # Caller has already verified _current_fh is not None.
        assert self._current_fh is not None
        try:
            self._current_fh.flush()
            os.fsync(self._current_fh.fileno())
        except OSError as exc:
            logger.debug("[JOURNAL] fsync failed: %s", exc)
        self._unflushed_count = 0
        self._last_fsync_ms = self._now_ms()

    def _rotate(self) -> None:
        """Close the current file, advance seq, open the next.

        Drop-oldest enforcement: if the file count would exceed
        ``max_files`` after opening the next file, the oldest existing
        file is unlinked first.

        Phase 12.C — after the rotation completes successfully, fires
        the ``on_rotate`` callback (if attached) with the from/to
        sequence numbers, the size of the just-closed file, and the
        oldest-remaining seq. Used by the lifespan to publish
        ``events_rotated`` on the bus.
        """
        # Capture the size of the file we're about to close, for the
        # telemetry payload.
        from_seq = self._current_seq
        from_path = self._file_for(from_seq)
        try:
            from_size = from_path.stat().st_size
        except OSError:
            from_size = 0

        # Close current.
        if self._current_fh is not None:
            try:
                self._current_fh.flush()
                os.fsync(self._current_fh.fileno())
            except OSError:
                pass
            self._current_fh.close()
            self._current_fh = None

        next_seq = self._current_seq + 1

        # Drop-oldest if we'd exceed max_files. After the rotation
        # there will be ``len(existing) + 1`` files (we've written
        # past current, about to open next). Trim down to ``max_files``.
        existing = self._list_existing_seqs()
        # Count of files after we open next_seq.
        projected = len(existing) + 1
        # Delete oldest until projected <= max_files.
        while projected > self._max_files and existing:
            oldest_seq = existing.pop(0)
            oldest_path = self._file_for(oldest_seq)
            try:
                oldest_path.unlink()
                logger.info("[JOURNAL] dropped oldest events.%03d.jsonl on rotate", oldest_seq)
            except OSError as exc:
                logger.warning("[JOURNAL] failed to drop oldest file: %s", exc)
            projected -= 1

        self._current_seq = next_seq
        new_path = self._file_for(next_seq)
        self._current_fh = new_path.open("ab")
        self._current_size = 0

        # Phase 12.C — fire telemetry callback. Failures swallowed so
        # the journal doesn't block on a misbehaving subscriber.
        if self._on_rotate is not None:
            remaining = self._list_existing_seqs()
            oldest_remaining = remaining[0] if remaining else None
            try:
                self._on_rotate(
                    from_seq=from_seq,
                    to_seq=next_seq,
                    file_size_bytes=from_size,
                    oldest_remaining_seq=oldest_remaining,
                )
            except Exception as exc:
                logger.debug("[JOURNAL] on_rotate callback failed: %s", exc)

    def close(self) -> None:
        """Flush + close. Idempotent."""
        if self._closed:
            return
        if self._current_fh is not None:
            try:
                self._current_fh.flush()
                os.fsync(self._current_fh.fileno())
            except OSError:
                pass
            with contextlib.suppress(OSError):
                self._current_fh.close()
            self._current_fh = None
        self._closed = True

    # ------------------------------------------------------------------
    # Public API — read side
    # ------------------------------------------------------------------

    def iter_records(self, *, since: int = 0) -> Iterator[JournalRecord]:
        """Yield every persisted record with ``offset >= since``.

        Order: ascending by file seq, then ascending by record offset
        within each file (which equals chronological order — bus
        publishes are strictly monotonic).

        Failures: malformed JSON lines are skipped with a logged
        warning. Truncated trailing line (write interrupted) is
        skipped. ``v > MAX_SUPPORTED_SCHEMA_VERSION`` records are
        skipped with a logged warning so a future writer rolling out
        v2 doesn't poison v1 readers.

        Concurrency: safe to call while another writer is appending —
        each call opens its own read-only file handle. Records added
        AFTER the iteration starts may or may not be yielded
        (depending on OS pagecache visibility); the bus' own ring
        buffer covers the recent tail anyway.
        """
        for seq in self._list_existing_seqs():
            path = self._file_for(seq)
            try:
                with path.open("rb") as fh:
                    for line_no, raw in enumerate(fh, start=1):
                        if not raw.strip():
                            continue
                        try:
                            obj = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            # Truncated or corrupt — skip.
                            continue
                        if not isinstance(obj, dict):
                            continue
                        v = obj.get("v")
                        if not isinstance(v, int) or v > MAX_SUPPORTED_SCHEMA_VERSION:
                            logger.warning(
                                "[JOURNAL] skipping record v=%r in %s line %d " "(reader supports v<=%d)",
                                v,
                                path.name,
                                line_no,
                                MAX_SUPPORTED_SCHEMA_VERSION,
                            )
                            continue
                        offset = obj.get("offset")
                        if not isinstance(offset, int) or offset < since:
                            continue
                        ts = obj.get("ts", "")
                        kind = obj.get("kind", "")
                        payload = obj.get("payload", {})
                        if not isinstance(payload, dict):
                            payload = {}
                        yield JournalRecord(
                            v=v,
                            offset=offset,
                            ts=str(ts),
                            kind=str(kind),
                            payload=payload,
                        )
            except OSError as exc:
                logger.warning(
                    "[JOURNAL] failed to read %s: %s — skipping",
                    path,
                    exc,
                )

    def newest_persisted_offset(self) -> int | None:
        """Largest offset present on disk; ``None`` when journal is empty.

        Used by :class:`EventBus.__init__` to reconcile its
        ``_next_offset`` after a runner restart so the next event
        published doesn't collide with an offset already on disk.
        """
        # Walk in reverse: newest file first; once we find ANY record
        # we know the max offset is in that file's tail.
        seqs = self._list_existing_seqs()
        if not seqs:
            return None
        newest: int | None = None
        for seq in reversed(seqs):
            path = self._file_for(seq)
            try:
                with path.open("rb") as fh:
                    for raw in fh:
                        if not raw.strip():
                            continue
                        try:
                            obj = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if not isinstance(obj, dict):
                            continue
                        offset = obj.get("offset")
                        if isinstance(offset, int) and (newest is None or offset > newest):
                            newest = offset
            except OSError:
                continue
            if newest is not None:
                return newest
        return newest

    def oldest_persisted_offset(self) -> int | None:
        """Smallest offset present on disk; ``None`` when empty."""
        for seq in self._list_existing_seqs():
            path = self._file_for(seq)
            try:
                with path.open("rb") as fh:
                    for raw in fh:
                        if not raw.strip():
                            continue
                        try:
                            obj = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if not isinstance(obj, dict):
                            continue
                        offset = obj.get("offset")
                        if isinstance(offset, int):
                            return offset
            except OSError:
                continue
        return None

    def total_bytes(self) -> int:
        """Cumulative on-disk size across all files."""
        total = 0
        for seq in self._list_existing_seqs():
            try:
                total += self._file_for(seq).stat().st_size
            except OSError:
                continue
        return total

    def file_count(self) -> int:
        """Number of journal files currently on disk."""
        return len(self._list_existing_seqs())

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def is_closed(self) -> bool:
        return self._closed
