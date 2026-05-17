"""Durable on-disk JSONL journal for the EventBus, Phase 2 envelope format.

Phase 2 rewrite (ethereal-tumbling-patterson) — the journal now stores
:class:`ryotenkai_shared.events.BaseEvent` envelopes one per line via
:func:`ryotenkai_shared.events.to_jsonl` (length-prefixed framing,
``<utf8_byte_length>\\t<json>\\n``). Torn writes (e.g. ``kill -9``
mid-``write()``) are detected on resume by reading the trailing file
backwards and truncating any partial last line via the atomic
``tmp + rename`` dance.

Storage layout (unchanged from Phase 12.B):

::

    <workspace>/events/
    ├── events.000.jsonl
    ├── events.001.jsonl
    └── ...               (current — being appended)

Rotation cap (5 files × 100 MiB) and on-rotate callback semantics carry
over so :class:`EventBus.attach_journal_rotation_listener` keeps working
without changes.

Resume contract:

* The journal scans ``root_dir`` for ``events.NNN.jsonl`` files at
  init.
* The current file's last line is verified via
  :func:`ryotenkai_shared.events.parse_length_prefix`. If the line
  fails parsing (length mismatch, missing trailing newline, etc.) the
  file is rewritten atomically with the bad tail stripped before any
  new appends.
* :meth:`iter_envelopes` reads the full journal in ascending order via
  :func:`ryotenkai_shared.events.from_jsonl` (``strict=False`` so torn
  lines become :class:`UnknownEvent` with diagnostic crumbs rather than
  failing the read).
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from ryotenkai_shared.events import (
    BaseEvent,
    MalformedEventError,
    UnknownEvent,
    from_jsonl,
    parse_length_prefix,
    to_jsonl,
)
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


__all__ = [
    "DEFAULT_FILE_SIZE_CAP",
    "DEFAULT_MAX_FILES",
    "EVENTS_DIR_REL",
    "EVENTS_FILE_FMT",
    "EventJournal",
]


# -- Constants ----------------------------------------------------------------

EVENTS_DIR_REL = "events"
EVENTS_FILE_FMT = "events.{seq:03d}.jsonl"
_FILE_NAME_RE = re.compile(r"^events\.(\d{3,})\.jsonl$")

DEFAULT_FILE_SIZE_CAP = 100 * 1024 * 1024  # 100 MiB
DEFAULT_MAX_FILES = 5
DEFAULT_FSYNC_BATCH = 50
DEFAULT_FSYNC_INTERVAL_MS = 1000


# -- Configuration validation -------------------------------------------------


def validate_journal_config(
    *,
    file_size_cap: int,
    max_files: int,
    fsync_batch: int,
    fsync_interval_ms: int,
) -> None:
    """Pure-function parameter validation (kept from Phase 14.E)."""
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
    """Append-only rotating JSONL journal over length-prefixed envelopes.

    Construct once in :func:`_lifespan`; pass to :class:`EventBus` so
    every :meth:`EventBus.publish` also persists the envelope.

    Lifecycle:
        ``__init__`` walks the directory, truncates a torn trailing
        line if found, and resumes at ``max(seq)``. ``append_envelope``
        writes one length-prefixed line. ``close()`` flushes and
        releases the handle. Idempotent close().
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

        self._current_fh: IO[bytes] | None = None
        self._current_seq: int = 0
        self._current_size: int = 0
        self._unflushed_count: int = 0
        self._last_fsync_ms: int = 0
        self._closed: bool = False

        self._initialize()

    def set_rotation_callback(self, callback: Any) -> None:
        """Post-construction binding for the rotation observer.

        Idempotent — calling twice replaces the previous callback. Pass
        ``None`` to detach.
        """
        self._on_rotate = callback

    # ------------------------------------------------------------------
    # Construction-time directory walk + truncate-invalid recovery
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Scan ``root_dir``; truncate torn tail; resume or start fresh.

        Crash recovery: if the previous process died mid-``write()`` the
        last line of the current file may be partial. We detect this
        by trying to parse the trailing line via
        :func:`parse_length_prefix`; on failure we rewrite the file with
        the bad tail stripped via ``tmp + rename`` so the rewrite is
        atomic and never produces a half-written replacement.
        """
        self._root_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

        existing = self._list_existing_seqs()
        self._current_seq = existing[-1] if existing else 0

        current_path = self._file_for(self._current_seq)
        if current_path.exists():
            self._truncate_torn_tail(current_path)
            size = current_path.stat().st_size
        else:
            size = 0
        self._current_size = size

        self._current_fh = current_path.open("ab")
        self._last_fsync_ms = self._now_ms()

    def _truncate_torn_tail(self, path: Path) -> None:
        """Detect and atomically strip a partial trailing line.

        Reads the file's last 64 KiB (enough for a few full envelopes
        even at the verbose end), then walks backwards line by line
        through that suffix. The last line is validated via
        :func:`parse_length_prefix`; if it fails, rewrite the file
        with the bad tail removed using ``tmp + rename``.
        """
        if not path.exists() or path.stat().st_size == 0:
            return

        try:
            with path.open("rb") as fh:
                file_size = path.stat().st_size
                tail_window = min(file_size, 64 * 1024)
                fh.seek(file_size - tail_window)
                tail = fh.read(tail_window)
        except OSError as exc:
            logger.warning("[JOURNAL] could not read tail of %s: %s", path, exc)
            return

        try:
            text = tail.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            # Partial multi-byte sequence at the front of the window —
            # safe to re-decode from a UTF-8-aligned boundary if there
            # is one; otherwise treat the whole tail as suspect.
            text = tail.decode("utf-8", errors="ignore")

        # Split keeping the line terminators so we can reconstruct.
        lines = text.splitlines(keepends=True)
        if not lines:
            return
        last = lines[-1]
        try:
            parse_length_prefix(last)
            # Good — nothing to do.
            return
        except ValueError:
            pass

        # Rewrite the file with the bad tail removed. Use an absolute
        # byte offset rather than the line index so we truncate at
        # exactly the right place even when the suffix spans an earlier
        # boundary the tail-window didn't include.
        keep_text = "".join(lines[:-1])
        keep_bytes_in_tail = len(keep_text.encode("utf-8"))
        keep_total = (file_size - tail_window) + keep_bytes_in_tail

        tmp_path = path.with_suffix(path.suffix + ".trunc.tmp")
        try:
            with path.open("rb") as src, tmp_path.open("wb") as dst:
                remaining = keep_total
                while remaining > 0:
                    chunk = src.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
                dst.flush()
                try:
                    os.fsync(dst.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, path)
            logger.warning(
                "[JOURNAL] truncated torn tail in %s "
                "(was %d bytes, now %d)",
                path.name, file_size, keep_total,
            )
        except OSError as exc:
            logger.warning(
                "[JOURNAL] could not atomically truncate %s: %s",
                path, exc,
            )
            with contextlib.suppress(OSError):
                tmp_path.unlink()

    def _list_existing_seqs(self) -> list[int]:
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

    def append_envelope(self, event: BaseEvent) -> None:
        """Append a single envelope as one length-prefixed JSONL line."""
        if self._closed:
            raise RuntimeError("event journal is closed")
        if self._current_fh is None:  # pragma: no cover — defensive
            raise RuntimeError("event journal not initialized")

        line = to_jsonl(event)
        line_bytes = line.encode("utf-8")

        # Rotate BEFORE writing if this would put us past the cap.
        if (
            self._current_size > 0
            and self._current_size + len(line_bytes) > self._file_size_cap
        ):
            self._rotate()

        self._current_fh.write(line_bytes)
        try:
            self._current_fh.flush()
        except OSError as exc:
            logger.debug("[JOURNAL] flush failed: %s", exc)
        self._current_size += len(line_bytes)
        self._unflushed_count += 1

        now = self._now_ms()
        if (
            self._unflushed_count >= self._fsync_batch
            or (now - self._last_fsync_ms) >= self._fsync_interval_ms
        ):
            self._fsync_now_locked()

    def fsync_now(self) -> None:
        """Force fsync immediately. Best-effort — silent on failure."""
        if self._closed or self._current_fh is None:
            return
        self._fsync_now_locked()

    def _fsync_now_locked(self) -> None:
        assert self._current_fh is not None
        try:
            self._current_fh.flush()
            os.fsync(self._current_fh.fileno())
        except OSError as exc:
            logger.debug("[JOURNAL] fsync failed: %s", exc)
        self._unflushed_count = 0
        self._last_fsync_ms = self._now_ms()

    def _rotate(self) -> None:
        """Close the current file, advance seq, open the next."""
        from_seq = self._current_seq
        from_path = self._file_for(from_seq)
        try:
            from_size = from_path.stat().st_size
        except OSError:
            from_size = 0

        if self._current_fh is not None:
            try:
                self._current_fh.flush()
                os.fsync(self._current_fh.fileno())
            except OSError:
                pass
            self._current_fh.close()
            self._current_fh = None

        next_seq = self._current_seq + 1

        # Drop-oldest if we'd exceed max_files.
        existing = self._list_existing_seqs()
        projected = len(existing) + 1
        while projected > self._max_files and existing:
            oldest_seq = existing.pop(0)
            oldest_path = self._file_for(oldest_seq)
            try:
                oldest_path.unlink()
                logger.info(
                    "[JOURNAL] dropped oldest events.%03d.jsonl on rotate",
                    oldest_seq,
                )
            except OSError as exc:
                logger.warning(
                    "[JOURNAL] failed to drop oldest file: %s", exc,
                )
            projected -= 1

        self._current_seq = next_seq
        new_path = self._file_for(next_seq)
        self._current_fh = new_path.open("ab")
        self._current_size = 0

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

    def iter_envelopes(self, *, since: int = 0) -> Iterator[BaseEvent]:
        """Yield every persisted envelope with ``offset >= since``.

        Order: ascending by file seq, then by offset within each file
        (matches the chronological write order).

        Failures: malformed JSON lines become :class:`UnknownEvent`
        with diagnostic crumbs via the codec's ``strict=False`` path —
        readers stay forward-compatible with future schemas and never
        choke on torn writes mid-iteration. Unrecoverable I/O errors
        on a file skip the whole file with a logged warning.
        """
        for seq in self._list_existing_seqs():
            path = self._file_for(seq)
            try:
                with path.open("rb") as fh:
                    for raw in fh:
                        if not raw.strip():
                            continue
                        line = raw.decode("utf-8", errors="replace")
                        try:
                            envelope = from_jsonl(line, strict=False)
                        except MalformedEventError:
                            continue
                        if envelope.offset >= since or isinstance(envelope, UnknownEvent):
                            yield envelope
            except OSError as exc:
                logger.warning(
                    "[JOURNAL] failed to read %s: %s — skipping",
                    path, exc,
                )

    def newest_persisted_offset(self) -> int | None:
        """Largest offset present on disk; ``None`` when empty.

        Used by :class:`EventBus.__init__` to reconcile its offset
        counter on resume so the next event published doesn't collide
        with an offset already on disk.
        """
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
                        line = raw.decode("utf-8", errors="replace")
                        try:
                            envelope = from_jsonl(line, strict=False)
                        except MalformedEventError:
                            continue
                        # UnknownEvent on a malformed framing line will
                        # carry UNKNOWN_OFFSET (-1) — skip it.
                        if envelope.offset < 0:
                            continue
                        if newest is None or envelope.offset > newest:
                            newest = envelope.offset
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
                        line = raw.decode("utf-8", errors="replace")
                        try:
                            envelope = from_jsonl(line, strict=False)
                        except MalformedEventError:
                            continue
                        if envelope.offset < 0:
                            continue
                        return envelope.offset
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
