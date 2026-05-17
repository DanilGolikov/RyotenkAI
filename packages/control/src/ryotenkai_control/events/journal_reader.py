"""Tail-able reader for ``<run_directory>/events.jsonl`` (Phase 3).

Three primary consumers:

* :class:`ControlEventEmitter` on resume — reconstructs per-source
  offset counters via :meth:`newest_persisted_offset_per_source`.
* SSE / HTTP catchup (Phase 6) — :meth:`replay_from` yields the
  bounded range ``(after_offset, current_tail]``.
* :class:`EventDedup` on restart — reads the last N envelopes per
  source to repopulate the dedup set so post-restart resends are still
  rejected.

The reader is read-mostly: the only side effect on construction is
:meth:`truncate_torn_tail`, which rewrites the file in place via
``tmp + rename`` when the last line fails framing validation. This
mirrors the pod-side :class:`ryotenkai_pod.runner.event_journal.EventJournal`
recovery dance.

Behaviour on missing file: all read methods return an empty result
(no exception). Truncation is a no-op when the file does not exist.
"""

from __future__ import annotations

import contextlib
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_shared.events import (
    UNKNOWN_OFFSET,
    BaseEvent,
    MalformedEventError,
    UnknownEvent,
    from_jsonl,
    parse_length_prefix,
)
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = get_logger(__name__)

# Tail window we re-read backwards to locate a torn last line. 64 KiB
# is comfortably more than a single envelope so the last line always
# fits.
_TAIL_WINDOW_BYTES = 64 * 1024


__all__ = ["JournalReader"]


class JournalReader:
    """Read-only view of an events journal with crash-recovery on init."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_envelopes(self) -> Iterator[BaseEvent]:
        """Yield every envelope in file order.

        Malformed lines become :class:`UnknownEvent` with diagnostic
        crumbs — readers stay forward-compatible with future schemas
        and never choke on torn writes mid-iteration. The journal
        codec's ``strict=False`` path handles this.
        """
        if not self._path.exists():
            return
        try:
            with self._path.open("rb") as fh:
                for raw in fh:
                    if not raw.strip():
                        continue
                    line = raw.decode("utf-8", errors="replace")
                    try:
                        yield from_jsonl(line, strict=False)
                    except MalformedEventError:  # pragma: no cover — strict=False above
                        continue
        except OSError as exc:
            logger.warning(
                "[JournalReader] failed to read %s: %s — yielding nothing",
                self._path,
                exc,
            )

    def replay_from(
        self,
        *,
        after_offset: int,
        limit: int | None = None,
    ) -> Iterator[BaseEvent]:
        """Yield envelopes with ``offset > after_offset`` in file order.

        :class:`UnknownEvent` instances with ``offset == UNKNOWN_OFFSET``
        are filtered out — they represent torn-write residue from the
        codec's ``strict=False`` path and would corrupt monotonic-cursor
        bookkeeping if included (Phase 1.5 invariant).
        """
        yielded = 0
        for envelope in self.iter_envelopes():
            if envelope.offset == UNKNOWN_OFFSET:
                continue
            if envelope.offset <= after_offset:
                continue
            yield envelope
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    def newest_persisted_offset_per_source(self) -> dict[str, int]:
        """Return ``{source: max_offset}`` for every distinct source seen.

        Used by :meth:`ControlEventEmitter.__init__` to seed per-source
        offset counters on resume so the next locally-emitted event
        doesn't collide with what's already on disk. :class:`UnknownEvent`
        envelopes with ``offset == UNKNOWN_OFFSET`` are skipped — they
        carry no usable offset.
        """
        per_source: dict[str, int] = {}
        for envelope in self.iter_envelopes():
            if envelope.offset == UNKNOWN_OFFSET:
                continue
            current = per_source.get(envelope.source)
            if current is None or envelope.offset > current:
                per_source[envelope.source] = envelope.offset
        return per_source

    def tail_per_source(
        self,
        *,
        max_entries_per_source: int,
    ) -> dict[str, list[BaseEvent]]:
        """Yield the most-recent ``max_entries_per_source`` envelopes per source.

        Used by :class:`EventDedup` on restart to repopulate its set.
        :class:`UnknownEvent` (offset=-1) entries are skipped because
        the dedup key includes ``offset``.

        Implementation streams the whole file but only keeps a bounded
        deque per source, so memory is O(num_sources × max_entries_per_source).
        """
        if max_entries_per_source < 0:
            raise ValueError("max_entries_per_source must be >= 0")
        buckets: dict[str, deque[BaseEvent]] = defaultdict(
            lambda: deque(maxlen=max_entries_per_source)
        )
        for envelope in self.iter_envelopes():
            if envelope.offset == UNKNOWN_OFFSET:
                continue
            buckets[envelope.source].append(envelope)
        return {src: list(d) for src, d in buckets.items()}

    # ------------------------------------------------------------------
    # Truncation (crash recovery)
    # ------------------------------------------------------------------

    def truncate_torn_tail(self) -> bool:
        """Atomically strip a partial trailing line, if any.

        Returns ``True`` if truncation happened, ``False`` otherwise.
        Uses ``tmp + rename`` so the rewrite is atomic and never produces
        a half-written replacement file. Mirrors the pod-side journal's
        recovery pass.
        """
        if not self._path.exists():
            return False
        try:
            file_size = self._path.stat().st_size
        except OSError:
            return False
        if file_size == 0:
            return False

        try:
            with self._path.open("rb") as fh:
                tail_window = min(file_size, _TAIL_WINDOW_BYTES)
                fh.seek(file_size - tail_window)
                tail = fh.read(tail_window)
        except OSError as exc:
            logger.warning(
                "[JournalReader] could not read tail of %s: %s",
                self._path,
                exc,
            )
            return False

        try:
            text = tail.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = tail.decode("utf-8", errors="ignore")

        lines = text.splitlines(keepends=True)
        if not lines:
            return False
        last = lines[-1]
        try:
            parse_length_prefix(last)
            # Last line is well-formed — nothing to do.
            return False
        except ValueError:
            pass

        keep_text = "".join(lines[:-1])
        keep_bytes_in_tail = len(keep_text.encode("utf-8"))
        keep_total = (file_size - tail_window) + keep_bytes_in_tail

        tmp_path = self._path.with_suffix(self._path.suffix + ".trunc.tmp")
        try:
            with self._path.open("rb") as src, tmp_path.open("wb") as dst:
                remaining = keep_total
                while remaining > 0:
                    chunk = src.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
                dst.flush()
                with contextlib.suppress(OSError):
                    os.fsync(dst.fileno())
            os.replace(tmp_path, self._path)  # noqa: PTH105 — atomic replace
            logger.warning(
                "[JournalReader] truncated torn tail in %s "
                "(was %d bytes, now %d)",
                self._path.name,
                file_size,
                keep_total,
            )
            return True
        except OSError as exc:
            logger.warning(
                "[JournalReader] could not atomically truncate %s: %s",
                self._path,
                exc,
            )
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            return False

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self._path.exists()

    def has_torn_tail(self) -> bool:
        """Cheap predicate used by tests; mirrors :meth:`truncate_torn_tail` w/o rewriting."""
        if not self._path.exists():
            return False
        try:
            file_size = self._path.stat().st_size
        except OSError:
            return False
        if file_size == 0:
            return False
        try:
            with self._path.open("rb") as fh:
                tail_window = min(file_size, _TAIL_WINDOW_BYTES)
                fh.seek(file_size - tail_window)
                tail = fh.read(tail_window)
        except OSError:
            return False
        text = tail.decode("utf-8", errors="ignore")
        lines = text.splitlines(keepends=True)
        if not lines:
            return False
        try:
            parse_length_prefix(lines[-1])
            return False
        except ValueError:
            return True

    def is_envelope_unknown_marker(self, envelope: BaseEvent) -> bool:
        """``True`` when ``envelope`` is the malformed-line UnknownEvent.

        Pure helper used by Phase 4-7 consumers that want to skip the
        torn-write residue without depending on the codec's internals.
        """
        return isinstance(envelope, UnknownEvent) and envelope.offset == UNKNOWN_OFFSET
