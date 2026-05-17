"""Offset-range slice helper for the Phase 6 HTTP replay endpoint.

Phase 3 keeps this module deliberately tiny: a pure function that the
SSE backfill / WS catchup adapters can call to produce a bounded
iterator over a journal. Phase 6 will introduce the FastAPI router that
turns this into an HTTP endpoint; until then the helper exists so
:class:`ControlEventEmitter` is the obvious composition point for both
live (bus) and replay (journal) data sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ryotenkai_control.events.journal_reader import JournalReader
    from ryotenkai_shared.events import BaseEvent


__all__ = ["slice_journal"]


def slice_journal(
    reader: JournalReader,
    *,
    after_offset: int = -1,
    limit: int | None = None,
) -> Iterator[BaseEvent]:
    """Yield envelopes from ``reader`` with offset > ``after_offset``.

    Thin wrapper over :meth:`JournalReader.replay_from` that exists so
    HTTP / SSE adapters in Phase 6 import a stable name rather than
    reaching into the reader directly. :class:`UnknownEvent` instances
    with ``offset == UNKNOWN_OFFSET`` are filtered upstream.
    """
    return reader.replay_from(after_offset=after_offset, limit=limit)
