"""Pub/sub event bus with bounded ring buffer — placeholder for Phase 1.

Design (Phase 1 will implement):
- Append-only ring buffer of ``Event`` records keyed by monotonic offset.
- ``subscribe(since: int) -> AsyncIterator[Event]`` for WebSocket streams.
- Default capacity 10k events × ~1 KB ≈ 10 MB RAM. Configurable through
  ``RYOTENKAI_EVENT_BUFFER_SIZE``.

Used by:
- ``src/runner/api/internal.py`` — trainer pushes via loopback HTTP.
- ``src/runner/api/events.py`` — Mac client subscribes via WS.
- ``src/runner/supervisor.py`` — emits stdout/stderr / lifecycle events.
- ``src/runner/health_reporter.py`` — emits periodic GPU/RAM snapshots.

Phase 0 only declares the public interface — no real implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Event"]


@dataclass(frozen=True, slots=True)
class Event:
    """Single event written to the bus.

    The ``offset`` field is assigned by the bus, not the publisher, and
    serves as the monotonic cursor a WebSocket subscriber resumes from.

    Attributes
    ----------
    offset:
        Monotonic position in the bus. Stable across reconnects until
        the buffer wraps; then the oldest events become unreachable
        from low offsets.
    timestamp:
        ISO-8601 UTC second-precision (via ``utc_now_iso`` from
        ``src.utils.atomic_fs``).
    kind:
        Short identifier such as ``"training_started"``,
        ``"step"``, ``"checkpoint_saved"``. Free-form for now;
        Phase 3 introduces a small enum once the trainer-side schema
        stabilises.
    payload:
        JSON-serialisable body. Keep small — large blobs (tail of
        ``training.log``) belong in the chunked log endpoint, not in
        events.
    """

    offset: int
    timestamp: str
    kind: str
    payload: dict[str, Any]
