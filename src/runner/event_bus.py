"""In-memory event bus with offset-based ring buffer + async pub/sub.

The bus is the **fan-out point** for everything that happens inside
the pod after :meth:`JobLifecycleFSM.submit`:

- Supervisor — emits stdout / stderr chunks, lifecycle events
  (``trainer_started`` / ``trainer_exited``), exit-code parsing
  results.
- HealthReporter — emits periodic GPU / RAM snapshots.
- IdleDetector — emits ``idle_detected`` warnings.
- TrainerCallback (loopback HTTP) — emits per-step / per-epoch
  metrics.

Subscribers (Mac client over WebSocket) consume events via
:meth:`EventBus.subscribe(since)` which is an async iterator that:

1. Replays everything in the ring buffer with offset ≥ ``since``.
2. Then live-streams new events as they arrive, in order.
3. Closes when the bus is closed (``close()``) or the subscriber
   cancels its task.

Buffer is a **bounded deque** of capacity ``RYOTENKAI_EVENT_BUFFER_SIZE``
(default 10 000 events ≈ 10 MB at ~1 KB / event). When the buffer
overflows, the oldest events are dropped — a subscriber that asked
for ``since=N`` where ``N`` is older than the oldest event in the
buffer gets a ``BufferTruncatedError`` so the client can decide
whether to fall back to the durable JSONL on disk (Phase 1+ may add
a chunked endpoint that streams the JSONL).

Threading model
---------------

The bus is **asyncio-native**. Publishers can be sync or async — the
``publish()`` method is sync (so the supervisor's stdout pump and
TrainerCallback's HTTP handler don't need an event loop reference) but
the wakeup signal goes through ``asyncio.Event.set()`` which is
thread-safe when called via ``loop.call_soon_threadsafe``. Phase 1
keeps everything single-loop; the cross-thread variant is documented
as "Phase 5+" — see :meth:`publish` docstring.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "BufferTruncatedError",
    "Event",
    "EventBus",
    "DEFAULT_BUFFER_SIZE",
    "MIN_BUFFER_SIZE",
    "MAX_BUFFER_SIZE",
]


# 10k × ~1 KB ≈ 10 MB — fits comfortably in memory; survives a 5 minute
# Mac sleep at 30 events/sec without truncation.
DEFAULT_BUFFER_SIZE = 10_000

# Lower bound — anything below renders detach/reattach useless even
# for short network blips.
MIN_BUFFER_SIZE = 100

# Upper bound — keeps memory under 100 MB even with verbose payloads.
MAX_BUFFER_SIZE = 100_000


# ---------------------------------------------------------------------------
# Event payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """Single record on the bus.

    Attributes
    ----------
    offset:
        Monotonic position assigned by :meth:`EventBus.publish`. Cursor
        the WebSocket subscriber resumes from. The first event after
        bus construction has offset 0; subsequent events strictly
        increase by 1 each.
    timestamp:
        ISO-8601 UTC second-precision (set by the bus, not the
        publisher — keeps ordering total even when publishers' clocks
        skew).
    kind:
        Short label, eg ``"training_started"``, ``"step"``,
        ``"checkpoint_saved"``, ``"gpu_snapshot"``. Free-form in
        Phase 1; Phase 3 may stabilise an enum once the trainer-side
        event vocabulary settles.
    payload:
        JSON-serialisable mapping. Keep small — large blobs (tail of
        ``training.log``) belong in the chunked log endpoint, not
        here.
    """

    offset: int
    timestamp: str
    kind: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BufferTruncatedError(LookupError):
    """The ``since`` offset requested by a subscriber is older than the
    oldest event still in the buffer.

    The client can recover by either (a) accepting the loss and
    re-subscribing with the bus's current oldest offset, or (b)
    fetching missing events from the durable ``state.jsonl`` /
    ``training.log`` if the data is preserved there.
    """

    def __init__(self, requested_offset: int, oldest_available: int) -> None:
        super().__init__(
            f"event buffer truncated: requested offset {requested_offset}, "
            f"oldest available {oldest_available}",
        )
        self.requested_offset = requested_offset
        self.oldest_available = oldest_available


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------


def _resolve_capacity() -> int:
    """Read capacity from the env var, fall back to default, clamp."""
    raw = os.environ.get("RYOTENKAI_EVENT_BUFFER_SIZE", "").strip()
    if not raw:
        return DEFAULT_BUFFER_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BUFFER_SIZE
    return max(MIN_BUFFER_SIZE, min(MAX_BUFFER_SIZE, value))


class EventBus:
    """Bounded ring buffer + async pub/sub with offset cursor.

    Construction is cheap; one bus per FastAPI app instance is plenty
    for Phase 1 (single-active job). Phase 5+ may bind one bus per
    job_id — the API stays the same.
    """

    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = capacity or _resolve_capacity()
        # Deque of Event objects; oldest at left, newest at right.
        # Bound enforced by deque(maxlen=...).
        self._buffer: deque[Event] = deque(maxlen=self._capacity)
        # Monotonic offset assigned to the next published event.
        self._next_offset = 0
        # Set whenever a new event is published; cleared when consumers
        # observe it. Subscribers wait on this in their async iter loop.
        self._wakeup = asyncio.Event()
        self._closed = False

    # --- public API -----------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def next_offset(self) -> int:
        """Offset that will be assigned to the *next* published event.

        Equal to the count of events ever published since construction.
        """
        return self._next_offset

    @property
    def oldest_offset(self) -> int | None:
        """Smallest offset still resident in the buffer, or ``None``
        if the buffer is empty."""
        if not self._buffer:
            return None
        return self._buffer[0].offset

    def publish(
        self, kind: str, payload: Mapping[str, Any], *, timestamp: str | None = None,
    ) -> Event:
        """Append an event to the buffer and wake all subscribers.

        Sync callable — usable from non-async code (supervisor's
        stdout pump, TrainerCallback HTTP handler dispatch). Phase 1
        assumes the caller is already on the FastAPI event loop's
        thread; Phase 5+ may add ``loop.call_soon_threadsafe`` if
        cross-thread publishers appear.

        Returns the freshly-constructed :class:`Event` so callers can
        echo it for logging / persistence without a second lookup.
        """
        if self._closed:
            raise RuntimeError("event bus is closed")

        # Inline timestamp helper to keep the bus dependency-free —
        # mirrors the same pattern used in :mod:`src.runner.state`.
        # Phase 6 cutover migrates these to ``src.utils.atomic_fs.utc_now_iso``.
        if timestamp is None:
            timestamp = (
                datetime.now(UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )

        event = Event(
            offset=self._next_offset,
            timestamp=timestamp,
            kind=kind,
            payload=dict(payload),  # defensive copy — frozen dataclass holds the dict
        )
        self._buffer.append(event)
        self._next_offset += 1

        # Wake every waiting subscribe() iterator.
        self._wakeup.set()
        return event

    async def subscribe(self, *, since: int = 0) -> AsyncIterator[Event]:
        """Replay buffered events from ``since`` onwards, then live-stream.

        Algorithm:

        1. Validate ``since`` against the buffer's oldest offset:

           * ``since < oldest`` → :class:`BufferTruncatedError` (the
             client missed events we no longer have).
           * ``since > next_offset`` → :class:`ValueError` (impossible
             cursor; client confused).
           * Otherwise replay from ``since`` to ``next_offset - 1``.

        2. After the snapshot, loop on the wakeup event:

           * ``await self._wakeup.wait()`` until a new event arrives
             or the bus closes.
           * Drain everything new into the consumer (they may be slow
             — that's fine, we yield each event one at a time).
           * Reset wakeup *only* if no new events have piled up since
             the last drain; otherwise keep it set.

        Cancellation safety: if the consumer task is cancelled
        (CancelledError propagates out of ``await self._wakeup.wait()``),
        the iterator simply terminates — no resources to release
        beyond Python GC.
        """
        if since < 0:
            raise ValueError(f"since must be non-negative, got {since}")
        if since > self._next_offset:
            raise ValueError(
                f"since={since} is beyond the current cursor "
                f"({self._next_offset}); client cursor is corrupt",
            )

        oldest = self.oldest_offset
        if oldest is not None and since < oldest:
            raise BufferTruncatedError(requested_offset=since, oldest_available=oldest)

        # Replay phase — capture a snapshot to avoid mutating the
        # deque while iterating.
        replay_cursor = since
        for event in list(self._buffer):
            if event.offset >= replay_cursor:
                yield event
                replay_cursor = event.offset + 1

        # Live phase.
        while True:
            if self._closed and replay_cursor >= self._next_offset:
                return

            # Yield anything that arrived between replay and our wait.
            if replay_cursor < self._next_offset:
                for event in list(self._buffer):
                    if event.offset >= replay_cursor:
                        yield event
                        replay_cursor = event.offset + 1
                continue

            await self._wakeup.wait()
            self._wakeup.clear()

    def close(self) -> None:
        """Mark the bus closed and wake every subscriber so their
        async iterators can drain and terminate.

        After ``close()``, :meth:`publish` raises :class:`RuntimeError`.
        """
        self._closed = True
        self._wakeup.set()
