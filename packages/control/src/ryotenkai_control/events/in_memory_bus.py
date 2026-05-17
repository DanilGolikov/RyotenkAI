"""Ray-style ``MultiConsumerEventBuffer`` for control-side fan-out (Phase 3).

A single global ring (``deque(maxlen=capacity)``) backed by per-consumer
cursors stored in a dict. When a consumer subscribes:

* It picks up at ``after_offset`` (or at the current tail if ``after_offset``
  is ``None``).
* If the consumer's cursor falls behind ``deque[0].offset`` (the oldest
  event still resident), the events between its cursor and ``deque[0]``
  are counted as drops for that consumer specifically. The bus does NOT
  re-yield them — they're gone from the ring.
* :meth:`publish` is non-blocking and lock-protected; on overflow the
  oldest event drops automatically because ``deque(maxlen=)`` evicts
  from the left side.

Why a ring buffer and not an unbounded queue? Slow consumers cannot
exert backpressure on producers (the orchestrator's main loop must not
stall on an SSE client's network). Bounded fan-out means slow consumers
get visible drop counters; producers stay fast.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import deque
from typing import TYPE_CHECKING

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ryotenkai_shared.events import BaseEvent


logger = get_logger(__name__)


DEFAULT_CAPACITY = 10_000
MIN_CAPACITY = 1
MAX_CAPACITY = 1_000_000


__all__ = [
    "DEFAULT_CAPACITY",
    "InMemoryBus",
]


class InMemoryBus:
    """Bounded ring buffer + per-consumer cursors over typed envelopes."""

    def __init__(self, *, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < MIN_CAPACITY:
            raise ValueError(f"capacity must be >= {MIN_CAPACITY}")
        if capacity > MAX_CAPACITY:
            raise ValueError(f"capacity must be <= {MAX_CAPACITY}")

        self._capacity = capacity
        self._buffer: deque[BaseEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

        # Per-consumer cursor — the offset the consumer LAST yielded.
        # On next ``publish``-triggered wakeup the consumer resumes at
        # ``cursor + 1`` (or higher if events between were evicted).
        self._consumer_cursors: dict[str, int] = {}

        # Drop counters. ``_dropped_total`` ticks each time the ring
        # ejects an event because ``deque.maxlen`` was reached;
        # ``_dropped_per_consumer`` ticks each time a consumer's cursor
        # falls behind the oldest resident event.
        self._dropped_total = 0
        self._dropped_per_consumer: dict[str, int] = {}

        # Phase 8 — observability counters scraped by the health endpoint
        # (``GET /api/v1/health/events``). ``_published_total`` ticks on
        # every publish (including the ones that immediately evict).
        # ``_subscriber_count`` tracks live ``subscribe()`` generators so
        # the health view can spot leaked SSE connections.
        self._published_total = 0
        self._subscriber_count = 0

        # Wakeup mechanism for async subscribers. We rebuild the Event
        # on every publish so concurrent subscribers don't race on
        # ``.clear()``.
        self._publish_signal = asyncio.Event()
        self._closed = False

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def dropped_total(self) -> int:
        return self._dropped_total

    @property
    def dropped_per_consumer(self) -> dict[str, int]:
        """Snapshot of per-consumer drop counts (defensive copy)."""
        with self._lock:
            return dict(self._dropped_per_consumer)

    @property
    def oldest_offset(self) -> int | None:
        with self._lock:
            return self._buffer[0].offset if self._buffer else None

    @property
    def newest_offset(self) -> int | None:
        with self._lock:
            return self._buffer[-1].offset if self._buffer else None

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def published_total(self) -> int:
        """Total ``publish()`` calls observed (including overflows)."""
        return self._published_total

    @property
    def current_depth(self) -> int:
        """Current ring length — alias of :attr:`buffered_count`.

        Exists so the Phase 8 metrics aggregator can call the same
        attribute name across emitter/bus/journal/dedup without
        special-casing.
        """
        with self._lock:
            return len(self._buffer)

    @property
    def subscriber_count(self) -> int:
        """Number of active :meth:`subscribe` generators.

        Incremented on entry to the async generator body, decremented in
        the ``finally`` block so a torn ``async for`` still releases the
        slot.
        """
        return self._subscriber_count

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, event: BaseEvent) -> None:
        """Append ``event`` to the ring and wake subscribers.

        Non-blocking. On overflow the oldest event is evicted by the
        deque's ``maxlen``; we increment ``_dropped_total`` to surface
        the loss in metrics. Closed buses silently drop publishes
        (matches the never-raises contract on the wrapping emitter).
        """
        if self._closed:
            logger.debug(
                "[InMemoryBus] publish after close ignored: kind=%s",
                event.kind,
            )
            return

        with self._lock:
            if len(self._buffer) == self._capacity:
                self._dropped_total += 1
            self._buffer.append(event)
            self._published_total += 1

            # Atomic swap pattern from pod-side EventBus — replace the
            # signal so subscribers snapshotting AFTER this point pick
            # up a fresh waitable.
            stale = self._publish_signal
            self._publish_signal = asyncio.Event()
        # ``set()`` is async-safe; deliberately release the lock first
        # so a subscriber waking up under the loop can grab the lock.
        # No running loop bound to this Event yet → swallow the
        # RuntimeError; subscribers will see the new buffer state when
        # they next snapshot.
        with contextlib.suppress(RuntimeError):
            stale.set()

    def record_consumer_drop(self, consumer_id: str, count: int = 1) -> None:
        """Increment the per-consumer drop counter by ``count``."""
        if count <= 0:
            return
        with self._lock:
            self._dropped_per_consumer[consumer_id] = (
                self._dropped_per_consumer.get(consumer_id, 0) + count
            )

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        consumer_id: str,
        *,
        after_offset: int | None = None,
    ) -> AsyncIterator[BaseEvent]:
        """Yield events for ``consumer_id``, optionally starting from a cursor.

        When ``after_offset`` is ``None`` the consumer tails — yielding
        only events published AFTER subscription. When it's an int the
        consumer first drains the resident range ``(after_offset,
        newest]`` then falls through to live tailing.

        Slow-consumer accounting: if ``after_offset < oldest_offset``
        the bus increments ``dropped_per_consumer[consumer_id]`` by
        ``oldest_offset - after_offset`` (the events that were already
        evicted from the ring when the consumer asked for them) before
        resuming at ``oldest_offset``.
        """
        # Resolve the starting cursor under the lock so we get a
        # consistent snapshot. The cursor is "the highest offset the
        # consumer has already received" — next yield is ``cursor + 1``.
        with self._lock:
            if after_offset is None:
                cursor = self._buffer[-1].offset if self._buffer else -1
            else:
                cursor = after_offset
                oldest = self._buffer[0].offset if self._buffer else None
                if oldest is not None and cursor < oldest - 1:
                    missed = (oldest - 1) - cursor
                    if missed > 0:
                        self._dropped_per_consumer[consumer_id] = (
                            self._dropped_per_consumer.get(consumer_id, 0) + missed
                        )
                        cursor = oldest - 1
            self._consumer_cursors[consumer_id] = cursor
            # Phase 8 — count this slot before yielding so a slow generator
            # body still observable to the health endpoint.
            self._subscriber_count += 1

        try:
            while True:
                # Drain any resident events ahead of the cursor under lock.
                pending: list[BaseEvent] = []
                with self._lock:
                    for ev in self._buffer:
                        if ev.offset > cursor:
                            pending.append(ev)
                    # Snapshot the signal AFTER the drain so we don't
                    # miss a wakeup from a publish racing in between.
                    signal = self._publish_signal
                    closed = self._closed

                if pending:
                    for ev in pending:
                        cursor = ev.offset
                        with self._lock:
                            self._consumer_cursors[consumer_id] = cursor
                        yield ev
                    continue

                if closed:
                    return

                await signal.wait()
        finally:
            # Final cursor write — defensive cleanup so a torn ``async
            # for`` still leaves an observable position. Decrement
            # subscriber count under the same lock so a leaked generator
            # body (no ``aclose()``) on the caller side eventually
            # releases the slot once Python finalizes it.
            with self._lock:
                self._consumer_cursors[consumer_id] = cursor
                if self._subscriber_count > 0:
                    self._subscriber_count -= 1

    def consumer_cursor(self, consumer_id: str) -> int | None:
        """Return the consumer's last-yielded offset, or ``None`` if unknown."""
        with self._lock:
            return self._consumer_cursors.get(consumer_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Mark closed and wake every subscriber so async iterators terminate."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stale = self._publish_signal
            self._publish_signal = asyncio.Event()
        with contextlib.suppress(RuntimeError):
            stale.set()
