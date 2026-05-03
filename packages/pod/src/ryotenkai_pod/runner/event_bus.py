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
overflows, the oldest events are dropped.

Phase 12.B — durable journal
----------------------------
An optional :class:`~src.runner.event_journal.EventJournal` can be
passed to :meth:`EventBus.__init__`. When attached, every published
event is **also** written to a rotated JSONL file on disk
(``<workspace>/.runner/events/``). The WS handler transparently
falls back to disk replay when the requested ``since`` cursor is
older than the ring's oldest offset but still present on disk. The
``BufferTruncatedError`` exception still fires for offsets older
than the journal's oldest record.

A subscriber that asked for ``since=N`` where ``N`` is older than
the oldest event in the buffer **AND** older than the oldest record
on disk gets a ``BufferTruncatedError`` so the client can fall back
to the durable ``state.jsonl`` / ``training.log`` if those preserved
the data.

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
    "DiskJournalExhausted",
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

    Phase 12.B: when an :class:`EventJournal` is attached, this is
    raised only after disk replay is also exhausted (the WS handler's
    ``_subscribe_with_disk_fallback`` raises a more specific
    :class:`DiskJournalExhausted` instead). On a journal-less bus it's
    raised immediately as the original Phase 1 contract.

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


class DiskJournalExhausted(LookupError):
    """The ``since`` offset is older than even the journal's oldest
    persisted record.

    Phase 12.B-specific. Subclass of :class:`LookupError` so existing
    catch sites that handle :class:`BufferTruncatedError` can be
    widened with a single ``except LookupError`` if they want the
    same fallback behaviour.

    The WS handler maps this to close code 4410 (Gone).
    """

    def __init__(
        self,
        requested_offset: int,
        oldest_in_ring: int | None,
        oldest_on_disk: int | None,
    ) -> None:
        super().__init__(
            f"disk journal exhausted: requested offset {requested_offset}, "
            f"oldest_in_ring={oldest_in_ring}, oldest_on_disk={oldest_on_disk}",
        )
        self.requested_offset = requested_offset
        self.oldest_in_ring = oldest_in_ring
        self.oldest_on_disk = oldest_on_disk


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

    def __init__(
        self,
        capacity: int | None = None,
        *,
        journal: Any | None = None,
    ) -> None:
        """
        Args:
            capacity: Ring-buffer size. Defaults to env-driven
                       ``RYOTENKAI_EVENT_BUFFER_SIZE``.
            journal:  Phase 12.B — optional
                       :class:`~src.runner.event_journal.EventJournal`.
                       When attached, every published event is also
                       written to disk. The bus also reconciles its
                       starting offset from
                       ``journal.newest_persisted_offset()`` on init,
                       so a runner restart resumes the offset
                       sequence without collisions.
        """
        self._capacity = capacity or _resolve_capacity()
        # Deque of Event objects; oldest at left, newest at right.
        # Bound enforced by deque(maxlen=...).
        self._buffer: deque[Event] = deque(maxlen=self._capacity)
        # Monotonic offset assigned to the next published event.
        self._next_offset = 0
        # Phase 12.B § 2.9 — offset reconciliation across runner
        # restarts. If a journal is attached and contains records,
        # advance ``_next_offset`` past the highest persisted one so
        # the next event we emit doesn't collide with what's already
        # on disk.
        self._journal = journal
        if journal is not None:
            try:
                persisted = journal.newest_persisted_offset()
            except Exception as exc:  # noqa: BLE001 — defensive
                # A journal that can't tell us its newest offset is
                # broken; fall back to fresh start at 0 and let the
                # operator notice via subsequent events_disk_pressure
                # signalling.
                import logging
                logging.getLogger(__name__).warning(
                    "[BUS] journal.newest_persisted_offset failed: %s "
                    "— starting from 0 (offsets may collide)", exc,
                )
                persisted = None
            if persisted is not None:
                self._next_offset = persisted + 1
        # Disk-pressure rate-limit: emit at most one
        # ``events_disk_pressure`` warning per minute even if every
        # publish fails to persist.
        self._last_disk_pressure_ms: int = 0
        # Per-publish wakeup signal — replaced (atomic swap) on every
        # publish so that **multi-subscriber** broadcasts are race-free.
        #
        # The naive single-Event design (set on publish, clear on
        # consume) has a known race: if a subscriber clears AFTER a
        # second publish has already set the same flag, the second
        # event's wakeup is lost. We dodge this by treating the
        # signal as immutable per publish: each publish creates a
        # fresh ``asyncio.Event``, sets the OLD one (releasing
        # everyone waiting on it), and stores the new one for the
        # next round. Subscribers snapshot the current signal under
        # a "double-checked cursor" pattern (see :meth:`subscribe`).
        self._publish_signal = asyncio.Event()
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

    def attach_journal_rotation_listener(self) -> None:
        """Phase 14.E (V1) — register self as the journal's rotation
        observer.

        Replaces the pre-14.E circular-binding-closure pattern
        (``rotate_publisher = {"bus": None}`` mutable cell) in the
        FastAPI lifespan. Called AFTER both the bus and journal
        exist; safe no-op if no journal is attached.

        Internally wires :meth:`_publish_rotation_event` as the
        journal's ``on_rotate`` callback so every rotation
        produces an :data:`EVENTS_ROTATED` event on the bus.
        """
        if self._journal is None:
            return
        self._journal.set_rotation_callback(self._publish_rotation_event)

    def _publish_rotation_event(
        self, *, from_seq: int, to_seq: int,
        file_size_bytes: int, oldest_remaining_seq: int | None,
    ) -> None:
        """Internal — invoked by the journal on rotation.

        Phase 14.E (V1). Failure-tolerant: if the bus is in any
        state where publish fails, swallow so the journal's
        rotation pipeline isn't blocked.
        """
        # Lazy import — :data:`EVENTS_ROTATED` lives in
        # cancellation_telemetry, which already imports the bus
        # module for the disk-pressure event kind. Avoid the
        # module-load cycle.
        try:
            from src.runner.cancellation_telemetry import EVENTS_ROTATED
            self.publish(
                EVENTS_ROTATED,
                {
                    "from_seq": from_seq,
                    "to_seq": to_seq,
                    "file_size_bytes": file_size_bytes,
                    "oldest_remaining_seq": oldest_remaining_seq,
                },
            )
        except Exception:  # noqa: BLE001 — defensive
            pass

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

        # Phase 12.B — durable persistence. Write to disk BEFORE
        # waking subscribers so a slow disk doesn't bottleneck the
        # event loop on a publish/wake round-trip. Failures are
        # rate-limited; the in-memory ring still holds the event so
        # current subscribers see it normally.
        if self._journal is not None:
            try:
                self._journal.append(
                    offset=event.offset,
                    ts=event.timestamp,
                    kind=event.kind,
                    payload=dict(event.payload),
                )
            except Exception as exc:  # noqa: BLE001 — best-effort persist
                self._signal_disk_pressure(exc)

        # Atomic swap: stash the current signal, install a fresh one
        # for future waiters, then release the stashed one. Any
        # subscriber that snapshotted ``self._publish_signal`` before
        # this point is waiting on ``stale`` — we wake them. Any
        # subscriber that snapshots AFTER this point picks up the
        # fresh event and won't be affected by the ``set()`` we
        # just did. This is how we side-step the "lost wakeup"
        # race that the naive single-Event design has.
        stale = self._publish_signal
        self._publish_signal = asyncio.Event()
        stale.set()
        return event

    def _signal_disk_pressure(self, exc: BaseException) -> None:
        """Rate-limited warning when journal append fails.

        Logs at WARN level at most once per minute (per bus instance).
        Operators see "disk pressure" as a sustained pattern rather
        than every single failed write spamming the log. We do NOT
        publish a follow-up event because that could trigger a
        cascading failure if the underlying disk pressure is itself
        the cause of the journal write failing.
        """
        import logging
        import time as _time
        now_ms = int(_time.monotonic() * 1000)
        if now_ms - self._last_disk_pressure_ms < 60_000:
            return
        self._last_disk_pressure_ms = now_ms
        logging.getLogger(__name__).warning(
            "[BUS] journal append failed (rate-limited 1/min): %s: %s",
            type(exc).__name__, exc,
        )

    async def subscribe(self, *, since: int = 0) -> AsyncIterator[Event]:
        """Replay buffered events from ``since`` onwards, then live-stream.

        Algorithm:

        1. Validate ``since`` against the buffer's oldest offset:

           * ``since < oldest`` → :class:`BufferTruncatedError` (the
             client missed events we no longer have).
           * ``since > next_offset`` → :class:`ValueError` (impossible
             cursor; client confused).
           * Otherwise replay from ``since`` to ``next_offset - 1``.

        2. After the snapshot, loop on the per-publish wakeup signal:

           * Drain everything new into the consumer (they may be slow
             — that's fine, we yield each event one at a time).
           * Snapshot ``self._publish_signal`` BEFORE re-checking the
             cursor. The ordering is what makes the wait race-free —
             see ``__init__`` for the swap-pattern rationale.
           * ``await signal.wait()`` until either a publish releases
             the signal we snapshotted, or the bus closes.

        Cancellation safety: if the consumer task is cancelled
        (CancelledError propagates out of ``signal.wait()``),
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

        # Live phase. The double-check around ``signal = ...`` is
        # what makes this race-free: any publish() that completes
        # before the snapshot has already populated the buffer (so
        # ``replay_cursor < self._next_offset`` will catch it on the
        # second check); any publish() after the snapshot installs a
        # fresh ``_publish_signal``, so the ``signal.set()`` issued
        # by THAT publish wakes us. There is no window where a
        # publish can be missed.
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

            # Snapshot the current per-publish signal BEFORE the
            # final cursor re-check. See ``__init__`` and
            # ``publish`` for the swap-pattern rationale.
            signal = self._publish_signal
            if replay_cursor < self._next_offset:
                continue
            if self._closed:
                return
            await signal.wait()

    def close(self) -> None:
        """Mark the bus closed and wake every subscriber so their
        async iterators can drain and terminate.

        After ``close()``, :meth:`publish` raises :class:`RuntimeError`.
        Also closes the attached journal (Phase 12.B) so its file
        handle is fsync'd and released.
        """
        self._closed = True
        # Release whatever signal subscribers are currently waiting
        # on. We do NOT swap here — a fresh signal would defeat
        # the purpose, since post-close there will be no further
        # publish() to populate it.
        self._publish_signal.set()
        # Phase 12.B — close the journal too. Idempotent.
        if self._journal is not None:
            try:
                self._journal.close()
            except Exception:  # noqa: BLE001 — defensive on shutdown
                pass

    @property
    def journal(self) -> Any | None:
        """Phase 12.B — accessor for the attached journal.

        Used by :class:`~src.runner.api.events` to perform disk
        replay when a subscriber's ``since`` cursor falls outside
        the ring buffer. ``None`` when no journal was attached.
        """
        return self._journal
