"""In-memory event bus + multi-consumer cursors over the typed envelope.

Phase 2 rewrite (ethereal-tumbling-patterson) — the bus now traffics in
:class:`ryotenkai_shared.events.BaseEvent` envelopes instead of the
legacy ``Event(offset, timestamp, kind, payload)`` dataclass. ``publish``
accepts a pre-built envelope, assigns an offset under a per-source
``threading.Lock`` (R-05 in the plan — closes the offset-collision risk),
optionally journals to disk via the attached :class:`EventJournal`, and
fans out to subscribers via an asyncio wakeup signal.

A legacy :meth:`publish_legacy` shim is kept for callsites that emit
short-lived telemetry kinds that have NOT yet been promoted to typed
events (cancellation telemetry, watchdog, stream errors). These wrap the
payload in :class:`UnknownEvent` so the on-wire format stays uniform
without forcing a per-event-type refactor of every internal call site.

Multi-consumer cursors (R-05 in the plan, "drop-oldest metrics"): the
ring buffer is still global, but each consumer subscribes with a unique
``consumer_id`` so the bus can track ``events_dropped_total{consumer}``
per slow reader independently. Two consumers reading at different paces
get independent drop counters even though they share the underlying
deque.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.events import (
    UNKNOWN_OFFSET,
    BaseEvent,
    UnknownEvent,
    new_uuid7,
    utc_now,
)

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_journal import EventJournal


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "MAX_BUFFER_SIZE",
    "MIN_BUFFER_SIZE",
    "BufferTruncatedError",
    "DiskJournalExhausted",
    "Event",
    "EventBus",
]


# 10k × ~1 KB ≈ 10 MB — fits comfortably in memory; survives a 5 minute
# Mac sleep at 30 events/sec without truncation.
DEFAULT_BUFFER_SIZE = 10_000

# Lower bound — anything below renders detach/reattach useless even
# for short network blips.
MIN_BUFFER_SIZE = 100

# Upper bound — keeps memory under 100 MB even with verbose payloads.
MAX_BUFFER_SIZE = 100_000

# Source label stamped on legacy / unknown events synthesised by
# :meth:`EventBus.publish_legacy`. Concrete callsites that want a more
# specific URI should migrate to the typed envelope path.
_LEGACY_SOURCE_DEFAULT = "pod://runner/legacy"
_LEGACY_RUN_ID = "unknown"


# ---------------------------------------------------------------------------
# Wire-shape adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """Wire-format projection of a stored envelope.

    The bus stores :class:`BaseEvent` envelopes (rich, typed) but the
    WebSocket handler and a handful of legacy tests still consume the
    flat ``(offset, timestamp, kind, payload)`` shape. :class:`Event`
    keeps that shape stable so the WS endpoint and the disk-replay path
    can emit dicts without leaking Pydantic types onto the wire.

    The ``timestamp`` field is the canonical ISO-8601 ``Z`` string the
    pre-Phase-2 control-side WS consumer reads. The new envelope carries
    a ``time`` datetime; :func:`_event_to_wire` produces both.
    """

    offset: int
    timestamp: str
    kind: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": dict(self.payload),
        }


def legacy_kind_for(event: BaseEvent) -> str:
    """Return the legacy ``kind`` string for ``event``.

    Typed events expose the dotted kind directly. Legacy events
    (wrapped in :class:`UnknownEvent` via :meth:`EventBus.publish_legacy`)
    carry the pre-Phase-2 kind in ``original_type``. This helper
    centralises the lookup so test introspection and the WS wire
    projection agree on the "name" of an event without forcing every
    test to handle both shapes.
    """
    if isinstance(event, UnknownEvent):
        return event.original_type
    # Strip the dotted-namespace prefix so ``ryotenkai.pod.lifecycle.
    # stop_requested`` reads back as ``stop_requested`` — that's what
    # legacy consumers (and the supervisor's existing grep patterns)
    # match against.
    legacy_aliases = {
        "ryotenkai.pod.lifecycle.stop_requested": "stop_requested",
        "ryotenkai.pod.lifecycle.job_submitted": "job_submitted",
        "ryotenkai.pod.lifecycle.trainer_spawned": "trainer_spawned",
        "ryotenkai.pod.lifecycle.trainer_spawn_failed": "spawn_failed",
        "ryotenkai.pod.lifecycle.plugins_unpacked": "plugins_unpacked",
        "ryotenkai.pod.health.snapshot": "health_snapshot",
        "ryotenkai.pod.health.idle_detected": "idle_detector_triggered",
        "ryotenkai.pod.journal.rotated": "events_rotated",
        "ryotenkai.pod.journal.disk_pressure": "events_disk_pressure",
    }
    return legacy_aliases.get(event.kind, event.kind)


def envelope_to_wire(event: BaseEvent) -> dict[str, Any]:
    """Render an envelope as the dict the WS handler ships to clients.

    Adds the legacy ``timestamp`` field (ISO-8601 ``Z``) alongside the
    canonical envelope ``time`` so control-side consumers reading
    ``event.get("timestamp")`` keep working unchanged through Phase 2.
    Phase 4 will switch them to ``time`` and this helper can drop the
    duplicate. The body is materialised via ``model_dump`` so timezone
    handling and UUID serialisation match the codec path.
    """
    import json as _json

    payload = _json.loads(event.model_dump_json())
    # Backward-compat alias: the control-side WS consumer still reads
    # ``timestamp`` (ISO-8601 ``Z`` string). The envelope's canonical
    # field is ``time``. Mirror it without mutating the envelope.
    payload["timestamp"] = payload.get("time", "")
    # Preserve the dotted ``kind`` AND publish a legacy alias under
    # ``kind`` so pre-Phase-4 control-side consumers keep matching on
    # the pre-rewrite identifiers (e.g. ``training_started``,
    # ``stop_requested``). The dotted form remains available under
    # ``kind_dotted`` for forward-looking consumers.
    legacy = legacy_kind_for(event)
    payload["kind_dotted"] = payload.get("kind", "")
    if legacy and legacy != payload.get("kind"):
        payload["kind"] = legacy
    # Legacy-payload projection: for :class:`UnknownEvent` envelopes
    # the open dict was the original payload. Surface it under
    # ``payload`` so control-side ``event.get("payload")`` still
    # returns the same dict the pre-rewrite consumer saw.
    if isinstance(event, UnknownEvent):
        payload["payload"] = dict(event.raw_payload)
    return payload


def _event_to_legacy_view(event: BaseEvent) -> Event:
    """Project a stored envelope into the legacy :class:`Event` shape.

    ``payload`` is the typed payload's dict form (or ``raw_payload`` for
    :class:`UnknownEvent`). The ISO-8601 ``Z`` timestamp matches the
    pre-Phase-2 wire format.
    """
    if isinstance(event, UnknownEvent):
        payload_dict: dict[str, Any] = dict(event.raw_payload)
        kind = event.original_type
    else:
        payload_obj = getattr(event, "payload", None)
        if hasattr(payload_obj, "model_dump"):
            payload_dict = payload_obj.model_dump()  # type: ignore[union-attr]
        elif isinstance(payload_obj, dict):
            payload_dict = dict(payload_obj)
        else:
            payload_dict = {}
        kind = event.kind

    ts = event.time.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z",
    )
    return Event(offset=event.offset, timestamp=ts, kind=kind, payload=payload_dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BufferTruncatedError(LookupError):
    """The ``since`` offset requested by a subscriber is older than the
    oldest event still in the buffer.

    When an :class:`EventJournal` is attached the WS handler may attempt
    a disk replay first; if even the journal lacks the requested offset
    it raises :class:`DiskJournalExhausted` instead.
    """

    def __init__(self, requested_offset: int, oldest_available: int) -> None:
        super().__init__(
            f"event buffer truncated: requested offset {requested_offset}, "
            f"oldest available {oldest_available}",
        )
        self.requested_offset = requested_offset
        self.oldest_available = oldest_available


class DiskJournalExhausted(LookupError):
    """The ``since`` offset is older than even the journal's oldest record."""

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
    """Bounded ring buffer + async pub/sub keyed on typed envelopes.

    Phase 2 contract:

    * :meth:`publish` accepts a fully-built :class:`BaseEvent` envelope
      and assigns ``offset`` if the caller passed :data:`UNKNOWN_OFFSET`.
      Caller MUST supply ``source`` (the URI is authoritative at the
      construction site).
    * Per-source offset counters are guarded by a ``threading.Lock`` so
      callbacks running on the trainer pump thread and the asyncio loop
      cannot collide on the same source's offset (R-05 in the plan).
    * :meth:`publish_legacy` is a back-compat shim for telemetry kinds
      that have not yet been promoted to typed events. It wraps the
      payload in :class:`UnknownEvent` so journal / WS / SSE consumers
      see a uniform envelope shape.
    * :meth:`subscribe` yields :class:`BaseEvent` envelopes per the
      Phase-2 contract. The WS handler uses :func:`envelope_to_wire` to
      keep the legacy ``timestamp`` field on the dict it ships to
      clients (no change required on control-side).
    """

    def __init__(
        self,
        capacity: int | None = None,
        *,
        journal: EventJournal | None = None,
    ) -> None:
        """
        Args:
            capacity: Ring-buffer size. Defaults to env-driven
                       ``RYOTENKAI_EVENT_BUFFER_SIZE``.
            journal:  Optional :class:`EventJournal`. When attached,
                       every published event is also persisted to disk
                       via :func:`ryotenkai_shared.events.to_jsonl`. On
                       construction the bus reconciles its next-offset
                       counter from the journal's newest persisted
                       record so a runner restart resumes the offset
                       sequence without collisions.
        """
        self._capacity = capacity or _resolve_capacity()
        self._buffer: deque[BaseEvent] = deque(maxlen=self._capacity)
        # Global monotonic offset counter — the WS subscriber uses a
        # single ``since`` integer so the wire-protocol offset is
        # globally monotonic. ``_offset_lock`` guards the counter so
        # concurrent publishers (asyncio task + HTTP loopback callback
        # in worker thread) cannot hand out colliding offsets (R-05).
        self._next_offset = 0
        # Per-source ``last_offset`` for telemetry / debugging. The
        # source-keyed counter is a strict subset of the global one and
        # is recomputed on each publish so callers can ask "where is
        # source X up to" without scanning the buffer.
        self._offset_counters: dict[str, int] = {}
        self._offset_lock = threading.Lock()
        self._journal = journal
        # Reconcile from journal on init so a runner restart resumes the
        # offset sequence without colliding with what's already on disk.
        if journal is not None:
            try:
                persisted = journal.newest_persisted_offset()
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "[BUS] journal.newest_persisted_offset failed: %s "
                    "— starting fresh", exc,
                )
                persisted = None
            if persisted is not None:
                self._next_offset = persisted + 1
        # Per-consumer drop counters; populated when a subscriber falls
        # behind the bus' wakeup pace. Keyed by ``consumer_id`` so a
        # slow WS subscriber doesn't poison the disk-replay path's
        # counter.
        self._dropped_per_consumer: dict[str, int] = {}
        self._last_disk_pressure_ms: int = 0
        # Per-publish wakeup signal — see :meth:`publish`.
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
        """Offset that will be assigned to the *next* published event."""
        with self._offset_lock:
            return self._next_offset

    @property
    def oldest_offset(self) -> int | None:
        """Smallest offset still resident in the ring, or ``None`` if empty."""
        if not self._buffer:
            return None
        return self._buffer[0].offset

    @property
    def dropped_per_consumer(self) -> Mapping[str, int]:
        """Read-only view of per-consumer drop counters (R-05 metric).

        Phase 2 reserves the field for future SSE / WS slow-consumer
        bookkeeping; the bus itself doesn't drop events from the ring
        (the deque's ``maxlen`` does), so the counter here is bumped
        only when a subscriber's local queue overflows in the consumer
        adapter layer. The accessor is exposed now so emitters and
        adapters can write through to it.
        """
        return self._dropped_per_consumer

    def record_consumer_drop(self, consumer_id: str, count: int = 1) -> None:
        """Increment the drop counter for ``consumer_id`` by ``count``.

        Adapters (SSE, WS, async-queue) call this when they shed events
        from their own bounded queue. Centralising the counter on the
        bus keeps observability surface uniform: one metric source per
        run regardless of how many fan-out paths exist.
        """
        if count <= 0:
            return
        self._dropped_per_consumer[consumer_id] = (
            self._dropped_per_consumer.get(consumer_id, 0) + count
        )

    def attach_journal_rotation_listener(self) -> None:
        """Register the bus as the journal's rotation observer.

        Wires :meth:`_publish_rotation_event` as the journal's
        ``on_rotate`` callback so every rotation emits a
        :class:`JournalRotatedEvent` on the bus. Idempotent no-op when
        no journal is attached.
        """
        if self._journal is None:
            return
        self._journal.set_rotation_callback(self._publish_rotation_event)

    def _publish_rotation_event(
        self, *, from_seq: int, to_seq: int,
        file_size_bytes: int, oldest_remaining_seq: int | None,
    ) -> None:
        """Internal — invoked by the journal on rotation.

        Failure-tolerant: if the bus is in any state where publish
        fails, swallow so the journal's rotation pipeline isn't
        blocked.
        """
        try:
            from ryotenkai_shared.events.types.pod_journal import (
                JournalRotatedEvent,
                JournalRotatedPayload,
            )
            ev = JournalRotatedEvent(
                source="pod://runner/journal",
                run_id=_LEGACY_RUN_ID,
                offset=UNKNOWN_OFFSET,
                payload=JournalRotatedPayload(
                    from_seq=from_seq,
                    to_seq=to_seq,
                    file_size_bytes=file_size_bytes,
                    oldest_remaining_seq=oldest_remaining_seq,
                ),
            )
            self.publish(ev)
        except Exception:
            pass

    def _assign_offset_locked(self, source: str) -> int:
        """Hand out the next global monotonic offset, bumping source bookkeeping.

        Thread-safe — the counter is mutated only while ``_offset_lock``
        is held. Concurrent publishers from different threads (asyncio
        loop + HTTP loopback callback executor) get distinct offsets.
        """
        with self._offset_lock:
            offset = self._next_offset
            self._next_offset = offset + 1
            self._offset_counters[source] = offset + 1
            return offset

    def publish(self, event: BaseEvent) -> int:
        """Append a pre-built envelope to the ring and wake subscribers.

        Contracts:

        * ``event.source`` is authoritative — never overwritten.
        * If ``event.offset == UNKNOWN_OFFSET`` (or negative) the bus
          assigns the next offset for that source under
          ``_offset_lock``. Otherwise the caller-supplied offset is
          preserved verbatim (this matches the ``emit_remote`` semantic
          where pod-relayed events arrive already-numbered).
        * ``event.event_id`` and ``event.time`` are kept as-is — the
          envelope's ``default_factory`` already produced sensible
          values at construction time.
        * Returns the assigned offset so callers can correlate.
        """
        if self._closed:
            raise RuntimeError("event bus is closed")

        if event.offset < 0 or event.offset == UNKNOWN_OFFSET:
            new_offset = self._assign_offset_locked(event.source)
            event = event.model_copy(update={"offset": new_offset})
        else:
            # Caller-numbered event (typically replayed from a remote
            # producer via emit_remote). Bump the global counter past
            # the caller-supplied offset so a future auto-assign can't
            # collide.
            with self._offset_lock:
                if event.offset >= self._next_offset:
                    self._next_offset = event.offset + 1
                self._offset_counters[event.source] = max(
                    self._offset_counters.get(event.source, 0),
                    event.offset + 1,
                )

        self._buffer.append(event)

        # Durable persistence. Write to disk BEFORE waking subscribers
        # so a slow disk doesn't bottleneck the event loop on a
        # publish/wake round-trip. Failures are rate-limited; the
        # in-memory ring still holds the event so live subscribers see
        # it normally.
        if self._journal is not None:
            try:
                self._journal.append_envelope(event)
            except Exception as exc:
                self._signal_disk_pressure(exc)

        # Atomic swap: stash the current signal, install a fresh one
        # for future waiters, then release the stashed one. Any
        # subscriber that snapshotted ``self._publish_signal`` before
        # this point wakes; any subscriber that snapshots AFTER this
        # point picks up the fresh event and won't be affected.
        stale = self._publish_signal
        self._publish_signal = asyncio.Event()
        stale.set()
        return event.offset

    def publish_legacy(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        source: str | None = None,
        run_id: str | None = None,
        timestamp: str | None = None,
    ) -> Event:
        """Legacy ``(kind, payload)`` shim — wraps into :class:`UnknownEvent`.

        Kept for callsites that emit telemetry kinds NOT yet promoted to
        typed events (cancellation_started/completed, watchdog_timeout,
        stop_escalated, gpu_idle_*, stream_error, spawn_failed,
        events_disk_pressure). Behaviour matches the pre-Phase-2 API
        closely enough that no test rewrite is required for these:

        * Synthesises an :class:`UnknownEvent` envelope with the given
          ``kind`` as ``original_type`` and the payload as
          ``raw_payload``.
        * Returns a legacy :class:`Event` view (``offset``,
          ``timestamp``, ``kind``, ``payload``) — the test surface that
          asserts on these fields keeps working unchanged.
        """
        if self._closed:
            raise RuntimeError("event bus is closed")

        envelope_source = source or _LEGACY_SOURCE_DEFAULT
        envelope_run_id = run_id or _LEGACY_RUN_ID

        if timestamp is None:
            now = utc_now()
        else:
            try:
                now = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                now = utc_now()

        unknown = UnknownEvent(
            event_id=new_uuid7(),
            source=envelope_source,
            time=now,
            run_id=envelope_run_id,
            offset=UNKNOWN_OFFSET,
            severity="info",
            original_type=kind,
            raw_payload=dict(payload),
        )
        assigned = self.publish(unknown)
        stored = self._buffer[-1]
        return Event(
            offset=assigned,
            timestamp=stored.time.astimezone(UTC).replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            kind=kind,
            payload=dict(payload),
        )

    def _signal_disk_pressure(self, exc: BaseException) -> None:
        """Rate-limited warning when journal append fails.

        Logs at WARN level at most once per minute. We do NOT publish a
        follow-up event because that could cascade if the underlying
        disk pressure is itself the cause.
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

    async def subscribe(
        self,
        *,
        since: int = 0,
        consumer_id: str = "default",
    ) -> AsyncIterator[BaseEvent]:
        """Replay buffered envelopes ≥ ``since``, then live-stream.

        Multi-consumer cursors: each subscriber names itself via
        ``consumer_id`` so :meth:`record_consumer_drop` can track per-
        consumer overflow independently. The bus does NOT itself drop
        from a per-consumer queue (the global ring + ``maxlen`` does
        the bounding); subscribers that want bounded fan-out should
        wrap this iterator with their own queue and call
        :meth:`record_consumer_drop` on overflow.

        Raises:
            BufferTruncatedError: when ``since`` is older than the
                oldest envelope still resident.
            ValueError: when ``since`` is negative or beyond the
                current cursor.
        """
        _ = consumer_id  # accepted for the counter API; bus is shared
        if since < 0:
            raise ValueError(f"since must be non-negative, got {since}")
        if since > self.next_offset:
            raise ValueError(
                f"since={since} is beyond the current cursor "
                f"({self.next_offset}); client cursor is corrupt",
            )

        oldest = self.oldest_offset
        if oldest is not None and since < oldest:
            raise BufferTruncatedError(requested_offset=since, oldest_available=oldest)

        # Replay phase — snapshot list() to avoid mutating the deque
        # while iterating.
        replay_cursor = since
        for event in list(self._buffer):
            if event.offset >= replay_cursor:
                yield event
                replay_cursor = event.offset + 1

        # Live phase. The double-check around ``signal = ...`` is
        # what makes this race-free (see ``publish`` for the swap
        # pattern rationale).
        while True:
            if self._closed and replay_cursor >= self.next_offset:
                return

            if replay_cursor < self.next_offset:
                for event in list(self._buffer):
                    if event.offset >= replay_cursor:
                        yield event
                        replay_cursor = event.offset + 1
                continue

            signal = self._publish_signal
            if replay_cursor < self.next_offset:
                continue
            if self._closed:
                return
            await signal.wait()

    async def subscribe_legacy(
        self, *, since: int = 0, consumer_id: str = "default",
    ) -> AsyncIterator[Event]:
        """Legacy WS-shaped subscribe — yields :class:`Event` projections.

        Phase 2 keeps the legacy shape available so the WS handler can
        emit ``{offset, timestamp, kind, payload}`` dicts without
        reaching into :class:`BaseEvent` reflection at each frame.
        """
        async for envelope in self.subscribe(since=since, consumer_id=consumer_id):
            yield _event_to_legacy_view(envelope)

    def iter_buffered_envelopes(self) -> Iterator[BaseEvent]:
        """Snapshot iterator over currently-resident envelopes.

        Used by the HTTP replay endpoint (``GET /jobs/.../events/replay``)
        to scan the ring without holding it locked across an HTTP write.
        """
        return iter(list(self._buffer))

    def close(self) -> None:
        """Mark the bus closed and wake every subscriber.

        After ``close()``, :meth:`publish` raises :class:`RuntimeError`.
        Also closes the attached journal (idempotent).
        """
        self._closed = True
        self._publish_signal.set()
        if self._journal is not None:
            with contextlib.suppress(Exception):
                self._journal.close()

    @property
    def journal(self) -> EventJournal | None:
        """Accessor for the attached journal (``None`` if not attached)."""
        return self._journal
