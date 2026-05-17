"""Phase 2 (ethereal-tumbling-patterson) — :class:`EventBus` envelope contract.

Replaces the legacy ``publish(kind, payload)`` test suite with coverage
of the typed envelope path:

* :meth:`publish` accepts a :class:`BaseEvent` envelope, assigns offset
  if ``UNKNOWN_OFFSET`` was passed, and exposes both ``next_offset``
  and ``oldest_offset`` accessors.
* :meth:`publish_legacy` keeps the (kind, payload) wrapper alive for
  internal telemetry kinds that have not yet been promoted to typed
  events — it produces an :class:`UnknownEvent` envelope under the
  hood so the on-wire format stays uniform.
* :meth:`subscribe` yields :class:`BaseEvent` envelopes; the WS layer
  uses :func:`envelope_to_wire` to ship dicts.
"""

from __future__ import annotations

import asyncio

import pytest

from ryotenkai_pod.runner.event_bus import (
    DEFAULT_BUFFER_SIZE,
    MAX_BUFFER_SIZE,
    MIN_BUFFER_SIZE,
    BufferTruncatedError,
    EventBus,
)
from ryotenkai_shared.events import UNKNOWN_OFFSET, BaseEvent
from ryotenkai_shared.events.types.pod_lifecycle import (
    TrainerSpawnedEvent,
    TrainerSpawnedPayload,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def bus() -> EventBus:
    return EventBus(capacity=10)


def _make_event(*, pid: int = 1, source: str = "pod://test/runner") -> TrainerSpawnedEvent:
    return TrainerSpawnedEvent(
        source=source,
        run_id="test",
        offset=UNKNOWN_OFFSET,
        payload=TrainerSpawnedPayload(pid=pid, cmdline="py", cwd="/tmp"),
    )


async def drain_n(
    bus: EventBus, *, n: int, since: int = 0, timeout: float = 1.0,
) -> list[BaseEvent]:
    async def _drain() -> list[BaseEvent]:
        out: list[BaseEvent] = []
        agen = bus.subscribe(since=since)
        try:
            async for event in agen:
                out.append(event)
                if len(out) >= n:
                    break
        finally:
            await agen.aclose()
        return out

    return await asyncio.wait_for(_drain(), timeout=timeout)


# ---------------------------------------------------------------------------
# Publish — typed envelope
# ---------------------------------------------------------------------------


class TestPublishOrdering:
    async def test_offsets_are_zero_indexed_and_monotonic(self, bus: EventBus) -> None:
        o0 = bus.publish(_make_event(pid=1))
        o1 = bus.publish(_make_event(pid=2))
        o2 = bus.publish(_make_event(pid=3))
        assert (o0, o1, o2) == (0, 1, 2)
        assert bus.next_offset == 3

    async def test_oldest_offset_is_none_when_empty(self, bus: EventBus) -> None:
        assert bus.oldest_offset is None
        bus.publish(_make_event(pid=1))
        assert bus.oldest_offset == 0

    async def test_envelope_offset_is_assigned_when_unknown(self, bus: EventBus) -> None:
        ev = _make_event(pid=1)
        new_offset = bus.publish(ev)
        # The stored envelope carries the assigned offset, not UNKNOWN.
        stored = next(iter(list(bus.iter_buffered_envelopes())))
        assert stored.offset == new_offset
        assert new_offset >= 0


class TestPublishLegacyShim:
    async def test_legacy_payload_wraps_into_unknown_event(self, bus: EventBus) -> None:
        from ryotenkai_shared.events import UnknownEvent

        legacy_event = bus.publish_legacy("custom_kind", {"foo": "bar"})
        assert legacy_event.kind == "custom_kind"
        assert legacy_event.payload == {"foo": "bar"}
        stored = next(iter(list(bus.iter_buffered_envelopes())))
        assert isinstance(stored, UnknownEvent)
        assert stored.original_type == "custom_kind"

    async def test_legacy_and_typed_share_offset_counter(self, bus: EventBus) -> None:
        typed_offset = bus.publish(_make_event(pid=1))
        legacy = bus.publish_legacy("legacy", {})
        assert legacy.offset == typed_offset + 1


# ---------------------------------------------------------------------------
# Subscribe — replay + live
# ---------------------------------------------------------------------------


class TestSubscribe:
    async def test_replay_from_zero_yields_all_envelopes(self, bus: EventBus) -> None:
        for i in range(3):
            bus.publish(_make_event(pid=i + 1))
        events = await drain_n(bus, n=3)
        assert [ev.offset for ev in events] == [0, 1, 2]

    async def test_replay_from_middle(self, bus: EventBus) -> None:
        for i in range(3):
            bus.publish(_make_event(pid=i + 1))
        events = await drain_n(bus, n=2, since=1)
        assert [ev.offset for ev in events] == [1, 2]

    async def test_live_subscriber_receives_post_attach(self, bus: EventBus) -> None:
        async def _producer() -> None:
            await asyncio.sleep(0.01)
            bus.publish(_make_event(pid=42))

        task = asyncio.create_task(_producer())
        events = await drain_n(bus, n=1)
        await task
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Errors + close semantics
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_since_beyond_cursor_raises(self, bus: EventBus) -> None:
        bus.publish(_make_event())
        agen = bus.subscribe(since=99)
        with pytest.raises(ValueError):
            await agen.__anext__()
        await agen.aclose()

    async def test_truncated_buffer_raises(self) -> None:
        bus = EventBus(capacity=2)
        for i in range(4):
            bus.publish(_make_event(pid=i + 1))
        agen = bus.subscribe(since=0)
        with pytest.raises(BufferTruncatedError):
            await agen.__anext__()
        await agen.aclose()


class TestCloseSemantics:
    async def test_publish_after_close_raises(self, bus: EventBus) -> None:
        bus.close()
        with pytest.raises(RuntimeError):
            bus.publish(_make_event())

    async def test_close_drains_subscriber(self, bus: EventBus) -> None:
        bus.publish(_make_event(pid=1))
        events: list[BaseEvent] = []
        agen = bus.subscribe(since=0)
        # Pull the single buffered event then close the bus; the
        # subscribe loop terminates without hanging.
        events.append(await agen.__anext__())
        bus.close()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(agen.__anext__(), timeout=1.0)


# ---------------------------------------------------------------------------
# Capacity / env
# ---------------------------------------------------------------------------


class TestEnvCapacity:
    def test_default_capacity_constants_consistent(self) -> None:
        assert MIN_BUFFER_SIZE <= DEFAULT_BUFFER_SIZE <= MAX_BUFFER_SIZE


class TestDropCounter:
    def test_record_consumer_drop_bumps_counter(self, bus: EventBus) -> None:
        bus.record_consumer_drop("ws-conn-1", 3)
        bus.record_consumer_drop("ws-conn-1", 2)
        assert bus.dropped_per_consumer["ws-conn-1"] == 5
