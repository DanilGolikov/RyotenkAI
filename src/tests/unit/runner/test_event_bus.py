"""Phase 1.2 — EventBus contract.

Coverage matrix:

- TestPublishOrdering        offset monotonicity, timestamps, defensive copy
- TestSubscribeReplay        replay from arbitrary offsets
- TestSubscribeLive          new events arrive after subscriber attaches
- TestSubscribeBoundaries    since=0, since=next_offset, since=truncated
- TestRingBufferOverflow     capacity enforcement, oldest_offset progression
- TestCloseSemantics         close stops publishers, drains subscribers
- TestEnvCapacity            RYOTENKAI_EVENT_BUFFER_SIZE override + clamps
- TestErrors                 BufferTruncatedError attrs, ValueError for invalid since
"""

from __future__ import annotations

import asyncio

import pytest

from src.runner.event_bus import (
    DEFAULT_BUFFER_SIZE,
    MAX_BUFFER_SIZE,
    MIN_BUFFER_SIZE,
    BufferTruncatedError,
    Event,
    EventBus,
)

# All async tests live here. The sync edge cases for the Event
# dataclass are covered in :mod:`test_runner_skeleton.TestEventBusContract`.
pytestmark = pytest.mark.asyncio


@pytest.fixture
def bus() -> EventBus:
    return EventBus(capacity=10)


# ---------------------------------------------------------------------------
# Helper — drain N events with a short timeout to keep tests responsive
# ---------------------------------------------------------------------------


async def drain_n(bus: EventBus, *, n: int, since: int = 0, timeout: float = 1.0) -> list[Event]:
    """Subscribe and pull exactly ``n`` events, then stop.

    The subscriber breaks out of the iterator as soon as ``n`` events
    arrive — the bus' infinite ``await self._wakeup.wait()`` loop is
    interrupted by the ``async for`` exit, which cleanly closes the
    async generator. ``asyncio.wait_for`` provides a hard upper bound
    so a hang in the bus doesn't freeze the suite.
    """

    async def _drain() -> list[Event]:
        out: list[Event] = []
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
# Publish
# ---------------------------------------------------------------------------


class TestPublishOrdering:
    async def test_offsets_are_zero_indexed_and_monotonic(self, bus: EventBus) -> None:
        e0 = bus.publish("a", {})
        e1 = bus.publish("b", {})
        e2 = bus.publish("c", {})
        assert (e0.offset, e1.offset, e2.offset) == (0, 1, 2)
        assert bus.next_offset == 3

    async def test_oldest_offset_is_none_when_empty(self, bus: EventBus) -> None:
        assert bus.oldest_offset is None
        bus.publish("a", {})
        assert bus.oldest_offset == 0

    async def test_timestamp_format_is_iso_z(self, bus: EventBus) -> None:
        event = bus.publish("a", {})
        # Format: 2026-04-26T12:00:00Z (no microseconds, no offset)
        assert event.timestamp.endswith("Z")
        assert "T" in event.timestamp
        assert "+" not in event.timestamp

    async def test_publisher_payload_is_defensively_copied(self, bus: EventBus) -> None:
        # Mutating the publisher's dict after publish must not leak
        # into the stored event.
        local = {"k": 1}
        event = bus.publish("a", local)
        local["k"] = 999
        assert event.payload["k"] == 1


# ---------------------------------------------------------------------------
# Subscribe — replay
# ---------------------------------------------------------------------------


class TestSubscribeReplay:
    async def test_replay_from_zero_yields_everything(self, bus: EventBus) -> None:
        for kind in ("a", "b", "c"):
            bus.publish(kind, {})

        events = await drain_n(bus, n=3)
        assert [e.kind for e in events] == ["a", "b", "c"]

    async def test_replay_from_middle(self, bus: EventBus) -> None:
        for kind in ("a", "b", "c", "d"):
            bus.publish(kind, {})

        events = await drain_n(bus, n=2, since=2)
        assert [e.kind for e in events] == ["c", "d"]


# ---------------------------------------------------------------------------
# Subscribe — live
# ---------------------------------------------------------------------------


class TestSubscribeLive:
    async def test_subscriber_receives_events_published_after_attach(
        self, bus: EventBus,
    ) -> None:
        async def _publisher() -> None:
            await asyncio.sleep(0)  # yield to subscriber first
            bus.publish("a", {})
            bus.publish("b", {})
            bus.publish("c", {})

        async with asyncio.TaskGroup() as tg:
            tg.create_task(_publisher())
            drain_task = tg.create_task(drain_n(bus, n=3))
        events = drain_task.result()
        assert [e.kind for e in events] == ["a", "b", "c"]

    async def test_replay_then_live(self, bus: EventBus) -> None:
        bus.publish("buffered", {})

        async def _publisher() -> None:
            # Wait long enough for the subscriber to consume the
            # replay event before producing the live one.
            await asyncio.sleep(0.01)
            bus.publish("live", {})

        async with asyncio.TaskGroup() as tg:
            tg.create_task(_publisher())
            drain_task = tg.create_task(drain_n(bus, n=2))
        events = drain_task.result()
        assert [e.kind for e in events] == ["buffered", "live"]


# ---------------------------------------------------------------------------
# Subscribe — boundaries
# ---------------------------------------------------------------------------


class TestSubscribeBoundaries:
    async def test_since_equals_next_offset_just_waits(self, bus: EventBus) -> None:
        # No buffered events; subscriber waits for the live one.
        async def _publisher() -> None:
            await asyncio.sleep(0)
            bus.publish("first", {})

        async with asyncio.TaskGroup() as tg:
            tg.create_task(_publisher())
            drain_task = tg.create_task(drain_n(bus, n=1, since=0))
        assert [e.kind for e in drain_task.result()] == ["first"]

    async def test_since_negative_raises(self, bus: EventBus) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            agen = bus.subscribe(since=-1)
            await agen.__anext__()

    async def test_since_beyond_cursor_raises(self, bus: EventBus) -> None:
        bus.publish("a", {})
        # next_offset = 1, since=5 → invalid
        with pytest.raises(ValueError, match="beyond the current cursor"):
            agen = bus.subscribe(since=5)
            await agen.__anext__()


# ---------------------------------------------------------------------------
# Ring buffer overflow
# ---------------------------------------------------------------------------


class TestRingBufferOverflow:
    async def test_capacity_enforced(self, bus: EventBus) -> None:
        # bus fixture has capacity=10
        for _ in range(15):
            bus.publish("x", {})
        # Buffer holds the last 10 events; offsets 5..14.
        assert bus.next_offset == 15
        assert bus.oldest_offset == 5

    async def test_truncated_replay_raises(self, bus: EventBus) -> None:
        for _ in range(15):
            bus.publish("x", {})
        # since=2 is older than oldest (5).
        with pytest.raises(BufferTruncatedError) as exc_info:
            agen = bus.subscribe(since=2)
            await agen.__anext__()
        assert exc_info.value.requested_offset == 2
        assert exc_info.value.oldest_available == 5


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestCloseSemantics:
    async def test_publish_after_close_raises(self, bus: EventBus) -> None:
        bus.close()
        with pytest.raises(RuntimeError, match="closed"):
            bus.publish("a", {})

    async def test_close_drains_subscriber(self, bus: EventBus) -> None:
        bus.publish("a", {})
        bus.publish("b", {})

        async def _drain() -> list[Event]:
            out: list[Event] = []
            async for event in bus.subscribe():
                out.append(event)
            return out

        task = asyncio.create_task(_drain())
        await asyncio.sleep(0)  # let it pick up the buffered events
        bus.close()
        events = await task
        assert [e.kind for e in events] == ["a", "b"]


# ---------------------------------------------------------------------------
# Env-driven capacity
# ---------------------------------------------------------------------------


class TestEnvCapacity:
    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            ("", DEFAULT_BUFFER_SIZE),
            ("not-a-number", DEFAULT_BUFFER_SIZE),
            ("500", 500),
            (str(MIN_BUFFER_SIZE - 1), MIN_BUFFER_SIZE),  # clamped low
            (str(MAX_BUFFER_SIZE + 1), MAX_BUFFER_SIZE),  # clamped high
        ],
    )
    async def test_env_drives_default_capacity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_value: str,
        expected: int,
    ) -> None:
        if env_value:
            monkeypatch.setenv("RYOTENKAI_EVENT_BUFFER_SIZE", env_value)
        else:
            monkeypatch.delenv("RYOTENKAI_EVENT_BUFFER_SIZE", raising=False)
        bus = EventBus()
        assert bus.capacity == expected

    async def test_explicit_capacity_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_EVENT_BUFFER_SIZE", "500")
        bus = EventBus(capacity=42)
        assert bus.capacity == 42


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_buffer_truncated_error_carries_attrs(self) -> None:
        err = BufferTruncatedError(requested_offset=2, oldest_available=5)
        assert err.requested_offset == 2
        assert err.oldest_available == 5
        assert "2" in str(err) and "5" in str(err)


# Sync-only Event dataclass invariants are covered in
# ``test_runner_skeleton.TestEventBusContract``.
