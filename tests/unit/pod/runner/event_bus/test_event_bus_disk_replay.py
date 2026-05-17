"""Phase 2 — disk-fallback subscribe contract over typed envelopes.

The WS handler's ``_subscribe_with_disk_fallback`` yields envelopes
from the journal first when ``since`` is older than the ring's tail,
then hands off to the live ring at exactly ``ring_oldest`` so the
consumer sees a monotonic stream with no overlap.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ryotenkai_pod.runner.api.events import _subscribe_with_disk_fallback
from ryotenkai_pod.runner.event_bus import (
    DiskJournalExhausted,
    EventBus,
)
from ryotenkai_pod.runner.event_journal import EVENTS_DIR_REL, EventJournal
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.pod_lifecycle import (
    TrainerSpawnedEvent,
    TrainerSpawnedPayload,
)


def _make_event(pid: int) -> TrainerSpawnedEvent:
    return TrainerSpawnedEvent(
        source="pod://test/runner",
        run_id="test",
        offset=UNKNOWN_OFFSET,
        payload=TrainerSpawnedPayload(pid=pid, cmdline="py", cwd="/tmp"),
    )


def _make_pair(tmp_path: Path, *, capacity: int = 1000) -> tuple[EventBus, EventJournal]:
    journal = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
    bus = EventBus(capacity=capacity, journal=journal)
    return bus, journal


async def _drain(
    bus: EventBus, *, since: int, max_events: int,
) -> list[int]:
    """Pull at most ``max_events`` offsets, then close the bus to
    break the live-stream wait so the test doesn't hang."""
    out: list[int] = []

    async def collector() -> None:
        async for event in _subscribe_with_disk_fallback(bus, since=since):
            out.append(event.offset)
            if len(out) >= max_events:
                return

    task = asyncio.create_task(collector())
    await asyncio.sleep(0.01)
    if not task.done():
        bus.close()
    await asyncio.wait_for(task, timeout=2.0)
    return out


class TestPositive:
    @pytest.mark.asyncio
    async def test_fast_path_no_disk_replay(self, tmp_path: Path) -> None:
        bus, _ = _make_pair(tmp_path, capacity=10)
        for i in range(5):
            bus.publish(_make_event(pid=i + 1))
        offsets = await _drain(bus, since=0, max_events=5)
        assert offsets == [0, 1, 2, 3, 4]


class TestDiskReplay:
    @pytest.mark.asyncio
    async def test_replays_older_records_from_disk(self, tmp_path: Path) -> None:
        # Ring capacity 2 so events 0-2 get evicted, but the journal
        # still has them.
        bus, _ = _make_pair(tmp_path, capacity=2)
        for i in range(5):
            bus.publish(_make_event(pid=i + 1))

        offsets = await _drain(bus, since=0, max_events=5)
        assert offsets == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_since_below_disk_oldest_raises(self, tmp_path: Path) -> None:
        bus, _ = _make_pair(tmp_path, capacity=2)
        for i in range(3):
            bus.publish(_make_event(pid=i + 1))

        # Patch journal.oldest_persisted_offset to fake "older offsets
        # were dropped" so we can validate the exhaustion path.
        bus.journal._list_existing_seqs = lambda: []  # type: ignore[union-attr]

        with pytest.raises(DiskJournalExhausted):
            await _drain(bus, since=0, max_events=1)
