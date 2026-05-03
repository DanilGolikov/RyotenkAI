"""Phase 12.B — :func:`_subscribe_with_disk_fallback` contract.

Pin the WS handler's transparent disk-replay logic. The test
fixtures here build minimal :class:`EventBus` + :class:`EventJournal`
pairs (no FastAPI, no WebSocket) and exercise the helper directly so
we can assert exact event ordering across the disk → ring boundary.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.runner.api.events import _subscribe_with_disk_fallback
from src.runner.event_bus import (
    BufferTruncatedError,
    DiskJournalExhausted,
    EventBus,
)
from src.runner.event_journal import EVENTS_DIR_REL, EventJournal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(
    tmp_path: Path, *, capacity: int = 1000,
) -> tuple[EventBus, EventJournal]:
    journal = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
    bus = EventBus(capacity=capacity, journal=journal)
    return bus, journal


async def _drain(
    bus: EventBus, *, since: int, max_events: int
) -> list[dict[str, Any]]:
    """Pull at most ``max_events`` from the disk-fallback iterator,
    then close the bus to break the live-stream wait."""
    out: list[dict[str, Any]] = []

    async def collector() -> None:
        async for event in _subscribe_with_disk_fallback(bus, since=since):
            out.append(event.to_dict())
            if len(out) >= max_events:
                return

    task = asyncio.create_task(collector())
    # Give the collector a tick; bus is already populated.
    await asyncio.sleep(0.01)
    if not task.done():
        bus.close()  # break out of live wait
    await asyncio.wait_for(task, timeout=2.0)
    return out


# ---------------------------------------------------------------------------
# 1. Positive — fast path (since within ring)
# ---------------------------------------------------------------------------


class TestPositive:
    @pytest.mark.asyncio
    async def test_fast_path_no_disk_replay(self, tmp_path: Path) -> None:
        bus, _ = _make_pair(tmp_path, capacity=10)
        for i in range(5):
            bus.publish(f"k{i}", {"i": i})

        out = await _drain(bus, since=0, max_events=5)
        assert [e["payload"]["i"] for e in out] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 2. Disk replay — since older than ring oldest
# ---------------------------------------------------------------------------


class TestDiskReplay:
    @pytest.mark.asyncio
    async def test_replays_older_records_from_disk(
        self, tmp_path: Path
    ) -> None:
        # Tiny ring (capacity=3) so old records get evicted but disk
        # keeps everything.
        bus, _ = _make_pair(tmp_path, capacity=3)
        for i in range(10):
            bus.publish(f"k{i}", {"i": i})

        # Subscribe from offset 2: ring oldest is now ~7 (last 3 in
        # ring = 7, 8, 9). Disk has 0..9. Helper should yield 2..9
        # in order, drawn from disk for 2..6 then ring for 7..9.
        out = await _drain(bus, since=2, max_events=8)
        assert [e["payload"]["i"] for e in out] == [2, 3, 4, 5, 6, 7, 8, 9]

    @pytest.mark.asyncio
    async def test_no_overlap_at_handoff_boundary(
        self, tmp_path: Path
    ) -> None:
        # Pin: disk replay STOPS exactly at ring_oldest, ring picks
        # up at ring_oldest. No record yielded twice.
        bus, _ = _make_pair(tmp_path, capacity=2)
        for i in range(5):
            bus.publish(f"k{i}", {"i": i})

        # Ring oldest = 3 (last 2 in ring = 3, 4). Disk has 0..4.
        out = await _drain(bus, since=0, max_events=5)
        assert [e["payload"]["i"] for e in out] == [0, 1, 2, 3, 4]
        # No duplicates.
        assert len(set(e["payload"]["i"] for e in out)) == 5


# ---------------------------------------------------------------------------
# 3. DiskJournalExhausted
# ---------------------------------------------------------------------------


class TestDiskExhausted:
    @pytest.mark.asyncio
    async def test_since_below_disk_oldest_raises(
        self, tmp_path: Path
    ) -> None:
        # Configure a tiny journal (1 file, very small cap) and
        # publish enough events to roll the journal too. Then
        # subscribe from offset 0 — should raise.
        journal = EventJournal(
            root_dir=tmp_path / EVENTS_DIR_REL,
            file_size_cap=80,
            max_files=2,
        )
        bus = EventBus(capacity=2, journal=journal)
        for i in range(20):
            bus.publish(f"k{i}", {"i": i, "data": "x" * 30})

        # Disk oldest is now well past 0 due to drop-oldest rotation.
        # Helper should raise DiskJournalExhausted.
        with pytest.raises(DiskJournalExhausted):
            async for _ in _subscribe_with_disk_fallback(bus, since=0):
                break  # raise happens BEFORE first yield
        bus.close()


# ---------------------------------------------------------------------------
# 4. No journal attached — old-style BufferTruncatedError
# ---------------------------------------------------------------------------


class TestNoJournal:
    @pytest.mark.asyncio
    async def test_falls_through_to_bus_subscribe(
        self, tmp_path: Path
    ) -> None:
        # Bus without journal — disk replay is a no-op; we hit the
        # ordinary BufferTruncatedError path.
        bus = EventBus(capacity=2)
        for i in range(5):
            bus.publish(f"k{i}", {"i": i})

        with pytest.raises(BufferTruncatedError):
            async for _ in _subscribe_with_disk_fallback(bus, since=0):
                break
        bus.close()


# ---------------------------------------------------------------------------
# 5. Empty ring — ring_oldest is None
# ---------------------------------------------------------------------------


class TestEmptyRing:
    @pytest.mark.asyncio
    async def test_empty_ring_subscribes_normally(
        self, tmp_path: Path
    ) -> None:
        # Empty bus → ring_oldest=None. Helper should fall through
        # to bus.subscribe which live-streams from publish forward.
        bus, _ = _make_pair(tmp_path)
        out: list[Any] = []

        async def consumer() -> None:
            async for event in _subscribe_with_disk_fallback(bus, since=0):
                out.append(event)
                if len(out) >= 1:
                    return

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)  # consumer is now waiting on signal
        bus.publish("k", {"x": 1})
        await asyncio.wait_for(task, timeout=1.0)

        assert len(out) == 1
        assert out[0].kind == "k"
        bus.close()
