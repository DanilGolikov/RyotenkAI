"""Phase 12.B — :class:`EventBus` + :class:`EventJournal` integration.

Pin the contract that publishes also persist to disk, that the bus
reconciles its starting offset on restart, and that close() flushes
the journal.

Slim-venv compatible: pure stdlib + asyncio.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.event_journal import EVENTS_DIR_REL, EventJournal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(
    tmp_path: Path, *, capacity: int = 1000,
) -> tuple[EventBus, EventJournal]:
    journal = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
    bus = EventBus(capacity=capacity, journal=journal)
    return bus, journal


# ---------------------------------------------------------------------------
# 1. Positive — every publish persists
# ---------------------------------------------------------------------------


class TestPositive:
    def test_publish_writes_to_disk(self, tmp_path: Path) -> None:
        bus, journal = _make_pair(tmp_path)
        bus.publish("step", {"loss": 0.5})
        bus.close()

        records = list(journal.iter_records(since=0))
        assert len(records) == 1
        assert records[0].kind == "step"
        assert records[0].payload == {"loss": 0.5}

    def test_publish_offset_matches_persisted_offset(self, tmp_path: Path) -> None:
        bus, journal = _make_pair(tmp_path)
        e0 = bus.publish("a", {})
        e1 = bus.publish("b", {})
        bus.close()

        records = list(journal.iter_records(since=0))
        assert [r.offset for r in records] == [e0.offset, e1.offset]


# ---------------------------------------------------------------------------
# 2. Negative — bus without journal
# ---------------------------------------------------------------------------


class TestNegative:
    def test_journal_optional_no_changes_for_unset(self, tmp_path: Path) -> None:
        # Bus constructed without journal — Phase 1 behaviour preserved.
        bus = EventBus(capacity=10)
        bus.publish("step", {})
        bus.close()
        # No exception.

    def test_disk_pressure_does_not_crash_publish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bus, journal = _make_pair(tmp_path)

        def _broken_append(**kwargs: Any) -> None:
            raise OSError("simulated EROFS")

        monkeypatch.setattr(journal, "append", _broken_append)

        # Publish must NOT raise.
        bus.publish("step", {})
        bus.close()


# ---------------------------------------------------------------------------
# 3. Boundary — empty journal at construction
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_starting_offset_zero_when_journal_empty(self, tmp_path: Path) -> None:
        bus, _ = _make_pair(tmp_path)
        assert bus.next_offset == 0
        bus.close()


# ---------------------------------------------------------------------------
# 4. Invariants — restart reconciliation
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_restart_reconciles_next_offset(self, tmp_path: Path) -> None:
        # First lifecycle: write records 0..2.
        bus1, journal1 = _make_pair(tmp_path)
        bus1.publish("a", {})
        bus1.publish("b", {})
        bus1.publish("c", {})
        bus1.close()

        # New bus on the same workspace inherits offsets via journal.
        journal2 = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
        bus2 = EventBus(capacity=10, journal=journal2)

        # New bus starts at offset 3, NOT 0 → no collision with disk.
        assert bus2.next_offset == 3

        e = bus2.publish("d", {})
        assert e.offset == 3
        bus2.close()

        # Disk now has 4 records (0..3).
        records = list(journal2.iter_records(since=0))
        assert [r.offset for r in records] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 5. Dependency errors — broken journal at init
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_broken_newest_offset_falls_back_to_zero(
        self, tmp_path: Path
    ) -> None:
        class _BrokenJournal:
            def newest_persisted_offset(self) -> int | None:
                raise RuntimeError("synthetic")
            def append(self, **kwargs: Any) -> None: ...
            def close(self) -> None: ...

        bus = EventBus(capacity=5, journal=_BrokenJournal())
        # Falls back to fresh start.
        assert bus.next_offset == 0
        bus.close()


# ---------------------------------------------------------------------------
# 6. Regressions — close fsyncs journal
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_close_propagates_to_journal(self, tmp_path: Path) -> None:
        bus, journal = _make_pair(tmp_path)
        bus.publish("step", {})
        bus.close()
        # Journal closed too.
        assert journal.is_closed is True

    def test_journal_property_exposes_attached(self, tmp_path: Path) -> None:
        bus, journal = _make_pair(tmp_path)
        assert bus.journal is journal
        bus.close()

    def test_journal_property_none_when_not_attached(
        self, tmp_path: Path
    ) -> None:
        bus = EventBus(capacity=5)
        assert bus.journal is None
        bus.close()


# ---------------------------------------------------------------------------
# 7. Logic-specific — async subscribe still works
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    @pytest.mark.asyncio
    async def test_subscribe_unaffected_by_journal(
        self, tmp_path: Path
    ) -> None:
        bus, _ = _make_pair(tmp_path)
        bus.publish("a", {})
        bus.publish("b", {})

        events: list[Any] = []

        async def consumer() -> None:
            async for event in bus.subscribe(since=0):
                events.append(event)
                if len(events) == 2:
                    break

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)
        await asyncio.wait_for(task, timeout=1.0)

        assert len(events) == 2
        bus.close()
