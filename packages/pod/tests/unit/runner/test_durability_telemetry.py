"""Phase 12.C — durability telemetry constants + emission contracts.

Pin:
* New event-kind constants are stable strings (operator dashboards
  key off these).
* :class:`EventJournal` emits ``events_rotated`` via the on_rotate
  callback after every file rotation, with the right payload shape.
* :class:`EventBus` emits ``events_disk_pressure`` when its
  ``_signal_disk_pressure`` runs (no event itself — only logged).
* :func:`_periodic_journal_health_check` publishes
  ``events_disk_pressure`` when the journal footprint crosses 90%
  of cap.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_shared.observability.cancellation_telemetry import (
    CANCELLATION_EVENT_KINDS,
    DURABILITY_EVENT_KINDS,
    EVENTS_DISK_PRESSURE,
    EVENTS_GC_RAN,
    EVENTS_ROTATED,
    METRICS_BUFFER_RETRIEVED,
    TERMINAL_EVENT_KINDS,
)
from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.event_journal import EVENTS_DIR_REL, EventJournal


# ---------------------------------------------------------------------------
# 1. Constants stability
# ---------------------------------------------------------------------------


class TestConstants:
    def test_durability_event_kind_strings_stable(self) -> None:
        # Operator dashboards / SLO alerts key off these strings.
        # Renaming requires deliberate cross-codebase grep.
        assert EVENTS_DISK_PRESSURE == "events_disk_pressure"
        assert EVENTS_ROTATED == "events_rotated"
        assert EVENTS_GC_RAN == "events_gc_ran"
        assert METRICS_BUFFER_RETRIEVED == "metrics_buffer_retrieved"

    def test_durability_set_contains_all_durability_kinds(self) -> None:
        assert EVENTS_DISK_PRESSURE in DURABILITY_EVENT_KINDS
        assert EVENTS_ROTATED in DURABILITY_EVENT_KINDS
        assert EVENTS_GC_RAN in DURABILITY_EVENT_KINDS
        assert METRICS_BUFFER_RETRIEVED in DURABILITY_EVENT_KINDS

    def test_terminal_set_is_superset(self) -> None:
        # TERMINAL = CANCELLATION | DURABILITY | {COMPLETION_FINALIZED}
        assert DURABILITY_EVENT_KINDS.issubset(TERMINAL_EVENT_KINDS)
        assert CANCELLATION_EVENT_KINDS.issubset(TERMINAL_EVENT_KINDS)

    def test_durability_disjoint_from_cancellation(self) -> None:
        # No accidental membership crossover.
        assert DURABILITY_EVENT_KINDS.isdisjoint(CANCELLATION_EVENT_KINDS)


# ---------------------------------------------------------------------------
# 2. on_rotate callback fires
# ---------------------------------------------------------------------------


class TestRotationCallback:
    def test_on_rotate_called_with_correct_shape(
        self, tmp_path: Path
    ) -> None:
        events: list[dict[str, Any]] = []

        def _capture(**kwargs: Any) -> None:
            events.append(kwargs)

        # cap=80, max_files=100 to avoid drop-oldest.
        j = EventJournal(
            root_dir=tmp_path / EVENTS_DIR_REL,
            file_size_cap=80,
            max_files=100,
            on_rotate=_capture,
        )
        # Each record is ~50 bytes → rotate after 1-2 records.
        for i in range(5):
            j.append(
                offset=i, ts="t", kind="k", payload={"data": "x" * 30}
            )
        j.close()

        # We rotated multiple times.
        assert len(events) >= 2
        # Payload shape pinned.
        first = events[0]
        assert "from_seq" in first
        assert "to_seq" in first
        assert "file_size_bytes" in first
        assert "oldest_remaining_seq" in first
        # Sequences progress.
        assert first["from_seq"] < first["to_seq"]

    def test_on_rotate_failure_does_not_break_journal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A misbehaving callback must not block the journal.
        def _broken(**kwargs: Any) -> None:
            raise RuntimeError("callback misbehaved")

        j = EventJournal(
            root_dir=tmp_path / EVENTS_DIR_REL,
            file_size_cap=50,
            max_files=10,
            on_rotate=_broken,
        )
        # Trigger a rotation. The journal MUST keep working.
        j.append(offset=0, ts="t", kind="k", payload={"data": "x" * 100})
        j.append(offset=1, ts="t", kind="k", payload={})
        j.close()


# ---------------------------------------------------------------------------
# 3. Bus → events_rotated wire-through
# ---------------------------------------------------------------------------


class TestBusRotationWire:
    def test_bus_publishes_rotation_via_callback(
        self, tmp_path: Path
    ) -> None:
        # Wire the journal's on_rotate to bus.publish — this is the
        # production wiring pattern used in `_lifespan`.
        rotate_publisher = {"bus": None}

        def _on_rotate(**kwargs: Any) -> None:
            target = rotate_publisher.get("bus")
            if target is None:
                return
            target.publish(EVENTS_ROTATED, kwargs)

        j = EventJournal(
            root_dir=tmp_path / EVENTS_DIR_REL,
            file_size_cap=80,
            max_files=100,
            on_rotate=_on_rotate,
        )
        bus = EventBus(capacity=100, journal=j)
        rotate_publisher["bus"] = bus

        for i in range(5):
            bus.publish(f"k{i}", {"data": "x" * 30})

        # Iterate the journal to find rotation records (events_rotated
        # is itself published, so it lands on disk too).
        records = list(j.iter_records(since=0))
        rotation_records = [r for r in records if r.kind == EVENTS_ROTATED]
        assert len(rotation_records) >= 1
        bus.close()


# ---------------------------------------------------------------------------
# 4. Bus disk-pressure rate limit
# ---------------------------------------------------------------------------


class TestBusDiskPressureRateLimit:
    def test_signal_disk_pressure_rate_limited_to_one_per_minute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from ryotenkai_pod.runner import event_bus as bus_mod

        # Pin monotonic clock for deterministic rate-limit window.
        # `_signal_disk_pressure` does ``import time as _time`` locally
        # so we patch the global ``time.monotonic``.
        import time as time_mod
        clock = [1000.0]
        monkeypatch.setattr(time_mod, "monotonic", lambda: clock[0])

        bus = EventBus(capacity=10)

        # Simulate two failures 30s apart — only the first should log.
        with caplog.at_level("WARNING", logger=bus_mod.__name__):
            bus._signal_disk_pressure(OSError("disk full 1"))
            clock[0] += 30  # 30s later, still inside 60s window
            bus._signal_disk_pressure(OSError("disk full 2"))

        warn_records = [r for r in caplog.records if "journal append failed" in r.message]
        assert len(warn_records) == 1
        bus.close()


# ---------------------------------------------------------------------------
# 5. Periodic health check
# ---------------------------------------------------------------------------


class TestPeriodicHealthCheck:
    @pytest.mark.asyncio
    async def test_emits_disk_pressure_when_above_threshold(
        self, tmp_path: Path
    ) -> None:
        from ryotenkai_pod.runner.main import _periodic_journal_health_check

        # Build a journal whose total_bytes > threshold.
        j = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
        # Stub total_bytes to exceed threshold.
        j.total_bytes = lambda: 999_999_999  # type: ignore[assignment]
        j.file_count = lambda: 5  # type: ignore[assignment]

        bus = EventBus(capacity=100, journal=j)

        task = asyncio.create_task(
            _periodic_journal_health_check(
                bus=bus, journal=j, interval_s=0.05, threshold_fraction=0.5,
            )
        )
        await asyncio.sleep(0.1)  # one tick
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        records = list(j.iter_records(since=0))
        pressure_records = [
            r for r in records if r.kind == EVENTS_DISK_PRESSURE
        ]
        assert len(pressure_records) >= 1
        # Payload shape.
        assert "total_bytes" in pressure_records[0].payload
        assert "threshold_bytes" in pressure_records[0].payload
        bus.close()

    @pytest.mark.asyncio
    async def test_no_event_when_below_threshold(
        self, tmp_path: Path
    ) -> None:
        from ryotenkai_pod.runner.main import _periodic_journal_health_check

        j = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
        j.total_bytes = lambda: 1_000  # well below
        j.file_count = lambda: 1

        bus = EventBus(capacity=100, journal=j)

        task = asyncio.create_task(
            _periodic_journal_health_check(
                bus=bus, journal=j, interval_s=0.05, threshold_fraction=0.5,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        records = list(j.iter_records(since=0))
        pressure_records = [
            r for r in records if r.kind == EVENTS_DISK_PRESSURE
        ]
        assert pressure_records == []
        bus.close()

    @pytest.mark.asyncio
    async def test_one_alert_per_threshold_crossing(
        self, tmp_path: Path
    ) -> None:
        # When we stay above threshold for multiple ticks, we should
        # emit only ONE event (until we drop back below and re-cross).
        from ryotenkai_pod.runner.main import _periodic_journal_health_check

        j = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
        j.total_bytes = lambda: 999_999_999
        j.file_count = lambda: 5

        bus = EventBus(capacity=100, journal=j)

        task = asyncio.create_task(
            _periodic_journal_health_check(
                bus=bus, journal=j, interval_s=0.02, threshold_fraction=0.5,
            )
        )
        await asyncio.sleep(0.1)  # multiple ticks, all above threshold
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        records = list(j.iter_records(since=0))
        pressure_records = [
            r for r in records if r.kind == EVENTS_DISK_PRESSURE
        ]
        # Exactly one event for the single sustained crossing.
        assert len(pressure_records) == 1
        bus.close()
