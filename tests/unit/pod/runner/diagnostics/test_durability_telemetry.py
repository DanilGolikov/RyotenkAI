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

import pytest

from ryotenkai_pod.runner.event_bus import EventBus, legacy_kind_for
from ryotenkai_pod.runner.event_journal import EVENTS_DIR_REL, EventJournal
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.pod_lifecycle import (
    TrainerSpawnedEvent,
    TrainerSpawnedPayload,
)
from ryotenkai_shared.observability.cancellation_telemetry import (
    CANCELLATION_EVENT_KINDS,
    DURABILITY_EVENT_KINDS,
    EVENTS_DISK_PRESSURE,
    EVENTS_GC_RAN,
    EVENTS_ROTATED,
    METRICS_BUFFER_RETRIEVED,
    TERMINAL_EVENT_KINDS,
)


def _envelope_for(offset: int) -> TrainerSpawnedEvent:
    return TrainerSpawnedEvent(
        source="pod://test/runner",
        run_id="test",
        offset=offset,
        payload=TrainerSpawnedPayload(pid=offset + 1, cmdline="py", cwd="/tmp"),
    )

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

        # cap=200, max_files=100 to avoid drop-oldest. Typed envelopes
        # are larger than the legacy free-form records (frozen Pydantic
        # serialisation), so the file_size_cap is bumped accordingly.
        j = EventJournal(
            root_dir=tmp_path / EVENTS_DIR_REL,
            file_size_cap=200,
            max_files=100,
            on_rotate=_capture,
        )
        for i in range(8):
            j.append_envelope(_envelope_for(i))
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
            file_size_cap=200,
            max_files=10,
            on_rotate=_broken,
        )
        # Trigger rotations. The journal MUST keep working.
        for i in range(5):
            j.append_envelope(_envelope_for(i))
        j.close()


# ---------------------------------------------------------------------------
# 3. Bus → events_rotated wire-through
# ---------------------------------------------------------------------------


class TestBusRotationWire:
    def test_bus_publishes_rotation_via_callback(
        self, tmp_path: Path
    ) -> None:
        # Phase 2: the bus owns the wiring via
        # :meth:`attach_journal_rotation_listener` and emits a typed
        # ``JournalRotatedEvent`` directly. The journal-resident
        # snapshot uses the legacy ``events_rotated`` alias via
        # :func:`legacy_kind_for`.
        j = EventJournal(
            root_dir=tmp_path / EVENTS_DIR_REL,
            file_size_cap=200,
            max_files=100,
        )
        bus = EventBus(capacity=100, journal=j)
        bus.attach_journal_rotation_listener()

        for i in range(8):
            bus.publish(_envelope_for(i))

        envelopes = list(j.iter_envelopes())
        rotation_envelopes = [e for e in envelopes if legacy_kind_for(e) == EVENTS_ROTATED]
        assert len(rotation_envelopes) >= 1
        bus.close()


# ---------------------------------------------------------------------------
# 4. Bus disk-pressure rate limit
# ---------------------------------------------------------------------------


class TestBusDiskPressureRateLimit:
    def test_signal_disk_pressure_rate_limited_to_one_per_minute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Pin monotonic clock for deterministic rate-limit window.
        # `_signal_disk_pressure` does ``import time as _time`` locally
        # so we patch the global ``time.monotonic``.
        import time as time_mod

        from ryotenkai_pod.runner import event_bus as bus_mod
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

        envelopes = list(j.iter_envelopes())
        pressure_envelopes = [
            e for e in envelopes if legacy_kind_for(e) == EVENTS_DISK_PRESSURE
        ]
        assert len(pressure_envelopes) >= 1
        payload_obj = getattr(pressure_envelopes[0], "payload", None)
        if hasattr(payload_obj, "total_bytes"):
            assert payload_obj.total_bytes is not None
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

        envelopes = list(j.iter_envelopes())
        pressure_envelopes = [
            e for e in envelopes if legacy_kind_for(e) == EVENTS_DISK_PRESSURE
        ]
        assert pressure_envelopes == []
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

        envelopes = list(j.iter_envelopes())
        pressure_envelopes = [
            e for e in envelopes if legacy_kind_for(e) == EVENTS_DISK_PRESSURE
        ]
        # Exactly one event for the single sustained crossing.
        assert len(pressure_envelopes) == 1
        bus.close()
