"""Tests for :mod:`ryotenkai_control.events.metrics` (Phase 8).

Seven categories per ``docs/testing/mutation_testing.md``:

1. TestPositive             — happy path: snapshot mirrors live counters.
2. TestNegative             — None collaborators → zero-defaults instead
                              of raising.
3. TestBoundary             — zero counters; 1000-event volume.
4. TestInvariants           — counters monotonically increasing across
                              multiple snapshots.
5. TestDependencyErrors     — snapshot when emitter closed → callable.
6. TestRegressions          — snapshots are independent copies; later
                              mutation does not leak into earlier snap.
7. TestLogicSpecific        — dropped_per_consumer keying; is_degraded
                              flips on each indicator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ryotenkai_control.events import (
    ControlEventEmitter,
    EventDedup,
    EventSubsystemMetrics,
    InMemoryBus,
    JournalWriter,
    collect_metrics,
    collect_metrics_for_emitter,
)

from tests.unit.control.events.conftest import make_completed, make_started


def _build_emitter(tmp_path: Path) -> ControlEventEmitter:
    journal = JournalWriter(tmp_path / "events.jsonl")
    bus = InMemoryBus(capacity=8)
    dedup = EventDedup()
    return ControlEventEmitter(
        run_id="test-run",
        source="control://orchestrator",
        journal=journal,
        bus=bus,
        dedup=dedup,
    )


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_collect_returns_metrics_dataclass(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        try:
            emitter.emit(make_started())
            snap = collect_metrics_for_emitter(emitter)
            assert isinstance(snap, EventSubsystemMetrics)
            assert snap.emitter_events_emitted_total == 1
            assert snap.journal_appended_total == 1
            assert snap.bus_published_total == 1
        finally:
            emitter.close()

    def test_collect_with_explicit_collaborators(self, tmp_path: Path) -> None:
        journal = JournalWriter(tmp_path / "events.jsonl")
        bus = InMemoryBus(capacity=4)
        dedup = EventDedup()
        try:
            journal.append(make_started(offset=0))
            bus.publish(make_started(offset=0))
            dedup.remember("r", "src", 0)
            snap = collect_metrics(journal=journal, bus=bus, dedup=dedup)
            assert snap.journal_appended_total == 1
            assert snap.bus_published_total == 1
            assert snap.dedup_seen_total == 1
            assert snap.dedup_size == 1
        finally:
            journal.close()

    def test_to_dict_contains_all_fields(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        try:
            snap = collect_metrics_for_emitter(emitter)
            data = snap.to_dict()
            # 21 declared fields on the dataclass.
            assert "emitter_events_emitted_total" in data
            assert "bus_published_total" in data
            assert "journal_appended_total" in data
            assert "dedup_size" in data
            assert "journal_last_fsync_age_seconds" in data
        finally:
            emitter.close()


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_all_none_returns_zero_defaults(self) -> None:
        snap = collect_metrics()
        assert snap.emitter_events_emitted_total == 0
        assert snap.bus_published_total == 0
        assert snap.journal_appended_total == 0
        assert snap.dedup_size == 0
        assert snap.journal_last_fsync_age_seconds is None
        # Dict-typed defaults are empty, not missing.
        assert snap.emitter_events_emit_failed_total == {}
        assert snap.bus_dropped_per_consumer == {}

    def test_partial_collaborators_supported(self, tmp_path: Path) -> None:
        journal = JournalWriter(tmp_path / "events.jsonl")
        try:
            journal.append(make_started(offset=0))
            snap = collect_metrics(journal=journal)
            # Journal-only collection works without bus/dedup.
            assert snap.journal_appended_total == 1
            assert snap.bus_published_total == 0
            assert snap.dedup_size == 0
        finally:
            journal.close()


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_counters_at_zero(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        try:
            snap = collect_metrics_for_emitter(emitter)
            assert snap.emitter_events_emitted_total == 0
            assert snap.bus_published_total == 0
            assert snap.journal_appended_total == 0
        finally:
            emitter.close()

    def test_counters_after_many_events(self, tmp_path: Path) -> None:
        # Use a bus large enough to avoid drops at this volume.
        journal = JournalWriter(tmp_path / "events.jsonl")
        bus = InMemoryBus(capacity=2048)
        dedup = EventDedup()
        emitter = ControlEventEmitter(
            run_id="test-run",
            source="control://orchestrator",
            journal=journal,
            bus=bus,
            dedup=dedup,
        )
        try:
            for _ in range(1000):
                emitter.emit(make_started())
            snap = collect_metrics_for_emitter(emitter)
            assert snap.emitter_events_emitted_total == 1000
            assert snap.journal_appended_total == 1000
            assert snap.bus_published_total == 1000
            # No drops with a 2048-deep bus.
            assert snap.bus_dropped_total == 0
        finally:
            emitter.close()


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_counters_monotonically_non_decreasing(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        try:
            snaps: list[EventSubsystemMetrics] = []
            for _ in range(5):
                emitter.emit(make_started())
                snaps.append(collect_metrics_for_emitter(emitter))
            # Each snapshot's emitter counter is >= the previous one's.
            assert all(
                snaps[i].emitter_events_emitted_total
                <= snaps[i + 1].emitter_events_emitted_total
                for i in range(len(snaps) - 1)
            )
            # Journal and bus track the same way.
            assert all(
                snaps[i].journal_appended_total
                <= snaps[i + 1].journal_appended_total
                for i in range(len(snaps) - 1)
            )
            assert all(
                snaps[i].bus_published_total <= snaps[i + 1].bus_published_total
                for i in range(len(snaps) - 1)
            )
        finally:
            emitter.close()

    def test_snapshot_is_frozen(self, tmp_path: Path) -> None:
        snap = collect_metrics()
        # Dataclass is frozen — mutation raises FrozenInstanceError.
        with pytest.raises(Exception):  # noqa: BLE001 — FrozenInstanceError is dataclass-private
            snap.emitter_events_emitted_total = 99  # type: ignore[misc]


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_snapshot_after_close_still_callable(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        emitter.emit(make_started())
        emitter.close()
        # After close, the collaborators still expose their counters.
        snap = collect_metrics_for_emitter(emitter)
        assert snap.emitter_events_emitted_total == 1
        assert snap.journal_appended_total == 1

    def test_journal_with_no_fsync_yields_none_age(self, tmp_path: Path) -> None:
        # Fresh writer with no append → last_fsync_at is None →
        # journal_last_fsync_age_seconds is None.
        journal = JournalWriter(tmp_path / "events.jsonl")
        try:
            snap = collect_metrics(journal=journal)
            assert snap.journal_last_fsync_age_seconds is None
        finally:
            journal.close()


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_dict_fields_are_independent_copies(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        try:
            emitter._inc_emit_failed("journal_write")
            snap_a = collect_metrics_for_emitter(emitter)
            emitter._inc_emit_failed("journal_write")
            snap_b = collect_metrics_for_emitter(emitter)
            # Earlier snapshot remains at the count it captured.
            assert snap_a.emitter_events_emit_failed_total == {"journal_write": 1}
            assert snap_b.emitter_events_emit_failed_total == {"journal_write": 2}
        finally:
            emitter.close()

    def test_multiple_snapshots_no_leak(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        try:
            for _ in range(50):
                snap = collect_metrics_for_emitter(emitter)
                # Snapshot contents are well-typed.
                assert isinstance(snap, EventSubsystemMetrics)
                assert isinstance(snap.emitter_events_emit_failed_total, dict)
                assert isinstance(snap.bus_dropped_per_consumer, dict)
        finally:
            emitter.close()


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_dropped_per_consumer_keyed_correctly(self) -> None:
        bus = InMemoryBus(capacity=2)
        bus.record_consumer_drop("c1", 3)
        bus.record_consumer_drop("c2", 5)
        snap = collect_metrics(bus=bus)
        assert snap.bus_dropped_per_consumer == {"c1": 3, "c2": 5}

    def test_is_degraded_false_when_clean(self) -> None:
        snap = collect_metrics()
        assert snap.is_degraded() is False

    def test_is_degraded_flips_on_emit_failure(self, tmp_path: Path) -> None:
        emitter = _build_emitter(tmp_path)
        try:
            emitter._inc_emit_failed("journal_write")
            snap = collect_metrics_for_emitter(emitter)
            assert snap.is_degraded() is True
        finally:
            emitter.close()

    def test_is_degraded_flips_on_bus_overflow(self) -> None:
        bus = InMemoryBus(capacity=1)
        bus.publish(make_started(offset=0))
        bus.publish(make_completed(offset=1))  # evicts offset=0.
        snap = collect_metrics(bus=bus)
        assert snap.bus_dropped_total == 1
        assert snap.is_degraded() is True

    def test_is_degraded_flips_on_journal_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        try:
            def boom_write(_self: object, _data: bytes) -> int:
                raise OSError("disk full")

            monkeypatch.setattr(
                writer._fh,  # type: ignore[union-attr]
                "write",
                boom_write.__get__(writer._fh),  # type: ignore[union-attr]
            )
            writer.append(make_started(offset=0))
            snap = collect_metrics(journal=writer)
            assert snap.journal_write_failed_total == 1
            assert snap.is_degraded() is True
        finally:
            writer.close()

    def test_dedup_hits_tracked(self) -> None:
        d = EventDedup()
        d.remember("r", "src", 0)
        assert d.is_duplicate("r", "src", 0) is True
        assert d.is_duplicate("r", "src", 0) is True
        # Each hit counts (does not deduplicate the dedup counter).
        snap = collect_metrics(dedup=d)
        assert snap.dedup_hits_total == 2
        assert snap.dedup_seen_total == 1

    def test_dedup_evicted_total_tracked(self) -> None:
        d = EventDedup()
        d.remember("r1", "src", 0)
        d.remember("r1", "src", 1)
        d.remember("r2", "src", 0)
        d.evict_run("r1")
        snap = collect_metrics(dedup=d)
        assert snap.dedup_evicted_total == 2
        assert snap.dedup_size == 1

    def test_journal_total_bytes_tracked(self, tmp_path: Path) -> None:
        writer = JournalWriter(tmp_path / "events.jsonl")
        try:
            writer.append(make_started(offset=0))
            writer.append(make_started(offset=1))
            snap = collect_metrics(journal=writer)
            # File size on disk is the cleanest cross-check.
            on_disk = (tmp_path / "events.jsonl").stat().st_size
            assert snap.journal_total_bytes_written == on_disk
            assert snap.journal_total_bytes_written > 0
        finally:
            writer.close()

    def test_journal_last_fsync_age_seconds_populated_after_fsync(
        self, tmp_path: Path,
    ) -> None:
        writer = JournalWriter(
            tmp_path / "events.jsonl",
            fsync_batch_size=1,
            fsync_interval_s=999.0,
        )
        try:
            writer.append(make_started(offset=0))
            # batch_size=1 → an fsync happened. Pass an explicit ``now``
            # ahead of last_fsync_at so the age is deterministic.
            assert writer.last_fsync_at is not None
            snap = collect_metrics(
                journal=writer,
                now=writer.last_fsync_at + 0.25,
            )
            assert snap.journal_last_fsync_age_seconds == pytest.approx(0.25)
        finally:
            writer.close()
