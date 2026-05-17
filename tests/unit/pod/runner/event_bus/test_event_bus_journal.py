"""Phase 2 — :class:`EventBus` + :class:`EventJournal` integration.

Coverage:

* Every typed-envelope ``publish`` persists to disk via the shared
  codec.
* The bus reconciles its offset counter from the journal's newest
  persisted record on restart.
* :meth:`EventBus.close` flushes the journal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.event_journal import EVENTS_DIR_REL, EventJournal
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.pod_lifecycle import (
    TrainerSpawnedEvent,
    TrainerSpawnedPayload,
)


def _make_pair(
    tmp_path: Path, *, capacity: int = 1000,
) -> tuple[EventBus, EventJournal]:
    journal = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
    bus = EventBus(capacity=capacity, journal=journal)
    return bus, journal


def _make_event(pid: int) -> TrainerSpawnedEvent:
    return TrainerSpawnedEvent(
        source="pod://test/runner",
        run_id="test",
        offset=UNKNOWN_OFFSET,
        payload=TrainerSpawnedPayload(pid=pid, cmdline="py", cwd="/tmp"),
    )


class TestPositive:
    def test_publish_writes_envelope_to_disk(self, tmp_path: Path) -> None:
        bus, journal = _make_pair(tmp_path)
        bus.publish(_make_event(pid=1))
        bus.close()

        envelopes = list(journal.iter_envelopes())
        assert len(envelopes) == 1
        assert envelopes[0].kind == "ryotenkai.pod.lifecycle.trainer_spawned"

    def test_offset_matches_persisted_offset(self, tmp_path: Path) -> None:
        bus, journal = _make_pair(tmp_path)
        o0 = bus.publish(_make_event(pid=1))
        o1 = bus.publish(_make_event(pid=2))
        bus.close()

        envelopes = list(journal.iter_envelopes())
        assert [ev.offset for ev in envelopes] == [o0, o1]


class TestRestartReconciliation:
    def test_restart_reconciles_next_offset(self, tmp_path: Path) -> None:
        bus, _ = _make_pair(tmp_path)
        for i in range(3):
            bus.publish(_make_event(pid=i + 1))
        bus.close()

        # Second bus on the same journal should pick up offsets from 3.
        journal = EventJournal(root_dir=tmp_path / EVENTS_DIR_REL)
        bus2 = EventBus(capacity=10, journal=journal)
        new_offset = bus2.publish(_make_event(pid=99))
        assert new_offset == 3
        bus2.close()


class TestDiskPressure:
    def test_publish_does_not_crash_on_journal_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bus, journal = _make_pair(tmp_path)

        def _broken_append(self: EventJournal, event):  # noqa: ANN001
            raise OSError("simulated EROFS")

        monkeypatch.setattr(EventJournal, "append_envelope", _broken_append)
        # Publish must not raise — the in-memory ring still gets the
        # envelope, the journal write is best-effort.
        new_offset = bus.publish(_make_event(pid=1))
        assert new_offset == 0
        bus.close()


class TestCloseSemantics:
    def test_close_idempotent(self, tmp_path: Path) -> None:
        bus, journal = _make_pair(tmp_path)
        bus.publish(_make_event(pid=1))
        bus.close()
        # Second close is a no-op.
        bus.close()
        assert journal.is_closed
