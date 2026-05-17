"""Tests for :class:`ryotenkai_control.events.JournalReader` (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ryotenkai_control.events import JournalReader, JournalWriter
from ryotenkai_shared.events import UNKNOWN_OFFSET, UnknownEvent

from tests.unit.control.events.conftest import (
    make_completed,
    make_started,
)


def _write_events(path: Path, *events: object) -> None:
    writer = JournalWriter(path)
    for ev in events:
        writer.append(ev)  # type: ignore[arg-type]
    writer.close()


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_iter_envelopes_returns_events_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(
            path,
            make_started(offset=0),
            make_completed(offset=1),
        )
        reader = JournalReader(path)
        out = list(reader.iter_envelopes())
        assert [e.offset for e in out] == [0, 1]
        assert out[0].kind == "ryotenkai.control.run.started"
        assert out[1].kind == "ryotenkai.control.run.completed"

    def test_path_property(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        reader = JournalReader(path)
        assert reader.path == path


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_partial_last_line_truncated(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(path, make_started(offset=0))
        # Append a torn last line.
        with path.open("ab") as fh:
            fh.write(b"99\t{\"partial\":")
        reader = JournalReader(path)
        assert reader.has_torn_tail()
        assert reader.truncate_torn_tail() is True
        assert reader.has_torn_tail() is False

    def test_malformed_mid_file_line_becomes_unknown(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(path, make_started(offset=0))
        # Inject a malformed line between two valid envelopes by hand-
        # crafting the journal contents.
        good = path.read_bytes()
        with path.open("wb") as fh:
            fh.write(good)
            fh.write(b"42\t{not json}\n")
        # Append a second valid envelope.
        writer = JournalWriter(path)
        writer.append(make_completed(offset=1))
        writer.close()

        envelopes = list(JournalReader(path).iter_envelopes())
        # 0: valid started, 1: UnknownEvent for malformed line, 2: valid completed.
        assert len(envelopes) == 3
        assert isinstance(envelopes[1], UnknownEvent)
        assert envelopes[2].offset == 1


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.touch()
        reader = JournalReader(path)
        assert list(reader.iter_envelopes()) == []
        assert reader.newest_persisted_offset_per_source() == {}

    def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.jsonl"
        reader = JournalReader(path)
        assert list(reader.iter_envelopes()) == []
        assert reader.newest_persisted_offset_per_source() == {}
        assert reader.truncate_torn_tail() is False


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_newest_persisted_offset_per_source_picks_max(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(
            path,
            make_started(source="control://orchestrator", offset=0),
            make_completed(source="control://orchestrator", offset=5),
            make_started(source="control://api", offset=2),
        )
        out = JournalReader(path).newest_persisted_offset_per_source()
        assert out["control://orchestrator"] == 5
        assert out["control://api"] == 2

    def test_unknown_marker_skipped_by_newest_offset(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(path, make_started(offset=0))
        # Inject malformed line whose UnknownEvent carries UNKNOWN_OFFSET.
        with path.open("ab") as fh:
            fh.write(b"5\t{bad}\n")
        out = JournalReader(path).newest_persisted_offset_per_source()
        # Only the well-formed started event contributes.
        assert out["control://orchestrator"] == 0


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_iter_on_missing_file_returns_empty(self, tmp_path: Path) -> None:
        reader = JournalReader(tmp_path / "nope.jsonl")
        assert list(reader.iter_envelopes()) == []

    def test_tail_per_source_rejects_negative(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.touch()
        with pytest.raises(ValueError):
            JournalReader(path).tail_per_source(max_entries_per_source=-1)


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_raw_json_line_legacy_format_is_decoded(self, tmp_path: Path) -> None:
        """The codec accepts a raw-JSON line (no length prefix) per
        Phase 1 back-compat. The reader must surface it without
        crashing — important for older fixtures.
        """
        from ryotenkai_shared.events import to_jsonl
        path = tmp_path / "events.jsonl"
        # Build a raw-JSON line by stripping the length prefix.
        line = to_jsonl(make_started(offset=0))
        _, _, body_with_nl = line.partition("\t")
        path.write_bytes(body_with_nl.encode("utf-8"))
        envelopes = list(JournalReader(path).iter_envelopes())
        # Codec recognises raw-JSON form and the event decodes fine.
        assert len(envelopes) == 1
        assert envelopes[0].offset == 0


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_replay_from_filters_by_offset(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(
            path,
            make_started(offset=0),
            make_completed(offset=1),
            make_completed(offset=2),
            make_completed(offset=3),
        )
        reader = JournalReader(path)
        out = list(reader.replay_from(after_offset=1))
        assert [e.offset for e in out] == [2, 3]

    def test_replay_from_with_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(
            path,
            make_started(offset=0),
            make_completed(offset=1),
            make_completed(offset=2),
        )
        reader = JournalReader(path)
        out = list(reader.replay_from(after_offset=-1, limit=2))
        assert [e.offset for e in out] == [0, 1]

    def test_replay_skips_unknown_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _write_events(path, make_started(offset=0))
        # Inject malformed mid-file.
        with path.open("ab") as fh:
            fh.write(b"5\t{bad}\n")
        writer = JournalWriter(path)
        writer.append(make_completed(offset=1))
        writer.close()
        reader = JournalReader(path)
        out = list(reader.replay_from(after_offset=-1))
        # The UnknownEvent (offset=UNKNOWN_OFFSET=-1) is filtered out.
        assert all(e.offset != UNKNOWN_OFFSET for e in out)
        assert [e.offset for e in out] == [0, 1]

    def test_tail_per_source_bounds_each_bucket(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        events = []
        for i in range(10):
            events.append(make_started(offset=i, source="control://orchestrator"))
        _write_events(path, *events)
        out = JournalReader(path).tail_per_source(max_entries_per_source=3)
        assert "control://orchestrator" in out
        offsets = [e.offset for e in out["control://orchestrator"]]
        # The last 3 by file order.
        assert offsets == [7, 8, 9]
