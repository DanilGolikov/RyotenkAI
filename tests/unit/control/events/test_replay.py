"""Tests for :func:`ryotenkai_control.events.slice_journal` (Phase 3)."""

from __future__ import annotations

from pathlib import Path

from ryotenkai_control.events import JournalReader, JournalWriter, slice_journal

from tests.unit.control.events.conftest import make_completed, make_started


def _seed(path: Path) -> None:
    writer = JournalWriter(path)
    writer.append(make_started(offset=0))
    writer.append(make_completed(offset=1))
    writer.append(make_completed(offset=2))
    writer.append(make_completed(offset=3))
    writer.close()


class TestSliceJournal:
    def test_default_after_offset_returns_everything(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _seed(path)
        out = list(slice_journal(JournalReader(path)))
        assert [e.offset for e in out] == [0, 1, 2, 3]

    def test_after_offset_filters(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _seed(path)
        out = list(slice_journal(JournalReader(path), after_offset=1))
        assert [e.offset for e in out] == [2, 3]

    def test_limit_caps_yielded_count(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        _seed(path)
        out = list(slice_journal(JournalReader(path), limit=2))
        assert [e.offset for e in out] == [0, 1]

    def test_empty_journal_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.touch()
        out = list(slice_journal(JournalReader(path)))
        assert out == []

    def test_filters_unknown_marker(self, tmp_path: Path) -> None:
        """The wrapper inherits the reader's behaviour: torn-write
        UnknownEvents with UNKNOWN_OFFSET are filtered.
        """
        path = tmp_path / "events.jsonl"
        _seed(path)
        # Inject a malformed line so the codec produces UnknownEvent.
        with path.open("ab") as fh:
            fh.write(b"5\t{bad}\n")
        out = list(slice_journal(JournalReader(path)))
        assert [e.offset for e in out] == [0, 1, 2, 3]
