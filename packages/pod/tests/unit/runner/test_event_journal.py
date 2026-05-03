"""Phase 12.B — :class:`EventJournal` contract.

7-category coverage for the durable JSONL journal that backs the
EventBus across long Mac sleeps and runner restarts.

Slim-venv compatible: pure stdlib + the runner module under test.
No FastAPI, no asyncio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_pod.runner.event_journal import (
    DEFAULT_FILE_SIZE_CAP,
    EVENTS_DIR_REL,
    EVENTS_FILE_FMT,
    EventJournal,
    JournalRecord,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_journal(
    tmp_path: Path,
    *,
    file_size_cap: int = DEFAULT_FILE_SIZE_CAP,
    max_files: int = 5,
    fsync_batch: int = 50,
    fsync_interval_ms: int = 1000,
) -> EventJournal:
    return EventJournal(
        root_dir=tmp_path / EVENTS_DIR_REL,
        file_size_cap=file_size_cap,
        max_files=max_files,
        fsync_batch=fsync_batch,
        fsync_interval_ms=fsync_interval_ms,
    )


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# 1. Positive — append + readback
# ---------------------------------------------------------------------------


class TestPositive:
    def test_append_persists_record(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        j.append(offset=0, ts="2026-04-27T00:00:00Z", kind="step", payload={"loss": 0.5})
        j.fsync_now()
        j.close()

        path = tmp_path / EVENTS_DIR_REL / "events.000.jsonl"
        records = _read_lines(path)
        assert len(records) == 1
        assert records[0]["v"] == SCHEMA_VERSION
        assert records[0]["offset"] == 0
        assert records[0]["kind"] == "step"
        assert records[0]["payload"] == {"loss": 0.5}

    def test_iter_records_yields_in_offset_order(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        for i in range(5):
            j.append(offset=i, ts="t", kind="k", payload={"i": i})
        j.close()

        offsets = [r.offset for r in j.iter_records(since=0)]
        assert offsets == [0, 1, 2, 3, 4]

    def test_iter_since_filters_out_older(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        for i in range(10):
            j.append(offset=i, ts="t", kind="k", payload={})
        j.close()

        offsets = [r.offset for r in j.iter_records(since=5)]
        assert offsets == [5, 6, 7, 8, 9]

    def test_oldest_and_newest_persisted_offset(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        for i in range(7):
            j.append(offset=i, ts="t", kind="k", payload={})
        j.close()

        assert j.oldest_persisted_offset() == 0
        assert j.newest_persisted_offset() == 6


# ---------------------------------------------------------------------------
# 2. Negative — corrupt / truncated lines
# ---------------------------------------------------------------------------


class TestNegative:
    def test_truncated_last_line_skipped_on_read(
        self, tmp_path: Path
    ) -> None:
        j = _make_journal(tmp_path)
        j.append(offset=0, ts="t", kind="k", payload={"a": 1})
        j.close()

        # Append a half-written line manually.
        path = tmp_path / EVENTS_DIR_REL / "events.000.jsonl"
        with path.open("ab") as fh:
            fh.write(b'{"v":1,"offset":1,"kind":"truncated"')

        # Reader skips it.
        # Re-construct journal to pick up the file.
        j2 = _make_journal(tmp_path)
        records = list(j2.iter_records(since=0))
        assert len(records) == 1
        assert records[0].offset == 0
        j2.close()

    def test_unsupported_schema_version_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        j = _make_journal(tmp_path)
        j.close()
        path = tmp_path / EVENTS_DIR_REL / "events.000.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"v":99,"offset":0,"ts":"t","kind":"k","payload":{}}\n')
        # New journal sees the v=99 record and skips it.
        j2 = _make_journal(tmp_path)
        records = list(j2.iter_records(since=0))
        assert records == []
        j2.close()

    def test_close_after_close_is_idempotent(
        self, tmp_path: Path
    ) -> None:
        j = _make_journal(tmp_path)
        j.close()
        j.close()  # MUST NOT raise

    def test_append_after_close_raises(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        j.close()
        with pytest.raises(RuntimeError):
            j.append(offset=0, ts="t", kind="k", payload={})


# ---------------------------------------------------------------------------
# 3. Boundary — rotation, fsync batching
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_rotation_at_cap(self, tmp_path: Path) -> None:
        # Tiny cap so rotation fires every ~2 records. max_files=100
        # so we don't lose any during this 20-record run — the
        # drop-oldest behaviour is exercised separately in
        # ``test_drop_oldest_when_max_files_exceeded``.
        j = _make_journal(tmp_path, file_size_cap=200, max_files=100)
        for i in range(20):
            j.append(offset=i, ts="t", kind="k", payload={"data": "x" * 50})
        j.close()

        # We should have multiple files now.
        events_dir = tmp_path / EVENTS_DIR_REL
        files = sorted(events_dir.glob("events.*.jsonl"))
        assert len(files) >= 2  # at least one rotation
        # Records are strictly monotonic across files.
        seen: list[int] = []
        for f in files:
            for rec in _read_lines(f):
                seen.append(rec["offset"])
        assert seen == list(range(20))

    def test_drop_oldest_when_max_files_exceeded(
        self, tmp_path: Path
    ) -> None:
        # cap=100, max_files=2 → after enough records we drop the
        # oldest file.
        j = _make_journal(tmp_path, file_size_cap=100, max_files=2)
        for i in range(50):
            j.append(offset=i, ts="t", kind="k", payload={"data": "x" * 30})
        j.close()

        files = sorted((tmp_path / EVENTS_DIR_REL).glob("events.*.jsonl"))
        # max_files=2 enforced → exactly 2 files left.
        assert len(files) == 2
        # Oldest record present is NO LONGER offset 0 (file 0 dropped).
        assert j.oldest_persisted_offset() is not None
        assert j.oldest_persisted_offset() > 0
        # Newest still 49.
        assert j.newest_persisted_offset() == 49

    def test_fsync_batch_triggers_flush(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Replace os.fsync with a counter to verify batching.
        from ryotenkai_pod.runner import event_journal as ej_mod

        fsync_calls: list[int] = []
        original_fsync = os.fsync

        def _counting_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            original_fsync(fd)

        monkeypatch.setattr(ej_mod.os, "fsync", _counting_fsync)

        # batch=10, interval=very large → only a single fsync after 10 appends.
        j = _make_journal(tmp_path, fsync_batch=10, fsync_interval_ms=10_000_000)
        for i in range(9):
            j.append(offset=i, ts="t", kind="k", payload={})
        # Below batch threshold → no fsync yet.
        assert len(fsync_calls) == 0
        # 10th append triggers the batch flush.
        j.append(offset=9, ts="t", kind="k", payload={})
        assert len(fsync_calls) == 1
        j.close()  # final fsync

    def test_fsync_interval_triggers_flush(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ryotenkai_pod.runner import event_journal as ej_mod

        # Pin monotonic so the interval check is deterministic.
        clock = [0]

        def _fake_monotonic() -> float:
            return clock[0] / 1000.0  # ms-resolution

        monkeypatch.setattr(ej_mod.time, "monotonic", _fake_monotonic)

        fsync_calls: list[int] = []
        original_fsync = os.fsync
        monkeypatch.setattr(
            ej_mod.os, "fsync", lambda fd: (fsync_calls.append(fd), original_fsync(fd))[1],
        )

        # batch=very large, interval=100ms → triggered by interval, not batch.
        j = _make_journal(tmp_path, fsync_batch=10000, fsync_interval_ms=100)
        # First append at clock=0 — sets last_fsync_ms=0.
        clock[0] = 0
        j.append(offset=0, ts="t", kind="k", payload={})
        # Advance to 50ms — still under interval.
        clock[0] = 50
        j.append(offset=1, ts="t", kind="k", payload={})
        # 1 fsync from initial empty interval check at __init__? No,
        # init does not call fsync. So far 0.
        # Advance to 200ms → next append triggers batch flush by interval.
        clock[0] = 200
        j.append(offset=2, ts="t", kind="k", payload={})
        assert len(fsync_calls) >= 1
        j.close()


# ---------------------------------------------------------------------------
# 4. Invariants — offset monotonicity across files
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_offset_monotonic_across_files(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path, file_size_cap=80)
        for i in range(30):
            j.append(offset=i, ts="t", kind="k", payload={"x": "y" * 20})
        j.close()

        all_offsets: list[int] = []
        for r in j.iter_records(since=0):
            all_offsets.append(r.offset)
        # Strictly monotonic.
        assert all(a < b for a, b in zip(all_offsets, all_offsets[1:]))
        assert all_offsets == sorted(all_offsets)

    def test_close_flushes_pending(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path, fsync_batch=1000, fsync_interval_ms=1000000)
        j.append(offset=0, ts="t", kind="k", payload={})
        j.close()

        # File contains the record on disk.
        path = tmp_path / EVENTS_DIR_REL / "events.000.jsonl"
        assert _read_lines(path) == [
            {"v": 1, "offset": 0, "ts": "t", "kind": "k", "payload": {}}
        ]


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_invalid_constructor_args(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _make_journal(tmp_path, file_size_cap=0)
        with pytest.raises(ValueError):
            _make_journal(tmp_path, max_files=0)
        with pytest.raises(ValueError):
            _make_journal(tmp_path, fsync_batch=0)
        with pytest.raises(ValueError):
            _make_journal(tmp_path, fsync_interval_ms=-1)


# ---------------------------------------------------------------------------
# 6. Regressions — crash recovery
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_crash_recovery_resumes_from_max_seq(
        self, tmp_path: Path
    ) -> None:
        # Simulate previous lifecycle: write some records, close.
        j = _make_journal(tmp_path)
        for i in range(3):
            j.append(offset=i, ts="t", kind="k", payload={})
        j.close()

        # New journal picks up where the old one left off.
        j2 = _make_journal(tmp_path)
        # Newest offset reconciled correctly.
        assert j2.newest_persisted_offset() == 2
        # Append continues; new record goes into events.000.jsonl
        # (current file under cap). Its offset is just monotonic from
        # the bus side — journal doesn't enforce ordering, only
        # records what it's told.
        j2.append(offset=3, ts="t", kind="k", payload={})
        j2.close()

        records = list(j2.iter_records(since=0))
        offsets = [r.offset for r in records]
        assert offsets == [0, 1, 2, 3]

    def test_partial_rotation_on_init(self, tmp_path: Path) -> None:
        # Pre-create files.000 (bigger than cap) — simulate "rotation
        # interrupted before next file was opened".
        events_dir = tmp_path / EVENTS_DIR_REL
        events_dir.mkdir(parents=True, exist_ok=True)
        oversized = events_dir / EVENTS_FILE_FMT.format(seq=0)
        oversized.write_bytes(
            b'{"v":1,"offset":0,"ts":"t","kind":"k","payload":{}}\n' * 100
        )

        # New journal opens; size > cap so the next append should
        # rotate immediately.
        j = _make_journal(tmp_path, file_size_cap=100)
        j.append(offset=100, ts="t", kind="k", payload={})
        j.close()

        files = sorted(events_dir.glob("events.*.jsonl"))
        # We should have rotated to events.001.jsonl on first append.
        assert (events_dir / "events.001.jsonl") in files


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_unicode_payload_persisted_compactly(
        self, tmp_path: Path
    ) -> None:
        # Cyrillic message should NOT be encoded as \uXXXX
        # (ensure_ascii=False).
        j = _make_journal(tmp_path)
        j.append(offset=0, ts="t", kind="msg", payload={"text": "Привет"})
        j.close()

        path = tmp_path / EVENTS_DIR_REL / "events.000.jsonl"
        raw = path.read_bytes()
        assert "Привет".encode() in raw
        assert b"\\u041f" not in raw  # not \u-escaped

    def test_non_serializable_payload_coerced_to_string(
        self, tmp_path: Path
    ) -> None:
        # default=str on json.dumps coerces datetime/Path/Enum to str.
        from datetime import datetime

        j = _make_journal(tmp_path)
        j.append(
            offset=0,
            ts="t",
            kind="dt",
            payload={"when": datetime(2026, 4, 27)},  # non-serializable by default
        )
        j.close()

        records = list(j.iter_records(since=0))
        assert records[0].payload["when"].startswith("2026-04-27")

    def test_total_bytes_and_file_count(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path, file_size_cap=200)
        for i in range(20):
            j.append(offset=i, ts="t", kind="k", payload={"x": "y" * 30})
        j.close()

        # Multiple files.
        assert j.file_count() >= 2
        # Total bytes = sum of file sizes.
        events_dir = tmp_path / EVENTS_DIR_REL
        actual_total = sum(p.stat().st_size for p in events_dir.glob("events.*.jsonl"))
        assert j.total_bytes() == actual_total

    def test_root_dir_property_exposes_path(self, tmp_path: Path) -> None:
        j = _make_journal(tmp_path)
        assert j.root_dir == tmp_path / EVENTS_DIR_REL
        j.close()

    def test_empty_journal_returns_none_for_offsets(
        self, tmp_path: Path
    ) -> None:
        j = _make_journal(tmp_path)
        assert j.oldest_persisted_offset() is None
        assert j.newest_persisted_offset() is None
        j.close()
