"""EventMirrorWriter — Mac-side append-only mirror of pod runner journal.

Categories covered (per project test policy):
* Positive — basic write + read-back, context manager closes file
* Negative — write after close, malformed input
* Boundary — empty events, unicode payloads, no-events run
* Invariant — JSONL each-line valid; offsets monotonic if input is
* Dependency-error — parent dir missing → auto-created
* Regression — fsync called periodically
* Logic-specific — context manager opens lazily; closing flushes
* Combinatorial — fsync_every_n × write_count interaction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.pipeline.stages.managers.event_mirror import EventMirrorWriter

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_event(offset: int, kind: str = "trainer_log", **payload: object) -> dict:
    return {
        "v": 1,
        "offset": offset,
        "ts": "2026-04-30T00:00:00Z",
        "kind": kind,
        "payload": payload,
    }


@pytest.fixture
def attempt_dir(tmp_path: Path) -> Iterator[Path]:
    """Empty attempt-dir; mirror auto-creates the events/ subdirectory
    on first write."""
    yield tmp_path


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_write_appends_jsonl_line(attempt_dir: Path) -> None:
    """One write → one JSON line in the mirror file with the expected shape."""
    with EventMirrorWriter(attempt_dir) as mirror:
        mirror.write(_make_event(0, kind="trainer_log", line="hello"))

    raw = mirror.path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 1
    parsed = json.loads(raw[0])
    assert parsed["offset"] == 0
    assert parsed["kind"] == "trainer_log"
    assert parsed["payload"]["line"] == "hello"


def test_context_manager_closes_file(attempt_dir: Path) -> None:
    """Exiting ``with`` block must close the underlying handle. After
    close, ``write()`` raises so the caller knows their data is lost.
    """
    mirror = EventMirrorWriter(attempt_dir)
    with mirror:
        mirror.write(_make_event(0))
    with pytest.raises(RuntimeError, match="closed"):
        mirror.write(_make_event(1))


def test_multiple_writes_each_get_their_own_line(attempt_dir: Path) -> None:
    with EventMirrorWriter(attempt_dir) as mirror:
        for i in range(5):
            mirror.write(_make_event(i, kind="trainer_log", line=f"l{i}"))
    lines = mirror.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    parsed_offsets = [json.loads(line)["offset"] for line in lines]
    assert parsed_offsets == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_write_after_close_raises(attempt_dir: Path) -> None:
    mirror = EventMirrorWriter(attempt_dir)
    mirror.close()
    with pytest.raises(RuntimeError):
        mirror.write(_make_event(0))


def test_double_close_is_idempotent(attempt_dir: Path) -> None:
    """``close()`` must be safe to call twice — defensive against
    callers wrapping the writer in their own try/finally on top of
    the context manager."""
    mirror = EventMirrorWriter(attempt_dir)
    mirror.write(_make_event(0))
    mirror.close()
    mirror.close()  # MUST NOT raise


def test_negative_fsync_every_n_rejected(attempt_dir: Path) -> None:
    with pytest.raises(ValueError, match="fsync_every_n"):
        EventMirrorWriter(attempt_dir, fsync_every_n=-1)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_no_writes_no_file_created(attempt_dir: Path) -> None:
    """The mirror is opened lazily — a context manager that does
    nothing should leave the disk untouched. Otherwise every short-
    lived monitor would litter empty files."""
    with EventMirrorWriter(attempt_dir):
        pass
    expected = (
        attempt_dir
        / EventMirrorWriter.EVENTS_DIR_NAME
        / EventMirrorWriter.MIRROR_FILE_NAME
    )
    assert not expected.exists()


def test_unicode_payload_round_trips(attempt_dir: Path) -> None:
    payload_line = "тест 🚀 multi-byte"
    with EventMirrorWriter(attempt_dir) as mirror:
        mirror.write(_make_event(0, line=payload_line))
    parsed = json.loads(mirror.path.read_text(encoding="utf-8"))
    assert parsed["payload"]["line"] == payload_line


def test_existing_file_appends_not_truncates(attempt_dir: Path) -> None:
    """A second monitor session on the same attempt dir should append
    to the existing mirror, not overwrite it. Prevents data loss on
    process restart with the same attempt."""
    with EventMirrorWriter(attempt_dir) as m1:
        m1.write(_make_event(0, line="first"))
    with EventMirrorWriter(attempt_dir) as m2:
        m2.write(_make_event(1, line="second"))
    lines = (
        attempt_dir
        / EventMirrorWriter.EVENTS_DIR_NAME
        / EventMirrorWriter.MIRROR_FILE_NAME
    ).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["payload"]["line"] == "first"
    assert json.loads(lines[1])["payload"]["line"] == "second"


# ---------------------------------------------------------------------------
# Invariant
# ---------------------------------------------------------------------------


def test_each_line_is_valid_json(attempt_dir: Path) -> None:
    """Mirror is a strict JSONL; no broken lines even if writes are
    interrupted mid-write (we don't simulate that here, just verify
    the happy-path correctness)."""
    with EventMirrorWriter(attempt_dir) as mirror:
        for i in range(20):
            mirror.write(_make_event(i, kind="trainer_log", line="x" * 50))
    for line in mirror.path.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # raises if not valid JSON


def test_offsets_preserved_in_file_order(attempt_dir: Path) -> None:
    """If the input stream had monotonic offsets, the mirror file
    has them too. We rely on this for reader catch-up arithmetic."""
    with EventMirrorWriter(attempt_dir) as mirror:
        for i in [0, 1, 2, 3, 5, 8, 13]:
            mirror.write(_make_event(i))
    parsed = [json.loads(line) for line in mirror.path.read_text().splitlines()]
    assert [e["offset"] for e in parsed] == [0, 1, 2, 3, 5, 8, 13]


# ---------------------------------------------------------------------------
# Dependency-error
# ---------------------------------------------------------------------------


def test_events_dir_auto_created(tmp_path: Path) -> None:
    """First write must create the ``events/`` subdir under attempt_dir.
    Tests caller can pass any attempt_dir without pre-mkdir."""
    attempt_dir = tmp_path / "attempts" / "attempt_1"
    attempt_dir.mkdir(parents=True)
    assert not (attempt_dir / EventMirrorWriter.EVENTS_DIR_NAME).exists()
    with EventMirrorWriter(attempt_dir) as mirror:
        mirror.write(_make_event(0))
    assert (attempt_dir / EventMirrorWriter.EVENTS_DIR_NAME).is_dir()


# ---------------------------------------------------------------------------
# Regression — fsync semantics
# ---------------------------------------------------------------------------


def test_fsync_called_every_n_writes(attempt_dir: Path) -> None:
    """``fsync_every_n=3`` should call ``os.fsync`` on writes 3, 6, 9."""
    with patch(
        "src.pipeline.stages.managers.event_mirror.os.fsync"
    ) as fsync_mock:
        with EventMirrorWriter(attempt_dir, fsync_every_n=3) as mirror:
            for i in range(7):
                mirror.write(_make_event(i))
        # close() also fsyncs; expect 2 in-loop fsyncs (writes 3 and 6)
        # plus 1 close-time fsync.
        assert fsync_mock.call_count >= 2


def test_fsync_zero_disables_periodic_sync(attempt_dir: Path) -> None:
    """Setting ``fsync_every_n=0`` skips periodic fsync; only the
    close-time fsync runs."""
    with patch(
        "src.pipeline.stages.managers.event_mirror.os.fsync"
    ) as fsync_mock:
        with EventMirrorWriter(attempt_dir, fsync_every_n=0) as mirror:
            for i in range(5):
                mirror.write(_make_event(i))
        # Only the close-time fsync (1 call total).
        assert fsync_mock.call_count == 1


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


def test_path_property_resolves_before_open(attempt_dir: Path) -> None:
    """``mirror.path`` must be the canonical mirror path even before
    any write happens — used by API endpoints to compute the path
    without instantiating a writer."""
    mirror = EventMirrorWriter(attempt_dir)
    assert mirror.path == (
        attempt_dir
        / EventMirrorWriter.EVENTS_DIR_NAME
        / EventMirrorWriter.MIRROR_FILE_NAME
    )


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_events", "fsync_every"),
    [
        (1, 1),
        (50, 50),
        (100, 50),
        (0, 50),
    ],
)
def test_fsync_count_matches_expected(
    attempt_dir: Path, n_events: int, fsync_every: int,
) -> None:
    with patch(
        "src.pipeline.stages.managers.event_mirror.os.fsync",
    ) as fsync_mock, EventMirrorWriter(attempt_dir, fsync_every_n=fsync_every) as mirror:
        for i in range(n_events):
            mirror.write(_make_event(i))

    expected_periodic = (
        n_events // fsync_every if fsync_every > 0 and n_events > 0 else 0
    )
    expected_close = 1 if n_events > 0 else 0
    assert fsync_mock.call_count == expected_periodic + expected_close
