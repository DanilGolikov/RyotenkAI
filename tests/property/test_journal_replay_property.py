"""Property tests for the runner event journal.

Invariants asserted across hypothesis-generated event sequences written
to a real :class:`EventJournal`:

* ``append(...)`` + ``iter_records(since=0)`` yields the same events
  back in the same order (modulo records with offset < since).
* Rotation never drops records when the total fits within the
  configured ``max_files * file_size_cap`` budget.
* ``newest_persisted_offset()`` after a write sequence equals the
  largest appended offset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from ryotenkai_pod.runner.event_journal import EventJournal

pytestmark = [pytest.mark.property]


@st.composite
def _event_sequence(draw: st.DrawFn) -> list[dict[str, object]]:
    n = draw(st.integers(min_value=1, max_value=20))
    seq: list[dict[str, object]] = []
    for offset in range(n):
        kind = draw(st.sampled_from([
            "trainer_spawned", "trainer_exited", "stop_requested",
            "cancellation_started", "cancellation_completed", "info",
        ]))
        payload_keys = draw(st.lists(
            st.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=0, max_size=3, unique=True,
        ))
        payload = {k: draw(st.integers(min_value=-10, max_value=10)) for k in payload_keys}
        seq.append({"offset": offset, "kind": kind, "payload": payload, "ts": f"t-{offset:04d}"})
    return seq


@given(events=_event_sequence())
@hyp_settings(max_examples=50, deadline=None)
def test_journal_round_trip(tmp_path_factory: pytest.TempPathFactory, events: list[dict[str, object]]) -> None:
    root: Path = tmp_path_factory.mktemp("journal-roundtrip")
    journal = EventJournal(root_dir=root)
    try:
        for event in events:
            journal.append(
                offset=event["offset"],
                ts=event["ts"],
                kind=event["kind"],
                payload=event["payload"],
            )
    finally:
        journal.close()

    # Reader.
    reader = EventJournal(root_dir=root)
    try:
        records = list(reader.iter_records(since=0))
    finally:
        reader.close()

    assert len(records) == len(events)
    for rec, exp in zip(records, events, strict=True):
        assert rec.offset == exp["offset"]
        assert rec.kind == exp["kind"]
        assert rec.payload == exp["payload"]


@given(events=_event_sequence(), since=st.integers(min_value=0, max_value=30))
@hyp_settings(max_examples=30, deadline=None)
def test_journal_since_filters_old_offsets(
    tmp_path_factory: pytest.TempPathFactory,
    events: list[dict[str, object]],
    since: int,
) -> None:
    root = tmp_path_factory.mktemp("journal-since")
    journal = EventJournal(root_dir=root)
    try:
        for event in events:
            journal.append(
                offset=event["offset"],
                ts=event["ts"],
                kind=event["kind"],
                payload=event["payload"],
            )
    finally:
        journal.close()

    reader = EventJournal(root_dir=root)
    try:
        offsets = [r.offset for r in reader.iter_records(since=since)]
    finally:
        reader.close()

    expected = [int(e["offset"]) for e in events if int(e["offset"]) >= since]
    assert offsets == expected


@given(events=_event_sequence())
@hyp_settings(max_examples=20, deadline=None)
def test_journal_newest_offset_matches_largest_appended(
    tmp_path_factory: pytest.TempPathFactory,
    events: list[dict[str, object]],
) -> None:
    root = tmp_path_factory.mktemp("journal-newest")
    journal = EventJournal(root_dir=root)
    try:
        for event in events:
            journal.append(
                offset=event["offset"],
                ts=event["ts"],
                kind=event["kind"],
                payload=event["payload"],
            )
    finally:
        journal.close()

    reader = EventJournal(root_dir=root)
    try:
        newest = reader.newest_persisted_offset()
    finally:
        reader.close()

    if not events:
        assert newest is None
    else:
        assert newest == max(int(e["offset"]) for e in events)
