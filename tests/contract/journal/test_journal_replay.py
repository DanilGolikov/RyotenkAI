"""Journal write→read round-trip.

The production writer (:class:`ryotenkai_pod.runner.event_journal.EventJournal`)
appends one JSONL line per published event. The reader
(:meth:`EventJournal.iter_records`) yields records monotonically. We
generate a deterministic event stream, write it through the writer,
read it back through the reader, and assert the reconstructed
sequence is bit-identical (modulo schema version).

We also validate every record against the committed JSON schema —
catches drift between the schema generator and the writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from ryotenkai_pod.runner.event_journal import (
    EventJournal,
    JournalRecord,
    SCHEMA_VERSION,
)
from tests._contracts import CONTRACTS_DIR

pytestmark = [pytest.mark.contract, pytest.mark.property]

_SCHEMA = json.loads(
    (CONTRACTS_DIR / "runner_events_schema.json").read_text(),
)
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


def _payload_strategy() -> st.SearchStrategy[dict]:
    """Generate JSON-serialisable payload dicts."""
    leaf = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**32), max_value=2**32),
        st.text(max_size=20),
    )
    return st.dictionaries(
        keys=st.text(min_size=1, max_size=12, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        values=leaf,
        max_size=5,
    )


@given(
    events=st.lists(
        st.tuples(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"), _payload_strategy()),
        min_size=1,
        max_size=20,
    ),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_event_journal_round_trip(tmp_path_factory: pytest.TempPathFactory, events: list[tuple[str, dict]]) -> None:
    """Write N events; read them back via iter_records; assert equal."""
    root = tmp_path_factory.mktemp("journal")
    journal = EventJournal(root_dir=root)
    try:
        offsets = []
        for i, (kind, payload) in enumerate(events):
            journal.append(offset=i, ts=f"2026-05-10T00:00:{i:02d}Z", kind=kind, payload=payload)
            offsets.append(i)
        journal.fsync_now()
    finally:
        journal.close()

    reread = list(EventJournal(root_dir=root).iter_records())
    # Reader yields ascending by offset.
    assert [r.offset for r in reread] == offsets
    for record, (kind, payload) in zip(reread, events, strict=True):
        assert record.kind == kind
        assert record.payload == payload
        assert record.v == SCHEMA_VERSION


def test_each_persisted_record_validates_against_schema(tmp_path: Path) -> None:
    journal = EventJournal(root_dir=tmp_path)
    try:
        for i in range(10):
            journal.append(offset=i, ts="2026-05-10T00:00:00Z", kind=f"kind_{i}", payload={"i": i})
        journal.fsync_now()
    finally:
        journal.close()

    # Read each line raw and feed through the schema validator.
    for f in sorted(tmp_path.glob("events.*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            _VALIDATOR.validate(record)


def test_schema_version_pin_matches_production() -> None:
    """If production bumps SCHEMA_VERSION the schema artifact is stale."""
    assert _SCHEMA["properties"]["v"]["maximum"] >= SCHEMA_VERSION


def test_record_dataclass_keys_match_schema() -> None:
    """JournalRecord field names must equal the schema's required keys."""
    fields = {f.name for f in JournalRecord.__dataclass_fields__.values()}
    schema_required = set(_SCHEMA["required"])
    assert fields == schema_required


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        # missing v
        {"offset": 0, "ts": "x", "kind": "k", "payload": {}},
        # negative offset
        {"v": 1, "offset": -1, "ts": "x", "kind": "k", "payload": {}},
        # kind too long
        {"v": 1, "offset": 0, "ts": "x", "kind": "k" * 100, "payload": {}},
        # extra key
        {"v": 1, "offset": 0, "ts": "x", "kind": "k", "payload": {}, "extra": 1},
        # payload not an object
        {"v": 1, "offset": 0, "ts": "x", "kind": "k", "payload": [1, 2]},
    ],
)
def test_schema_rejects_malformed_records(record: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        _VALIDATOR.validate(record)
