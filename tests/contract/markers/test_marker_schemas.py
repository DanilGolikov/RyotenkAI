"""Marker-file payload schemas — property + golden validation.

The production writers in ``packages/pod/src/ryotenkai_pod/trainer/
callbacks/{cancellation,completion}_callback.py`` build payloads as
Python dicts and serialise them via ``json.dumps``. We assert via
hypothesis that any plausible writer-side input produces a payload
that JSON-schema validates against the committed contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tests._contracts import CONTRACTS_DIR

pytestmark = [pytest.mark.contract, pytest.mark.property]


def _load(name: str) -> dict:
    return json.loads((CONTRACTS_DIR / name).read_text())


_CANCELLED_SCHEMA = _load("marker_cancelled_schema.json")
_COMPLETION_SCHEMA = _load("marker_completion_schema.json")
_CANCELLED_VALIDATOR = jsonschema.Draft202012Validator(_CANCELLED_SCHEMA)
_COMPLETION_VALIDATOR = jsonschema.Draft202012Validator(_COMPLETION_SCHEMA)


def _emit_cancelled(*, run_id: str | None, drained: int, ts_ms: int) -> dict:
    """Mirror :meth:`CancellationCallback._write_cancelled_marker` payload."""
    return {
        "run_id": run_id,
        "flushed_count": int(drained),
        "ts_ms": int(ts_ms),
        "reason": "flush_budget_exceeded",
    }


def _emit_completion(
    *, run_id: str | None, drained: int, flush_timed_out: bool, ts_ms: int,
) -> dict:
    """Mirror :meth:`CompletionCallback._write_completion_marker` payload."""
    return {
        "run_id": run_id,
        "flushed_count": int(drained),
        "flush_timed_out": bool(flush_timed_out),
        "ts_ms": int(ts_ms),
        "reason": "flush_budget_exceeded" if flush_timed_out else "natural_completion",
    }


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_cancelled_schema_well_formed() -> None:
    jsonschema.Draft202012Validator.check_schema(_CANCELLED_SCHEMA)
    assert _CANCELLED_SCHEMA["title"]
    assert _CANCELLED_SCHEMA["version"]
    assert _CANCELLED_SCHEMA["$id"]


def test_completion_schema_well_formed() -> None:
    jsonschema.Draft202012Validator.check_schema(_COMPLETION_SCHEMA)
    assert _COMPLETION_SCHEMA["title"]
    assert _COMPLETION_SCHEMA["version"]
    assert _COMPLETION_SCHEMA["$id"]


# ---------------------------------------------------------------------------
# Property: anything the writer could produce validates
# ---------------------------------------------------------------------------


@given(
    run_id=st.one_of(st.none(), st.text(min_size=0, max_size=40)),
    drained=st.integers(min_value=0, max_value=10_000),
    ts_ms=st.integers(min_value=0, max_value=2**60),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_cancelled_writer_output_validates(
    run_id: str | None, drained: int, ts_ms: int,
) -> None:
    payload = _emit_cancelled(run_id=run_id, drained=drained, ts_ms=ts_ms)
    # Writer always serialises through json.dumps then we read back.
    roundtripped = json.loads(json.dumps(payload, indent=2))
    _CANCELLED_VALIDATOR.validate(roundtripped)


@given(
    run_id=st.one_of(st.none(), st.text(min_size=0, max_size=40)),
    drained=st.integers(min_value=0, max_value=10_000),
    flush_timed_out=st.booleans(),
    ts_ms=st.integers(min_value=0, max_value=2**60),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_completion_writer_output_validates(
    run_id: str | None, drained: int, flush_timed_out: bool, ts_ms: int,
) -> None:
    payload = _emit_completion(
        run_id=run_id, drained=drained, flush_timed_out=flush_timed_out, ts_ms=ts_ms,
    )
    roundtripped = json.loads(json.dumps(payload, indent=2))
    _COMPLETION_VALIDATOR.validate(roundtripped)


# ---------------------------------------------------------------------------
# Negative: malformed payloads are rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": "flush_budget_exceeded", "ts_ms": 1, "flushed_count": -1},  # negative
        {"reason": "wrong_reason", "ts_ms": 1, "flushed_count": 0},  # bad enum
        {"reason": "flush_budget_exceeded", "ts_ms": 1},  # missing flushed_count
        {"reason": "flush_budget_exceeded", "ts_ms": 1, "flushed_count": 0, "extra": 1},  # extra
    ],
)
def test_cancelled_schema_rejects_malformed(payload: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        _CANCELLED_VALIDATOR.validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        # missing reason
        {"flushed_count": 0, "flush_timed_out": False, "ts_ms": 1},
        # invalid reason
        {"flushed_count": 0, "flush_timed_out": False, "ts_ms": 1, "reason": "x"},
        # wrong type
        {"flushed_count": "0", "flush_timed_out": False, "ts_ms": 1, "reason": "natural_completion"},
    ],
)
def test_completion_schema_rejects_malformed(payload: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        _COMPLETION_VALIDATOR.validate(payload)


# ---------------------------------------------------------------------------
# Real-file fixture (writer exercised end-to-end)
# ---------------------------------------------------------------------------


def test_writer_emit_then_validate_via_atomic_write(tmp_path: Path) -> None:
    """End-to-end: produce a payload exactly as the writer does
    (``json.dumps(..., indent=2)``), then validate the on-disk text."""
    target = tmp_path / "cancelled.marker"
    payload = _emit_cancelled(run_id="abc", drained=42, ts_ms=1234567890)
    target.write_text(json.dumps(payload, indent=2))
    parsed = json.loads(target.read_text())
    _CANCELLED_VALIDATOR.validate(parsed)
