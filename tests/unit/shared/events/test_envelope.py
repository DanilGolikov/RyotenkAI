"""Unit tests: :mod:`ryotenkai_shared.events.envelope`.

The envelope module is foundational — every concrete event class extends
:class:`BaseEvent` and inherits its ``frozen=True`` / ``extra=forbid``
invariants, its UUIDv7 default factory, and its UTC datetime factory.
Breaking any of these silently would corrupt every downstream consumer,
so the tests are organised by the seven-class CLAUDE.md template even
though the module itself is tiny.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events.envelope import BaseEvent, new_uuid7, utc_now
from ryotenkai_shared.events.severity import Severity  # noqa: TC001 — Pydantic field type


class _DummyEvent(BaseEvent):
    """Minimal subclass used by tests to exercise the envelope contract."""

    kind: Literal["test.dummy"] = "test.dummy"
    severity: Severity = "info"


def _make_event(**overrides) -> _DummyEvent:
    """Build a :class:`_DummyEvent` with sensible defaults."""
    kwargs = {
        "source": "pod://run-0/trainer",
        "run_id": "run-0",
        "offset": 0,
    }
    kwargs.update(overrides)
    return _DummyEvent(**kwargs)


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_construct_with_defaults_yields_uuid7_and_utc_now(self) -> None:
        event = _make_event()
        assert isinstance(event.event_id, UUID)
        # UUIDv7 sets version = 7. RFC 9562 carries the version in the
        # high nibble of the 7th byte.
        assert event.event_id.version == 7
        assert event.time.tzinfo is UTC
        assert event.schema_version == 1

    def test_explicit_fields_round_trip_through_model_dump(self) -> None:
        event = _make_event(run_id="run-42", offset=17, stage_id="train")
        dumped = event.model_dump()
        assert dumped["run_id"] == "run-42"
        assert dumped["offset"] == 17
        assert dumped["stage_id"] == "train"
        assert dumped["kind"] == "test.dummy"
        assert dumped["severity"] == "info"


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            _DummyEvent(source="pod://x", run_id="x")  # offset omitted

    def test_extra_field_rejected_by_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            _DummyEvent(  # type: ignore[call-arg]
                source="pod://x",
                run_id="x",
                offset=0,
                surprise="boom",
            )

    def test_unknown_severity_rejected(self) -> None:
        with pytest.raises(ValidationError):

            class _BadSev(BaseEvent):
                kind: Literal["test.bad"] = "test.bad"
                severity: Severity = "info"

            _BadSev(  # type: ignore[arg-type]
                source="x", run_id="x", offset=0, severity="oops",
            )


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_offset_zero_is_valid(self) -> None:
        event = _make_event(offset=0)
        assert event.offset == 0

    def test_large_offset_does_not_overflow(self) -> None:
        big = 2**62
        event = _make_event(offset=big)
        assert event.offset == big

    def test_empty_string_source_is_accepted(self) -> None:
        # Pydantic's str does not impose a min length — the emitter is
        # responsible for refusing empty sources in production. We pin
        # the envelope-level behaviour here so a regression surfaces.
        event = _make_event(source="")
        assert event.source == ""

    def test_stage_id_optional_default_none(self) -> None:
        event = _make_event()
        assert event.stage_id is None


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_frozen_blocks_mutation_after_construction(self) -> None:
        event = _make_event()
        with pytest.raises(ValidationError):
            event.offset = 999  # type: ignore[misc]

    def test_uuid7_is_monotonic_within_session(self) -> None:
        from itertools import pairwise

        ids = [new_uuid7() for _ in range(50)]
        sorted_ids = sorted(ids)
        # UUIDv7 has ms resolution so back-to-back ids can tie. We
        # require a strong (not strict) monotonicity ratio.
        monotonic_pairs = sum(1 for a, b in pairwise(ids) if a <= b)
        assert monotonic_pairs >= len(ids) - 5
        assert ids == sorted_ids or monotonic_pairs > 40

    def test_utc_now_carries_utc_tzinfo(self) -> None:
        assert utc_now().tzinfo is UTC

    def test_schema_version_default_is_1(self) -> None:
        event = _make_event()
        assert event.schema_version == 1


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_explicit_naive_datetime_is_passed_through(self) -> None:
        # The envelope does not enforce timezone-awareness on caller-supplied
        # values; the emitter is the boundary that does. We pin the
        # current shape so a future tightening is intentional.
        naive = datetime(2026, 1, 1, 12, 0, 0)
        event = _make_event(time=naive)
        assert event.time == naive
        assert event.time.tzinfo is None

    def test_invalid_uuid_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _DummyEvent(  # type: ignore[arg-type]
                source="x",
                run_id="x",
                offset=0,
                event_id="not-a-uuid",
            )


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_severity_literal_default_pins_through_subclass(self) -> None:
        # Regression for the failure mode where a subclass forgets to
        # pin ``severity`` via Literal and silently inherits "info".
        # The dummy event has severity="info" — exercising the inheritance
        # path defensively.
        event = _make_event()
        assert event.severity == "info"

    def test_uuid7_version_byte_is_7(self) -> None:
        # Guard against silently swapping to uuid4 — uuid4().version == 4
        # which would break journal natural ordering.
        for _ in range(20):
            assert new_uuid7().version == 7


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_extra_forbid_inherited_by_subclass(self) -> None:
        # The model_config sits on BaseEvent; subclasses MUST inherit it.
        # The dummy subclass does not redeclare model_config, so unknown
        # fields should still be rejected.
        with pytest.raises(ValidationError):
            _DummyEvent(  # type: ignore[call-arg]
                source="x",
                run_id="x",
                offset=0,
                tagged_for_audit=True,
            )

    def test_frozen_inherited_by_subclass(self) -> None:
        event = _make_event()
        with pytest.raises(ValidationError):
            event.run_id = "mutated"  # type: ignore[misc]
