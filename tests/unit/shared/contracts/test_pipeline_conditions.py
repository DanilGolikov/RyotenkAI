"""Tests for :mod:`ryotenkai_shared.contracts.pipeline_conditions` (Phase G).

Seven test classes covering the :class:`Condition` Pydantic model,
the :func:`update_condition` helper, and behavioural invariants from
the plan (idempotency, last_transition_time stickiness, CamelCase
validation).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ryotenkai_shared.contracts.pipeline_conditions import (
    CAMEL_CASE_RE,
    STANDARD_CONDITION_TYPES,
    Condition,
    ConditionStatus,
    update_condition,
)


# ---------------------------------------------------------------------------
# Class 1 — Model construction + Pydantic field validation
# ---------------------------------------------------------------------------


class TestConditionModel:
    """Pydantic-level invariants on :class:`Condition`."""

    def test_constructs_with_required_fields(self) -> None:
        now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
        c = Condition(
            type="Progressing",
            status=ConditionStatus.TRUE,
            reason="StageStarted",
            last_transition_time=now,
        )
        assert c.type == "Progressing"
        assert c.status is ConditionStatus.TRUE
        assert c.reason == "StageStarted"
        assert c.message is None
        assert c.last_transition_time == now

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Condition.model_validate(
                {
                    "type": "Progressing",
                    "status": "True",
                    "reason": "StageStarted",
                    "last_transition_time": datetime.now(UTC).isoformat(),
                    "rogue_key": "x",
                },
            )

    def test_status_enum_serialises_as_camelcase_value(self) -> None:
        c = Condition(
            type="Available",
            status=ConditionStatus.FALSE,
            reason="StageFailed",
            last_transition_time=datetime.now(UTC),
        )
        dumped = c.model_dump(mode="json")
        # k8s convention: status is "True" / "False" / "Unknown" — not lowercase.
        assert dumped["status"] == "False"

    def test_message_optional(self) -> None:
        c = Condition(
            type="Available",
            status=ConditionStatus.TRUE,
            reason="AsExpected",
            last_transition_time=datetime.now(UTC),
        )
        assert c.message is None
        d = c.model_dump()
        assert d["message"] is None


# ---------------------------------------------------------------------------
# Class 2 — CamelCase validation on type / reason
# ---------------------------------------------------------------------------


class TestCamelCaseValidation:
    """Reason and type must be CamelCase (k8s convention)."""

    def _make(self, *, type: str = "Progressing", reason: str = "AsExpected") -> Condition:
        return Condition(
            type=type,
            status=ConditionStatus.TRUE,
            reason=reason,
            last_transition_time=datetime.now(UTC),
        )

    @pytest.mark.parametrize("reason", ["AsExpected", "StageStarted", "GPUMemoryHigh", "X", "Abc123"])
    def test_camelcase_reason_accepted(self, reason: str) -> None:
        c = self._make(reason=reason)
        assert c.reason == reason

    @pytest.mark.parametrize("reason", ["as_expected", "lowercase", "Has-Hyphen", "has space", "1Numeric", ""])
    def test_non_camelcase_reason_rejected(self, reason: str) -> None:
        with pytest.raises(ValidationError):
            self._make(reason=reason)

    @pytest.mark.parametrize("type_", ["snake_case", "lower", "Has-Dash", " Leading", ""])
    def test_non_camelcase_type_rejected(self, type_: str) -> None:
        with pytest.raises(ValidationError):
            self._make(type=type_)

    def test_camel_case_re_constant(self) -> None:
        # Sanity: the exported regex is what the validator pins on.
        assert CAMEL_CASE_RE.match("ValidCamel")
        assert not CAMEL_CASE_RE.match("invalid_snake")


# ---------------------------------------------------------------------------
# Class 3 — update_condition: idempotency on same status
# ---------------------------------------------------------------------------


class TestUpdateConditionIdempotent:
    """Invariant: same-status updates keep ``last_transition_time``."""

    def test_first_call_appends_with_now(self) -> None:
        conditions: list[Condition] = []
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        update_condition(
            conditions,
            type="Progressing",
            status=ConditionStatus.TRUE,
            reason="StageStarted",
            now=ts,
        )
        assert len(conditions) == 1
        assert conditions[0].last_transition_time == ts

    def test_second_call_same_status_keeps_timestamp(self) -> None:
        conditions: list[Condition] = []
        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(minutes=5)
        update_condition(
            conditions,
            type="Progressing",
            status=ConditionStatus.TRUE,
            reason="StageStarted",
            now=ts1,
        )
        update_condition(
            conditions,
            type="Progressing",
            status=ConditionStatus.TRUE,
            reason="StageStartedAgain",
            now=ts2,
        )
        assert len(conditions) == 1
        # Timestamp unchanged — no transition.
        assert conditions[0].last_transition_time == ts1
        # Reason WAS refreshed (k8s convention).
        assert conditions[0].reason == "StageStartedAgain"

    def test_message_refreshed_on_same_status(self) -> None:
        conditions: list[Condition] = []
        update_condition(
            conditions,
            type="Available",
            status=ConditionStatus.TRUE,
            reason="AsExpected",
            message="initial",
        )
        update_condition(
            conditions,
            type="Available",
            status=ConditionStatus.TRUE,
            reason="AsExpected",
            message="refreshed",
        )
        assert conditions[0].message == "refreshed"


# ---------------------------------------------------------------------------
# Class 4 — update_condition: status flip bumps last_transition_time
# ---------------------------------------------------------------------------


class TestUpdateConditionTransition:
    """Invariant: status flip updates ``last_transition_time``."""

    def test_true_to_false_bumps_timestamp(self) -> None:
        conditions: list[Condition] = []
        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(hours=1)
        update_condition(
            conditions,
            type="Progressing",
            status=ConditionStatus.TRUE,
            reason="StageStarted",
            now=ts1,
        )
        update_condition(
            conditions,
            type="Progressing",
            status=ConditionStatus.FALSE,
            reason="StageCompleted",
            now=ts2,
        )
        assert len(conditions) == 1
        assert conditions[0].status is ConditionStatus.FALSE
        assert conditions[0].last_transition_time == ts2
        assert conditions[0].reason == "StageCompleted"

    @pytest.mark.parametrize(
        "from_status,to_status",
        [
            (ConditionStatus.TRUE, ConditionStatus.FALSE),
            (ConditionStatus.FALSE, ConditionStatus.TRUE),
            (ConditionStatus.TRUE, ConditionStatus.UNKNOWN),
            (ConditionStatus.UNKNOWN, ConditionStatus.TRUE),
            (ConditionStatus.FALSE, ConditionStatus.UNKNOWN),
        ],
    )
    def test_all_transitions_update_timestamp(
        self,
        from_status: ConditionStatus,
        to_status: ConditionStatus,
    ) -> None:
        conditions: list[Condition] = []
        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(minutes=10)
        update_condition(conditions, type="X", status=from_status, reason="Start", now=ts1)
        update_condition(conditions, type="X", status=to_status, reason="Flip", now=ts2)
        assert conditions[0].status is to_status
        assert conditions[0].last_transition_time == ts2

    def test_unknown_to_same_unknown_keeps_timestamp(self) -> None:
        conditions: list[Condition] = []
        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(seconds=42)
        update_condition(conditions, type="X", status=ConditionStatus.UNKNOWN, reason="Probing", now=ts1)
        update_condition(conditions, type="X", status=ConditionStatus.UNKNOWN, reason="StillProbing", now=ts2)
        assert conditions[0].last_transition_time == ts1


# ---------------------------------------------------------------------------
# Class 5 — multiple condition types coexist independently
# ---------------------------------------------------------------------------


class TestMultipleConditions:
    """Distinct ``type`` values are independent entries."""

    def test_two_types_two_entries(self) -> None:
        conditions: list[Condition] = []
        update_condition(conditions, type="Progressing", status=ConditionStatus.TRUE, reason="StageStarted")
        update_condition(conditions, type="OOMRisk", status=ConditionStatus.TRUE, reason="GPUMemoryHigh")
        assert len(conditions) == 2
        types = {c.type for c in conditions}
        assert types == {"Progressing", "OOMRisk"}

    def test_update_to_one_type_does_not_touch_others(self) -> None:
        conditions: list[Condition] = []
        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        update_condition(
            conditions,
            type="Progressing",
            status=ConditionStatus.TRUE,
            reason="StageStarted",
            now=ts1,
        )
        update_condition(
            conditions,
            type="OOMRisk",
            status=ConditionStatus.TRUE,
            reason="GPUMemoryHigh",
            now=ts1 + timedelta(minutes=5),
        )
        # Flip OOMRisk only — Progressing entry must be untouched.
        update_condition(
            conditions,
            type="OOMRisk",
            status=ConditionStatus.FALSE,
            reason="GPUMemoryOK",
            now=ts1 + timedelta(minutes=10),
        )
        progressing = next(c for c in conditions if c.type == "Progressing")
        assert progressing.status is ConditionStatus.TRUE
        assert progressing.last_transition_time == ts1

    def test_return_value_is_input_list(self) -> None:
        conditions: list[Condition] = []
        returned = update_condition(
            conditions,
            type="Available",
            status=ConditionStatus.TRUE,
            reason="AsExpected",
        )
        assert returned is conditions


# ---------------------------------------------------------------------------
# Class 6 — Regression: last_transition_time NEVER updates without a flip
# ---------------------------------------------------------------------------


class TestRegressionsLastTransitionTime:
    """Regression suite: the k8s-style timestamp stickiness invariant."""

    def test_n_same_status_updates_keep_first_timestamp(self) -> None:
        conditions: list[Condition] = []
        ts0 = datetime(2026, 1, 1, tzinfo=UTC)
        update_condition(
            conditions,
            type="Available",
            status=ConditionStatus.TRUE,
            reason="AsExpected",
            now=ts0,
        )
        for i in range(1, 50):
            update_condition(
                conditions,
                type="Available",
                status=ConditionStatus.TRUE,
                reason=f"Tick{i}",
                now=ts0 + timedelta(seconds=i),
            )
        assert conditions[0].last_transition_time == ts0
        assert conditions[0].reason == "Tick49"

    def test_flip_back_uses_latest_ts_not_original(self) -> None:
        conditions: list[Condition] = []
        ts1 = datetime(2026, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(minutes=1)
        ts3 = ts1 + timedelta(minutes=2)
        update_condition(conditions, type="X", status=ConditionStatus.TRUE, reason="A", now=ts1)
        update_condition(conditions, type="X", status=ConditionStatus.FALSE, reason="B", now=ts2)
        update_condition(conditions, type="X", status=ConditionStatus.TRUE, reason="C", now=ts3)
        # Last flip was ts3 — that's the one we expose.
        assert conditions[0].last_transition_time == ts3

    def test_now_default_is_close_to_real_clock(self) -> None:
        conditions: list[Condition] = []
        before = datetime.now(UTC)
        update_condition(conditions, type="X", status=ConditionStatus.TRUE, reason="Now")
        after = datetime.now(UTC)
        assert before <= conditions[0].last_transition_time <= after


# ---------------------------------------------------------------------------
# Class 7 — Standard types + JSON round-trip
# ---------------------------------------------------------------------------


class TestStandardTypesAndSerialisation:
    """Phase G seeds five standard condition types; payloads must
    round-trip through JSON cleanly."""

    @pytest.mark.parametrize("type_", STANDARD_CONDITION_TYPES)
    def test_all_standard_types_construct(self, type_: str) -> None:
        c = Condition(
            type=type_,
            status=ConditionStatus.TRUE,
            reason="AsExpected",
            last_transition_time=datetime(2026, 5, 16, tzinfo=UTC),
        )
        assert c.type == type_

    def test_json_round_trip_preserves_fields(self) -> None:
        original = Condition(
            type="OOMRisk",
            status=ConditionStatus.UNKNOWN,
            reason="GPUMemoryHigh",
            message="GPU memory usage > 90%",
            last_transition_time=datetime(2026, 5, 16, 14, 30, tzinfo=UTC),
        )
        raw = original.model_dump_json()
        reloaded = Condition.model_validate_json(raw)
        assert reloaded == original

    def test_seeded_standard_types_set(self) -> None:
        # The seeded tuple is what the CLI table columns pin on; the
        # ordering is documented as stable.
        assert STANDARD_CONDITION_TYPES == (
            "Available",
            "Progressing",
            "Degraded",
            "OOMRisk",
            "RateLimited",
        )
