"""Phase G — verify the API response surface carries ``conditions[]``.

The Web UI consumes the API directly; missing this field would force
clients to re-read ``pipeline_state.json``, defeating the purpose of
the surface. This test exercises
:func:`ryotenkai_control.api.services.run_service._stage_run_to_schema`
to pin the contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ryotenkai_control.api.services.run_service import _stage_run_to_schema
from ryotenkai_control.pipeline.state.models import StageRunState
from ryotenkai_shared.contracts.pipeline_conditions import Condition, ConditionStatus


def _make_condition(*, status: ConditionStatus = ConditionStatus.TRUE) -> Condition:
    return Condition(
        type="Progressing",
        status=status,
        reason="StageStarted",
        message="started",
        last_transition_time=datetime(2026, 5, 16, tzinfo=UTC),
    )


def test_empty_conditions_serialised_as_empty_list() -> None:
    stage = StageRunState(stage_name="alpha", status="pending")
    schema = _stage_run_to_schema(stage)
    assert schema.conditions == []


def test_conditions_passed_through_to_schema() -> None:
    stage = StageRunState(
        stage_name="alpha",
        status="running",
        conditions=[_make_condition()],
    )
    schema = _stage_run_to_schema(stage)
    assert len(schema.conditions) == 1
    assert schema.conditions[0].type == "Progressing"
    assert schema.conditions[0].status is ConditionStatus.TRUE


def test_multiple_conditions_round_trip_through_json() -> None:
    stage = StageRunState(
        stage_name="alpha",
        status="running",
        conditions=[
            _make_condition(status=ConditionStatus.TRUE),
            Condition(
                type="OOMRisk",
                status=ConditionStatus.UNKNOWN,
                reason="GPUMemoryHigh",
                last_transition_time=datetime(2026, 5, 16, tzinfo=UTC),
            ),
        ],
    )
    schema = _stage_run_to_schema(stage)
    payload = schema.model_dump(mode="json")
    types = {c["type"]: c for c in payload["conditions"]}
    assert types["Progressing"]["status"] == "True"
    assert types["OOMRisk"]["status"] == "Unknown"
