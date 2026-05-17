"""Unit tests: control-stage event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_stage import (
    StageCompletedEvent,
    StageCompletedPayload,
    StageFailedEvent,
    StageFailedPayload,
    StageInterruptedEvent,
    StageInterruptedPayload,
    StageSkippedEvent,
    StageSkippedPayload,
    StageStartedEvent,
    StageStartedPayload,
)


def _started() -> StageStartedEvent:
    return StageStartedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=0,
        stage_id="dataset_validator",
        payload=StageStartedPayload(
            stage_name="dataset_validator",
            stage_index=1,
            total_stages=7,
            inputs_summary={"dataset": "d-1"},
        ),
    )


def _completed() -> StageCompletedEvent:
    return StageCompletedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=1,
        payload=StageCompletedPayload(
            stage_name="dataset_validator",
            duration_s=5.0,
            outputs_summary={"samples": 100},
        ),
    )


def _failed() -> StageFailedEvent:
    return StageFailedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=2,
        payload=StageFailedPayload(
            stage_name="dataset_validator",
            error_type="ValidationError",
            message="bad data",
            traceback_excerpt="...",
        ),
    )


def _skipped() -> StageSkippedEvent:
    return StageSkippedEvent(
        source="control://orchestrator/eval",
        run_id="r",
        offset=3,
        payload=StageSkippedPayload(stage_name="eval", reason="config disabled"),
    )


def _interrupted() -> StageInterruptedEvent:
    return StageInterruptedEvent(
        source="control://orchestrator/train",
        run_id="r",
        offset=4,
        payload=StageInterruptedPayload(
            stage_name="train", signal=15, cleanup_completed=True,
        ),
    )


_ALL = [_started, _completed, _failed, _skipped, _interrupted]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_failed_payload_retry_count_default_zero(self) -> None:
        payload = StageFailedPayload(
            stage_name="x", error_type="y", message="m", traceback_excerpt="t",
        )
        assert payload.retry_count == 0

    def test_started_inputs_summary_required(self) -> None:
        with pytest.raises(ValidationError):
            StageStartedPayload(  # type: ignore[call-arg]
                stage_name="x", stage_index=0, total_stages=1,
            )


class TestInvariants:
    def test_severities(self) -> None:
        assert _started().severity == "info"
        assert _completed().severity == "info"
        assert _failed().severity == "error"
        assert _skipped().severity == "info"
        assert _interrupted().severity == "warning"
