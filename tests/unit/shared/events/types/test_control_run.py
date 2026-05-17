"""Unit tests: control-run lifecycle event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_run import (
    RunCancelledEvent,
    RunCancelledPayload,
    RunCompletedEvent,
    RunCompletedPayload,
    RunFailedEvent,
    RunFailedPayload,
    RunStartedEvent,
    RunStartedPayload,
)


def _started() -> RunStartedEvent:
    return RunStartedEvent(
        source="control://orchestrator",
        run_id="r",
        offset=0,
        payload=RunStartedPayload(
            run_name="alpha",
            algorithm="sft",
            model_id="m-1",
            dataset_id="d-1",
            config_hash="h",
        ),
    )


def _completed() -> RunCompletedEvent:
    return RunCompletedEvent(
        source="control://orchestrator",
        run_id="r",
        offset=1,
        payload=RunCompletedPayload(duration_s=600.0, final_status="success"),
    )


def _failed() -> RunFailedEvent:
    return RunFailedEvent(
        source="control://orchestrator",
        run_id="r",
        offset=2,
        payload=RunFailedPayload(
            failing_stage="train", error_type="X", message="m", traceback_excerpt="t",
        ),
    )


def _cancelled() -> RunCancelledEvent:
    return RunCancelledEvent(
        source="control://orchestrator",
        run_id="r",
        offset=3,
        payload=RunCancelledPayload(reason="user_abort"),
    )


_ALL = [_started, _completed, _failed, _cancelled]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_started_payload_rejects_unknown_algorithm(self) -> None:
        with pytest.raises(ValidationError):
            RunStartedPayload(  # type: ignore[arg-type]
                run_name="a",
                algorithm="xxx",  # not in Literal
                model_id="m",
                dataset_id="d",
                config_hash="h",
            )

    def test_completed_payload_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunCompletedPayload(  # type: ignore[call-arg]
                duration_s=1.0, final_status="ok", surprise="x",
            )


class TestInvariants:
    def test_run_severities(self) -> None:
        assert _started().severity == "info"
        assert _completed().severity == "info"
        assert _failed().severity == "error"
        assert _cancelled().severity == "warning"
