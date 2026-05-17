"""Unit tests: control-evaluation event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_evaluation import (
    EvaluationCompletedEvent,
    EvaluationCompletedPayload,
    EvaluationPluginCompletedEvent,
    EvaluationPluginCompletedPayload,
    EvaluationPluginFailedEvent,
    EvaluationPluginFailedPayload,
    EvaluationPluginStartedEvent,
    EvaluationPluginStartedPayload,
    EvaluationStartedEvent,
    EvaluationStartedPayload,
)


def _started() -> EvaluationStartedEvent:
    return EvaluationStartedEvent(
        source="control://orchestrator/evaluator",
        run_id="r",
        offset=0,
        payload=EvaluationStartedPayload(
            plugin_names=["judge"], model_path="/m",
        ),
    )


def _plugin_started() -> EvaluationPluginStartedEvent:
    return EvaluationPluginStartedEvent(
        source="control://orchestrator/evaluator",
        run_id="r",
        offset=1,
        payload=EvaluationPluginStartedPayload(
            plugin_name="judge", plugin_version="1.0",
        ),
    )


def _plugin_completed() -> EvaluationPluginCompletedEvent:
    return EvaluationPluginCompletedEvent(
        source="control://orchestrator/evaluator",
        run_id="r",
        offset=2,
        payload=EvaluationPluginCompletedPayload(
            plugin_name="judge", metrics={"score": 0.9}, duration_s=10.0,
        ),
    )


def _plugin_failed() -> EvaluationPluginFailedEvent:
    return EvaluationPluginFailedEvent(
        source="control://orchestrator/evaluator",
        run_id="r",
        offset=3,
        payload=EvaluationPluginFailedPayload(
            plugin_name="judge", error_type="X", message="m",
        ),
    )


def _completed() -> EvaluationCompletedEvent:
    return EvaluationCompletedEvent(
        source="control://orchestrator/evaluator",
        run_id="r",
        offset=4,
        payload=EvaluationCompletedPayload(
            aggregated_metrics={"score": 0.9}, total_duration_s=15.0,
        ),
    )


_ALL = [_started, _plugin_started, _plugin_completed, _plugin_failed, _completed]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original


class TestNegative:
    def test_plugin_completed_metrics_must_be_dict_of_floats(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationPluginCompletedPayload(
                plugin_name="x",
                metrics={"x": "non-float"},  # type: ignore[dict-item]
                duration_s=1.0,
            )

    def test_started_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationStartedPayload(  # type: ignore[call-arg]
                plugin_names=["x"], model_path="/m", extra=True,
            )


class TestInvariants:
    def test_plugin_failed_severity_is_error(self) -> None:
        assert _plugin_failed().severity == "error"

    def test_other_severities_info(self) -> None:
        for ev in (_started(), _plugin_started(), _plugin_completed(), _completed()):
            assert ev.severity == "info"
