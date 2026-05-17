"""Unit tests: control-training (monitor) event types."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_training import (
    TrainingMonitorStartedEvent,
    TrainingMonitorStartedPayload,
    TrainingMonitorTimeoutEvent,
    TrainingMonitorTimeoutPayload,
)


def _started() -> TrainingMonitorStartedEvent:
    return TrainingMonitorStartedEvent(
        source="control://orchestrator/training_monitor",
        run_id="r",
        offset=0,
        payload=TrainingMonitorStartedPayload(
            pod_endpoint="http://pod", poll_interval_s=5.0,
        ),
    )


def _timeout() -> TrainingMonitorTimeoutEvent:
    return TrainingMonitorTimeoutEvent(
        source="control://orchestrator/training_monitor",
        run_id="r",
        offset=1,
        payload=TrainingMonitorTimeoutPayload(
            last_event_at=datetime(2026, 5, 16, tzinfo=UTC),
            timeout_s=300.0,
        ),
    )


_ALL = [_started, _timeout]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original


class TestNegative:
    def test_timeout_payload_requires_last_event_at(self) -> None:
        with pytest.raises(ValidationError):
            TrainingMonitorTimeoutPayload(timeout_s=1.0)  # type: ignore[call-arg]


class TestInvariants:
    def test_timeout_severity_is_error(self) -> None:
        assert _timeout().severity == "error"

    def test_started_severity_is_info(self) -> None:
        assert _started().severity == "info"
