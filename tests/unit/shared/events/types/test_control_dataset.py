"""Unit tests: control-dataset event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_dataset import (
    DatasetValidationCompletedEvent,
    DatasetValidationCompletedPayload,
    DatasetValidationFailedEvent,
    DatasetValidationFailedPayload,
    DatasetValidationPluginCompletedEvent,
    DatasetValidationPluginCompletedPayload,
    DatasetValidationPluginFailedEvent,
    DatasetValidationPluginFailedPayload,
    DatasetValidationPluginStartedEvent,
    DatasetValidationPluginStartedPayload,
    DatasetValidationStartedEvent,
    DatasetValidationStartedPayload,
)


def _started() -> DatasetValidationStartedEvent:
    return DatasetValidationStartedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=0,
        payload=DatasetValidationStartedPayload(
            dataset_path="/tmp/d", validator_chain=["schema", "size"],
        ),
    )


def _completed() -> DatasetValidationCompletedEvent:
    return DatasetValidationCompletedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=1,
        payload=DatasetValidationCompletedPayload(
            num_samples=100, num_rejected=2, schema_version="1.0",
            checks_passed=["schema", "size"],
        ),
    )


def _failed() -> DatasetValidationFailedEvent:
    return DatasetValidationFailedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=2,
        payload=DatasetValidationFailedPayload(
            failed_check="schema", details="missing required field",
        ),
    )


def _plugin_started() -> DatasetValidationPluginStartedEvent:
    return DatasetValidationPluginStartedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=3,
        payload=DatasetValidationPluginStartedPayload(
            plugin_name="min_samples", plugin_version="1.0.0",
            dataset_path="/tmp/d/train.jsonl",
        ),
    )


def _plugin_completed() -> DatasetValidationPluginCompletedEvent:
    return DatasetValidationPluginCompletedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=4,
        payload=DatasetValidationPluginCompletedPayload(
            plugin_name="min_samples",
            num_checked=100, num_passed=98, num_failed=2,
            duration_s=0.5,
        ),
    )


def _plugin_failed() -> DatasetValidationPluginFailedEvent:
    return DatasetValidationPluginFailedEvent(
        source="control://orchestrator/dataset_validator",
        run_id="r",
        offset=5,
        payload=DatasetValidationPluginFailedPayload(
            plugin_name="min_samples", error_type="RuntimeError", message="boom",
            traceback_excerpt='File "x.py", line 1, in validate\n    raise RuntimeError("boom")',
        ),
    )


_ALL = [_started, _completed, _failed]
_ALL_PLUGIN = [_plugin_started, _plugin_completed, _plugin_failed]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL + _ALL_PLUGIN, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original


class TestNegative:
    def test_completed_payload_requires_checks_passed(self) -> None:
        with pytest.raises(ValidationError):
            DatasetValidationCompletedPayload(  # type: ignore[call-arg]
                num_samples=1, num_rejected=0, schema_version="1",
            )

    def test_plugin_started_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetValidationPluginStartedPayload(  # type: ignore[call-arg]
                plugin_name="x", plugin_version="1",
                dataset_path="/d", rogue=True,
            )

    def test_plugin_completed_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetValidationPluginCompletedPayload(  # type: ignore[call-arg]
                plugin_name="x", num_checked=1, num_passed=1, num_failed=0,
                duration_s=0.0, rogue_field=42,
            )

    def test_plugin_failed_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetValidationPluginFailedPayload(  # type: ignore[call-arg]
                plugin_name="x", error_type="Y", message="z",
                traceback_excerpt="tb", rogue="x",
            )


class TestInvariants:
    def test_failed_severity_is_error(self) -> None:
        assert _failed().severity == "error"

    def test_started_completed_info(self) -> None:
        assert _started().severity == "info"
        assert _completed().severity == "info"

    def test_plugin_kinds_pinned(self) -> None:
        assert _plugin_started().kind == "ryotenkai.control.dataset.validation_plugin_started"
        assert _plugin_completed().kind == "ryotenkai.control.dataset.validation_plugin_completed"
        assert _plugin_failed().kind == "ryotenkai.control.dataset.validation_plugin_failed"

    def test_plugin_severities(self) -> None:
        assert _plugin_started().severity == "info"
        assert _plugin_completed().severity == "info"
        assert _plugin_failed().severity == "error"
