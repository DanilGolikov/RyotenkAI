"""
Isolated unit tests for MLflowEventLog.
No dependency on MLflowManager or any real MLflow SDK calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.training.mlflow.event_log import MLflowEventLog


class TestMLflowEventLogBasics:
    def test_initial_state(self):
        log = MLflowEventLog()
        assert log.event_count == 0
        assert log.has_errors is False
        assert log.get_events() == []

    def test_log_event_returns_dict(self):
        log = MLflowEventLog()
        event = log.log_event("start", "Training started", category="training", source="test")
        assert isinstance(event, dict)
        assert event["message"] == "Training started"
        assert event["category"] == "training"
        assert event["source"] == "test"
        assert event["event_type"] == "start"

    def test_event_counter_increments(self):
        log = MLflowEventLog()
        log.log_event("info", "msg1")
        log.log_event("info", "msg2")
        assert log.event_count == 2

    def test_log_event_metadata_in_attributes(self):
        log = MLflowEventLog()
        event = log.log_event("info", "msg", category="pipeline", gpu_name="A100", vram_gb=40.0)
        assert "attributes" in event
        assert event["attributes"]["gpu_name"] == "A100"
        assert event["attributes"]["vram_gb"] == 40.0

    def test_log_event_no_metadata_no_attributes_key(self):
        log = MLflowEventLog()
        event = log.log_event("info", "no extra")
        assert "attributes" not in event


class TestMLflowEventLogSeverity:
    def test_start_severity(self):
        log = MLflowEventLog()
        event = log.log_event("start", "started")
        assert event["severity"] == "INFO"
        assert event["severity_number"] == 9

    def test_warning_severity(self):
        log = MLflowEventLog()
        event = log.log_event("warning", "warn")
        assert event["severity"] == "WARN"
        assert event["severity_number"] == 13

    def test_error_severity(self):
        log = MLflowEventLog()
        event = log.log_event("error", "err")
        assert event["severity"] == "ERROR"
        assert event["severity_number"] == 17

    def test_unknown_type_defaults_to_info(self):
        log = MLflowEventLog()
        event = log.log_event("custom_type", "msg")
        assert event["severity"] == "INFO"
        assert event["severity_number"] == 9


class TestMLflowEventLogErrorFlag:
    def test_has_errors_false_without_error(self):
        log = MLflowEventLog()
        log.log_event("info", "ok")
        log.log_event("warning", "warn")
        assert log.has_errors is False

    def test_has_errors_set_on_error(self):
        log = MLflowEventLog()
        log.log_event("error", "fail")
        assert log.has_errors is True

    def test_has_errors_not_reset_after_info(self):
        log = MLflowEventLog()
        log.log_event("error", "fail")
        log.log_event("info", "recovery")
        assert log.has_errors is True


class TestMLflowEventLogConvenienceMethods:
    def test_log_event_start(self):
        log = MLflowEventLog()
        ev = log.log_event_start("started")
        assert ev["event_type"] == "start"

    def test_log_event_complete(self):
        log = MLflowEventLog()
        ev = log.log_event_complete("done")
        assert ev["event_type"] == "complete"

    def test_log_event_error(self):
        log = MLflowEventLog()
        ev = log.log_event_error("error!")
        assert ev["event_type"] == "error"

    def test_log_event_warning(self):
        log = MLflowEventLog()
        ev = log.log_event_warning("warn!")
        assert ev["event_type"] == "warning"

    def test_log_event_info(self):
        log = MLflowEventLog()
        ev = log.log_event_info("info msg")
        assert ev["event_type"] == "info"

    def test_log_event_checkpoint(self):
        log = MLflowEventLog()
        ev = log.log_event_checkpoint("checkpoint")
        assert ev["event_type"] == "checkpoint"


class TestMLflowEventLogGetEvents:
    def test_get_all_events(self):
        log = MLflowEventLog()
        log.log_event("info", "m1", category="training")
        log.log_event("info", "m2", category="pipeline")
        all_events = log.get_events()
        assert len(all_events) == 2

    def test_get_events_filter_by_category(self):
        log = MLflowEventLog()
        log.log_event("info", "m1", category="training")
        log.log_event("info", "m2", category="pipeline")
        log.log_event("info", "m3", category="training")
        training_events = log.get_events(category="training")
        assert len(training_events) == 2
        assert all(e["category"] == "training" for e in training_events)

    def test_get_events_returns_copy(self):
        log = MLflowEventLog()
        log.log_event("info", "m1")
        events = log.get_events()
        events.append({"fake": "event"})
        assert len(log.get_events()) == 1

    def test_get_events_empty_when_no_events(self):
        log = MLflowEventLog()
        assert log.get_events() == []
        assert log.get_events(category="training") == []


class TestMLflowEventLogArtifact:
    def test_log_events_artifact_calls_log_dict_fn(self):
        log = MLflowEventLog()
        log.log_event("info", "msg")
        mock_log_dict = MagicMock(return_value=True)

        result = log.log_events_artifact("events.json", log_dict_fn=mock_log_dict)

        assert result is True
        mock_log_dict.assert_called_once()
        call_args = mock_log_dict.call_args
        data = call_args[0][0]
        assert "events" in data
        assert data["total"] == 1

    def test_log_events_artifact_returns_false_when_empty(self):
        log = MLflowEventLog()
        mock_log_dict = MagicMock()
        result = log.log_events_artifact("events.json", log_dict_fn=mock_log_dict)
        assert result is False
        mock_log_dict.assert_not_called()

    def test_log_events_artifact_passes_run_id(self):
        log = MLflowEventLog()
        log.log_event("info", "msg")
        mock_log_dict = MagicMock(return_value=True)
        log.log_events_artifact("events.json", log_dict_fn=mock_log_dict, run_id="run123")
        call_args = mock_log_dict.call_args
        assert call_args[0][2] == "run123"

    def test_log_events_artifact_handles_exception(self):
        log = MLflowEventLog()
        log.log_event("info", "msg")
        mock_log_dict = MagicMock(side_effect=RuntimeError("boom"))
        result = log.log_events_artifact("events.json", log_dict_fn=mock_log_dict)
        assert result is False


class TestMLflowEventLogClear:
    def test_clear_resets_state(self):
        log = MLflowEventLog()
        log.log_event("error", "fail")
        log.log_event("info", "msg")
        log.clear()
        assert log.event_count == 0
        assert log.has_errors is False
        assert log.get_events() == []

    def test_clear_allows_fresh_events_after(self):
        log = MLflowEventLog()
        log.log_event("error", "fail")
        log.clear()
        log.log_event("info", "fresh")
        assert log.event_count == 1
        assert log.has_errors is False


class TestMLflowEventLogSummarySection:
    def test_generate_summary_section_with_events(self):
        log = MLflowEventLog()
        log.log_event("start", "Training started", category="training")
        log.log_event("complete", "Done", category="training")
        lines = log.generate_summary_section()
        text = "\n".join(lines)
        assert "Events Timeline" in text
        assert "Training started" in text

    def test_generate_summary_section_empty(self):
        log = MLflowEventLog()
        lines = log.generate_summary_section()
        text = "\n".join(lines)
        assert "No events recorded" in text

    def test_generate_summary_section_shows_statistics(self):
        log = MLflowEventLog()
        log.log_event("error", "fail")
        log.log_event("warning", "warn")
        lines = log.generate_summary_section()
        text = "\n".join(lines)
        assert "Total events:" in text
