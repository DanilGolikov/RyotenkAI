"""Tests for ValidationArtifactManager (extracted from PipelineOrchestrator).

Covers:
- Dataset lifecycle callbacks (scheduled → loaded → completed/failed)
- Plugin lifecycle callbacks (start → complete/failed)
- Artifact flushing (all passed, some failed, empty, already flushed)
- State output building (with stage context, with error, no datasets)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.validation.artifact_manager import ValidationArtifactManager

pytestmark = pytest.mark.unit


def _make_manager(
    *,
    collector: MagicMock | None = None,
    context: dict | None = None,
) -> ValidationArtifactManager:
    """Create a ValidationArtifactManager with a mock collector."""
    if collector is None:
        collector = MagicMock()
        collector.is_flushed = False
    collectors = {"Dataset Validator": collector}
    return ValidationArtifactManager(
        collectors=collectors,
        context=context or {},
    )


class TestDatasetLifecycle:
    """Callbacks: on_dataset_scheduled, on_dataset_loaded, on_validation_completed/failed."""

    def test_schedule_creates_accumulator_entry(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        assert "/tmp/ds1" in mgr._validation_accumulator
        entry = mgr._validation_accumulator["/tmp/ds1"]
        assert entry["name"] == "ds1"
        assert entry["status"] == "scheduled"
        assert entry["plugins"] == []

    def test_loaded_updates_sample_count(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_dataset_loaded("ds1", "/tmp/ds1", 100, 2)
        entry = mgr._validation_accumulator["/tmp/ds1"]
        assert entry["sample_count"] == 100
        assert entry["critical_failures"] == 2

    def test_loaded_ignores_unknown_path(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_loaded("ds1", "/nonexistent", 5, 0)  # must not raise

    def test_completed_marks_passed(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_validation_completed("ds1", "/tmp/ds1", {}, [])
        assert mgr._validation_accumulator["/tmp/ds1"]["status"] == "passed"

    def test_failed_marks_failed(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_validation_failed("ds1", "/tmp/ds1", ["error"])
        assert mgr._validation_accumulator["/tmp/ds1"]["status"] == "failed"

    def test_multiple_datasets(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("a", "/a", "full")
        mgr.on_dataset_scheduled("b", "/b", "full")
        assert len(mgr._validation_accumulator) == 2


class TestPluginLifecycle:
    """Callbacks: on_plugin_start, on_plugin_complete, on_plugin_failed."""

    def test_plugin_complete_appends_to_entry(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_plugin_start("ds1", "/tmp/ds1", "p1", "checker", "Checks stuff")
        mgr.on_plugin_complete(
            "ds1", "/tmp/ds1", "p1", "checker",
            params={"k": 1}, thresholds={"t": 0.5}, metrics={"m": 0.9},
            duration_ms=42.0,
        )
        plugins = mgr._validation_accumulator["/tmp/ds1"]["plugins"]
        assert len(plugins) == 1
        assert plugins[0]["passed"] is True
        assert plugins[0]["description"] == "Checks stuff"
        assert plugins[0]["duration_ms"] == 42.0

    def test_plugin_failed_appends_with_errors(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_plugin_start("ds1", "/tmp/ds1", "p2", "validator", "Validates")
        mgr.on_plugin_failed(
            "ds1", "/tmp/ds1", "p2", "validator",
            params={}, thresholds={}, metrics={},
            duration_ms=10.0,
            errors=["bad data"], recommendations=["fix it"],
        )
        plugins = mgr._validation_accumulator["/tmp/ds1"]["plugins"]
        assert len(plugins) == 1
        assert plugins[0]["passed"] is False
        assert plugins[0]["errors"] == ["bad data"]

    def test_description_consumed_on_complete(self) -> None:
        """Plugin description is popped from cache after complete/failed."""
        mgr = _make_manager()
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_plugin_start("ds1", "/tmp/ds1", "p1", "checker", "desc")
        mgr.on_plugin_complete(
            "ds1", "/tmp/ds1", "p1", "checker",
            params={}, thresholds={}, metrics={}, duration_ms=1.0,
        )
        assert ("/tmp/ds1", "p1") not in mgr._validation_plugin_descriptions


class TestFlushValidationArtifact:
    """flush_validation_artifact: writes artifact via collector."""

    def test_all_passed_flushes_ok(self) -> None:
        collector = MagicMock()
        collector.is_flushed = False
        mgr = _make_manager(collector=collector)
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_validation_completed("ds1", "/tmp/ds1", {}, [])

        mgr.flush_validation_artifact(started_at="t0", duration_seconds=1.0)
        collector.flush_ok.assert_called_once()
        collector.flush_error.assert_not_called()

    def test_any_failed_flushes_error(self) -> None:
        collector = MagicMock()
        collector.is_flushed = False
        mgr = _make_manager(collector=collector)
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        mgr.on_validation_failed("ds1", "/tmp/ds1", ["err"])

        mgr.flush_validation_artifact(started_at="t0", duration_seconds=2.0)
        collector.flush_error.assert_called_once()
        assert "ds1" in collector.flush_error.call_args.kwargs["error"]

    def test_already_flushed_is_noop(self) -> None:
        collector = MagicMock()
        collector.is_flushed = True
        mgr = _make_manager(collector=collector)
        mgr.flush_validation_artifact(started_at="t0", duration_seconds=0.0)
        collector.put.assert_not_called()

    def test_no_collector_is_noop(self) -> None:
        mgr = ValidationArtifactManager(collectors={}, context={})
        mgr.flush_validation_artifact(started_at="t0", duration_seconds=0.0)  # must not raise

    def test_scheduled_treated_as_passed(self) -> None:
        collector = MagicMock()
        collector.is_flushed = False
        mgr = _make_manager(collector=collector)
        mgr.on_dataset_scheduled("ds1", "/tmp/ds1", "full")
        # status remains "scheduled" — should be treated as passed
        mgr.flush_validation_artifact(started_at="t0", duration_seconds=0.0)
        collector.flush_ok.assert_called_once()


class TestBuildStateOutputs:
    """build_dataset_validation_state_outputs: creates pipeline state dict."""

    def test_empty_accumulator_returns_defaults(self) -> None:
        mgr = _make_manager()
        outputs = mgr.build_dataset_validation_state_outputs()
        assert "validation_artifact_ref" in outputs
        assert outputs["validation_status"] == "passed"

    def test_with_datasets_returns_counts(self) -> None:
        mgr = _make_manager()
        mgr.on_dataset_scheduled("a", "/a", "full")
        mgr.on_validation_completed("a", "/a", {}, [])
        mgr.on_dataset_scheduled("b", "/b", "full")
        mgr.on_validation_failed("b", "/b", ["err"])

        outputs = mgr.build_dataset_validation_state_outputs()
        assert outputs["datasets_validated"] == 2
        assert outputs["datasets_passed"] == 1
        assert outputs["datasets_failed"] == 1
        assert outputs["failed_datasets"] == ["b"]
        assert outputs["validation_status"] == "failed"

    def test_with_stage_ctx(self) -> None:
        mgr = _make_manager()
        outputs = mgr.build_dataset_validation_state_outputs(
            stage_ctx={"validation_status": "warning", "warnings": ["w1"], "message": "ok"},
        )
        assert outputs["validation_status"] == "warning"
        assert outputs["validation_warning_count"] == 1
        assert outputs["validation_message"] == "ok"

    def test_with_error(self) -> None:
        mgr = _make_manager()
        outputs = mgr.build_dataset_validation_state_outputs(error="boom")
        assert outputs["validation_status"] == "failed"
        assert outputs["validation_message"] == "boom"
