from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.reports.core.builder import ReportBuilder
from src.reports.domain.entities import (
    ExperimentData,
    MetricHistory,
    PhaseData,
    RunStatus,
)


def _mh(key: str, values: list[float], start: datetime, step_ms: int = 1000) -> MetricHistory:
    base = int(start.timestamp() * 1000)
    return MetricHistory(
        key=key,
        values=values,
        steps=list(range(len(values))),
        timestamps=[base + i * step_ms for i in range(len(values))],
    )


def _phase(idx: int, start: datetime, end: datetime, *, loss_values: list[float]) -> PhaseData:
    return PhaseData(
        idx=idx,
        name="sft",
        strategy="SFT",
        status=RunStatus.FINISHED,
        duration_seconds=(end - start).total_seconds(),
        start_time=start,
        end_time=end,
        config={"strategy_type": "sft", "model_name": "base"},
        metrics={"epoch": 1.0, "global_step": 2},
        history={"loss": _mh("loss", loss_values, start)},
    )


class TestReportBuilderEndToEnd:
    def test_build_includes_sliced_phase_resources_and_validation_and_issues(self) -> None:
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(seconds=10)

        p = _phase(0, start, end, loss_values=[1.0, 1.2])  # loss increased -> BAD -> issue

        # Global resource history has points inside phase window -> slicing should populate phase.history keys
        gpu_util = MetricHistory(
            key="system/gpu_0_utilization_percentage",
            values=[50.0, 60.0],
            steps=[1, 2],
            timestamps=[
                int((start + timedelta(seconds=1)).timestamp() * 1000),
                int((start + timedelta(seconds=9)).timestamp() * 1000),
            ],
        )

        data = ExperimentData(
            run_id="run_12345678",
            run_name="run",
            experiment_name="exp",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=end,
            duration_seconds=10.0,
            phases=[p],
            source_config={"model": {"name": "base"}},
            root_params={},
            memory_events=[],
            validation_results={
                "datasets": [
                    {
                        "name": "d1",
                        "path": "/p",
                        "sample_count": 10,
                        "status": "passed",
                        "critical_failures": 0,
                        "plugins": [
                            {
                                "name": "schema",
                                "passed": True,
                                "duration_ms": 1.0,
                                "description": "desc",
                                "metrics": {"rows": 10},
                                "params": {"max_len": 512},
                                "errors": [],
                                "recommendations": [],
                            }
                        ],
                    }
                ]
            },
            gpu_info={"name": "GPU", "tier": "t", "vram": 8.0},
            resource_history={"system/gpu_0_utilization_percentage": gpu_util},
        )

        builder = ReportBuilder(data)
        # Ensure memory analysis contributes an insight (covers issues-from-memory branch deterministically)
        builder._memory_analyzer.analyze = lambda **kw: SimpleNamespace(insights=["insight"])  # type: ignore[method-assign]

        report = builder.build()
        assert report.summary.total_epochs == 1
        assert report.summary.total_steps == 2

        # Phase resource slicing should have produced a non-empty avg
        assert report.phases and report.phases[0].gpu_utilization is not None

        # Issues should include at least one ERROR (loss BAD) and WARN (timeline + memory insight)
        severities = {i.severity for i in report.issues}
        assert "ERROR" in severities
        assert "WARN" in severities

        # Validation report should be present and include dataset
        assert report.validation is not None
        assert report.validation.total_datasets == 1
        assert report.validation.datasets[0].dataset_name == "d1"


class TestResourceHistorySlicing:
    def test_slice_returns_nearest_point_when_no_points_in_window_but_close(self) -> None:
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(seconds=5)

        # point is 10s away from center (~2.5s) -> within 30s window
        ts = int((start + timedelta(seconds=12)).timestamp() * 1000)
        hist = MetricHistory(key="k", values=[1.0], steps=[1], timestamps=[ts])

        data = ExperimentData(
            run_id="r",
            run_name="r",
            experiment_name="e",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=end,
            duration_seconds=5.0,
            phases=[],
            resource_history={"k": hist},
        )
        b = ReportBuilder(data)
        out = b._slice_resource_history(start, end, "k")
        assert out is not None
        assert out.values == [1.0]

    def test_slice_returns_none_when_nearest_point_is_too_far(self) -> None:
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(seconds=5)

        # point is 60s away from center -> too far
        ts = int((start + timedelta(seconds=65)).timestamp() * 1000)
        hist = MetricHistory(key="k", values=[1.0], steps=[1], timestamps=[ts])

        data = ExperimentData(
            run_id="r",
            run_name="r",
            experiment_name="e",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=end,
            duration_seconds=5.0,
            phases=[],
            resource_history={"k": hist},
        )
        b = ReportBuilder(data)
        assert b._slice_resource_history(start, end, "k") is None


class TestResourcesFallbackAggregation:
    def test_build_resources_falls_back_to_phase_metrics_when_no_global_history(self) -> None:
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(seconds=10)
        p = _phase(0, start, end, loss_values=[1.0, 0.5])
        # No history for the key -> fallback to scalar metric value
        p.metrics["system/gpu_0_utilization_percentage"] = 42.0

        data = ExperimentData(
            run_id="r",
            run_name="r",
            experiment_name="e",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=end,
            duration_seconds=10.0,
            phases=[p],
            resource_history={},
            gpu_info={},
        )
        b = ReportBuilder(data)
        resources = b._build_resources()
        assert resources.gpu_utilization.avg == pytest.approx(42.0)


class TestValidationParsing:
    def test_parse_dataset_validation_handles_failed_load_when_only_scheduled(self) -> None:
        """New behavior: no validation_results → None returned."""
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        data = ExperimentData(
            run_id="r",
            run_name="r",
            experiment_name="e",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=start,
            duration_seconds=0.0,
            phases=[],
        )
        b = ReportBuilder(data)
        out = b._build_validation()
        assert out is None

    def test_parse_dataset_validation_failed_plugin_bad_recommendations_json_is_ignored(self) -> None:
        """Plugin with errors but no recommendations still parses cleanly."""
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        data = ExperimentData(
            run_id="r",
            run_name="r",
            experiment_name="e",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=start,
            duration_seconds=0.0,
            phases=[],
            validation_results={
                "datasets": [
                    {
                        "name": "d1",
                        "path": "/p",
                        "sample_count": 1,
                        "status": "failed",
                        "critical_failures": 1,
                        "plugins": [
                            {
                                "name": "schema",
                                "passed": False,
                                "duration_ms": 0.0,
                                "description": "",
                                "metrics": {"metric_y": 2},
                                "params": {"config_x": 1},
                                "errors": ["2 errors"],
                                "recommendations": [],
                            }
                        ],
                    }
                ]
            },
        )
        b = ReportBuilder(data)
        out = b._build_validation()
        assert out is not None
        assert out.datasets[0].status == "failed"
        assert out.datasets[0].failed_plugins == 1
        pr = out.datasets[0].plugin_results[0]
        assert pr.status == "failed"
        assert pr.errors

    def test_validation_failures_generate_warnings_in_issues(self) -> None:
        """Test that failed dataset validations generate WARN issues."""
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        data = ExperimentData(
            run_id="r",
            run_name="r",
            experiment_name="e",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=start,
            duration_seconds=0.0,
            phases=[],
            validation_results={
                "datasets": [
                    {
                        "name": "test_dataset",
                        "path": "/path/to/data",
                        "sample_count": 100,
                        "status": "failed",
                        "critical_failures": 1,
                        "plugins": [
                            {
                                "id": "deduplication_main",
                                "plugin_name": "deduplication",
                                "passed": False,
                                "duration_ms": 5.0,
                                "description": "Check for duplicates",
                                "metrics": {},
                                "params": {},
                                "errors": ["duplicates found"],
                                "recommendations": [],
                            }
                        ],
                    }
                ]
            },
        )

        builder = ReportBuilder(data)
        report = builder.build()

        assert report.validation is not None
        assert report.validation.failed_datasets == 1
        assert report.validation.datasets[0].status == "failed"
        assert report.validation.datasets[0].dataset_name == "test_dataset"

        validation_issues = [i for i in report.issues if i.context == "Dataset Validation"]
        assert len(validation_issues) == 1
        assert validation_issues[0].severity == "WARN"
        assert "test_dataset" in validation_issues[0].message
        assert "deduplication" in validation_issues[0].message

    def test_partial_failure_generates_warning_in_issues(self) -> None:
        """Test that partial dataset validation failures generate WARN issues."""
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        data = ExperimentData(
            run_id="r",
            run_name="r",
            experiment_name="e",
            status=RunStatus.FINISHED,
            start_time=start,
            end_time=start,
            duration_seconds=0.0,
            phases=[],
            validation_results={
                "datasets": [
                    {
                        "name": "partial_dataset",
                        "path": "/path",
                        "sample_count": 100,
                        "status": "passed",
                        "critical_failures": 3,
                        "plugins": [
                            {
                                "id": "min_samples_main",
                                "plugin_name": "min_samples",
                                "passed": True,
                                "duration_ms": 1.0,
                                "description": "",
                                "metrics": {},
                                "params": {},
                                "errors": [],
                                "recommendations": [],
                            },
                            {
                                "id": "dedup_main",
                                "plugin_name": "dedup",
                                "passed": False,
                                "duration_ms": 2.0,
                                "description": "",
                                "metrics": {},
                                "params": {},
                                "errors": ["1 error"],
                                "recommendations": [],
                            },
                        ],
                    }
                ]
            },
        )

        builder = ReportBuilder(data)
        report = builder.build()

        assert report.validation is not None
        assert report.validation.datasets[0].status == "passed"
        assert report.validation.datasets[0].has_partial_failure is True

        validation_issues = [i for i in report.issues if i.context == "Dataset Validation"]
        assert len(validation_issues) == 1
        assert validation_issues[0].severity == "WARN"
        assert "partial_dataset" in validation_issues[0].message
        assert "partial failure" in validation_issues[0].message
        assert "dedup" in validation_issues[0].message
