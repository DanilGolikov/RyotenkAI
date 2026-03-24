"""
Unit tests for Stage Artifacts — builder methods and EvaluationBlockPlugin.

Tests:
 - ReportBuilder._build_validation() from ValidationArtifactData
 - ReportBuilder._build_evaluation() from EvalArtifactData
 - ReportBuilder._build_timeline() from stage_envelopes
 - ReportBuilder._build_issues() from stage_envelopes + evaluation
 - EvaluationBlockPlugin.render() — happy path + empty
"""

import logging
from datetime import datetime
from typing import Any

import pytest

from src.pipeline.artifacts.base import StageArtifactEnvelope
from src.reports.core.builder import ReportBuilder
from src.reports.domain.entities import (
    DatasetValidation,
    EvalPluginResult,
    EvaluationReport,
    ExperimentData,
    RunStatus,
    ValidationPluginResults,
    ValidationReport,
)
from src.reports.models.report import (
    ConfigInfo,
    ExperimentHealth,
    ExperimentReport,
    ModelInfo,
    ReportSummary,
    ResourcesInfo,
)
from src.reports.plugins.builtins.dataset_validation import DatasetValidationBlockPlugin
from src.reports.plugins.builtins.evaluation_block import EvaluationBlockPlugin
from src.reports.plugins.interfaces import ReportPluginContext
from src.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer

# =============================================================================
# FIXTURES
# =============================================================================


def _base_data(**kwargs: Any) -> ExperimentData:
    """Minimal ExperimentData for testing."""
    defaults = dict(
        run_id="test-run-id",
        run_name="test-run",
        experiment_name="test-exp",
        status=RunStatus.FINISHED,
        start_time=datetime(2026, 3, 1, 12, 0, 0),
        end_time=datetime(2026, 3, 1, 12, 30, 0),
        duration_seconds=1800.0,
        phases=[],
        source_config={},
        root_params={},
    )
    defaults.update(kwargs)
    return ExperimentData(**defaults)


def _make_envelope(
    stage: str,
    status: str = "passed",
    started_at: str = "2026-03-01T12:00:00",
    duration_seconds: float = 10.0,
    error: str | None = None,
    data: dict[str, Any] | None = None,
) -> StageArtifactEnvelope:
    return StageArtifactEnvelope(
        stage=stage,
        status=status,
        started_at=started_at,
        duration_seconds=duration_seconds,
        error=error,
        data=data or {},
    )


def _minimal_report(
    evaluation: EvaluationReport | None = None,
    validation: ValidationReport | None = None,
) -> ExperimentReport:
    summary = ReportSummary(
        run_id="r1",
        run_name="run",
        experiment_name="exp",
        status=RunStatus.FINISHED,
        health=ExperimentHealth.GREEN,
        health_explanation="OK",
        duration_total_seconds=1800.0,
    )
    return ExperimentReport(
        generated_at=datetime.now(),
        summary=summary,
        model=ModelInfo(),
        config=ConfigInfo(),
        phases=[],
        resources=ResourcesInfo(),
        timeline=[],
        issues=[],
        validation=validation,
        evaluation=evaluation,
    )


# =============================================================================
# ReportBuilder._build_validation() — from ValidationArtifactData
# =============================================================================


class TestBuildValidation:
    """Tests for _build_validation() reading from validation_results."""

    def test_returns_none_when_no_data(self):
        data = _base_data(validation_results=None)
        builder = ReportBuilder(data)
        result = builder._build_validation()
        assert result is None

    def test_returns_none_when_empty_datasets(self):
        data = _base_data(validation_results={"datasets": []})
        builder = ReportBuilder(data)
        result = builder._build_validation()
        assert result is None

    def test_parses_passed_dataset(self):
        data = _base_data(
            validation_results={
                "datasets": [
                    {
                        "name": "train",
                        "path": "/data/train.jsonl",
                        "sample_count": 1000,
                        "status": "passed",
                        "critical_failures": 0,
                        "plugins": [
                            {
                                "id": "avg_length_main",
                                "plugin_name": "avg_length",
                                "passed": True,
                                "duration_ms": 12.5,
                                "description": "Checks average length",
                                "metrics": {"avg_length": 128.0},
                                "params": {"min_length": 10},
                                "errors": [],
                                "recommendations": [],
                            }
                        ],
                    }
                ]
            }
        )
        builder = ReportBuilder(data)
        result = builder._build_validation()
        assert result is not None
        assert len(result.datasets) == 1
        ds = result.datasets[0]
        assert ds.dataset_name == "train"
        assert ds.sample_count == 1000
        assert ds.status == "passed"
        assert ds.total_plugins == 1
        assert ds.passed_plugins == 1
        assert len(ds.plugin_results) == 1
        p = ds.plugin_results[0]
        assert p.id == "avg_length_main"
        assert p.plugin_name == "avg_length"
        assert p.passed is True
        assert p.duration_ms == pytest.approx(12.5)
        assert p.metrics["avg_length"] == pytest.approx(128.0)

    def test_parses_failed_plugin(self):
        data = _base_data(
            validation_results={
                "datasets": [
                    {
                        "name": "eval",
                        "path": "/data/eval.jsonl",
                        "sample_count": 50,
                        "status": "failed",
                        "critical_failures": 1,
                        "plugins": [
                            {
                                "id": "empty_ratio_main",
                                "plugin_name": "empty_ratio",
                                "passed": False,
                                "duration_ms": 5.0,
                                "description": "Checks empty ratio",
                                "metrics": {"empty_ratio": 0.8},
                                "params": {},
                                "errors": ["Too many empty samples"],
                                "recommendations": ["Check data pipeline"],
                            }
                        ],
                    }
                ]
            }
        )
        builder = ReportBuilder(data)
        result = builder._build_validation()
        assert result is not None
        ds = result.datasets[0]
        assert ds.status == "failed"
        assert ds.failed_plugins == 1
        p = ds.plugin_results[0]
        assert p.passed is False
        assert "Too many empty samples" in p.errors


# =============================================================================
# ReportBuilder._build_evaluation() — from EvalArtifactData
# =============================================================================


class TestBuildEvaluation:
    """Tests for _build_evaluation() reading from evaluation_results."""

    def test_returns_none_when_no_data(self):
        data = _base_data(evaluation_results=None)
        builder = ReportBuilder(data)
        result = builder._build_evaluation()
        assert result is None

    def test_parses_passed_evaluation(self):
        data = _base_data(
            evaluation_results={
                "overall_passed": True,
                "sample_count": 42,
                "duration_seconds": 27.5,
                "skipped_plugins": [],
                "errors": [],
                "plugins": {
                    "syntax_main": {
                        "plugin_name": "helixql_syntax",
                        "passed": True,
                        "metrics": {"syntax_valid_ratio": 0.95},
                        "errors": [],
                        "recommendations": [],
                        "sample_count": 42,
                        "failed_samples": 2,
                    }
                },
            }
        )
        builder = ReportBuilder(data)
        result = builder._build_evaluation()
        assert result is not None
        assert result.overall_passed is True
        assert result.sample_count == 42
        assert result.duration_seconds == pytest.approx(27.5)
        assert len(result.plugins) == 1
        p = result.plugins[0]
        assert p.name == "syntax_main"
        assert p.plugin_name == "helixql_syntax"
        assert p.passed is True
        assert p.metrics["syntax_valid_ratio"] == pytest.approx(0.95)
        assert p.sample_count == 42
        assert p.failed_samples == 2

    def test_parses_failed_evaluation(self):
        data = _base_data(
            evaluation_results={
                "overall_passed": False,
                "sample_count": 10,
                "duration_seconds": 5.0,
                "skipped_plugins": ["llm_judge"],
                "errors": ["endpoint error"],
                "plugins": {
                    "syntax_main": {
                        "plugin_name": "helixql_syntax",
                        "passed": False,
                        "metrics": {"syntax_valid_ratio": 0.3},
                        "errors": ["3 invalid queries"],
                        "recommendations": ["Fix prompts"],
                        "sample_count": 10,
                        "failed_samples": 7,
                    }
                },
            }
        )
        builder = ReportBuilder(data)
        result = builder._build_evaluation()
        assert result is not None
        assert result.overall_passed is False
        assert "endpoint error" in result.errors
        assert "llm_judge" in result.skipped_plugins
        p = result.plugins[0]
        assert p.plugin_name == "helixql_syntax"
        assert p.passed is False
        assert "3 invalid queries" in p.errors


# =============================================================================
# ReportBuilder._build_timeline() — from stage_envelopes
# =============================================================================


class TestBuildTimeline:
    """Tests for _build_timeline() reading from stage_envelopes."""

    def test_empty_when_no_envelopes(self):
        data = _base_data(stage_envelopes=[])
        builder = ReportBuilder(data)
        result = builder._build_timeline()
        assert result == []

    def test_builds_events_for_each_envelope(self):
        envelopes = [
            _make_envelope("dataset_validator", "passed", "2026-03-01T12:00:00", 1.3),
            _make_envelope("gpu_deployer", "passed", "2026-03-01T12:00:01", 402.1),
            _make_envelope("training_monitor", "failed", "2026-03-01T12:06:43", 128.0, error="OOM"),
        ]
        data = _base_data(stage_envelopes=envelopes)
        builder = ReportBuilder(data)
        result = builder._build_timeline()
        assert len(result) == 3
        # Check sorted by timestamp
        sources = [e.source for e in result]
        assert sources[0] == "dataset_validator"
        assert sources[1] == "gpu_deployer"
        assert sources[2] == "training_monitor"

    def test_failed_stage_gets_error_severity(self):
        envelopes = [
            _make_envelope("model_evaluator", "failed", error="timeout"),
        ]
        data = _base_data(stage_envelopes=envelopes)
        builder = ReportBuilder(data)
        result = builder._build_timeline()
        assert len(result) == 1
        assert result[0].severity == "ERROR"

    def test_passed_stage_gets_info_severity(self):
        envelopes = [_make_envelope("dataset_validator", "passed")]
        data = _base_data(stage_envelopes=envelopes)
        builder = ReportBuilder(data)
        result = builder._build_timeline()
        assert result[0].severity == "INFO"


# =============================================================================
# ReportBuilder._build_issues() — from stage_envelopes
# =============================================================================


class TestBuildIssues:
    """Tests for _build_issues() reading from stage_envelopes."""

    def _make_builder(self, **kwargs: Any) -> ReportBuilder:
        return ReportBuilder(_base_data(**kwargs))

    def test_no_issues_when_all_passed(self):
        envelopes = [
            _make_envelope("dataset_validator", "passed"),
            _make_envelope("gpu_deployer", "passed"),
        ]
        builder = self._make_builder(stage_envelopes=envelopes)
        from src.reports.models.report import MemoryManagementInfo

        issues = builder._build_issues([], MemoryManagementInfo(), None)
        assert len(issues) == 0

    def test_failed_envelope_adds_error_issue(self):
        envelopes = [
            _make_envelope("training_monitor", "failed", error="OOM crash"),
        ]
        builder = self._make_builder(stage_envelopes=envelopes)
        from src.reports.models.report import MemoryManagementInfo

        issues = builder._build_issues([], MemoryManagementInfo(), None)
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 1
        assert "OOM crash" in errors[0].message

    def test_interrupted_envelope_adds_warn_issue(self):
        envelopes = [
            _make_envelope("model_retriever", "interrupted"),
        ]
        builder = self._make_builder(stage_envelopes=envelopes)
        from src.reports.models.report import MemoryManagementInfo

        issues = builder._build_issues([], MemoryManagementInfo(), None)
        warns = [i for i in issues if i.severity == "WARN"]
        assert any("interrupted" in w.message for w in warns)

    def test_failed_eval_plugin_adds_warn(self):
        evaluation = EvaluationReport(
            overall_passed=False,
            sample_count=10,
            duration_seconds=5.0,
            plugins=[
                EvalPluginResult(
                    name="helixql_syntax",
                    passed=False,
                    errors=["3 failures"],
                ),
            ],
        )
        builder = self._make_builder()
        from src.reports.models.report import MemoryManagementInfo

        issues = builder._build_issues([], MemoryManagementInfo(), None, evaluation)
        warns = [i for i in issues if i.severity == "WARN"]
        assert any("helixql_syntax" in w.message for w in warns)

    def test_missing_artifacts_add_warn(self):
        """training_events.json in missing_artifacts yields WARN; stage artifacts do not."""
        builder = self._make_builder(
            missing_artifacts=["training_events.json"]
        )
        from src.reports.models.report import MemoryManagementInfo

        issues = builder._build_issues([], MemoryManagementInfo(), None)
        warns = [i for i in issues if i.severity == "WARN"]
        # training_events.json → special WARN about remote PC
        training_warns = [w for w in warns if "training_events" in w.message.lower()]
        assert len(training_warns) == 1

    def test_missing_stage_artifact_no_warn(self):
        """Missing stage artifact (stage did not run) must not produce WARN."""
        builder = self._make_builder(
            missing_artifacts=["evaluation_results.json", "gpu_deployer_results.json"]
        )
        from src.reports.models.report import MemoryManagementInfo

        issues = builder._build_issues([], MemoryManagementInfo(), None)
        artifact_warns = [i for i in issues if "Artifact missing" in i.message]
        assert len(artifact_warns) == 0


# =============================================================================
# EvaluationBlockPlugin
# =============================================================================


class _DummyProvider:
    def load(self, run_id: str):  # pragma: no cover
        raise NotImplementedError


class TestDatasetValidationBlockPlugin:
    """Tests for DatasetValidationBlockPlugin.render()."""

    def _render(self, validation: ValidationReport | None) -> str:
        plugin = DatasetValidationBlockPlugin()
        report = _minimal_report(validation=validation)
        data = _base_data()
        ctx = ReportPluginContext(
            run_id=data.run_id,
            data_provider=_DummyProvider(),
            data=data,
            report=report,
            logger=logging.getLogger(__name__),
        )
        block = plugin.render(ctx)
        renderer = MarkdownBlockRenderer()
        return renderer.render([block])

    def test_render_shows_unique_plugins_table_from_plugin_name(self):
        validation = ValidationReport(
            datasets=[
                DatasetValidation(
                    dataset_name="train",
                    dataset_path="/data/train.jsonl",
                    sample_count=100,
                    status="passed",
                    total_plugins=1,
                    passed_plugins=1,
                    failed_plugins=0,
                    plugin_results=[
                        ValidationPluginResults(
                            id="avg_length_main",
                            plugin_name="avg_length",
                            status="passed",
                            duration_ms=12.5,
                            description="Checks average length",
                            metrics={"avg_length": 128.0},
                            params={},
                            thresholds={"min": 50, "max": 2048},
                            errors=[],
                            recommendations=[],
                        )
                    ],
                )
            ]
        )

        output = self._render(validation)
        assert "Plugins in use" in output
        assert "avg_length" in output
        assert "Checks average length" in output
        assert "**id**: avg_length_main" in output
        assert "**plugin**: avg_length" in output
        assert "<br>" in output


class TestEvaluationBlockPlugin:
    """Tests for EvaluationBlockPlugin.render()."""

    def _render(self, evaluation: EvaluationReport | None) -> str:
        plugin = EvaluationBlockPlugin()
        report = _minimal_report(evaluation=evaluation)
        data = _base_data()
        ctx = ReportPluginContext(
            run_id=data.run_id,
            data_provider=_DummyProvider(),
            data=data,
            report=report,
            logger=logging.getLogger(__name__),
        )
        block = plugin.render(ctx)
        renderer = MarkdownBlockRenderer()
        return renderer.render([block])

    def test_render_no_evaluation(self):
        output = self._render(None)
        assert "no model evaluation" in output.lower() or "model evaluation" in output.lower()

    def test_render_passed_evaluation(self):
        evaluation = EvaluationReport(
            overall_passed=True,
            sample_count=42,
            duration_seconds=27.5,
            plugins=[
                EvalPluginResult(
                    name="helixql_syntax",
                    passed=True,
                    metrics={"syntax_valid_ratio": 0.95},
                    sample_count=42,
                    failed_samples=2,
                )
            ],
        )
        output = self._render(evaluation)
        assert "Passed" in output or "passed" in output.lower()
        assert "42" in output
        assert "helixql_syntax" in output
        assert "0.9500" in output

    def test_render_failed_evaluation(self):
        evaluation = EvaluationReport(
            overall_passed=False,
            sample_count=10,
            duration_seconds=5.0,
            plugins=[
                EvalPluginResult(
                    name="helixql_syntax",
                    passed=False,
                    errors=["3 invalid queries"],
                    sample_count=10,
                    failed_samples=7,
                )
            ],
        )
        output = self._render(evaluation)
        assert "Failed" in output or "failed" in output.lower()
        assert "helixql_syntax" in output
        assert "3 invalid queries" in output

    def test_render_with_skipped_plugins(self):
        evaluation = EvaluationReport(
            overall_passed=True,
            sample_count=5,
            duration_seconds=2.0,
            skipped_plugins=["llm_judge"],
        )
        output = self._render(evaluation)
        assert "llm_judge" in output

    def test_render_with_errors(self):
        evaluation = EvaluationReport(
            overall_passed=False,
            sample_count=0,
            duration_seconds=0.5,
            errors=["endpoint unreachable"],
        )
        output = self._render(evaluation)
        assert "endpoint unreachable" in output

    def test_render_shows_plugin_description_table(self):
        """If a plugin has description or plugin_name — show 'Plugins in use' table."""
        evaluation = EvaluationReport(
            overall_passed=True,
            sample_count=20,
            duration_seconds=10.0,
            plugins=[
                EvalPluginResult(
                    name="syntax_main",
                    passed=True,
                    plugin_name="helixql_syntax",
                    description="Validates HelixQL syntax correctness",
                    sample_count=20,
                )
            ],
        )
        output = self._render(evaluation)
        assert "Plugins in use" in output
        assert "helixql_syntax" in output
        assert "syntax_main" in output
        assert "Validates HelixQL syntax correctness" in output
        assert "**id**: syntax_main" in output
        assert "**plugin**: helixql_syntax" in output
        assert "<br>" in output

    def test_render_no_description_table_when_empty_meta(self):
        """If description and plugin_name are empty — 'Plugins in use' table is hidden."""
        evaluation = EvaluationReport(
            overall_passed=True,
            sample_count=5,
            duration_seconds=2.0,
            plugins=[
                EvalPluginResult(
                    name="helixql_syntax",
                    passed=True,
                    plugin_name="",
                    description="",
                    sample_count=5,
                )
            ],
        )
        output = self._render(evaluation)
        assert "Plugins in use" not in output

    def test_render_recommendations_section_shown_when_present(self):
        """Recommendations render in their own section when present."""
        evaluation = EvaluationReport(
            overall_passed=False,
            sample_count=10,
            duration_seconds=5.0,
            plugins=[
                EvalPluginResult(
                    name="helixql_syntax",
                    passed=False,
                    recommendations=["Fix JOIN syntax", "Add semicolons"],
                    sample_count=10,
                )
            ],
        )
        output = self._render(evaluation)
        assert "Recommendations" in output
        assert "Fix JOIN syntax" in output
        assert "Add semicolons" in output

    def test_render_recommendations_section_empty_when_all_passed(self):
        """When all plugins pass — show 'No recommendations' message."""
        evaluation = EvaluationReport(
            overall_passed=True,
            sample_count=10,
            duration_seconds=3.0,
            plugins=[
                EvalPluginResult(
                    name="helixql_syntax",
                    passed=True,
                    recommendations=[],
                    sample_count=10,
                )
            ],
        )
        output = self._render(evaluation)
        assert "No recommendations" in output


class TestBuildEvaluationWithMeta:
    """Tests for description/plugin_name fields in _build_evaluation."""

    def test_parses_description_and_plugin_name(self):
        data = _base_data(
            evaluation_results={
                "overall_passed": True,
                "sample_count": 10,
                "duration_seconds": 5.0,
                "skipped_plugins": [],
                "errors": [],
                "plugins": {
                    "syntax_main": {
                        "plugin_name": "helixql_syntax",
                        "passed": True,
                        "description": "Validates HelixQL syntax",
                        "metrics": {"valid_ratio": 0.9},
                        "errors": [],
                        "recommendations": [],
                        "sample_count": 10,
                        "failed_samples": 1,
                    }
                },
            }
        )
        builder = ReportBuilder(data)
        result = builder._build_evaluation()
        assert result is not None
        p = result.plugins[0]
        assert p.name == "syntax_main"
        assert p.plugin_name == "helixql_syntax"
        assert p.description == "Validates HelixQL syntax"

    def test_defaults_to_empty_string_when_missing(self):
        """Backward compatibility: legacy artifacts without description/plugin_name."""
        data = _base_data(
            evaluation_results={
                "overall_passed": True,
                "sample_count": 5,
                "duration_seconds": 2.0,
                "skipped_plugins": [],
                "errors": [],
                "plugins": {
                    "helixql_syntax": {
                        "passed": True,
                        "metrics": {},
                        "errors": [],
                        "recommendations": [],
                        "sample_count": 5,
                        "failed_samples": 0,
                    }
                },
            }
        )
        builder = ReportBuilder(data)
        result = builder._build_evaluation()
        assert result is not None
        p = result.plugins[0]
        assert p.description == ""
        assert p.plugin_name == ""


class TestRunSummaryToDict:
    """Tests for RunSummary.to_dict() with plugin_meta."""

    def test_to_dict_includes_description_and_plugin_name(self):
        from src.evaluation.plugins.base import EvalResult
        from src.evaluation.runner import RunSummary

        summary = RunSummary()
        summary.plugin_results["helixql_syntax"] = EvalResult(
            plugin_name="helixql_syntax",
            passed=True,
            metrics={"valid_ratio": 0.95},
            sample_count=10,
        )
        summary.plugin_meta["helixql_syntax"] = {
            "plugin_name": "helixql_syntax",
            "description": "Validates HelixQL syntax",
        }
        d = summary.to_dict()
        p = d["plugins"]["helixql_syntax"]
        assert p["plugin_name"] == "helixql_syntax"
        assert p["description"] == "Validates HelixQL syntax"

    def test_to_dict_empty_meta_when_not_set(self):
        """When plugin_meta is unset, description/plugin_name are empty strings."""
        from src.evaluation.plugins.base import EvalResult
        from src.evaluation.runner import RunSummary

        summary = RunSummary()
        summary.plugin_results["helixql_syntax"] = EvalResult(
            plugin_name="helixql_syntax",
            passed=True,
            sample_count=5,
        )
        d = summary.to_dict()
        p = d["plugins"]["helixql_syntax"]
        assert p["description"] == ""
        assert p["plugin_name"] == ""
