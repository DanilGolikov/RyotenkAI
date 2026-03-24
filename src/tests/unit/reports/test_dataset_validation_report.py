"""
Unit tests for Dataset Validation Report rendering.
Tests various scenarios: Success, Failure, Empty, None.
"""

import logging
from datetime import datetime

import pytest

from src.reports.domain.entities import (
    DatasetValidation,
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
from src.reports.plugins.interfaces import ReportPluginContext
from src.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer


class _DummyProvider:
    def load(self, run_id: str):  # pragma: no cover
        raise NotImplementedError


def _make_dummy_data(report: ExperimentReport) -> ExperimentData:
    # Minimal domain snapshot for plugin context (plugin uses only report.validation).
    return ExperimentData(
        run_id=report.summary.run_id,
        run_name=report.summary.run_name,
        experiment_name=report.summary.experiment_name,
        status=report.summary.status,
        start_time=datetime.now(),
        end_time=None,
        duration_seconds=0.0,
        phases=[],
    )


def _render_validation_block(report: ExperimentReport) -> str:
    plugin = DatasetValidationBlockPlugin()
    data = _make_dummy_data(report)
    ctx = ReportPluginContext(
        run_id=data.run_id,
        data_provider=_DummyProvider(),
        data=data,
        report=report,
        logger=logging.getLogger(__name__),
    )
    block = plugin.render(ctx)
    return MarkdownBlockRenderer().render([block])


@pytest.fixture
def base_report():
    """Create a base ExperimentReport with minimal fields populated."""
    return ExperimentReport(
        generated_at=datetime.now(),
        summary=ReportSummary(
            run_id="test_run",
            run_name="test_experiment",
            experiment_name="unit_test",
            status=RunStatus.FINISHED,
            health=ExperimentHealth.GREEN,
            health_explanation="All good",
            duration_total_seconds=100,
        ),
        model=ModelInfo(name="TestModel"),
        phases=[],
        resources=ResourcesInfo(),
        timeline=[],
        issues=[],
        config=ConfigInfo(),
        memory_management=None,
        validation=None,
    )


def test_render_validation_success(base_report):
    """Test rendering of a successful validation report."""
    base_report.validation = ValidationReport(
        datasets=[
            DatasetValidation(
                dataset_name="train.jsonl",
                dataset_path="/data/train.jsonl",
                status="passed",
                sample_count=5000,
                total_plugins=2,
                passed_plugins=2,
                failed_plugins=0,
                plugin_results=[
                    ValidationPluginResults(
                        id="plugin_1",
                        plugin_name="plugin_1",
                        status="passed",
                        duration_ms=10.5,
                        params={"threshold": 100},
                        metrics={"score": 1.0},
                    ),
                    ValidationPluginResults(
                        id="plugin_2",
                        plugin_name="plugin_2",
                        status="passed",
                        duration_ms=20.0,
                        params={"min": 10, "max": 100},
                        metrics={"count": 5000},
                    ),
                ],
            ),
        ],
    )

    output = _render_validation_block(base_report)

    assert "## Dataset Validation" in output
    assert "✅ Passed" in output
    assert "train.jsonl" in output
    assert "5000" in output
    assert "plugin_1" in output
    assert "plugin_2" in output
    assert "**id**: plugin_1" in output
    assert "**plugin**: plugin_1" in output


def test_render_validation_failure(base_report):
    """Test rendering of a failed validation report."""
    base_report.validation = ValidationReport(
        datasets=[
            DatasetValidation(
                dataset_name="bad_data.jsonl",
                dataset_path="/data/bad_data.jsonl",
                status="failed",
                sample_count=100,
                total_plugins=2,
                passed_plugins=1,
                failed_plugins=1,
                plugin_results=[
                    ValidationPluginResults(
                        id="ok_plugin",
                        plugin_name="ok_plugin",
                        status="passed",
                        duration_ms=5.0,
                    ),
                    ValidationPluginResults(
                        id="bad_plugin",
                        plugin_name="bad_plugin",
                        status="failed",
                        duration_ms=15.0,
                        errors=["Error 1", "Error 2"],
                    ),
                ],
            ),
        ],
    )

    output = _render_validation_block(base_report)

    assert "## Dataset Validation" in output
    assert "❌ Failed" in output
    assert "bad_plugin" in output
    assert "Error 1" in output


def test_render_validation_multiple_datasets(base_report):
    """Test rendering with multiple datasets."""
    base_report.validation = ValidationReport(
        datasets=[
            DatasetValidation(
                dataset_name="dataset_1.jsonl",
                dataset_path="/data/dataset_1.jsonl",
                status="passed",
                sample_count=1000,
                total_plugins=1,
                passed_plugins=1,
                failed_plugins=0,
                plugin_results=[
                    ValidationPluginResults(id="check", plugin_name="check", status="passed", duration_ms=5.0),
                ],
            ),
            DatasetValidation(
                dataset_name="dataset_2.jsonl",
                dataset_path="/data/dataset_2.jsonl",
                status="failed",
                sample_count=500,
                total_plugins=1,
                passed_plugins=0,
                failed_plugins=1,
                plugin_results=[
                    ValidationPluginResults(
                        id="check",
                        plugin_name="check",
                        status="failed",
                        duration_ms=3.0,
                        errors=["Failed check"],
                    ),
                ],
            ),
        ],
    )

    output = _render_validation_block(base_report)

    assert "## Dataset Validation" in output
    assert "dataset_1.jsonl" in output
    assert "dataset_2.jsonl" in output
    assert "1/2" in output  # Check for dataset count (format may vary)


def test_render_validation_no_datasets(base_report):
    """Test rendering when validation object exists but no datasets."""
    base_report.validation = ValidationReport(datasets=[])

    output = _render_validation_block(base_report)

    assert "## Dataset Validation" in output


def test_render_validation_none(base_report):
    """Test rendering when validation is None (No Data)."""
    base_report.validation = None

    output = _render_validation_block(base_report)

    assert "## Dataset Validation" in output
    assert "No dataset validation data available" in output


def test_render_validation_with_params_and_metrics(base_report):
    """Test rendering with complex params and metrics."""
    base_report.validation = ValidationReport(
        datasets=[
            DatasetValidation(
                dataset_name="test.jsonl",
                dataset_path="/data/test.jsonl",
                status="passed",
                sample_count=100,
                total_plugins=1,
                passed_plugins=1,
                failed_plugins=0,
                plugin_results=[
                    ValidationPluginResults(
                        id="complex_plugin",
                        plugin_name="complex_plugin",
                        status="passed",
                        duration_ms=10.0,
                        params={"threshold": 50, "min": 10, "max": 100},
                        metrics={"avg_length": 75.5, "count": 100},
                    ),
                ],
            ),
        ],
    )

    output = _render_validation_block(base_report)

    assert "complex_plugin" in output
    assert "threshold" in output.lower() or "Threshold" in output
    assert "avg_length" in output.lower() or "Avg Length" in output


def test_render_validation_shows_plugin_description_table(base_report):
    """Plugin-description table is rendered when descriptions are present."""
    base_report.validation = ValidationReport(
        datasets=[
            DatasetValidation(
                dataset_name="train.jsonl",
                dataset_path="/data/train.jsonl",
                status="passed",
                sample_count=100,
                total_plugins=1,
                passed_plugins=1,
                failed_plugins=0,
                plugin_results=[
                    ValidationPluginResults(
                        id="avg_length_main",
                        plugin_name="avg_length",
                        status="passed",
                        duration_ms=3.0,
                        description="Checks average text length",
                    )
                ],
            )
        ],
    )

    output = _render_validation_block(base_report)

    assert "Plugins in use" in output
    assert "avg_length" in output
    assert "Checks average text length" in output


def test_render_validation_groups_sample_errors(base_report):
    """Repeated per-sample errors are grouped by shared message."""
    base_report.validation = ValidationReport(
        datasets=[
            DatasetValidation(
                dataset_name="grouped.jsonl",
                dataset_path="/data/grouped.jsonl",
                status="failed",
                sample_count=10,
                total_plugins=1,
                passed_plugins=0,
                failed_plugins=1,
                plugin_results=[
                    ValidationPluginResults(
                        id="backend_check_main",
                        plugin_name="backend_check",
                        status="failed",
                        duration_ms=7.0,
                        errors=[
                            "Sample 1: backend=check failed with validation_error",
                            "Sample 2: backend=check failed with parse_error",
                            "Sample 4: backend=check failed with parse_error",
                            "Sample 6: backend=check failed with parse_error",
                            "Sample 10: backend=check failed with parse_error",
                        ],
                    )
                ],
            )
        ],
    )

    output = _render_validation_block(base_report)

    assert "Sample 1 - backend=check failed with validation_error" in output
    assert "Samples 2,4,6,10 - backend=check failed with parse_error" in output


def test_render_validation_truncates_grouped_index_errors_in_report(base_report):
    """Grouped validation errors in the report are capped at 20 examples."""
    grouped_indices = ", ".join(str(index) for index in range(1, 26))
    base_report.validation = ValidationReport(
        datasets=[
            DatasetValidation(
                dataset_name="grouped.jsonl",
                dataset_path="/data/grouped.jsonl",
                status="failed",
                sample_count=100,
                total_plugins=1,
                passed_plugins=0,
                failed_plugins=1,
                plugin_results=[
                    ValidationPluginResults(
                        id="gold_syntax_train",
                        plugin_name="helixql_gold_syntax_backend",
                        status="failed",
                        duration_ms=12.0,
                        errors=[
                            f"parse_error: [{grouped_indices}]",
                        ],
                    )
                ],
            )
        ],
    )

    output = _render_validation_block(base_report)

    assert (
        "parse_error: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] ... (+5 more errors)"
        in output
    )
    assert "21, 22, 23, 24, 25" not in output
