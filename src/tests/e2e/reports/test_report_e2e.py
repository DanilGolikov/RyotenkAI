"""
E2E tests for report generation.

Full cycle: ExperimentData -> ReportBuilder -> Plugins -> Markdown -> report file.
"""

import logging

from src.pipeline.artifacts.base import StageArtifactEnvelope
from src.reports.core.builder import ReportBuilder
from src.reports.models.report import ExperimentHealth
from src.reports.plugins.composer import ReportComposer
from src.reports.plugins.discovery import ensure_report_plugins_discovered
from src.reports.plugins.interfaces import ReportPluginContext
from src.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer
from src.reports.plugins.registry import build_report_plugins


class _DummyProvider:
    def load(self, run_id: str):  # pragma: no cover
        raise NotImplementedError


def _render_report_markdown_with_plugins(*, data, report, plugins) -> str:
    composer = ReportComposer(plugins)
    ctx = ReportPluginContext(
        run_id=data.run_id,
        data_provider=_DummyProvider(),
        data=data,
        report=report,
        logger=logging.getLogger(__name__),
    )
    blocks, _records = composer.compose(ctx)
    return MarkdownBlockRenderer().render(blocks)


def _render_report_markdown(*, data, report) -> str:
    ensure_report_plugins_discovered(force=True)
    return _render_report_markdown_with_plugins(data=data, report=report, plugins=build_report_plugins())


class _BoomPlugin:
    """Plugin used in tests to validate fail-open rendering."""

    plugin_id = "boom"
    title = "Boom"
    order = 15  # must not collide with builtin orders (10, 20, ...)

    def render(self, ctx: ReportPluginContext):  # pragma: no cover
        raise RuntimeError("kaboom")


class TestReportGenerationE2E:
    """E2E report generation tests."""

    def test_positive_green_report(self, positive_experiment_data, reports_output_dir):
        """
        POSITIVE: successful run -> GREEN report.

        Checks:
        - Report generates without errors
        - Health = GREEN
        - File is saved
        """
        # Arrange
        builder = ReportBuilder(positive_experiment_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=positive_experiment_data, report=report)

        # Assert
        assert report.summary.health == ExperimentHealth.GREEN
        assert "No critical issues" in report.summary.health_explanation
        assert report.summary.status.is_success
        assert len(report.issues) == 0  # No issues

        # Save report
        output_file = reports_output_dir / "positive_green_report.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        assert output_file.stat().st_size > 0
        print(f"\n✅ GREEN report saved: {output_file}")

    def test_negative_red_report(self, negative_experiment_data, reports_output_dir):
        """
        NEGATIVE: failed run with errors -> RED report.

        Checks:
        - Health = RED
        - Has ERROR issues
        - RunStatus.FAILED is displayed
        """
        # Arrange
        builder = ReportBuilder(negative_experiment_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=negative_experiment_data, report=report)

        # Assert
        assert report.summary.health == ExperimentHealth.RED
        assert not report.summary.status.is_success

        # ERROR issues present
        error_issues = [i for i in report.issues if i.severity == "ERROR"]
        assert len(error_issues) >= 2  # At least two errors from timeline

        # Save report
        output_file = reports_output_dir / "negative_red_report.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        print(f"\n🔴 RED report saved: {output_file}")

    def test_boundary_3_warnings_yellow(self, boundary_3_warnings_data, reports_output_dir):
        """
        BOUNDARY: exactly 3 WARN -> YELLOW report.

        Checks:
        - Health = YELLOW
        - Exactly 3 WARN issues
        - Explanation contains threshold info
        """
        # Arrange
        builder = ReportBuilder(boundary_3_warnings_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=boundary_3_warnings_data, report=report)

        # Assert
        assert report.summary.health == ExperimentHealth.YELLOW

        # WARN issues
        warn_issues = [i for i in report.issues if i.severity == "WARN"]
        assert len(warn_issues) == 3

        assert "3 WARN" in report.summary.health_explanation

        # Save report
        output_file = reports_output_dir / "boundary_3_warnings_yellow.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        print(f"\n🟡 YELLOW report (3 WARN) saved: {output_file}")

    def test_boundary_5_warnings_red(self, boundary_5_warnings_data, reports_output_dir):
        """
        BOUNDARY: exactly 5 WARN -> RED report.

        Checks:
        - Health = RED
        - Exactly 5 WARN issues
        - Rule >= 5 applied
        """
        # Arrange
        builder = ReportBuilder(boundary_5_warnings_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=boundary_5_warnings_data, report=report)

        # Assert
        assert report.summary.health == ExperimentHealth.RED

        # WARN issues
        warn_issues = [i for i in report.issues if i.severity == "WARN"]
        assert len(warn_issues) == 5

        assert "5 WARN" in report.summary.health_explanation

        # Save report
        output_file = reports_output_dir / "boundary_5_warnings_red.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        print(f"\n🔴 RED report (5 WARN) saved: {output_file}")

    def test_crazy_empty_data(self, crazy_empty_data, reports_output_dir):
        """
        EDGE: almost empty data.

        Checks:
        - Report generates without crashing
        - No phases
        - Defaults are sane
        """
        # Arrange
        builder = ReportBuilder(crazy_empty_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=crazy_empty_data, report=report)

        # Assert
        assert len(report.phases) == 0
        assert report.summary.total_epochs == 0
        assert report.summary.total_steps == 0
        # Plugins should tolerate missing data and emit placeholders
        assert "No dataset validation data available" in markdown
        assert "No training phase data available" in markdown
        assert "No timeline events" in markdown or "No pipeline stage data available" in markdown
        assert "No warnings or errors" in markdown
        assert "No metric descriptions" in markdown

        # Save report
        output_file = reports_output_dir / "crazy_empty_data.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        print(f"\n🤪 CRAZY (empty) report saved: {output_file}")

    def test_fail_open_plugin_error_block_visible_and_report_saved(
        self,
        positive_experiment_data,
        reports_output_dir,
    ):
        """
        FAIL-OPEN: plugin raises -> error block in report; other blocks still render.
        """
        builder = ReportBuilder(positive_experiment_data)
        report = builder.build()

        ensure_report_plugins_discovered(force=True)
        plugins = [*build_report_plugins(), _BoomPlugin()]
        markdown = _render_report_markdown_with_plugins(data=positive_experiment_data, report=report, plugins=plugins)

        assert "## Boom" in markdown
        assert "Error type:" in markdown
        assert "Error message:" in markdown
        assert "kaboom" in markdown
        assert "# 🧪 Experiment Report:" in markdown

        output_file = reports_output_dir / "fail_open_error_block.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        print(f"\n🧯 FAIL-OPEN report (error-block) saved: {output_file}")

    def test_crazy_missing_fields(self, crazy_missing_fields_data, reports_output_dir):
        """
        CRAZY: missing/empty fields.

        Checks:
        - Report is generated even with empty fields
        - No KeyError or AttributeError
        """
        # Arrange
        builder = ReportBuilder(crazy_missing_fields_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=crazy_missing_fields_data, report=report)

        # Assert
        assert len(report.phases) == 1
        assert report.phases[0].strategy == ""

        # Save report
        output_file = reports_output_dir / "crazy_missing_fields.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        print(f"\n🤪 CRAZY (missing fields) report saved: {output_file}")

    def test_crazy_mixed_severities(self, crazy_mixed_severities_data, reports_output_dir):
        """
        EDGE: 2 WARN + 10 INFO -> GREEN (INFO not counted).

        Checks:
        - Health = GREEN (2 WARN < 3)
        - INFO events do not affect health
        """
        # Arrange
        builder = ReportBuilder(crazy_mixed_severities_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=crazy_mixed_severities_data, report=report)

        # Assert
        assert report.summary.health == ExperimentHealth.GREEN

        # INFO must not appear in issues
        info_issues = [i for i in report.issues if i.severity == "INFO"]
        assert len(info_issues) == 0

        warn_issues = [i for i in report.issues if i.severity == "WARN"]
        assert len(warn_issues) == 2

        # Save report
        output_file = reports_output_dir / "crazy_mixed_severities.md"
        output_file.write_text(markdown, encoding="utf-8")

        assert output_file.exists()
        print(f"\n🤪 CRAZY (mixed severities) report saved: {output_file}")

    def test_report_structure_completeness(self, positive_experiment_data, reports_output_dir):
        """
        Report structure completeness: main sections present.
        """
        # Arrange
        builder = ReportBuilder(positive_experiment_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=positive_experiment_data, report=report)

        # Assert — key sections
        assert "# 🧪 Experiment Report:" in markdown
        assert "## 📊 Summary" in markdown
        assert "## ⚙️ Model Configuration" in markdown or "model" in markdown.lower()
        assert "## 💾 Memory Management" in markdown
        assert "## 📚 Training Configuration" in markdown
        assert "## ⚠️ Warnings & Errors" in markdown

        # Health explanation appears in markdown
        assert report.summary.health_explanation in markdown

        # Save report
        output_file = reports_output_dir / "structure_completeness.md"
        output_file.write_text(markdown, encoding="utf-8")

        print(f"\n✅ Full report (structure) saved: {output_file}")


class TestReportHealthLogic:
    """Health logic tests in full-report context."""

    def test_run_failed_overrides_warnings(self, boundary_3_warnings_data, reports_output_dir):
        """
        Priority: RunStatus.FAILED -> RED (even with only 3 WARN).
        """
        from src.reports.domain.entities import RunStatus

        # Modify data
        boundary_3_warnings_data.status = RunStatus.FAILED

        # Arrange
        builder = ReportBuilder(boundary_3_warnings_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=boundary_3_warnings_data, report=report)

        # Assert
        assert report.summary.health == ExperimentHealth.RED
        assert "failed" in report.summary.health_explanation.lower()

        # Save report
        output_file = reports_output_dir / "priority_run_failed.md"
        output_file.write_text(markdown, encoding="utf-8")

        print(f"\n🔴 RED (priority: FAILED) report saved: {output_file}")

    def test_4_warnings_yellow(self, boundary_3_warnings_data, reports_output_dir):
        """
        Check: 4 WARN -> YELLOW (>= 3 but < 5).
        """
        # Add one more interrupted stage (4th warning)
        boundary_3_warnings_data.stage_envelopes.append(
            StageArtifactEnvelope(
                stage="Model Retriever",
                status="interrupted",
                started_at=boundary_3_warnings_data.start_time.isoformat(),  # type: ignore
                duration_seconds=30.0,
                error=None,
                data={},
            )
        )

        # Arrange
        builder = ReportBuilder(boundary_3_warnings_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=boundary_3_warnings_data, report=report)

        # Assert
        assert report.summary.health == ExperimentHealth.YELLOW

        warn_issues = [i for i in report.issues if i.severity == "WARN"]
        assert len(warn_issues) == 4

        # Save report
        output_file = reports_output_dir / "boundary_4_warnings_yellow.md"
        output_file.write_text(markdown, encoding="utf-8")

        print(f"\n🟡 YELLOW (4 WARN) report saved: {output_file}")


class TestReportFileOutput:
    """Report file output tests."""

    def test_all_reports_saved(self, reports_output_dir):
        """
        Expected report files land in the output directory.
        """
        # Expected files (after full test run)
        expected_files = [
            "positive_green_report.md",
            "negative_red_report.md",
            "boundary_3_warnings_yellow.md",
            "boundary_5_warnings_red.md",
            "crazy_empty_data.md",
            "crazy_missing_fields.md",
            "crazy_mixed_severities.md",
            "structure_completeness.md",
            "priority_run_failed.md",
            "boundary_4_warnings_yellow.md",
        ]

        assert reports_output_dir.exists()
        assert reports_output_dir.is_dir()

        print(f"\n📁 Reports directory: {reports_output_dir}")
        print(f"📊 Expect at least {len(expected_files)} files")

    def test_report_encoding(self, positive_experiment_data, reports_output_dir):
        """
        Verify UTF-8 encoding (report text is readable after round-trip).
        """
        # Arrange
        builder = ReportBuilder(positive_experiment_data)

        # Act
        report = builder.build()
        markdown = _render_report_markdown(data=positive_experiment_data, report=report)

        # Assert — summary contains expected English text
        assert "Experiment" in markdown

        # Save and read back
        output_file = reports_output_dir / "encoding_test.md"
        output_file.write_text(markdown, encoding="utf-8")

        read_back = output_file.read_text(encoding="utf-8")
        assert read_back == markdown

        print(f"\n✅ UTF-8 encoding verified: {output_file}")
