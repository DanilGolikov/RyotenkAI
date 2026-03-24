from __future__ import annotations

import logging
from datetime import datetime

from src.reports.document.nodes import Heading, HorizontalRule, inlines, txt
from src.reports.domain.entities import ExperimentData, RunStatus
from src.reports.models.report import (
    ConfigInfo,
    ExperimentHealth,
    ExperimentReport,
    ModelInfo,
    ReportSummary,
    ResourcesInfo,
)
from src.reports.plugins.composer import ReportComposer
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext
from src.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer


class _DummyProvider:
    def load(self, run_id: str):  # pragma: no cover
        raise NotImplementedError


class _OkPlugin:
    plugin_id = "ok"
    title = "OK"
    order = 10

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        return ReportBlock(
            block_id=self.plugin_id,
            title=self.title,
            order=self.order,
            nodes=[Heading(2, inlines(txt("OK"))), HorizontalRule()],
        )


class _BoomPlugin:
    plugin_id = "boom"
    title = "Boom"
    order = 20

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        raise RuntimeError("kaboom")


def test_plugins_fail_open_and_render_error_block() -> None:
    report = ExperimentReport(
        generated_at=datetime.now(),
        summary=ReportSummary(
            run_id="test_run",
            run_name="test",
            experiment_name="unit",
            status=RunStatus.FINISHED,
            health=ExperimentHealth.GREEN,
            health_explanation="ok",
            duration_total_seconds=1.0,
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
    data = ExperimentData(
        run_id=report.summary.run_id,
        run_name=report.summary.run_name,
        experiment_name=report.summary.experiment_name,
        status=report.summary.status,
        start_time=datetime.now(),
        end_time=None,
        duration_seconds=0.0,
        phases=[],
    )

    ctx = ReportPluginContext(
        run_id=data.run_id,
        data_provider=_DummyProvider(),
        data=data,
        report=report,
        logger=logging.getLogger(__name__),
    )

    composer = ReportComposer([_OkPlugin(), _BoomPlugin()])
    blocks, records = composer.compose(ctx)
    markdown = MarkdownBlockRenderer().render(blocks)

    assert len(blocks) == 2
    assert len(records) == 2
    assert any(r.plugin_id == "boom" and r.status == "failed" for r in records)

    # Error should be visible in the report and also logged via exception handler.
    assert "## Boom" in markdown
    assert "Error type:" in markdown
    assert "Error message:" in markdown
    assert "kaboom" in markdown
    assert "## OK" in markdown

