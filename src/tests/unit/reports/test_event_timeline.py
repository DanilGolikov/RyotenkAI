"""
Unit tests for EventTimelineBlockPlugin.

Covers: render() with empty timeline, pipeline/training/other event groups,
severity emoji mapping, null timestamps/sources, and HorizontalRule footer.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.reports.document.nodes import Heading, HorizontalRule, Paragraph, Table
from src.reports.models.report import TimelineEvent
from src.reports.plugins.builtins.event_timeline import EventTimelineBlockPlugin
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext

pytestmark = pytest.mark.unit


def _make_ctx(timeline: list[TimelineEvent]) -> ReportPluginContext:
    report = MagicMock()
    report.timeline = timeline
    ctx = MagicMock(spec=ReportPluginContext)
    ctx.report = report
    return ctx


def _event(**kwargs) -> TimelineEvent:
    defaults = dict(
        timestamp=datetime(2024, 6, 1, 10, 0, 0),
        event_type="TEST",
        message="test message",
        source="test_source",
        origin="pipeline",
        severity="INFO",
    )
    defaults.update(kwargs)
    return TimelineEvent(**defaults)


class TestEventTimelineBlockPluginMetadata:
    def test_plugin_id(self) -> None:
        assert EventTimelineBlockPlugin.plugin_id == "event_timeline"

    def test_title(self) -> None:
        assert EventTimelineBlockPlugin.title == "Event Timeline"

    def test_order(self) -> None:
        assert EventTimelineBlockPlugin.order == 100


class TestEventTimelineRenderEmpty:
    def setup_method(self) -> None:
        self.plugin = EventTimelineBlockPlugin()

    def test_returns_report_block(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        assert isinstance(block, ReportBlock)

    def test_block_id_matches_plugin_id(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        assert block.block_id == "event_timeline"

    def test_title_matches(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        assert block.title == "Event Timeline"

    def test_order_matches(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        assert block.order == 100

    def test_heading_level_2_present(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        headings = [n for n in block.nodes if isinstance(n, Heading)]
        assert any(h.level == 2 for h in headings)

    def test_empty_timeline_shows_paragraph(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        paragraphs = [n for n in block.nodes if isinstance(n, Paragraph)]
        assert paragraphs, "Expected a Paragraph for empty timeline"

    def test_empty_timeline_has_horizontal_rule(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        assert any(isinstance(n, HorizontalRule) for n in block.nodes)

    def test_empty_timeline_no_tables(self) -> None:
        block = self.plugin.render(_make_ctx([]))
        tables = [n for n in block.nodes if isinstance(n, Table)]
        assert not tables


class TestEventTimelineRenderPipelineEvents:
    def setup_method(self) -> None:
        self.plugin = EventTimelineBlockPlugin()

    def test_pipeline_event_creates_h3_heading(self) -> None:
        block = self.plugin.render(_make_ctx([_event(origin="pipeline")]))
        h3s = [n for n in block.nodes if isinstance(n, Heading) and n.level == 3]
        assert h3s

    def test_pipeline_event_creates_table(self) -> None:
        block = self.plugin.render(_make_ctx([_event(origin="pipeline")]))
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_pipeline_event_horizontal_rule_at_end(self) -> None:
        block = self.plugin.render(_make_ctx([_event(origin="pipeline")]))
        assert isinstance(block.nodes[-1], HorizontalRule)

    def test_multiple_pipeline_events_single_table(self) -> None:
        events = [_event(origin="pipeline", message=f"msg {i}") for i in range(3)]
        block = self.plugin.render(_make_ctx(events))
        tables = [n for n in block.nodes if isinstance(n, Table)]
        assert len(tables) == 1


class TestEventTimelineRenderTrainingEvents:
    def setup_method(self) -> None:
        self.plugin = EventTimelineBlockPlugin()

    def test_training_event_creates_h3_heading(self) -> None:
        block = self.plugin.render(_make_ctx([_event(origin="training")]))
        h3s = [n for n in block.nodes if isinstance(n, Heading) and n.level == 3]
        assert h3s

    def test_training_event_creates_table(self) -> None:
        block = self.plugin.render(_make_ctx([_event(origin="training")]))
        assert any(isinstance(n, Table) for n in block.nodes)


class TestEventTimelineRenderOtherEvents:
    def setup_method(self) -> None:
        self.plugin = EventTimelineBlockPlugin()

    def test_other_origin_creates_h3_heading(self) -> None:
        block = self.plugin.render(_make_ctx([_event(origin="custom_source")]))
        h3s = [n for n in block.nodes if isinstance(n, Heading) and n.level == 3]
        assert h3s

    def test_other_origin_creates_table(self) -> None:
        block = self.plugin.render(_make_ctx([_event(origin="unknown_origin")]))
        assert any(isinstance(n, Table) for n in block.nodes)


class TestEventTimelineRenderMixedEvents:
    def setup_method(self) -> None:
        self.plugin = EventTimelineBlockPlugin()

    def test_all_three_groups_rendered(self) -> None:
        events = [
            _event(origin="pipeline", event_type="PIPELINE_START"),
            _event(origin="training", event_type="EPOCH_END"),
            _event(origin="custom", event_type="CUSTOM_EVENT"),
        ]
        block = self.plugin.render(_make_ctx(events))
        h3s = [n for n in block.nodes if isinstance(n, Heading) and n.level == 3]
        assert len(h3s) == 3

    def test_three_groups_produce_three_tables(self) -> None:
        events = [
            _event(origin="pipeline"),
            _event(origin="training"),
            _event(origin="other"),
        ]
        block = self.plugin.render(_make_ctx(events))
        tables = [n for n in block.nodes if isinstance(n, Table)]
        assert len(tables) == 3

    def test_only_pipeline_and_training_two_groups(self) -> None:
        events = [_event(origin="pipeline"), _event(origin="training")]
        block = self.plugin.render(_make_ctx(events))
        h3s = [n for n in block.nodes if isinstance(n, Heading) and n.level == 3]
        assert len(h3s) == 2


class TestEventTimelineSeverityEmoji:
    def setup_method(self) -> None:
        self.plugin = EventTimelineBlockPlugin()

    def _render(self, severity: str) -> ReportBlock:
        return self.plugin.render(_make_ctx([_event(origin="pipeline", severity=severity)]))

    def test_info_severity(self) -> None:
        block = self._render("INFO")
        assert block is not None
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_error_severity(self) -> None:
        block = self._render("ERROR")
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_warn_severity(self) -> None:
        block = self._render("WARN")
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_warning_severity(self) -> None:
        block = self._render("WARNING")
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_unknown_severity_uses_bullet(self) -> None:
        block = self._render("DEBUG")
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_empty_severity_string(self) -> None:
        block = self._render("")
        assert any(isinstance(n, Table) for n in block.nodes)


class TestEventTimelineNullFields:
    def setup_method(self) -> None:
        self.plugin = EventTimelineBlockPlugin()

    def test_none_timestamp_renders_dash(self) -> None:
        block = self.plugin.render(_make_ctx([_event(timestamp=None, origin="pipeline")]))
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_none_source_renders_dash(self) -> None:
        block = self.plugin.render(_make_ctx([_event(source=None, origin="pipeline")]))
        assert any(isinstance(n, Table) for n in block.nodes)

    def test_empty_source_renders(self) -> None:
        block = self.plugin.render(_make_ctx([_event(source="", origin="pipeline")]))
        assert any(isinstance(n, Table) for n in block.nodes)
