"""Stage Timeline Block Plugin — 2-column table: Stage | Duration."""

from __future__ import annotations

from types import MappingProxyType

from src.reports.document.nodes import (
    DocBlock,
    Heading,
    HorizontalRule,
    Paragraph,
    Table,
    emph,
    inlines,
    table_rows,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext

_STATUS_EMOJI: MappingProxyType[str, str] = MappingProxyType(
    {
        "passed": "✅",
        "failed": "❌",
        "skipped": "⏭️",
        "interrupted": "⚠️",
    }
)


def _status_emoji(status: str) -> str:
    return _STATUS_EMOJI.get((status or "").lower(), "❓")


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


class StageTimelineBlockPlugin:
    """Renders a 2-column table: Stage | Duration from stage artifact envelopes."""

    plugin_id = "stage_timeline"
    title = "Pipeline Stages"
    order = 100

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        timeline = ctx.report.timeline
        nodes: list[DocBlock] = [Heading(2, inlines(txt("📋 Pipeline Stages")))]

        if not timeline:
            nodes.extend([Paragraph(inlines(emph("No pipeline stage data available."))), HorizontalRule()])
            return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)

        headers = (inlines(txt("Stage")), inlines(txt("Duration")))
        rows = []
        for event in timeline:
            attrs = event.attributes or {}
            stage_name = attrs.get("stage") or event.source or "—"
            status = attrs.get("status") or ""
            duration_seconds = float(attrs.get("duration_seconds") or 0.0)

            emoji = _status_emoji(status)
            display_name = stage_name.replace("_", " ").title()

            rows.append(
                [
                    inlines(txt(f"{emoji} {display_name}")),
                    inlines(txt(_fmt_duration(duration_seconds))),
                ]
            )

        nodes.append(Table(headers=headers, rows=table_rows(rows)))
        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["StageTimelineBlockPlugin"]
