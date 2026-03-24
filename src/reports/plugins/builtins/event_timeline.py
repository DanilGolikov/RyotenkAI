from __future__ import annotations

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


class EventTimelineBlockPlugin:
    plugin_id = "event_timeline"
    title = "Event Timeline"
    order = 100

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        timeline = ctx.report.timeline
        nodes: list[DocBlock] = [Heading(2, inlines(txt("📋 Event Timeline")))]

        if not timeline:
            nodes.extend([Paragraph(inlines(emph("No timeline events."))), HorizontalRule()])
            return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)

        def sev_emoji(sev: str) -> str:
            s = (sev or "").upper()
            return {"ERROR": "❌", "WARN": "⚠️", "WARNING": "⚠️", "INFO": "ℹ️"}.get(s, "•")

        def add_group(title: str, events) -> None:
            if not events:
                return
            nodes.append(Heading(3, inlines(txt(title))))
            headers = (
                inlines(txt("Time")),
                inlines(txt("Type")),
                inlines(txt("Source")),
                inlines(txt("Message")),
            )
            rows = []
            for e in events:
                ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else "—"
                rows.append(
                    [
                        inlines(txt(ts)),
                        inlines(txt(f"{sev_emoji(e.severity)} {e.event_type}")),
                        inlines(txt(e.source or "—")),
                        inlines(txt(e.message)),
                    ]
                )
            nodes.append(Table(headers=headers, rows=table_rows(rows)))

        pipeline_events = [e for e in timeline if e.origin == "pipeline"]
        training_events = [e for e in timeline if e.origin == "training"]
        other_events = [e for e in timeline if e.origin not in ("pipeline", "training")]

        add_group("Pipeline Events (Control Plane)", pipeline_events)
        add_group("Training Events (Data Plane)", training_events)
        add_group("Other Events", other_events)

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["EventTimelineBlockPlugin"]
