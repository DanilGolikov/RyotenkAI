from __future__ import annotations

from src.reports.document.nodes import DocBlock, Paragraph, emph, inlines
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext


class FooterBlockPlugin(ReportPlugin):
    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        now = ctx.clock().strftime("%Y-%m-%d %H:%M:%S")
        nodes: list[DocBlock] = [Paragraph(inlines(emph(f"Generated automatically by RyotenkAI at {now}")))]
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["FooterBlockPlugin"]
