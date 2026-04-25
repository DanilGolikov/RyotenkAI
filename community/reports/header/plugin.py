from __future__ import annotations

from src.reports.document.nodes import (
    BlockQuote,
    DocBlock,
    Heading,
    HorizontalRule,
    Paragraph,
    br,
    code,
    inlines,
    strong,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext


class HeaderBlockPlugin(ReportPlugin):
    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        s = ctx.report.summary
        started = s.start_time.strftime("%Y-%m-%d %H:%M:%S") if s.start_time else "—"

        nodes: list[DocBlock] = [
            Heading(1, inlines(txt("🧪 Experiment Report: "), txt(s.run_name))),
            BlockQuote(
                (
                    Paragraph(
                        inlines(
                            strong("Run ID:"),
                            txt(" "),
                            code(s.run_id),
                            br(),
                            strong("Experiment:"),
                            txt(" "),
                            txt(s.experiment_name),
                            br(),
                            strong("Run Status:"),
                            txt(" "),
                            txt(f"{s.status.emoji} {s.status.value}"),
                            br(),
                            strong("Health:"),
                            txt(" "),
                            txt(f"{s.health.emoji} {s.health.value.upper()}"),
                            br(),
                            strong("Duration:"),
                            txt(" "),
                            txt(s.duration_formatted),
                            br(),
                            strong("Started:"),
                            txt(" "),
                            txt(started),
                        )
                    ),
                )
            ),
            HorizontalRule(),
        ]

        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["HeaderBlockPlugin"]
