from __future__ import annotations

from src.reports.core.constants import MarkdownSymbols
from src.reports.document.nodes import (
    DocBlock,
    Heading,
    HorizontalRule,
    Paragraph,
    Table,
    inlines,
    strong,
    table_rows,
    txt,
)
from src.reports.models.report import RunStatus
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext


class SummaryBlockPlugin(ReportPlugin):
    @staticmethod
    def _status_description(status: RunStatus) -> str:
        return {
            RunStatus.FINISHED: "finished successfully",
            RunStatus.FAILED: "failed",
            RunStatus.RUNNING: "running",
            RunStatus.KILLED: "was stopped",
            RunStatus.UNKNOWN: "unknown",
        }.get(status, "unknown")

    @staticmethod
    def _format_train_loss_cell(p) -> str:
        t_trend = p.train_loss_trend
        if t_trend and t_trend.data_points > 0:
            t_start = f"{t_trend.first:.4f}" if t_trend.first is not None else MarkdownSymbols.DASH
            t_end = f"{t_trend.last:.4f}" if t_trend.last is not None else MarkdownSymbols.DASH
            t_icon = {
                "decreased": "↘️",
                "increased": "↗️",
                "stable": "➡️",
            }.get(t_trend.direction, "")
            return f"{t_start} → {t_end} {t_icon}".strip()
        if p.final_loss is not None:
            return f"{p.final_loss:.4f}"
        return MarkdownSymbols.DASH

    @staticmethod
    def _format_step_loss_cell(p) -> str:
        s_trend = p.loss_trend
        if s_trend and s_trend.data_points > 0:
            s_start = f"{s_trend.first:.4f}" if s_trend.first is not None else MarkdownSymbols.DASH
            s_end = f"{s_trend.last:.4f}" if s_trend.last is not None else MarkdownSymbols.DASH
            return f"{s_start} → {s_end}"
        return MarkdownSymbols.DASH

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        s = ctx.report.summary
        phases = ctx.report.phases
        status_text = self._status_description(s.status)
        chain_str = " → ".join(s.strategy_chain) if s.strategy_chain else MarkdownSymbols.DASH

        nodes: list[DocBlock] = [
            Heading(2, inlines(txt("📊 Summary"))),
            Paragraph(
                inlines(
                    txt(f"{s.health.emoji} "),
                    strong(f"Experiment {status_text}"),
                    txt(f" (status: {s.status.value})."),
                )
            ),
            Paragraph(inlines(strong("Health Status:"), txt(f" {s.health.value.upper()} — {s.health_explanation}"))),
            Paragraph(inlines(txt("⏱️ "), strong("Total duration:"), txt(f" {s.duration_formatted}"))),
            Paragraph(inlines(txt("🔗 "), strong("Strategy chain:"), txt(f" {chain_str}"))),
        ]

        if phases:
            nodes.append(Heading(3, inlines(txt("📈 Metrics by strategy"))))
            headers = (
                inlines(txt("#")),
                inlines(txt("Strategy")),
                inlines(txt("Train Loss (Start→End)")),
                inlines(txt("Step Loss (Start→End)")),
                inlines(txt("Epochs")),
                inlines(txt("Steps")),
                inlines(txt("Time")),
            )
            rows = []
            for i, p in enumerate(phases, 1):
                duration_str = f"{p.duration_seconds:.1f}s" if p.duration_seconds else MarkdownSymbols.DASH
                epochs_str = str(p.epochs) if p.epochs is not None else MarkdownSymbols.DASH
                steps_str = str(p.steps) if p.steps is not None else MarkdownSymbols.DASH

                rows.append(
                    [
                        inlines(txt(str(i))),
                        inlines(txt(p.strategy)),
                        inlines(txt(self._format_train_loss_cell(p))),
                        inlines(txt(self._format_step_loss_cell(p))),
                        inlines(txt(epochs_str)),
                        inlines(txt(steps_str)),
                        inlines(txt(duration_str)),
                    ]
                )

            nodes.append(Table(headers=headers, rows=table_rows(rows)))

        nodes.append(HorizontalRule())

        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["SummaryBlockPlugin"]
