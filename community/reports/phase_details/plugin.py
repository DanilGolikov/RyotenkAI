from __future__ import annotations

from src.reports.core.constants import MarkdownSymbols
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


class PhaseDetailsBlockPlugin:
    plugin_id = "phase_details"
    title = "Phase Details"
    order = 80

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        phases = ctx.report.phases
        nodes: list[DocBlock] = [Heading(2, inlines(txt("🔄 Phase Details")))]

        if not phases:
            nodes.extend([Paragraph(inlines(emph("No training phase data available."))), HorizontalRule()])
            return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)

        headers = (
            inlines(txt("#")),
            inlines(txt("Strategy")),
            inlines(txt("Status")),
            inlines(txt("Duration")),
            inlines(txt("Final Loss")),
            inlines(txt("Train Loss Trend")),
            inlines(txt("GPU util")),
            inlines(txt("GPU mem")),
            inlines(txt("CPU util")),
            inlines(txt("RAM")),
        )
        rows = []
        for p in phases:
            dur = f"{p.duration_seconds:.1f}s" if p.duration_seconds is not None else MarkdownSymbols.DASH
            fin = f"{p.final_loss:.4f}" if p.final_loss is not None else MarkdownSymbols.DASH
            if p.train_loss_trend and p.train_loss_trend.data_points > 0:
                icon = {"decreased": "↘️", "increased": "↗️", "stable": "➡️"}.get(p.train_loss_trend.direction, "")
                t = f"{p.train_loss_trend.first:.4f} → {p.train_loss_trend.last:.4f} {icon}".strip()
            else:
                t = MarkdownSymbols.DASH

            gpu_u = f"{p.gpu_utilization:.1f}%" if p.gpu_utilization is not None else MarkdownSymbols.DASH
            gpu_m = (
                f"{p.gpu_memory_mb:.0f}MB ({p.gpu_memory_percent:.1f}%)"
                if p.gpu_memory_mb is not None and p.gpu_memory_percent is not None
                else MarkdownSymbols.DASH
            )
            cpu_u = f"{p.cpu_utilization:.1f}%" if p.cpu_utilization is not None else MarkdownSymbols.DASH
            ram = (
                f"{p.system_memory_mb:.0f}MB ({p.system_memory_percent:.1f}%)"
                if p.system_memory_mb is not None and p.system_memory_percent is not None
                else MarkdownSymbols.DASH
            )

            rows.append(
                [
                    inlines(txt(str(p.phase_idx))),
                    inlines(txt(p.strategy)),
                    inlines(txt(f"{p.status.emoji} {p.status.value}")),
                    inlines(txt(dur)),
                    inlines(txt(fin)),
                    inlines(txt(t)),
                    inlines(txt(gpu_u)),
                    inlines(txt(gpu_m)),
                    inlines(txt(cpu_u)),
                    inlines(txt(ram)),
                ]
            )

        nodes.append(Table(headers=headers, rows=table_rows(rows)))
        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["PhaseDetailsBlockPlugin"]
