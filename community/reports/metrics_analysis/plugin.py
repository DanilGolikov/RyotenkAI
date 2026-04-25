from __future__ import annotations

from src.reports.core.constants import MarkdownSymbols
from src.reports.document.nodes import (
    BlockQuote,
    DocBlock,
    DocInline,
    Heading,
    HorizontalRule,
    Paragraph,
    Table,
    code,
    emph,
    inlines,
    strong,
    table_rows,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext


class MetricsAnalysisBlockPlugin(ReportPlugin):
    plugin_id = "metrics_analysis"
    title = "Metrics Analysis"
    order = 90

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        phases = ctx.report.phases
        resources = ctx.report.resources

        nodes: list[DocBlock] = [Heading(2, inlines(txt("🔬 Metrics Analysis")))]

        # Metric docs (best-effort)
        metric_docs: dict[str, tuple[str, str]] = {}
        for p in phases:
            for m in p.metrics_analysis:
                metric_docs.setdefault(m.name, (m.display_name, m.description))

        nodes.append(Heading(3, inlines(txt("📖 Metric descriptions"))))
        if metric_docs:
            quote_lines: list[DocBlock] = []
            for name, (display, desc) in sorted(metric_docs.items()):
                quote_lines.append(
                    Paragraph(
                        inlines(
                            strong(display),
                            txt(" ("),
                            code(name),
                            txt(f") — {desc}"),
                        )
                    )
                )
            nodes.append(BlockQuote(tuple(quote_lines)))
        else:
            nodes.append(Paragraph(inlines(emph("No metric descriptions."))))

        nodes.append(HorizontalRule())

        # Per-phase deep dive
        for p in phases:
            nodes.append(Heading(3, inlines(txt(f"📉 Training Deep Dive: {p.display_name}"))))
            if not p.metrics_analysis:
                nodes.append(Paragraph(inlines(emph("No detailed metrics."))))
                continue

            headers_metrics: tuple[tuple[DocInline, ...], ...] = (
                inlines(txt("Metric")),
                inlines(txt("Trend")),
                inlines(txt("Status")),
                inlines(txt("Verdict")),
            )
            rows = []
            for m in p.metrics_analysis:
                t = m.trend
                if t and t.data_points > 0 and t.first is not None and t.last is not None:
                    change = f"{t.change_pct:+.1f}%" if t.change_pct is not None else MarkdownSymbols.DASH
                    icon = {"decreased": "↘️", "increased": "↗️", "stable": "➡️"}.get(t.direction, "")
                    trend_str = f"{t.first:.4f} → {t.last:.4f} ({change}) {icon}".strip()
                else:
                    trend_str = MarkdownSymbols.DASH

                rows.append(
                    [
                        inlines(txt(m.display_name)),
                        inlines(txt(trend_str)),
                        inlines(txt(f"{m.status.emoji} {m.status.value}")),
                        inlines(txt(m.verdict)),
                    ]
                )

            nodes.append(Table(headers=headers_metrics, rows=table_rows(rows)))

        nodes.append(HorizontalRule())

        # Resource utilization
        nodes.append(Heading(3, inlines(txt("🖥️ Resource Utilization (from charts)"))))

        def stats_row(label: str, s) -> list[tuple[DocInline, ...]]:
            if not s or s.data_points <= 0:
                return [
                    inlines(txt(label)),
                    inlines(txt(MarkdownSymbols.DASH)),
                    inlines(txt(MarkdownSymbols.DASH)),
                    inlines(txt(MarkdownSymbols.DASH)),
                    inlines(txt(MarkdownSymbols.DASH)),
                    inlines(txt(MarkdownSymbols.DASH)),
                ]
            return [
                inlines(txt(label)),
                inlines(txt(f"{s.avg:.2f}" if s.avg is not None else MarkdownSymbols.DASH)),
                inlines(txt(f"{s.min_val:.2f}" if s.min_val is not None else MarkdownSymbols.DASH)),
                inlines(txt(f"{s.max_val:.2f}" if s.max_val is not None else MarkdownSymbols.DASH)),
                inlines(txt(f"{s.p95:.2f}" if s.p95 is not None else MarkdownSymbols.DASH)),
                inlines(txt(f"{s.p99:.2f}" if s.p99 is not None else MarkdownSymbols.DASH)),
            ]

        headers_resources: tuple[tuple[DocInline, ...], ...] = (
            inlines(txt("Metric")),
            inlines(txt("Avg")),
            inlines(txt("Min")),
            inlines(txt("Max")),
            inlines(txt("P95")),
            inlines(txt("P99")),
        )
        resource_rows: list[list[tuple[DocInline, ...]]] = [
            stats_row("GPU Util (%)", resources.gpu_utilization),
            stats_row("GPU Memory (MB)", resources.gpu_memory_mb),
            stats_row("GPU Memory (%)", resources.gpu_memory_percent),
            stats_row("CPU Util (%)", resources.cpu_utilization),
            stats_row("RAM (MB)", resources.system_memory_mb),
            stats_row("RAM (%)", resources.system_memory_percent),
        ]
        nodes.append(Table(headers=headers_resources, rows=table_rows(resource_rows)))

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["MetricsAnalysisBlockPlugin"]
