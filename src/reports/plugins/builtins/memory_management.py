from __future__ import annotations

from src.reports.core.constants import MarkdownSymbols
from src.reports.document.nodes import (
    BulletList,
    DocBlock,
    Heading,
    HorizontalRule,
    Paragraph,
    Table,
    inlines,
    list_items,
    strong,
    table_rows,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext
from src.reports.plugins.registry import ReportPluginRegistry


@ReportPluginRegistry.register
class MemoryManagementBlockPlugin:
    plugin_id = "memory_management"
    title = "Memory Management"
    order = 60

    @staticmethod
    def _fmt(v) -> str:
        if v is None:
            return MarkdownSymbols.DASH
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    @classmethod
    def _events_table(cls, title: str, events) -> list[DocBlock]:
        if not events:
            return [Paragraph(inlines(txt(f"{title}: {MarkdownSymbols.DASH}")))]

        nodes: list[DocBlock] = [Heading(3, inlines(txt(title)))]
        headers = (
            inlines(txt("Time")),
            inlines(txt("Phase")),
            inlines(txt("Type")),
            inlines(txt("Util%")),
            inlines(txt("Freed MB")),
            inlines(txt("Message")),
        )
        rows = []
        for e in events:
            ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else MarkdownSymbols.DASH
            phase = e.phase or MarkdownSymbols.DASH
            util = f"{e.utilization_percent:.1f}%" if e.utilization_percent is not None else MarkdownSymbols.DASH
            freed = str(e.freed_mb) if e.freed_mb is not None else MarkdownSymbols.DASH
            rows.append(
                [
                    inlines(txt(ts)),
                    inlines(txt(phase)),
                    inlines(txt(e.event_type)),
                    inlines(txt(util)),
                    inlines(txt(freed)),
                    inlines(txt(e.message)),
                ]
            )
        nodes.append(Table(headers=headers, rows=table_rows(rows)))
        return nodes

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        mm = ctx.report.memory_management

        nodes: list[DocBlock] = [Heading(2, inlines(txt("💾 Memory Management")))]

        if not mm:
            nodes.extend(
                [
                    Paragraph(inlines(txt("MemoryManager was not activated or there are no events."))),
                    HorizontalRule(),
                ]
            )
            return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)

        # Analysis (optional)
        if mm.analysis:
            a = mm.analysis
            nodes.append(Heading(3, inlines(txt("🩺 Memory Health Analysis"))))
            nodes.append(Paragraph(inlines(txt(f"{a.status.emoji} "), strong(a.verdict))))

            analysis_rows = [
                [inlines(txt("Efficiency Score")), inlines(txt(str(a.efficiency_score)))],
                [inlines(txt("Overhead (seconds)")), inlines(txt(f"{a.overhead_seconds:.2f}"))],
                [inlines(txt("Fragmentation Warnings")), inlines(txt(str(a.fragmentation_warnings)))],
                [inlines(txt("OOM Count")), inlines(txt(str(a.oom_count)))],
            ]
            nodes.append(Table(headers=(inlines(txt("Metric")), inlines(txt("Value"))), rows=table_rows(analysis_rows)))

            if a.insights:
                nodes.append(Paragraph(inlines(strong("Insights:"))))
                nodes.append(BulletList(items=list_items([[txt(x)] for x in a.insights])))
            if a.recommendations:
                nodes.append(Paragraph(inlines(strong("Recommendations:"))))
                nodes.append(BulletList(items=list_items([[txt(x)] for x in a.recommendations])))

        # GPU & configuration
        nodes.append(Heading(3, inlines(txt("GPU & Configuration"))))
        kv_pairs = [
            ("GPU", mm.gpu_name),
            ("GPU Tier", mm.gpu_tier),
            ("Total VRAM (GB)", mm.total_vram_gb),
            ("Memory Margin (MB)", mm.memory_margin_mb),
            ("Critical Threshold", mm.critical_threshold),
            ("Warning Threshold", mm.warning_threshold),
            ("Max Retries", mm.max_retries),
            ("Max Model", mm.max_model),
            ("Actual Model Size", mm.actual_model_size),
        ]
        kv_rows = [[inlines(txt(k)), inlines(txt(self._fmt(v)))] for k, v in kv_pairs if v is not None]
        if kv_rows:
            nodes.append(Table(headers=(inlines(txt("Parameter")), inlines(txt("Value"))), rows=table_rows(kv_rows)))

        if mm.notes:
            nodes.append(Paragraph(inlines(strong("Notes:"), txt(f" {mm.notes}"))))

        if mm.config_warnings:
            nodes.append(Paragraph(inlines(strong("Config Warnings:"))))
            nodes.append(BulletList(items=list_items([[txt(x)] for x in mm.config_warnings])))

        # Events
        nodes.extend(self._events_table("Cache Clears", mm.cache_clears))
        nodes.extend(self._events_table("OOM Events", mm.oom_events))
        nodes.extend(self._events_table("Memory Warnings", mm.memory_warnings))

        # Phase stats
        if mm.phase_stats:
            nodes.append(Heading(3, inlines(txt("Memory by Phase"))))
            headers = (
                inlines(txt("#")),
                inlines(txt("Strategy")),
                inlines(txt("Peak MB")),
                inlines(txt("Peak %")),
                inlines(txt("Avg MB")),
                inlines(txt("Avg %")),
                inlines(txt("P95 MB")),
                inlines(txt("P99 MB")),
            )
            phase_rows = []
            for p in mm.phase_stats:
                phase_rows.append(
                    [
                        inlines(txt(str(p.phase_idx))),
                        inlines(txt(p.strategy)),
                        inlines(txt(str(p.peak_mb))),
                        inlines(txt(f"{p.peak_percent:.1f}%")),
                        inlines(txt(str(p.avg_mb))),
                        inlines(txt(f"{p.avg_percent:.1f}%")),
                        inlines(txt(str(p.p95_mb))),
                        inlines(txt(str(p.p99_mb))),
                    ]
                )
            nodes.append(Table(headers=headers, rows=table_rows(phase_rows)))

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["MemoryManagementBlockPlugin"]
