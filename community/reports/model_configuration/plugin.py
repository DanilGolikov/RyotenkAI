from __future__ import annotations

from src.reports.document.nodes import (
    DocBlock,
    Heading,
    HorizontalRule,
    Paragraph,
    Table,
    code,
    inlines,
    strong,
    table_rows,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext


def _format_number(value: int) -> str:
    """Format integer with thousands separator (dot)."""
    return f"{value:,}".replace(",", ".")


class ModelConfigurationBlockPlugin(ReportPlugin):
    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        m = ctx.report.model

        rows = [
            ("Training Type", m.training_type),
            ("Total Parameters", _format_number(m.total_parameters) if m.total_parameters else None),
            ("Trainable Parameters", _format_number(m.trainable_parameters) if m.trainable_parameters else None),
            ("Trainable %", f"{m.trainable_percent:.2f}%" if m.trainable_percent is not None else None),
            ("Loading Time", f"{m.loading_time_seconds:.2f}s" if m.loading_time_seconds is not None else None),
            ("Model Size", f"{m.model_size_mb:.1f} MB" if m.model_size_mb is not None else None),
            ("Quantization", m.quantization),
            ("Adapter Path", m.adapter_path),
            ("LoRA Rank", m.lora_rank),
            ("LoRA Alpha", m.lora_alpha),
            ("Target Modules", ", ".join(m.target_modules) if m.target_modules else None),
        ]

        table_rows_data = [[inlines(txt(k)), inlines(txt(str(v)))] for k, v in rows if v is not None and v != ""]

        nodes: list[DocBlock] = [
            Heading(2, inlines(txt("⚙️ Model Configuration"))),
            Paragraph(inlines(strong("Model:"), txt(" "), code(m.name or "—"))),
        ]

        if table_rows_data:
            nodes.append(
                Table(headers=(inlines(txt("Parameter")), inlines(txt("Value"))), rows=table_rows(table_rows_data))
            )

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["ModelConfigurationBlockPlugin"]
