from __future__ import annotations

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
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext
from src.reports.plugins.registry import ReportPluginRegistry


@ReportPluginRegistry.register
class TrainingConfigurationBlockPlugin:
    plugin_id = "training_configuration"
    title = "Training Configuration"
    order = 70

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        report = ctx.report
        c = report.config
        chain = " → ".join(report.summary.strategy_chain) if report.summary.strategy_chain else "—"

        def row(k: str, v) -> list:
            if v is None or v == "":
                return []
            return [inlines(txt(k)), inlines(txt(str(v)))]

        rows = []
        rows.append([inlines(txt("Strategy Chain")), inlines(txt(chain))])
        for r in [
            row("Batch Size", c.batch_size),
            row("Gradient Accumulation", c.grad_accum),
            row("Learning Rate", f"{c.learning_rate:g}" if c.learning_rate is not None else None),
            row("Num Epochs", c.num_epochs),
            row("Max Steps", c.max_steps),
            row("Optimizer", c.optimizer),
            row("Scheduler", c.scheduler),
            row("Warmup Ratio", c.warmup_ratio),
            row("Weight Decay", c.weight_decay),
            row("Max Seq Length", c.max_seq_length),
            row("FP16", c.fp16),
            row("BF16", c.bf16),
            row("Gradient Checkpointing", c.gradient_checkpointing),
        ]:
            if r:
                rows.append(r)

        nodes: list[DocBlock] = [
            Heading(2, inlines(txt("📚 Training Configuration"))),
            Paragraph(inlines(strong("Effective training hyperparameters (best-effort)."))),
        ]

        if rows:
            nodes.append(Table(headers=(inlines(txt("Parameter")), inlines(txt("Value"))), rows=table_rows(rows)))

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["TrainingConfigurationBlockPlugin"]
