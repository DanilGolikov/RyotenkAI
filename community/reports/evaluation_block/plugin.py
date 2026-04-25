"""Evaluation Block Plugin — displays model evaluation results."""

from __future__ import annotations

from src.reports.document.nodes import (
    BulletList,
    DocBlock,
    DocInline,
    Heading,
    HorizontalRule,
    Paragraph,
    Table,
    br,
    inlines,
    list_items,
    strong,
    table_rows,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext

_EMPTY_CELL = "—"  # noqa: WPS226


class EvaluationBlockPlugin(ReportPlugin):
    """Renders evaluation_results.json data in the experiment report."""

    plugin_id = "evaluation_block"
    title = "Model Evaluation"
    order = 45

    @staticmethod
    def _format_kv(d: dict) -> tuple[DocInline, ...]:
        if not d:
            return (txt(_EMPTY_CELL),)
        parts: list[DocInline] = []
        for idx, (k, v) in enumerate(d.items()):
            nice_key = str(k).replace("_", " ").title()
            val = f"{v:.2f}" if isinstance(v, float) else str(v)
            parts.append(txt(f"{nice_key}: {val}"))
            if idx < len(d) - 1:
                parts.append(br())
        return tuple(parts)

    @staticmethod
    def _format_metrics(metrics: dict) -> tuple[DocInline, ...]:
        if not metrics:
            return (txt(_EMPTY_CELL),)
        parts: list[DocInline] = []
        for idx, (k, v) in enumerate(metrics.items()):
            parts.append(txt(f"{k}: {v:.4f}"))
            if idx < len(metrics) - 1:
                parts.append(br())
        return tuple(parts)

    @staticmethod
    def _format_lines(items: list[str]) -> tuple[DocInline, ...]:
        if not items:
            return (txt(_EMPTY_CELL),)
        parts: list[DocInline] = []
        for idx, item in enumerate(items):
            parts.append(txt(item))
            if idx < len(items) - 1:
                parts.append(br())
        return tuple(parts)

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        evaluation = ctx.report.evaluation
        nodes: list[DocBlock] = [Heading(2, inlines(txt("Model Evaluation")))]

        if evaluation is None:
            nodes.extend(
                [
                    Paragraph(
                        inlines(txt("No model evaluation data (evaluation did not run or results were not saved)."))
                    ),
                    HorizontalRule(),
                ]
            )
            return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)

        overall_emoji = "✅" if evaluation.overall_passed else "❌"
        overall_text = "Passed" if evaluation.overall_passed else "Failed"

        nodes.append(Paragraph(inlines(strong("Overall Status:"), txt(f" {overall_emoji} {overall_text}"))))
        nodes.append(Paragraph(inlines(strong("Samples Evaluated:"), txt(f" {evaluation.sample_count}"))))
        nodes.append(Paragraph(inlines(strong("Duration:"), txt(f" {evaluation.duration_seconds:.1f}s"))))

        if evaluation.skipped_plugins:
            nodes.append(
                Paragraph(inlines(strong("Skipped Plugins:"), txt(f" {', '.join(evaluation.skipped_plugins)}")))
            )

        if evaluation.errors:
            nodes.append(Paragraph(inlines(strong("Errors:"))))
            for err in evaluation.errors:
                nodes.append(Paragraph(inlines(txt(f"  • {err}"))))

        # Unique plugins table with descriptions (mirrors DatasetValidationBlockPlugin)
        unique_plugins: dict[str, tuple[str, str]] = {}  # instance id -> (plugin_name, description)
        if evaluation.plugins:
            for plugin in evaluation.plugins:
                if plugin.name not in unique_plugins and (plugin.description or plugin.plugin_name):
                    unique_plugins[plugin.name] = (plugin.plugin_name, plugin.description)

        if unique_plugins:
            nodes.append(Heading(3, inlines(txt("Plugins in use"))))
            headers_plugins = (
                inlines(txt("Plugin")),
                inlines(txt("Description")),
            )
            rows = [
                [
                    inlines(
                        strong("id"),
                        txt(f": {instance_id}"),
                        br(),
                        strong("plugin"),
                        txt(f": {plugin_name or _EMPTY_CELL}"),
                    ),
                    inlines(txt(desc or _EMPTY_CELL)),
                ]
                for instance_id, (plugin_name, desc) in sorted(unique_plugins.items())
            ]
            nodes.append(Table(headers=headers_plugins, rows=table_rows(rows)))

        if evaluation.plugins:
            nodes.append(Heading(3, inlines(txt("Plugin Results"))))

            headers = (
                inlines(txt("Plugin")),
                inlines(txt("Status")),
                inlines(txt("Samples")),
                inlines(txt("Params")),
                inlines(txt("Thresholds")),
                inlines(txt("Metrics")),
                inlines(txt("Errors")),
            )
            rows = []
            for plugin in evaluation.plugins:
                p_emoji = "✅" if plugin.passed else "❌"
                rows.append(
                    [
                        inlines(
                            txt(p_emoji),
                            br(),
                            strong("id"),
                            txt(f": {plugin.name}"),
                            br(),
                            strong("plugin"),
                            txt(f": {plugin.plugin_name or _EMPTY_CELL}"),
                        ),
                        inlines(txt("passed" if plugin.passed else "failed")),
                        inlines(txt(f"{plugin.sample_count} ({plugin.failed_samples} failed)")),
                        self._format_kv(plugin.params),
                        self._format_kv(plugin.thresholds),
                        self._format_metrics(plugin.metrics),
                        self._format_lines(plugin.errors),
                    ]
                )

            nodes.append(Table(headers=headers, rows=table_rows(rows)))

            # Recommendations section (mirrors DatasetValidationBlockPlugin)
            recs_by_plugin = [(p.name, p.recommendations) for p in evaluation.plugins if p.recommendations]
            nodes.append(Paragraph(inlines(strong("Recommendations:"))))
            if recs_by_plugin:
                for plugin_name, recs in recs_by_plugin:
                    nodes.append(Paragraph(inlines(txt(f"[{plugin_name}]"))))
                    nodes.append(BulletList(items=list_items([[txt(r)] for r in recs])))
            else:
                nodes.append(Paragraph(inlines(txt("All checks passed. No recommendations."))))

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["EvaluationBlockPlugin"]
