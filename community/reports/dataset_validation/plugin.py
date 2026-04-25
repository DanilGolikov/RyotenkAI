from __future__ import annotations

import re
from collections import defaultdict

from src.reports.document.nodes import (
    BulletList,
    DocBlock,
    DocInline,
    Heading,
    HorizontalRule,
    Paragraph,
    Table,
    br,
    code,
    inlines,
    list_items,
    strong,
    table_rows,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext

_PASSED = "passed"
_MAX_REPORT_ERROR_EXAMPLES = 20
_SAMPLE_ERROR_RE = re.compile(r"^Sample\s+(?P<index>\d+):\s+(?P<message>.+)$")
_GROUPED_INDEX_ERROR_RE = re.compile(r"^(?P<label>[^:\n]+): \[(?P<body>[^\]]*)\]$")
_GROUPED_SAMPLES_ERROR_RE = re.compile(r"^(?P<label>Samples?) (?P<body>\d+(?:,\d+)*) - (?P<message>.+)$")


class DatasetValidationBlockPlugin(ReportPlugin):
    plugin_id = "dataset_validation"
    title = "Dataset Validation"
    order = 40

    @staticmethod
    def _format_kv_lines(d: dict) -> tuple[DocInline, ...]:
        if not d:
            return (txt("—"),)

        parts: list[DocInline] = []
        for idx, (k, v) in enumerate(d.items()):
            nice_key = str(k).replace("_", " ").title()
            if isinstance(v, float):
                val = f"{v:.2f}"
            else:
                val = str(v)

            parts.append(txt(f"{nice_key}: {val}"))
            if idx < len(d) - 1:
                parts.append(br())

        return tuple(parts)

    @staticmethod
    def _truncate_examples(values: list[str], *, separator: str) -> tuple[str, int]:
        if len(values) <= _MAX_REPORT_ERROR_EXAMPLES:
            return separator.join(values), 0

        visible_values = values[:_MAX_REPORT_ERROR_EXAMPLES]
        hidden_count = len(values) - _MAX_REPORT_ERROR_EXAMPLES
        return separator.join(visible_values), hidden_count

    @classmethod
    def _truncate_grouped_error(cls, error: str) -> str:
        grouped_match = _GROUPED_INDEX_ERROR_RE.match(error)
        if grouped_match:
            values = [part.strip() for part in grouped_match.group("body").split(",") if part.strip()]
            joined_values, hidden_count = cls._truncate_examples(values, separator=", ")
            suffix = f" ... (+{hidden_count} more errors)" if hidden_count else ""
            return f"{grouped_match.group('label')}: [{joined_values}]{suffix}"

        samples_match = _GROUPED_SAMPLES_ERROR_RE.match(error)
        if samples_match:
            values = [part.strip() for part in samples_match.group("body").split(",") if part.strip()]
            joined_values, hidden_count = cls._truncate_examples(values, separator=",")
            suffix = f" ... (+{hidden_count} more errors)" if hidden_count else ""
            return f"{samples_match.group('label')} {joined_values} - {samples_match.group('message')}{suffix}"

        return error

    @classmethod
    def _format_errors(cls, errors: list[str]) -> tuple[DocInline, ...]:
        if not errors:
            return (txt("—"),)

        grouped_sample_errors: dict[str, list[int]] = defaultdict(list)
        other_errors: list[str] = []

        for error in errors:
            match = _SAMPLE_ERROR_RE.match(error)
            if not match:
                other_errors.append(error)
                continue

            grouped_sample_errors[match.group("message")].append(int(match.group("index")))

        normalized_errors: list[str] = []
        normalized_errors.extend(other_errors)

        for message, indices in grouped_sample_errors.items():
            sorted_indices = sorted(indices)
            label = "Sample" if len(sorted_indices) == 1 else "Samples"
            joined_indices = ",".join(str(index) for index in sorted_indices)
            normalized_errors.append(f"{label} {joined_indices} - {message}")

        parts: list[DocInline] = []
        for i, e in enumerate(normalized_errors):
            parts.append(txt(cls._truncate_grouped_error(e)))
            if i < len(normalized_errors) - 1:
                parts.append(br())
        return tuple(parts)

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        validation = ctx.report.validation

        nodes: list[DocBlock] = [Heading(2, inlines(txt("Dataset Validation")))]

        if not validation or not getattr(validation, "datasets", None):
            nodes.extend(
                [
                    Paragraph(inlines(txt("No dataset validation data available."))),
                    HorizontalRule(),
                ]
            )
            return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)

        emoji = "✅" if validation.overall_status == _PASSED else "❌"
        status_text = "Passed" if validation.overall_status == _PASSED else "Failed"
        if validation.overall_status == "unknown":
            emoji = "❓"
            status_text = "Unknown"

        nodes.append(Paragraph(inlines(strong("Overall Status:"), txt(f" {emoji} {status_text}"))))
        nodes.append(
            Paragraph(
                inlines(
                    strong("Validated Datasets:"),
                    txt(f" {validation.passed_datasets}/{validation.total_datasets}"),
                )
            )
        )

        # Unique plugins table (if descriptions exist)
        unique_plugins: dict[str, str] = {}
        for dataset_val in validation.datasets:
            for plugin_result in dataset_val.plugin_results:
                if plugin_result.plugin_name not in unique_plugins and plugin_result.description:
                    unique_plugins[plugin_result.plugin_name] = plugin_result.description

        if unique_plugins:
            nodes.append(Heading(3, inlines(txt("Plugins in use"))))
            headers_plugins = (inlines(txt("Plugin")), inlines(txt("Description")))
            rows = [[inlines(txt(name)), inlines(txt(desc))] for name, desc in sorted(unique_plugins.items())]
            nodes.append(Table(headers=headers_plugins, rows=table_rows(rows)))

        nodes.append(HorizontalRule())

        for idx, dataset_val in enumerate(validation.datasets, 1):
            d = dataset_val

            # Dataset header
            nodes.append(Heading(3, inlines(txt(f"{idx}. Dataset: "), code(d.dataset_name))))

            # Status line (+ partial failure warning)
            if d.status == "passed":
                st_emoji = "✅"
                st_text = "Passed"
                warning_text = None
            elif d.has_partial_failure:
                st_emoji = "⚠️"
                st_text = "Partial Failure"
                warning_text = (
                    f"⚠️ Warning: {d.failed_plugins} plugin(s) did not pass validation, "
                    f"but this is below the critical threshold ({d.critical_failures}). "
                    "The pipeline will continue; review the data when possible."
                )
            else:
                st_emoji = "❌"
                st_text = "Failed"
                warning_text = None

            nodes.append(Paragraph(inlines(strong("Status:"), txt(f" {st_emoji} {st_text}"))))
            if warning_text:
                nodes.append(Paragraph(inlines(txt(warning_text))))

            # Summary table
            headers_kv = (inlines(txt("Parameter")), inlines(txt("Value")))
            rows = []
            if d.dataset_path:
                rows.append([inlines(txt("Path")), inlines(code(d.dataset_path))])
            if d.sample_count:
                rows.append([inlines(txt("Sample count")), inlines(txt(str(d.sample_count)))])

            rows.append([inlines(txt("Plugins passed")), inlines(txt(f"{d.passed_plugins} of {d.total_plugins}"))])
            if d.failed_plugins > 0:
                rows.append([inlines(txt("Plugins failed")), inlines(txt(str(d.failed_plugins)))])
            if d.critical_failures > 0:
                rows.append([inlines(txt("Critical failure threshold")), inlines(txt(str(d.critical_failures)))])

            nodes.append(Table(headers=headers_kv, rows=table_rows(rows)))

            # Plugin results
            if d.plugin_results:
                nodes.append(Paragraph(inlines(strong("Plugin results:"))))

                headers_results = (
                    inlines(txt("Plugin")),
                    inlines(txt("Status")),
                    inlines(txt("Duration")),
                    inlines(txt("Params")),
                    inlines(txt("Thresholds")),
                    inlines(txt("Metrics")),
                    inlines(txt("Errors")),
                )

                rows = []
                for p in d.plugin_results:
                    p_emoji = "✅" if p.status == "passed" else "❌"
                    duration = f"{p.duration_ms:.1f}ms"

                    rows.append(
                        [
                            inlines(
                                txt(p_emoji),
                                br(),
                                strong("id"),
                                txt(f": {p.id}"),
                                br(),
                                strong("plugin"),
                                txt(f": {p.plugin_name}"),
                            ),
                            inlines(txt(p.status)),
                            inlines(txt(duration)),
                            self._format_kv_lines(p.params),
                            self._format_kv_lines(p.thresholds),
                            self._format_kv_lines(
                                {
                                    k: v
                                    for k, v in p.metrics.items()
                                    if k
                                    not in (
                                        "threshold",
                                        "min_threshold",
                                        "max_threshold",
                                        "min_score",
                                        "max_duplicate_ratio",
                                    )
                                }
                            ),
                            self._format_errors(p.errors),
                        ]
                    )

                nodes.append(Table(headers=headers_results, rows=table_rows(rows)))

            # Recommendations
            recs_by_plugin = [(p.id, p.recommendations) for p in d.plugin_results if p.recommendations]
            nodes.append(Paragraph(inlines(strong("Recommendations:"))))
            if recs_by_plugin:
                for plugin_name, recs in recs_by_plugin:
                    nodes.append(Paragraph(inlines(txt(f"[{plugin_name}]"))))
                    nodes.append(BulletList(items=list_items([[txt(r)] for r in recs])))
            else:
                nodes.append(Paragraph(inlines(txt("All checks passed. No recommendations."))))

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["DatasetValidationBlockPlugin"]
