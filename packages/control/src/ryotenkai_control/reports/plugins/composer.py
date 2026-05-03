"""
ReportComposer: orchestrates report block plugins.

Fail-open policy:
- Any plugin exception results in a stub error block in the report,
  and a full traceback in logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.reports.document.nodes import DocBlock, Heading, HorizontalRule, Paragraph, inlines, strong, txt
from src.reports.plugins.interfaces import (
    IReportBlockPlugin,
    PluginExecutionRecord,
    ReportBlock,
    ReportPluginContext,
)


def _sanitize_error_message(message: str, *, max_len: int = 500) -> str:
    # Avoid breaking Markdown formatting and keep the report readable.
    sanitized = message.replace("`", "'").strip()
    if len(sanitized) > max_len:
        sanitized = sanitized[: max_len - 3] + "..."
    return sanitized


def _make_error_block(*, plugin: IReportBlockPlugin, error_type: str, error_message: str) -> ReportBlock:
    nodes: list[DocBlock] = [
        Heading(2, inlines(txt(plugin.title))),
        Paragraph(inlines(strong("Error type:"), txt(f" {error_type}"))),
        Paragraph(inlines(strong("Error message:"), txt(f" {error_message}"))),
        HorizontalRule(),
    ]

    return ReportBlock(
        block_id=plugin.plugin_id,
        title=plugin.title,
        order=plugin.order,
        nodes=nodes,
        meta={
            "status": "failed",
            "error_type": error_type,
            "error_message": error_message,
        },
    )


@dataclass(frozen=True, slots=True)
class ReportComposer:
    plugins: list[IReportBlockPlugin]

    def __post_init__(self) -> None:
        # Determinism and safety: fail-fast on ambiguous ordering/IDs.
        plugin_ids = [p.plugin_id for p in self.plugins]
        if len(set(plugin_ids)) != len(plugin_ids):
            raise ValueError(f"Duplicate report plugin_id detected: {plugin_ids}")

        orders = [p.order for p in self.plugins]
        if len(set(orders)) != len(orders):
            raise ValueError(f"Duplicate report plugin order detected: {orders}")

    def compose(self, ctx: ReportPluginContext) -> tuple[list[ReportBlock], list[PluginExecutionRecord]]:
        blocks: list[ReportBlock] = []
        records: list[PluginExecutionRecord] = []

        for plugin in sorted(self.plugins, key=lambda p: p.order):
            start = time.perf_counter()
            ctx.logger.info(f"[REPORT:PLUGIN] start id={plugin.plugin_id} order={plugin.order}")

            try:
                block = plugin.render(ctx)
                if block.block_id != plugin.plugin_id:
                    raise ValueError(
                        f"Plugin returned mismatched block_id='{block.block_id}', expected '{plugin.plugin_id}'"
                    )

                duration_ms = (time.perf_counter() - start) * 1000.0
                blocks.append(
                    ReportBlock(
                        block_id=block.block_id,
                        title=block.title,
                        order=block.order,
                        nodes=block.nodes,
                        meta={**block.meta, "duration_ms": duration_ms, "status": "ok"},
                    )
                )
                records.append(
                    PluginExecutionRecord(
                        plugin_id=plugin.plugin_id,
                        status="ok",
                        duration_ms=duration_ms,
                    )
                )
                ctx.logger.info(f"[REPORT:PLUGIN] ok id={plugin.plugin_id} duration_ms={duration_ms:.1f}")

            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000.0
                error_type = type(e).__name__
                error_message = _sanitize_error_message(str(e))

                ctx.logger.exception(
                    f"[REPORT:PLUGIN] failed id={plugin.plugin_id} duration_ms={duration_ms:.1f} "
                    f"error_type={error_type}"
                )

                blocks.append(
                    _make_error_block(
                        plugin=plugin,
                        error_type=error_type,
                        error_message=error_message,
                    )
                )
                records.append(
                    PluginExecutionRecord(
                        plugin_id=plugin.plugin_id,
                        status="failed",
                        duration_ms=duration_ms,
                        error_type=error_type,
                        error_message=error_message,
                    )
                )

        return blocks, records


__all__ = ["ReportComposer"]
