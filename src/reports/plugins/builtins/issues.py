from __future__ import annotations

from src.reports.document.nodes import (
    BulletList,
    DocBlock,
    Heading,
    HorizontalRule,
    Paragraph,
    emph,
    inlines,
    list_items,
    strong,
    txt,
)
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext
from src.reports.plugins.registry import ReportPluginRegistry


@ReportPluginRegistry.register
class IssuesBlockPlugin:
    plugin_id = "issues"
    title = "Warnings & Errors"
    order = 30

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        issues = ctx.report.issues
        nodes: list[DocBlock] = [Heading(2, inlines(txt("⚠️ Warnings & Errors")))]

        if not issues:
            nodes.append(Paragraph(inlines(emph("No warnings or errors."))))
        else:
            items = []
            for issue in issues:
                emoji = "❌" if issue.severity == "ERROR" else "⚠️"
                ctx_prefix = f"[{issue.context}] " if issue.context else ""
                items.append(
                    (
                        txt(f"{emoji} "),
                        strong(issue.severity),
                        txt(" "),
                        txt(ctx_prefix + issue.message),
                    )
                )
            nodes.append(BulletList(items=list_items(items)))

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["IssuesBlockPlugin"]
