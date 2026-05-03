"""
Markdown renderer for report blocks.

This is intentionally minimal: it sorts blocks by `order` and concatenates
their IR nodes, then renders to Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.reports.renderers.markdown_ir import MarkdownIRRenderer

if TYPE_CHECKING:
    from src.reports.document.nodes import DocBlock
    from src.reports.plugins.interfaces import ReportBlock


@dataclass(frozen=True, slots=True)
class MarkdownBlockRenderer:
    def render(self, blocks: list[ReportBlock]) -> str:
        nodes: list[DocBlock] = []
        for b in sorted(blocks, key=lambda x: x.order):
            nodes.extend(b.nodes)
        return MarkdownIRRenderer().render(nodes)


__all__ = ["MarkdownBlockRenderer"]
