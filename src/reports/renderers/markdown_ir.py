"""
Markdown renderer for the internal document IR (nodes).

This renderer is intentionally minimal and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.reports.document.nodes import (
    BlockQuote,
    BulletList,
    CodeBlock,
    DocBlock,
    DocInline,
    Emphasis,
    Heading,
    HorizontalRule,
    InlineCode,
    LineBreak,
    Link,
    OrderedList,
    Paragraph,
    Strong,
    Table,
    Text,
)


def _escape_text(text: str) -> str:
    # Keep it minimal: escape pipe to keep tables stable.
    return text.replace("|", "\\|")


@dataclass(frozen=True, slots=True)
class MarkdownIRRenderer:
    """
    Render DocBlocks to GitHub-flavored Markdown (best-effort).
    """

    def render(self, blocks: list[DocBlock]) -> str:
        parts = [self._render_block(b) for b in blocks]
        # Remove empty parts but preserve order.
        parts = [p for p in parts if p.strip() != ""]
        return "\n\n".join(parts).rstrip() + "\n"

    def _render_block(self, block: DocBlock) -> str:
        if isinstance(block, Heading):
            level = max(1, min(6, int(block.level)))
            return f"{'#' * level} {self._render_inlines(block.children)}"

        if isinstance(block, Paragraph):
            return self._render_inlines(block.children)

        if isinstance(block, BulletList):
            lines = []
            for item in block.items:
                lines.append(f"- {self._render_inlines(item)}")
            return "\n".join(lines)

        if isinstance(block, OrderedList):
            lines = []
            n = int(block.start)
            for item in block.items:
                lines.append(f"{n}. {self._render_inlines(item)}")
                n += 1
            return "\n".join(lines)

        if isinstance(block, CodeBlock):
            lang = block.language or ""
            return f"```{lang}\n{block.code}\n```".rstrip()

        if isinstance(block, HorizontalRule):
            return "---"

        if isinstance(block, Table):
            return self._render_table(block)

        if isinstance(block, BlockQuote):
            inner = self.render(list(block.blocks)).rstrip()
            return "\n".join(f"> {line}" if line.strip() else ">" for line in inner.splitlines())

        # Should not happen (exhaustive union), but keep safe.
        raise TypeError(f"Unsupported block type: {type(block)}")

    def _render_table(self, table: Table) -> str:
        # Markdown tables require at least header + separator.
        headers = [self._render_inlines(cell, context="table") for cell in table.headers]

        if table.align:
            align_map = {
                "left": ":---",
                "center": ":---:",
                "right": "---:",
            }
            sep = [align_map[a] for a in table.align]
        else:
            sep = ["---" for _ in headers]

        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(sep) + "|",
        ]

        for row in table.rows:
            cells = [self._render_inlines(cell, context="table") for cell in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _render_inlines(self, inlines: tuple[DocInline, ...], *, context: str = "paragraph") -> str:
        out: list[str] = []
        for node in inlines:
            if isinstance(node, Text):
                out.append(_escape_text(node.value))
            elif isinstance(node, Strong):
                out.append(f"**{self._render_inlines(node.children, context=context)}**")
            elif isinstance(node, Emphasis):
                out.append(f"*{self._render_inlines(node.children, context=context)}*")
            elif isinstance(node, InlineCode):
                # Keep it safe: avoid backticks in code.
                safe = node.code.replace("`", "'")
                out.append(f"`{safe}`")
            elif isinstance(node, Link):
                label = self._render_inlines(node.children, context=context)
                out.append(f"[{label}]({node.url})")
            elif isinstance(node, LineBreak):
                if context == "table":
                    out.append("<br>")
                else:
                    out.append("  \n")
            else:
                raise TypeError(f"Unsupported inline type: {type(node)}")

        return "".join(out)


__all__ = ["MarkdownIRRenderer"]
