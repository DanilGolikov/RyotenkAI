"""
Document IR (intermediate representation) nodes.

Goals:
- Simple, explicit, typed.
- Format-agnostic: plugins describe *content*, renderers decide *output* (md/html/pdf).
- Minimal but sufficient for our current experiment report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

# -----------------------------------------------------------------------------
# Inline nodes
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Text:
    value: str


@dataclass(frozen=True, slots=True)
class Strong:
    children: tuple[DocInline, ...]


@dataclass(frozen=True, slots=True)
class Emphasis:
    children: tuple[DocInline, ...]


@dataclass(frozen=True, slots=True)
class InlineCode:
    code: str


@dataclass(frozen=True, slots=True)
class Link:
    children: tuple[DocInline, ...]
    url: str


@dataclass(frozen=True, slots=True)
class LineBreak:
    """A soft line break inside a paragraph."""


DocInline = Text | Strong | Emphasis | InlineCode | Link | LineBreak


def _t(*items: DocInline) -> tuple[DocInline, ...]:
    """Helper: build a tuple of inline nodes."""
    return tuple(items)


def txt(value: str) -> Text:
    return Text(value)


def strong(text: str) -> Strong:
    return Strong(_t(Text(text)))


def emph(text: str) -> Emphasis:
    return Emphasis(_t(Text(text)))


def code(value: str) -> InlineCode:
    return InlineCode(value)


def br() -> LineBreak:
    return LineBreak()


# -----------------------------------------------------------------------------
# Block nodes
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Heading:
    level: int
    children: tuple[DocInline, ...]


@dataclass(frozen=True, slots=True)
class Paragraph:
    children: tuple[DocInline, ...]


@dataclass(frozen=True, slots=True)
class BulletList:
    items: tuple[tuple[DocInline, ...], ...]


@dataclass(frozen=True, slots=True)
class OrderedList:
    items: tuple[tuple[DocInline, ...], ...]
    start: int = 1


@dataclass(frozen=True, slots=True)
class Table:
    headers: tuple[tuple[DocInline, ...], ...]
    rows: tuple[tuple[tuple[DocInline, ...], ...], ...]
    align: tuple[Literal["left", "center", "right"], ...] | None = None


@dataclass(frozen=True, slots=True)
class CodeBlock:
    code: str
    language: str | None = None


@dataclass(frozen=True, slots=True)
class BlockQuote:
    blocks: tuple[DocBlock, ...]


@dataclass(frozen=True, slots=True)
class HorizontalRule:
    """A horizontal rule separator."""


DocBlock = Heading | Paragraph | BulletList | OrderedList | Table | CodeBlock | BlockQuote | HorizontalRule


def inlines(*nodes: DocInline) -> tuple[DocInline, ...]:
    return tuple(nodes)


def list_items(items: Sequence[Sequence[DocInline]]) -> tuple[tuple[DocInline, ...], ...]:
    return tuple(tuple(x) for x in items)


def table_rows(rows: Sequence[Sequence[Sequence[DocInline]]]) -> tuple[tuple[tuple[DocInline, ...], ...], ...]:
    return tuple(tuple(tuple(cell) for cell in row) for row in rows)


__all__ = [
    "BlockQuote",
    "BulletList",
    "CodeBlock",
    "DocBlock",
    "DocInline",
    "Emphasis",
    "Heading",
    "HorizontalRule",
    "InlineCode",
    "LineBreak",
    "Link",
    "OrderedList",
    "Paragraph",
    "Strong",
    "Table",
    "Text",
    "br",
    "code",
    "emph",
    "inlines",
    "list_items",
    "strong",
    "table_rows",
    "txt",
]
