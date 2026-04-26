"""Output rendering: ``TextRenderer`` (Rich) and ``JsonRenderer`` behind a
common ``Renderer`` protocol. Commands ask the active renderer to emit
rows / key-value pairs / status blocks; the concrete class decides whether
that becomes a pretty Rich table or a single ``json.dumps`` on exit.

Design notes:

- The ``JsonRenderer`` buffers everything internally and only emits on
  ``flush()`` (called from the root callback epilogue) so multiple render
  calls within one command compose into a single valid JSON document.
- Every ``table(...)`` call in JSON mode becomes a list of dicts (headers
  are keys). ``kv(...)`` becomes an object. Commands that want to emit a
  single top-level value should use ``emit(...)`` directly.
- All Rich output goes through ``cli.style.console`` — no extra Console
  instances, so colour / NO_COLOR stays consistent.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rich import box
from rich.console import Console
from rich.table import Table

from src.cli.context import CLIContext
from src.cli.style import COLOR_DIM, COLOR_LABEL
from src.cli.style import console as _default_console


class Renderer(Protocol):
    """Minimal surface every command can rely on."""

    def heading(self, text: str) -> None: ...
    def kv(self, pairs: Mapping[str, Any], *, title: str | None = None) -> None: ...
    def table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        title: str | None = None,
    ) -> None: ...
    def text(self, line: str = "") -> None: ...
    def emit(self, payload: Any) -> None: ...
    def flush(self) -> None: ...


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------


@dataclass
class TextRenderer:
    """Rich-backed human-readable output."""

    console: Console = field(default_factory=lambda: _default_console)

    def heading(self, text: str) -> None:
        self.console.print(f"[{COLOR_LABEL}]{text}[/{COLOR_LABEL}]")

    def kv(self, pairs: Mapping[str, Any], *, title: str | None = None) -> None:
        if title:
            self.heading(title)
        # Align the keys for readability without wrapping a table around them.
        width = max((len(k) for k in pairs), default=0)
        for key, value in pairs.items():
            display = "-" if value in (None, "") else value
            self.console.print(f"  {key:<{width}}  {display}")

    def table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        title: str | None = None,
    ) -> None:
        table = Table(
            title=title,
            box=box.SIMPLE,
            header_style=COLOR_LABEL,
            show_edge=False,
            pad_edge=False,
        )
        for header in headers:
            table.add_column(str(header))
        any_rows = False
        for row in rows:
            table.add_row(*(str(cell) if cell is not None else "-" for cell in row))
            any_rows = True
        if not any_rows:
            self.console.print(f"  [{COLOR_DIM}](no rows)[/{COLOR_DIM}]")
            return
        self.console.print(table)

    def text(self, line: str = "") -> None:
        self.console.print(line)

    def emit(self, payload: Any) -> None:
        # For text mode, ``emit`` is a freeform escape hatch — just print.
        self.console.print(payload)

    def flush(self) -> None:
        return None  # nothing to buffer


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


@dataclass
class JsonRenderer:
    """Collects render calls into a single JSON document.

    - First ``emit(obj)`` wins: later calls raise (commands should pick
      one output shape per invocation).
    - ``table`` / ``kv`` / ``heading`` are NO-OPS unless explicitly
      buffered via ``emit``. This prevents commands from accidentally
      leaking human-only output into the JSON stream.
    """

    payload: Any = None
    _emitted: bool = False

    def heading(self, text: str) -> None:  # noqa: ARG002 — intentionally unused
        return None

    def kv(
        self,
        pairs: Mapping[str, Any],
        *,
        title: str | None = None,  # noqa: ARG002
    ) -> None:
        self.emit(dict(pairs))

    def table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        title: str | None = None,  # noqa: ARG002
    ) -> None:
        header_list = list(headers)
        rows_as_dicts = [
            {header_list[i]: row[i] if i < len(row) else None for i in range(len(header_list))}
            for row in rows
        ]
        self.emit(rows_as_dicts)

    def text(self, line: str = "") -> None:  # noqa: ARG002
        return None

    def emit(self, payload: Any) -> None:
        if self._emitted:
            raise RuntimeError(
                "JsonRenderer.emit() called twice — pick a single payload per command"
            )
        self.payload = payload
        self._emitted = True

    def flush(self) -> None:
        if not self._emitted:
            return
        json.dump(self.payload, sys.stdout, indent=2, default=_json_default)
        sys.stdout.write("\n")


def _json_default(value: Any) -> Any:
    """Best-effort serialiser for common non-JSON types."""
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


# ---------------------------------------------------------------------------
# YAML renderer
# ---------------------------------------------------------------------------


@dataclass
class YamlRenderer:
    """YAML mirror of :class:`JsonRenderer`.

    Same buffer-then-flush contract: ``emit / kv / table`` accumulate one
    payload, ``flush`` dumps it as a single YAML document on stdout.
    Useful for pipelines that prefer YAML's readability for config-like
    payloads (``ryotenkai config show``, ``ryotenkai preset show``).
    """

    payload: Any = None
    _emitted: bool = False

    def heading(self, text: str) -> None:  # noqa: ARG002 — intentionally unused
        return None

    def kv(
        self,
        pairs: Mapping[str, Any],
        *,
        title: str | None = None,  # noqa: ARG002
    ) -> None:
        self.emit(dict(pairs))

    def table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        title: str | None = None,  # noqa: ARG002
    ) -> None:
        header_list = list(headers)
        rows_as_dicts = [
            {header_list[i]: row[i] if i < len(row) else None for i in range(len(header_list))}
            for row in rows
        ]
        self.emit(rows_as_dicts)

    def text(self, line: str = "") -> None:  # noqa: ARG002
        return None

    def emit(self, payload: Any) -> None:
        if self._emitted:
            raise RuntimeError(
                "YamlRenderer.emit() called twice — pick a single payload per command"
            )
        self.payload = payload
        self._emitted = True

    def flush(self) -> None:
        if not self._emitted:
            return
        # Lazy import: PyYAML is heavy enough that we don't want to pay
        # for it on every CLI invocation when YAML isn't requested.
        import yaml

        yaml.safe_dump(
            _yaml_normalise(self.payload),
            sys.stdout,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def _yaml_normalise(value: Any) -> Any:
    """Recursively coerce non-YAML-native types into safe equivalents.

    ``yaml.safe_dump`` rejects arbitrary objects — we map ``Path`` /
    ``datetime`` / dataclasses the same way :func:`_json_default` does.
    """
    if isinstance(value, dict):
        return {k: _yaml_normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_normalise(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_renderer(ctx: CLIContext) -> Renderer:
    """Pick the renderer that matches the context's output mode."""
    if ctx.is_json:
        return JsonRenderer()
    if ctx.is_yaml:
        return YamlRenderer()
    return TextRenderer()


__all__ = [
    "JsonRenderer",
    "Renderer",
    "TextRenderer",
    "YamlRenderer",
    "get_renderer",
]
