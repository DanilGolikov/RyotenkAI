"""Minimal TOML serializer tailored to community manifests.

The project does not depend on ``tomli_w``; we only need to emit the narrow
subset of TOML used by plugin/preset manifests — nested tables, scalar leaves
and list-of-scalar leaves. Nested dicts are **always** emitted as dedicated
``[section]`` headers (never as inline tables), so the shape matches the
handwritten manifests authors have seen so far and ``community sync`` diffs
stay small.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# Keys that should appear first inside ``[plugin]`` / ``[preset]`` for
# readability (others fall back to alphabetical order).
_TOP_KEY_ORDER = (
    # Plugin/preset root identity.
    "id",
    "kind",
    "name",
    "version",
    "priority",
    "category",
    "stability",
    "size_tier",
    "description",
    # Entry point — module before class.
    "module",
    "class",
    "file",
    # Schema entry fields — type and default first, then constraints.
    "type",
    "default",
    "min",
    "max",
    "options",
)

# Top-level section order of a plugin manifest.
_PLUGIN_SECTION_ORDER = (
    "plugin",
    "plugin.entry_point",
    "params_schema",
    "thresholds_schema",
    "suggested_params",
    "suggested_thresholds",
    "secrets",
    "compat",
)

# Top-level section order of a preset manifest.
_PRESET_SECTION_ORDER = ("preset", "preset.entry_point")


def _escape_str(value: str) -> str:
    out = value.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{out}"'


def _format_scalar(value: Any) -> str:
    """Render a scalar or list-of-scalars as TOML."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return '""'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _escape_str(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_scalar(x) for x in value) + "]"
    raise TypeError(
        f"cannot render {type(value).__name__} as scalar; "
        f"nested dicts must be emitted as dedicated [section] headers"
    )


def _sort_keys(keys: Iterable[str]) -> list[str]:
    priority = {k: i for i, k in enumerate(_TOP_KEY_ORDER)}

    def key_fn(k: str) -> tuple[int, str]:
        return (priority.get(k, len(_TOP_KEY_ORDER)), k)

    return sorted(keys, key=key_fn)


def _split_scalars_and_tables(
    body: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    scalar_keys: list[str] = []
    table_keys: list[str] = []
    for key, value in body.items():
        if isinstance(value, Mapping):
            table_keys.append(key)
        else:
            scalar_keys.append(key)
    return _sort_keys(scalar_keys), _sort_keys(table_keys)


def _emit_scalars(
    lines: list[str],
    header: str,
    body: Mapping[str, Any],
    scalar_keys: list[str],
    *,
    todo_fields: set[str],
) -> None:
    for key in scalar_keys:
        path = f"{header}.{key}"
        value = body[key]
        if path in todo_fields:
            lines.append(f'{key} = {_format_scalar(value)}  # TODO: fill in')
        else:
            lines.append(f"{key} = {_format_scalar(value)}")


def _emit_section(
    lines: list[str],
    header: str,
    body: Mapping[str, Any],
    *,
    todo_fields: set[str],
    scheduled: frozenset[str],
) -> None:
    """Emit ``[header]`` and everything beneath it.

    Nested dict children whose full path is listed in ``scheduled`` are skipped
    — they'll be emitted later in the top-level order pass.
    """
    scalar_keys, table_keys = _split_scalars_and_tables(body)

    lines.append(f"[{header}]")
    _emit_scalars(lines, header, body, scalar_keys, todo_fields=todo_fields)
    lines.append("")

    for key in table_keys:
        child = f"{header}.{key}"
        if child in scheduled:
            continue
        _emit_section(
            lines,
            child,
            body[key],
            todo_fields=todo_fields,
            scheduled=scheduled,
        )


def _resolve(doc: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, Mapping) else None


def dump_manifest_toml(
    manifest: Mapping[str, Any],
    *,
    todo_fields: set[str] | frozenset[str] = frozenset(),
    section_order: Iterable[str] | None = None,
) -> str:
    """Serialise a manifest dict to TOML text with a stable key order.

    ``todo_fields`` — dotted paths (``"plugin.category"``) whose scalar
    value should carry a trailing ``# TODO: fill in`` comment.
    ``section_order`` — ordered tuple of dotted section paths. Paths
    listed here are emitted in order; any additional sections present
    in ``manifest`` but missing from the order land at the end in a
    deterministic (sorted) position.
    """
    order = tuple(section_order) if section_order is not None else (
        _PLUGIN_SECTION_ORDER if "plugin" in manifest else _PRESET_SECTION_ORDER
    )
    scheduled: frozenset[str] = frozenset(order)

    lines: list[str] = []
    todo = set(todo_fields)

    for path in order:
        body = _resolve(manifest, path)
        if not body:
            continue
        scalar_keys, table_keys = _split_scalars_and_tables(body)
        # Skip emitting an empty header when the section has no scalar
        # content of its own (e.g. [params_schema] only holds sub-sections).
        if scalar_keys:
            lines.append(f"[{path}]")
            _emit_scalars(lines, path, body, scalar_keys, todo_fields=todo)
            lines.append("")
        # Emit nested dict children that are NOT explicitly scheduled.
        for key in table_keys:
            child = f"{path}.{key}"
            if child in scheduled:
                continue
            _emit_section(lines, child, body[key], todo_fields=todo, scheduled=scheduled)

    # Trailing blank → single newline at EOF
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


__all__ = ["dump_manifest_toml"]
