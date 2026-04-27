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

# Keys that should appear first inside ``[plugin]`` / ``[preset]`` /
# ``[lib]`` for readability (others fall back to alphabetical order).
_TOP_KEY_ORDER = (
    # Schema versioning — always first so it shows up at the top of
    # the file when present at the document root.
    "schema_version",
    # Plugin/preset/lib root identity.
    "id",
    "kind",
    "name",
    "version",
    "category",
    "stability",
    "size_tier",
    "description",
    # Reward-only (appended last in [plugin] so existing non-reward
    # manifests don't shift).
    "supported_strategies",
    # Author attribution. Lives under [plugin] / [lib]; comes after
    # description so the human-facing fields cluster together.
    "author",
    # Entry point — module before class.
    "module",
    "class",
    "file",
)

#: Schema-field (``[params_schema.X]`` / ``[thresholds_schema.X]``)
#: key order. Kept separate from ``_TOP_KEY_ORDER`` because ``title``
#: / ``description`` need to land AFTER the type/constraint block in
#: schema fields, not in the [plugin]-body position.
_SCHEMA_FIELD_KEY_ORDER = (
    "type",
    "default",
    "min",
    "max",
    "options",
    "title",
    "description",
    "required",
    "secret",
)

# Top-level section order of a plugin manifest.
_PLUGIN_SECTION_ORDER = (
    "plugin",
    "plugin.entry_point",
    "params_schema",
    "thresholds_schema",
    "suggested_params",
    "suggested_thresholds",
    "compat",
)

#: Top-level array-of-tables sections, in emission order. Always
#: rendered AFTER the named section pass so they land at the bottom of
#: the manifest in a deterministic order.
_PLUGIN_AOT_KEYS = ("required_env", "lib_requirements")

#: Field order within a single ``[[required_env]]`` block. Mirrors the
#: ParamFieldSchema-style "type/default first, descriptive bits after"
#: pattern so hand-crafted manifests stay diff-stable.
_REQUIRED_ENV_FIELD_ORDER = (
    "name",
    "description",
    "optional",
    "secret",
    "managed_by",
)

#: Field order within a single ``[[lib_requirements]]`` block. ``name``
#: first, then ``version`` — the only two fields that exist for now.
_LIB_REQ_FIELD_ORDER = ("name", "version")

# Top-level section order of a preset manifest.
_PRESET_SECTION_ORDER = ("preset", "preset.entry_point")

# Top-level section order of a lib manifest.
_LIB_SECTION_ORDER = ("lib",)


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


def _sort_keys(keys: Iterable[str], *, schema_field: bool = False) -> list[str]:
    order = _SCHEMA_FIELD_KEY_ORDER if schema_field else _TOP_KEY_ORDER
    priority = {k: i for i, k in enumerate(order)}

    def key_fn(k: str) -> tuple[int, str]:
        return (priority.get(k, len(order)), k)

    return sorted(keys, key=key_fn)


def _is_schema_field_section(path: str) -> bool:
    """True for dotted paths like ``params_schema.sample_size`` — leaves under
    a ``*_schema`` parent. These render a :class:`ParamFieldSchema` and need
    the schema-field key order, not the plugin-body one."""
    parts = path.split(".")
    return len(parts) >= 2 and parts[-2] in {"params_schema", "thresholds_schema"}


def _split_scalars_and_tables(
    body: Mapping[str, Any],
    *,
    schema_field: bool = False,
) -> tuple[list[str], list[str]]:
    scalar_keys: list[str] = []
    table_keys: list[str] = []
    for key, value in body.items():
        if isinstance(value, Mapping):
            table_keys.append(key)
        else:
            scalar_keys.append(key)
    return (
        _sort_keys(scalar_keys, schema_field=schema_field),
        _sort_keys(table_keys, schema_field=schema_field),
    )


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
    schema_field = _is_schema_field_section(header)
    scalar_keys, table_keys = _split_scalars_and_tables(body, schema_field=schema_field)

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


def _emit_array_of_tables(
    lines: list[str],
    key: str,
    items: list[Mapping[str, Any]],
    *,
    field_order: tuple[str, ...] = (),
) -> None:
    """Emit ``[[key]]`` blocks for each dict in ``items``.

    ``field_order`` lists keys whose order should be pinned at the top
    of each block; remaining keys fall back to alphabetical order.
    Empty / missing scalars are still rendered so the diff against an
    operator-edited manifest stays predictable.
    """
    priority = {k: i for i, k in enumerate(field_order)}
    for item in items:
        sorted_keys = sorted(
            item.keys(),
            key=lambda k: (priority.get(k, len(priority)), k),
        )
        lines.append(f"[[{key}]]")
        for sk in sorted_keys:
            value = item[sk]
            if isinstance(value, Mapping):
                # Nested-dict inside an AOT entry would need
                # `[[parent.child]]` — community manifests don't use that
                # shape today, so reject early instead of emitting
                # something that can't round-trip.
                raise TypeError(
                    f"nested dicts inside [[{key}]] are not supported; "
                    f"got mapping under {sk!r}"
                )
            lines.append(f"{sk} = {_format_scalar(value)}")
        lines.append("")


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
    if section_order is not None:
        order = tuple(section_order)
    elif "plugin" in manifest:
        order = _PLUGIN_SECTION_ORDER
    elif "lib" in manifest:
        order = _LIB_SECTION_ORDER
    else:
        order = _PRESET_SECTION_ORDER
    scheduled: frozenset[str] = frozenset(order)

    lines: list[str] = []
    todo = set(todo_fields)

    # Top-level scalars (currently just ``schema_version``) — emitted
    # before any section header so they appear at the top of the file,
    # matching how authors hand-write manifests.
    root_scalars = [
        k
        for k, v in manifest.items()
        if not isinstance(v, Mapping) and not isinstance(v, list)
    ]
    if root_scalars:
        for key in _sort_keys(root_scalars):
            lines.append(f"{key} = {_format_scalar(manifest[key])}")
        lines.append("")

    for path in order:
        body = _resolve(manifest, path)
        if not body:
            continue
        schema_field = _is_schema_field_section(path)
        scalar_keys, table_keys = _split_scalars_and_tables(body, schema_field=schema_field)
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

    # Top-level array-of-tables (e.g. [[required_env]],
    # [[lib_requirements]]) — always after the section order pass, in
    # a fixed key order so diffs stay small.
    if "plugin" in manifest:
        for aot_key in _PLUGIN_AOT_KEYS:
            entries = manifest.get(aot_key)
            if not entries:
                continue
            if aot_key == "required_env":
                field_order = _REQUIRED_ENV_FIELD_ORDER
            elif aot_key == "lib_requirements":
                field_order = _LIB_REQ_FIELD_ORDER
            else:
                field_order = ()
            _emit_array_of_tables(
                lines, aot_key, list(entries), field_order=field_order
            )

    # Trailing blank → single newline at EOF
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


__all__ = ["dump_manifest_toml"]
