"""AST-based inference for community manifests.

Pure, side-effect-free: reads plugin source files as AST, never imports or
executes plugin code. Used by ``scaffold`` to generate a fresh manifest and
by ``sync`` to detect changes in the code vs the manifest.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# PluginKind is a type alias used at runtime in the annotation of
# ``_BASE_TO_KIND`` below — keep the import outside TYPE_CHECKING.
from src.community.manifest import PluginKind

logger = logging.getLogger(__name__)

# Base-class names we look for in plugin source files. The mapping is by
# name only (no import resolution) — if a plugin subclasses via an alias,
# the author needs to state the kind explicitly. Keeping inference AST-only
# avoids importing (and executing) plugin code during scaffolding.
_BASE_TO_KIND: dict[str, PluginKind] = {
    "ValidationPlugin": "validation",
    "EvaluatorPlugin": "evaluation",
    "RewardPlugin": "reward",
}

# Report plugins use duck-typing (plugin_id + order + render), not a base
# class — detected separately.
_REPORT_REQUIRED_ATTRS = ("plugin_id", "order")
_REPORT_REQUIRED_METHODS = ("render",)


@dataclass(frozen=True, slots=True)
class InferredField:
    """One inferred entry for ``params_schema`` / ``thresholds_schema``."""

    type: str
    default: object | None  # None if plugin did not pass a default


@dataclass(frozen=True, slots=True)
class InferredPlugin:
    """Everything AST inference can tell us about a plugin folder."""

    entry_module: str         # "plugin" (file) or "plugin" (package)
    entry_class: str
    kind: PluginKind
    description: str
    params: dict[str, InferredField]
    thresholds: dict[str, InferredField]
    required_secrets: tuple[str, ...]


# ---------------------------------------------------------------------------
# Module discovery & AST loading
# ---------------------------------------------------------------------------


def find_entry_module(plugin_dir: Path) -> Path:
    """Return the path to the plugin's entry module (single-file or package)."""
    single = plugin_dir / "plugin.py"
    if single.is_file():
        return single
    package_init = plugin_dir / "plugin" / "__init__.py"
    if package_init.is_file():
        return package_init
    raise FileNotFoundError(
        f"no plugin entry module under {plugin_dir}: expected plugin.py or plugin/__init__.py"
    )


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------------------
# Entry class + kind detection
# ---------------------------------------------------------------------------


def _class_bases(node: ast.ClassDef) -> list[str]:
    """Extract base-class names from a ClassDef (flat names only)."""
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _is_report_plugin(node: ast.ClassDef) -> bool:
    body_names = set()
    for child in node.body:
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    body_names.add(target.id)
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            body_names.add(child.target.id)
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            body_names.add(child.name)
    return (
        all(attr in body_names for attr in _REPORT_REQUIRED_ATTRS)
        and all(m in body_names for m in _REPORT_REQUIRED_METHODS)
    )


def _classify(node: ast.ClassDef) -> PluginKind | None:
    for base in _class_bases(node):
        if base in _BASE_TO_KIND:
            return _BASE_TO_KIND[base]
    if _is_report_plugin(node):
        return "reports"
    return None


def find_entry_class(
    module_ast: ast.Module, *, prefer_name: str | None = None
) -> tuple[ast.ClassDef, PluginKind]:
    """Return the single ``ClassDef`` that looks like a plugin entry.

    Raises ``ValueError`` if zero or multiple candidates are found (unless
    one of the multiple matches ``prefer_name``).
    """
    candidates: list[tuple[ast.ClassDef, PluginKind]] = []
    for node in ast.walk(module_ast):
        if isinstance(node, ast.ClassDef):
            kind = _classify(node)
            if kind is not None:
                candidates.append((node, kind))

    if not candidates:
        raise ValueError(
            "no plugin entry class found — expected a subclass of "
            "ValidationPlugin / EvaluatorPlugin / RewardPlugin, or a class "
            "exposing plugin_id + order + render() (reports kind)"
        )

    if len(candidates) == 1:
        return candidates[0]

    if prefer_name:
        for node, kind in candidates:
            if node.name == prefer_name:
                return node, kind

    names = sorted(c.name for c, _ in candidates)
    raise ValueError(
        f"multiple plugin-like classes found: {names}. "
        "Disambiguate by keeping only one entry class per plugin or renaming the others."
    )


def infer_kind_from_class(node: ast.ClassDef) -> PluginKind:
    kind = _classify(node)
    if kind is None:
        raise ValueError(f"class {node.name!r} does not match any plugin kind")
    return kind


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------


def infer_docstring_summary(node: ast.ClassDef) -> str:
    doc = ast.get_docstring(node) or ""
    first = doc.strip().split("\n", 1)[0].strip()
    return first


# ---------------------------------------------------------------------------
# params / thresholds
# ---------------------------------------------------------------------------


_TYPE_BY_PY: dict[type, str] = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    list: "array",
    tuple: "array",
    dict: "object",
}


def _literal_value(node: ast.expr) -> tuple[bool, object | None]:
    """Return (is_literal, value). Handles -N as UnaryOp."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return False, None


def _collect_module_constants(module: ast.Module) -> dict[str, object]:
    """Harvest ``NAME = literal`` bindings at module top level.

    Lets us infer defaults written as ``self._param("model", DEFAULT_MODEL)``
    when ``DEFAULT_MODEL`` is a module-level literal constant. This is a
    common pattern in our plugins and very cheap to support.
    """
    consts: dict[str, object] = {}
    for node in module.body:
        targets: list[str] = []
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            value_node = node.value
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value_node = node.value
            targets.append(node.target.id)
        if not targets or value_node is None:
            continue
        is_lit, val = _literal_value(value_node)
        if not is_lit:
            continue
        for name in targets:
            consts[name] = val
    return consts


def _resolve_expr(
    node: ast.expr, constants: dict[str, object]
) -> tuple[bool, object | None]:
    """Like ``_literal_value`` but also resolves ``ast.Name`` via
    the module-level constants dict supplied by the caller."""
    if isinstance(node, ast.Name) and node.id in constants:
        return True, constants[node.id]
    return _literal_value(node)


def _infer_type(value: object | None) -> str:
    if value is None:
        return "string"
    # bool is a subclass of int — check it first
    if isinstance(value, bool):
        return "boolean"
    for py_type, toml_type in _TYPE_BY_PY.items():
        if isinstance(value, py_type):
            return toml_type
    return "string"


def _is_self_call(call: ast.Call, *, func_name: str) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == func_name
    )


def _is_self_attr_get(call: ast.Call, *, attr: str) -> bool:
    """Match ``self.<attr>.get(...)``."""
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "get"
        and isinstance(call.func.value, ast.Attribute)
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "self"
        and call.func.value.attr == attr
    )


def _collect_calls(
    node: ast.AST,
    *,
    method: str,
    attr: str,
    constants: dict[str, object] | None = None,
) -> dict[str, InferredField]:
    """Walk ``node`` and collect ``self.<method>(key, default)`` and
    ``self.<attr>.get(key, default)`` — both feed into the same schema.

    ``constants`` — optional module-level literal bindings used to resolve
    defaults written as ``self._param("model", DEFAULT_MODEL)``.
    """
    consts = constants or {}
    out: dict[str, InferredField] = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        if _is_self_call(sub, func_name=method) or _is_self_attr_get(sub, attr=attr):
            if not sub.args:
                continue
            is_lit, key = _literal_value(sub.args[0])
            if not is_lit or not isinstance(key, str):
                logger.warning(
                    "[INFERENCE] skipping dynamic %s(...) call: first arg not a string literal",
                    method,
                )
                continue
            default: object | None = None
            if len(sub.args) >= 2:
                is_lit, default = _resolve_expr(sub.args[1], consts)
                if not is_lit:
                    default = None
            # Don't overwrite existing entry if we already saw a better (non-None default) one
            if key in out and out[key].default is not None and default is None:
                continue
            out[key] = InferredField(type=_infer_type(default), default=default)
    return out


def infer_params(
    node: ast.ClassDef, *, constants: dict[str, object] | None = None
) -> dict[str, InferredField]:
    return _collect_calls(node, method="_param", attr="params", constants=constants)


def infer_thresholds(
    node: ast.ClassDef, *, constants: dict[str, object] | None = None
) -> dict[str, InferredField]:
    return _collect_calls(node, method="_threshold", attr="thresholds", constants=constants)


# ---------------------------------------------------------------------------
# secrets
# ---------------------------------------------------------------------------


def infer_required_secrets(node: ast.ClassDef) -> tuple[str, ...]:
    """Collect literal keys referenced via ``self._secrets["KEY"]``."""
    keys: list[str] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Subscript):
            continue
        if (
            isinstance(sub.value, ast.Attribute)
            and isinstance(sub.value.value, ast.Name)
            and sub.value.value.id == "self"
            and sub.value.attr == "_secrets"
        ):
            # py3.9+: sub.slice is the expression directly
            is_lit, key = _literal_value(sub.slice)
            if is_lit and isinstance(key, str):
                if key not in keys:
                    keys.append(key)
            else:
                logger.warning(
                    "[INFERENCE] skipping dynamic self._secrets[...] access: not a string literal",
                )
    return tuple(keys)


# ---------------------------------------------------------------------------
# Full inference entry point
# ---------------------------------------------------------------------------


def _collect_module_asts(entry_path: Path) -> list[ast.Module]:
    """Return all module ASTs to scan for the entry class.

    - Single-file plugin (``plugin.py``) → one module.
    - Package plugin (``plugin/__init__.py``) → ``__init__.py`` plus every
      other ``*.py`` sibling. Plugin code often splits concerns into
      ``main.py`` / ``provider.py`` and only re-exports from
      ``__init__.py``, so the entry class lives in a sibling file.
    """
    asts = [parse_module(entry_path)]
    if entry_path.name == "__init__.py":
        for sibling in sorted(entry_path.parent.glob("*.py")):
            if sibling.name == "__init__.py":
                continue
            asts.append(parse_module(sibling))
    return asts


def _find_entry_across_modules(
    modules: list[ast.Module], *, prefer_name: str | None
) -> tuple[ast.ClassDef, PluginKind]:
    candidates: list[tuple[ast.ClassDef, PluginKind]] = []
    for module in modules:
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                kind = _classify(node)
                if kind is not None:
                    candidates.append((node, kind))
    if not candidates:
        raise ValueError(
            "no plugin entry class found — expected a subclass of "
            "ValidationPlugin / EvaluatorPlugin / RewardPlugin, or a class "
            "exposing plugin_id + order + render() (reports kind)"
        )
    if len(candidates) == 1:
        return candidates[0]
    if prefer_name:
        for node, kind in candidates:
            if node.name == prefer_name:
                return node, kind
    names = sorted(c.name for c, _ in candidates)
    raise ValueError(
        f"multiple plugin-like classes found: {names}. "
        "Rename or remove extras so exactly one entry class is declared per plugin."
    )


def infer_plugin(plugin_dir: Path) -> InferredPlugin:
    """Run all inference passes on a plugin folder."""
    entry_path = find_entry_module(plugin_dir)
    module_asts = _collect_module_asts(entry_path)
    cls_node, kind = _find_entry_across_modules(
        module_asts, prefer_name=_camel_case(plugin_dir.name)
    )
    # Merge module-level constants across every scanned module so defaults
    # written as ``self._param("model", DEFAULT_MODEL)`` get resolved even
    # when the constant lives in a sibling file (e.g. main.py references
    # a constant declared in provider.py).
    constants: dict[str, object] = {}
    for module in module_asts:
        constants.update(_collect_module_constants(module))
    entry_module = "plugin"  # convention; loader accepts both file and package
    return InferredPlugin(
        entry_module=entry_module,
        entry_class=cls_node.name,
        kind=kind,
        description=infer_docstring_summary(cls_node),
        params=infer_params(cls_node, constants=constants),
        thresholds=infer_thresholds(cls_node, constants=constants),
        required_secrets=infer_required_secrets(cls_node),
    )


def _camel_case(snake: str) -> str:
    """Rough convention: snake_case folder → CamelCase class name (for disambiguation)."""
    return "".join(part.capitalize() for part in snake.split("_"))


# ---------------------------------------------------------------------------
# Semver bump
# ---------------------------------------------------------------------------


def bump_version(version: str, bump: Literal["patch", "minor", "major"]) -> str:
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"version must be semver X.Y.Z, got {version!r}")
    major, minor, patch = (int(p) for p in parts)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"bump must be patch|minor|major, got {bump!r}")


__all__ = [
    "InferredField",
    "InferredPlugin",
    "bump_version",
    "find_entry_class",
    "find_entry_module",
    "infer_docstring_summary",
    "infer_kind_from_class",
    "infer_params",
    "infer_plugin",
    "infer_required_secrets",
    "infer_thresholds",
    "parse_module",
]
