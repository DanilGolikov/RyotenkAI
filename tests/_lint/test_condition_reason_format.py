"""Sentinel: every literal ``update_condition(..., reason=...)`` /
``record_condition(..., reason=...)`` call uses a CamelCase reason.

Phase G — k8s conventions require Operator condition reasons to be
CamelCase (matching ``meta/v1.Condition.Reason``). The Pydantic
validator on :class:`Condition` rejects non-CamelCase reasons at
runtime, but a sentinel catches drift in production call sites at
**lint time** so a stray ``reason="rate_limited"`` doesn't ship
silently behind an untested branch.

Scope
-----
- Walks ``packages/*/src/**/*.py`` only (production code).
- Inspects ``update_condition`` and ``record_condition`` calls.
- Checks the ``reason=...`` keyword argument literal. Dynamic
  expressions (variable references, f-strings, function calls) are
  skipped — the runtime validator handles those.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# Same regex as in :class:`Condition` — start with uppercase letter,
# ASCII alphanumerics only.
_CAMEL_CASE_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


_TARGET_CALLEES: frozenset[str] = frozenset({"update_condition", "record_condition"})


def _packages_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "packages"


def _iter_production_files() -> list[Path]:
    base = _packages_dir()
    if not base.exists():
        return []
    return [
        p
        for p in base.rglob("*.py")
        if "__pycache__" not in p.parts and "/src/" in str(p).replace("\\", "/")
    ]


def _call_callee_name(node: ast.Call) -> str | None:
    """Return the unqualified callable name of an AST Call."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_offending_callsites() -> list[tuple[Path, int, str]]:
    offenders: list[tuple[Path, int, str]] = []
    for path in _iter_production_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_callee_name(node)
            if name not in _TARGET_CALLEES:
                continue
            for kw in node.keywords:
                if kw.arg != "reason":
                    continue
                value = kw.value
                # Only check literal strings — let the runtime validator
                # cover dynamic / interpolated values.
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if not _CAMEL_CASE_RE.match(value.value):
                        offenders.append((path, node.lineno, value.value))
    return offenders


def test_condition_reasons_are_camelcase() -> None:
    offenders = _collect_offending_callsites()
    if offenders:
        lines = [
            f"  {path.relative_to(_packages_dir().parent)}:{lineno} — reason={reason!r}"
            for path, lineno, reason in offenders
        ]
        pytest.fail(
            "Non-CamelCase reasons in update_condition/record_condition calls:\n"
            + "\n".join(lines)
            + "\nReasons must match ^[A-Z][A-Za-z0-9]*$ (k8s metav1.Condition convention).",
        )
