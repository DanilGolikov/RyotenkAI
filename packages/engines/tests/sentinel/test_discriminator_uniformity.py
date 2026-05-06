"""Sentinel: every Tag-based discriminated union in this monorepo MUST
use ``"kind"`` as its discriminator field — uniformly.

AD-6 of the discriminated-unions plan: divergent prior art (``type``,
``source_type``, ``engine``, ``strategy_type``) is consolidated to one
name. New code MUST follow.

This test walks every config module, finds ``Discriminator(...)``
arguments, and asserts the field name is ``"kind"``. Failures here are
either:
  (a) a new union was added with the wrong discriminator name; rename
      to ``kind`` per AD-6, OR
  (b) someone is intentionally diverging — open an ADR and update this
      test (deliberate change, not silent).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _python_files() -> list[Path]:
    """Walk all packages/*/src/ Python files."""
    out: list[Path] = []
    for pkg in (REPO_ROOT / "packages").iterdir():
        src = pkg / "src"
        if src.is_dir():
            out.extend(p for p in src.rglob("*.py") if p.is_file())
    return out


def _discriminator_field_names(file: Path) -> list[tuple[str, int]]:
    """Find every literal-string argument to ``Discriminator(...)`` AST node."""
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match Discriminator("..."), regardless of how it was imported.
        callee_name = ""
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr
        if callee_name != "Discriminator":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.append((arg.value, node.lineno))
    return found


def test_all_discriminators_named_kind() -> None:
    violations: list[str] = []
    for file in _python_files():
        for field, lineno in _discriminator_field_names(file):
            if field != "kind":
                rel = file.relative_to(REPO_ROOT)
                violations.append(f"  {rel}:{lineno} — Discriminator({field!r})")

    assert not violations, (
        "AD-6: all discriminators must be named ``'kind'``; found:\n"
        + "\n".join(violations)
    )
