"""Sentinel — shared MUST be a graph leaf (plan §7.3).

Defence-in-depth on top of importlinter (``uv run lint-imports``):

* importlinter is a static-graph check; if its config is silently
  broken (typo in a contract, accidental ``--ignore`` flag) the check
  becomes a no-op and the boundary drifts.
* This test does an AST-level scan of every ``ryotenkai_shared/*.py``
  file and asserts no ``ryotenkai_<other>`` import slips in.

Two failure modes one belt-and-braces protects against:

1. Someone adds an import in shared that pulls one of the four
   downstream packages — gets caught by either layer.
2. importlinter config drifts — only this AST sentinel catches that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FORBIDDEN = (
    "ryotenkai_community",
    "ryotenkai_pod",
    "ryotenkai_providers",
    "ryotenkai_control",
)


def _shared_src() -> Path:
    """Resolve ``packages/shared/src/ryotenkai_shared/`` from the test file.

    Anchor: ``tests/_lint/test_shared_is_leaf.py`` → ``parents[2]`` is the
    worktree root, then ``packages/shared/src/ryotenkai_shared``.
    """
    return (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "shared"
        / "src"
        / "ryotenkai_shared"
    )


def test_shared_does_not_import_any_internal_package() -> None:
    """``ryotenkai_shared`` is the leaf — must not depend on any other workspace member."""
    src = _shared_src()
    assert src.exists(), f"sentinel mis-anchored: {src} is missing"
    violations: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            pytest.fail(f"shared file has syntax error: {path}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in _FORBIDDEN:
                    if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                        violations.append(f"{path.relative_to(src.parent)}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in _FORBIDDEN:
                        if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                            violations.append(f"{path.relative_to(src.parent)}: import {alias.name}")
    # All historic shared→{community,pod,providers,control} drifts have
    # been closed (ADR rows 1, 2, 8 fixed). The leaf invariant is now
    # enforceable directly — any new violation fails this assertion.
    assert not violations, (
        "shared must be a graph leaf — new internal import detected:\n  "
        + "\n  ".join(violations)
        + "\nMove the dependency the other way (downstream depends on shared)."
    )
