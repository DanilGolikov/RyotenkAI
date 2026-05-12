"""Sentinel: ``ryotenkai_engines`` must not import ``ryotenkai_providers``,
``ryotenkai_control``, ``ryotenkai_pod``, or ``ryotenkai_community``.

Engines is a leaf workspace member — depends only on stdlib + pydantic
(+ ryotenkai_shared for Result/AppError types). The dependency direction
is providers → engines, NEVER engines → providers.

This test re-asserts the importlinter ``engines is leaf`` contract at the
unit-test layer so failures show up in pytest output (clearer error
attribution than CI-only importlinter runs).
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINES_PKG_ROOT = _REPO_ROOT / "packages" / "engines" / "src" / "ryotenkai_engines"

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "ryotenkai_providers",
    "ryotenkai_control",
    "ryotenkai_pod",
    "ryotenkai_community",
)


def _python_files() -> list[Path]:
    return [p for p in ENGINES_PKG_ROOT.rglob("*.py") if p.is_file()]


def _imports_in(file: Path) -> list[str]:
    """Top-level dotted module names imported by ``file``."""
    tree = ast.parse(file.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def test_engines_pkg_exists() -> None:
    assert ENGINES_PKG_ROOT.is_dir(), f"{ENGINES_PKG_ROOT} not found"


def test_no_forbidden_imports() -> None:
    """Walk every .py under ``src/ryotenkai_engines`` and assert none
    import a forbidden ryotenkai_* package."""
    violations: list[tuple[Path, str]] = []
    for path in _python_files():
        for imp in _imports_in(path):
            if any(imp == p or imp.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                violations.append((path.relative_to(ENGINES_PKG_ROOT.parent.parent), imp))
    assert not violations, (
        "engines must be a leaf — found imports of forbidden packages:\n"
        + "\n".join(f"  {file}: {imp}" for file, imp in violations)
    )
