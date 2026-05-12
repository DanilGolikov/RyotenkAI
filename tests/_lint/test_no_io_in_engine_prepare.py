"""Sentinel: engine ``prepare_model`` must be IO-free.

The engine ↔ provider contract is **engine describes, provider executes**.
``prepare_model`` returns a structured :class:`PreparePlan`; it must NOT
open files, spawn subprocesses, dial SSH, or do anything else that
implies "now". Side-effects belong to the provider's plan-runner.

This sentinel walks every ``ryotenkai_engines/*/runtime.py`` and asserts
the runtime module — the file containing the engine's ``prepare_model``
implementation — does not import any forbidden IO module at module level.

Module-level imports are the practical proxy for "this code does IO":
the engine tests themselves are IO-free fixtures, and a runtime module
that doesn't import paramiko / subprocess / requests / urllib /
fs-write helpers can't perform those operations on its callers.

Pairs with the ``engines is leaf`` importlinter contract and
``test_no_provider_imports.py`` — three layers of defense in depth so
this invariant breaking shows up in pytest output, not just CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINES_PKG_ROOT = _REPO_ROOT / "packages" / "engines" / "src" / "ryotenkai_engines"


# Modules whose presence in an engine runtime would imply side-effecty work
# at "prepare time" — exactly what we forbid. Synced with the design
# constraint AD-A2 (engine pure, provider executes) and the docstring
# contract on ``IInferenceEngine.prepare_model``.
FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "paramiko",
        "subprocess",
        "requests",
        "urllib",
        "urllib.request",
        "urllib.parse",  # network-y enough to ban; engines don't need it
        "shutil",  # writes/copies to fs
        "tempfile",
        "socket",
        "http",
        "http.client",
        "asyncio",
        "aiohttp",
        "httpx",
    }
)


def _runtime_modules() -> list[Path]:
    """Return every ``runtime.py`` under ``ryotenkai_engines/<engine_id>/``."""
    return [
        p
        for p in ENGINES_PKG_ROOT.rglob("runtime.py")
        if p.is_file() and p.parent.name not in {"ryotenkai_engines"}
    ]


def _module_level_imports(file: Path) -> list[str]:
    """Return dotted names imported at MODULE level (not inside functions)."""
    tree = ast.parse(file.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:  # only top-level statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def test_runtime_modules_discovered() -> None:
    """Sanity: at least one engine runtime exists (vLLM ships in PR-3)."""
    found = _runtime_modules()
    assert found, f"no runtime.py modules found under {ENGINES_PKG_ROOT}"


def test_no_forbidden_io_imports_at_module_level() -> None:
    """Walk every engine ``runtime.py`` and forbid IO-implying imports.

    Imports inside method bodies are allowed (rarely needed; if added,
    they should still avoid IO during ``prepare_model`` execution).
    """
    violations: list[tuple[Path, str]] = []
    for path in _runtime_modules():
        for imp in _module_level_imports(path):
            for forbidden in FORBIDDEN_MODULES:
                if imp == forbidden or imp.startswith(forbidden + "."):
                    violations.append(
                        (path.relative_to(ENGINES_PKG_ROOT.parent.parent), imp)
                    )
    assert not violations, (
        "engine runtime modules must be IO-free — found forbidden imports:\n"
        + "\n".join(f"  {file}: {imp}" for file, imp in violations)
    )
