"""Sentinel — ``ryotenkai_community`` MUST not import ``ryotenkai_control``,
``ryotenkai_pod``, or ``ryotenkai_providers`` (plan §7.3, ADR rows 3+4).

community sits one level above shared in the dependency graph. Its only
legitimate downstream packages are the loader/manifest machinery for
plugins themselves — registry locations and report defaults must be
resolved at call time via ``importlib.import_module`` (see
:class:`CommunityCatalog._REGISTRY_LOCATORS`), not via static imports.

Defence-in-depth complement to the importlinter contract
"community must not import control/pod/providers". The AST scanner here
catches drift that importlinter's TYPE_CHECKING-blind walker would miss.
"""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN = (
    "ryotenkai_control",
    "ryotenkai_pod",
    "ryotenkai_providers",
)


def _community_src() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "ryotenkai_community"


def test_community_does_not_import_downstream() -> None:
    src = _community_src()
    assert src.exists(), f"sentinel mis-anchored: {src} is missing"
    violations: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in _FORBIDDEN:
                    if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                        violations.append(
                            f"{path.relative_to(src.parent)}: from {node.module}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in _FORBIDDEN:
                        if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                            violations.append(
                                f"{path.relative_to(src.parent)}: import {alias.name}"
                            )

    assert not violations, (
        "community must not import control/pod/providers — "
        "use importlib.import_module at call time instead:\n  "
        + "\n  ".join(violations)
    )
