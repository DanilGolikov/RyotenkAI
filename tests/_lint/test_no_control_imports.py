"""Sentinel — ``ryotenkai_pod`` MUST not import ``ryotenkai_control`` or
``ryotenkai_providers`` (plan §7.3).

After deployment the pod runs on a remote node and only ships the
``ryotenkai_pod`` + ``ryotenkai_community`` + ``ryotenkai_shared``
wheels. Any ``ryotenkai_control`` / ``ryotenkai_providers`` import means
the pod can't start outside the dev workstation. ADR rows 5+6 documents
the legacy violations; once the trainer/runner refactors land, the
``expected_known`` list should empty out.

Defence-in-depth complement to the importlinter contract
"pod must not import control/providers". Both fire on the same
violation, but only this AST sentinel keeps working if the importlinter
config drifts.
"""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN = (
    "ryotenkai_control",
    "ryotenkai_providers",
)


def _pod_src() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "pod"
        / "src"
        / "ryotenkai_pod"
    )


def test_pod_does_not_import_control_or_providers() -> None:
    src = _pod_src()
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

    # Track legacy violations so the assertion fires only on NEW drift.
    # See ADR rows 5+6. Empty this set once each follow-up PR lands and
    # flip the assertion to ``assert not violations`` outright.
    expected_known: set[str] = set()
    unexpected = [v for v in violations if v not in expected_known]
    assert not unexpected, (
        "NEW pod→{control,providers} import detected (not in known-violations list):\n  "
        + "\n  ".join(unexpected)
        + "\nIf intentional, document it in the ADR and add to expected_known above."
    )
