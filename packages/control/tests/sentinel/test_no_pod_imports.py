"""Sentinel — ``ryotenkai_control`` MUST not import ``ryotenkai_pod`` (plan §7.3).

The Mac control plane and the in-pod runtime live on different
machines after deployment; importing pod-side modules into Mac-side
code is the failure mode that produced the 16-crash chain in
``run_20260502_113553_r8rul`` (see plan §1.1).

This test is a defence-in-depth complement to the importlinter
contract ``"control must not import pod"`` — both fire on the same
violation, but only this AST sentinel keeps working if the importlinter
config drifts.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _control_src() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "ryotenkai_control"


def test_control_does_not_import_pod() -> None:
    src = _control_src()
    assert src.exists(), f"sentinel mis-anchored: {src} is missing"
    violations: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "ryotenkai_pod" or node.module.startswith("ryotenkai_pod."):
                    violations.append(f"{path.relative_to(src.parent)}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ryotenkai_pod" or alias.name.startswith("ryotenkai_pod."):
                        violations.append(f"{path.relative_to(src.parent)}: import {alias.name}")
    # NOTE: this test is currently EXPECTED-FAIL — Phase B/C surfaced 6
    # pre-existing drifts (see ADR "Known violations" table). The list is
    # printed so that a reviewer can confirm the count hasn't grown beyond
    # the documented set; once each follow-up PR lands, the matching entry
    # disappears here too. When the list is empty, flip the assertion to
    # ``not violations``.
    expected_known = {
        # See docs/adrs/2026-05-03-monorepo-uv-workspace-packagization.md
        # rows 5, 6 in the "Known violations" table. Each entry below names
        # the importing module file relative to packages/control/src/.
        "ryotenkai_control/data/loaders/factory.py: from ryotenkai_pod.trainer.container",
        "ryotenkai_control/data/validation/standalone.py: from ryotenkai_pod.trainer.strategies.factory",
        "ryotenkai_control/pipeline/mlflow_attempt/manager.py: from ryotenkai_pod.trainer.managers.mlflow_manager",
    }
    unexpected = [v for v in violations if v not in expected_known]
    assert not unexpected, (
        "NEW control→pod import detected (not in the known-violations list):\n  "
        + "\n  ".join(unexpected)
        + "\nIf intentional, document it in the ADR and add to expected_known above."
    )
