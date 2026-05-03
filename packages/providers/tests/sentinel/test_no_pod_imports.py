"""Sentinel — ``ryotenkai_providers`` MUST not import ``ryotenkai_pod``
or ``ryotenkai_control`` (plan §7.3).

providers is a Mac-side package whose only legitimate downstream
dependency is shared. Pulling pod or control into providers would
re-introduce the deployment cycle that the packagization specifically
removed.

Defence-in-depth complement to the importlinter contract
"providers depend only on shared". See plan §19 / Q4.1.
"""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN = (
    "ryotenkai_community",
    "ryotenkai_pod",
    "ryotenkai_control",
)


def _providers_src() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "ryotenkai_providers"


def test_providers_does_not_import_pod_or_control() -> None:
    src = _providers_src()
    assert src.exists()
    violations: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
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

    # ADR row 7: providers→pod.runner.{lifecycle_client, pod_terminator}
    # — provider adapters import the runner-side Protocol + outcome enum.
    # ADR row 9: providers reach into control for cross-cutting types
    # (RunContext, PodAvailability, system_prompt resolver) via TYPE_CHECKING
    # / lazy imports. Each entry is a focused follow-up PR — extract the
    # type into shared or invert ownership.
    expected_known = {
        "ryotenkai_providers/training/interfaces.py: from ryotenkai_control.pipeline.state",
        "ryotenkai_providers/runpod/_status_mapper.py: from ryotenkai_control.pipeline.launch.pod_availability",
        "ryotenkai_providers/single_node/training/provider.py: from ryotenkai_control.pipeline.state",
        "ryotenkai_providers/single_node/inference/provider.py: from ryotenkai_control.evaluation.system_prompt",
        "ryotenkai_providers/runpod/training/provider.py: from ryotenkai_control.pipeline.state",
        "ryotenkai_providers/runpod/training/provider.py: from ryotenkai_control.pipeline.launch.pod_availability",
        "ryotenkai_providers/runpod/inference/pods/provider.py: from ryotenkai_control.evaluation.system_prompt",
    }
    unexpected = [v for v in violations if v not in expected_known]
    assert not unexpected, (
        "NEW providers→{pod,control,community} import detected:\n  "
        + "\n  ".join(unexpected)
        + "\nIf intentional, document in ADR and add to expected_known."
    )
