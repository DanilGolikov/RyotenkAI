"""Sentinel — ``ryotenkai_control`` MUST not introduce new ``ryotenkai_pod`` imports.

Greenfield migration of
``packages/control/tests/sentinel/test_no_pod_imports.py``.

Why this lives alongside importlinter rather than as a duplicate of it:

- The importlinter contract ``control must not import pod`` is currently
  **BROKEN** by five legacy import sites (see
  ``docs/adrs/2026-05-03-monorepo-uv-workspace-packagization.md`` "Known
  violations" table, rows 5+6). Those are tracked-and-tolerated drifts
  awaiting Phase D extraction follow-up PRs.
- importlinter's BROKEN signal is binary — it can't distinguish "no
  regression" from "regression added". A PR that introduces a NEW
  ``ryotenkai_control → ryotenkai_pod`` import would still show BROKEN
  and nothing else.
- This AST sentinel locks in the *known* violations as a baseline and
  fails CI ONLY on NEW drift. When each Phase D follow-up lands the
  matching entry disappears here and we tighten the baseline.

Once ``expected_known`` is empty, this file can be retired — at that
point the importlinter contract will flip from BROKEN to KEPT and the
diff-aware check becomes redundant.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTROL_SRC = _REPO_ROOT / "packages" / "control" / "src" / "ryotenkai_control"


# Known violations are intentionally tolerated until Phase D
# extraction PRs land. Each entry must have a matching row in the
# packagization ADR; new entries are NOT to be added without a
# corresponding ADR update.
_EXPECTED_KNOWN: frozenset[str] = frozenset(
    {
        "ryotenkai_control/pipeline/stages/dataset_validator/split_loader.py: from ryotenkai_pod.trainer.data_loaders.factory",
        "ryotenkai_control/pipeline/stages/dataset_validator/stage.py: from ryotenkai_pod.trainer.data_loaders.factory",
        "ryotenkai_control/pipeline/mlflow_attempt/manager.py: from ryotenkai_pod.trainer.managers.mlflow_manager",
        # Migration note: ``data/__init__.py`` was in the legacy baseline
        # but the live import has been removed (only the docstring still
        # mentions ``ryotenkai_pod.trainer.data_loaders``). One known-pod
        # import disappeared since the legacy test was written; this
        # baseline reflects current reality.
        "ryotenkai_control/data/validation/standalone.py: from ryotenkai_pod.trainer.strategies.factory",
    },
)


def _scan_for_pod_imports(src_root: Path) -> list[str]:
    violations: list[str] = []
    for path in src_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "ryotenkai_pod" or node.module.startswith(
                    "ryotenkai_pod.",
                ):
                    violations.append(
                        f"{path.relative_to(src_root.parent)}: from {node.module}",
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ryotenkai_pod" or alias.name.startswith(
                        "ryotenkai_pod.",
                    ):
                        violations.append(
                            f"{path.relative_to(src_root.parent)}: import {alias.name}",
                        )
    return violations


def test_no_new_control_to_pod_imports() -> None:
    assert _CONTROL_SRC.exists(), f"sentinel mis-anchored: {_CONTROL_SRC}"
    violations = _scan_for_pod_imports(_CONTROL_SRC)
    unexpected = sorted(v for v in violations if v not in _EXPECTED_KNOWN)
    assert not unexpected, (
        "NEW control→pod import detected (not in known-violations list):\n  "
        + "\n  ".join(unexpected)
        + "\nIf intentional, update the packagization ADR and the "
        "_EXPECTED_KNOWN frozenset above."
    )


def test_expected_known_baseline_does_not_grow_silently() -> None:
    """If a previously-known violation is gone (good!), the baseline
    must shrink. Catches a stale baseline that no longer reflects
    reality."""
    assert _CONTROL_SRC.exists(), f"sentinel mis-anchored: {_CONTROL_SRC}"
    violations = set(_scan_for_pod_imports(_CONTROL_SRC))
    stale = sorted(_EXPECTED_KNOWN - violations)
    assert not stale, (
        "Entries in _EXPECTED_KNOWN that no longer exist in source — "
        "remove them from the baseline so the boundary tightens:\n  "
        + "\n  ".join(stale)
    )


def test_sentinel_detects_synthetic_violation(tmp_path: Path) -> None:
    """Demonstrate the scanner catches a real ``ryotenkai_pod`` import
    when a file outside ``_EXPECTED_KNOWN`` introduces one."""
    fake_src = tmp_path / "ryotenkai_control"
    fake_src.mkdir()
    (fake_src / "__init__.py").write_text("")
    bad = fake_src / "new_drift.py"
    bad.write_text(
        "from ryotenkai_pod.runner.api import schemas  # noqa\n",
        encoding="utf-8",
    )
    found = _scan_for_pod_imports(fake_src)
    assert any("ryotenkai_pod.runner.api" in v for v in found), found
