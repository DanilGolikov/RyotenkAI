"""Sentinel: every top-level test tree must collect without errors.

Running ``pytest --collect-only`` on each tree is the L0 gate. If a tree
fails to collect (typically because of an import error or duplicate
registration), the rest of the lane never even runs — and CI silently
loses coverage.

This sentinel runs ``pytest --collect-only -q`` as a subprocess against
every ``tests/<tree>/`` we care about and asserts the trailing line
matches ``N tests collected`` (no ``ERROR collecting``). It's
intentionally subprocess-based because pytest cannot easily re-enter
itself in the same process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Trees that MUST collect cleanly. Order matters only for diagnostics.
# Note: ``tests/stack`` and ``tests/contract`` are marker-gated and
# sometimes parametrized over hardware — they're verified separately by
# their own CI workflows.
COLLECTABLE_TREES = (
    "tests/unit",
    "tests/integration",
    "tests/e2e",
    "tests/chaos",
    "tests/load",
    "tests/golden",
    "tests/_lint",
    "tests/_harness",
    "tests/_fakes",
)


@pytest.mark.parametrize("tree", COLLECTABLE_TREES)
def test_tree_collects_without_errors(tree: str) -> None:
    """Run ``pytest --collect-only`` on the tree; assert no collection errors."""
    tree_path = REPO_ROOT / tree
    if not tree_path.exists():
        pytest.skip(f"{tree!r} does not exist in this checkout")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tree_path),
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )

    combined = (proc.stdout + "\n" + proc.stderr).strip()
    # pytest --collect-only exits 0 on success, 2 on collection errors.
    if proc.returncode == 2 or "ERROR collecting" in combined or "errors during collection" in combined:
        pytest.fail(
            f"{tree!r} failed to collect cleanly. Tail:\n"
            f"{combined[-2000:]}"
        )

    # Sanity: a collectable tree should report N tests collected.
    last_line = next(
        (ln for ln in reversed(combined.splitlines()) if ln.strip()),
        "",
    )
    assert "collected" in last_line, (
        f"unexpected last line for {tree!r}: {last_line!r}\n\nfull tail:\n{combined[-2000:]}"
    )
