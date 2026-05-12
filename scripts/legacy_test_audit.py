#!/usr/bin/env python3
"""Phase 7 quarterly audit — stats on the legacy `packages/<pkg>/tests/`
lane.

Run from the repo root as ``make legacy-audit`` or
``python scripts/legacy_test_audit.py`` and read the resulting table.
The script is intentionally side-effect-free (read-only): it prints,
it does not delete. Deletion happens via PR per the policy in
``docs/adrs/2026-05-11-legacy-test-decommissioning.md``.

Outputs three sections:

1. **Per-package counts** — total test files, total ``def test_`` functions.
2. **Recency heuristic** — last git mtime per file, "files not touched in
   ≥365 days" is the prune candidate set.
3. **Flake hints** — files referenced by quarantine markers in
   ``tests/.telemetry/flake_board.json`` if present.

The script does NOT collect tests via pytest (heavy + slow); a static
AST scan is fast and accurate enough for the audit cadence.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"
TELEMETRY_DIR = REPO_ROOT / "tests" / ".telemetry"
FLAKE_BOARD = TELEMETRY_DIR / "flake_board.json"


@dataclass
class PackageStats:
    name: str
    test_files: int = 0
    test_functions: int = 0
    stale_files: int = 0  # not touched in ≥365 days
    flaky_files: int = 0
    files: list[Path] = field(default_factory=list)


def count_test_functions(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            count += 1
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test_"):
            count += 1
    return count


def last_commit_ts(path: Path) -> datetime | None:
    """Most recent git commit timestamp touching ``path``."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    try:
        return datetime.fromisoformat(out)
    except ValueError:
        return None


def load_flake_board() -> set[str]:
    if not FLAKE_BOARD.is_file():
        return set()
    try:
        data = json.loads(FLAKE_BOARD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    flaky: set[str] = set()
    for entry in data.get("entries", []):
        path = entry.get("path")
        if isinstance(path, str):
            flaky.add(path)
    return flaky


def gather() -> list[PackageStats]:
    flaky_set = load_flake_board()
    stale_cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    out: list[PackageStats] = []
    if not PACKAGES_DIR.is_dir():
        return out
    for pkg_dir in sorted(PACKAGES_DIR.iterdir()):
        if not pkg_dir.is_dir():
            continue
        tests_dir = pkg_dir / "tests"
        if not tests_dir.is_dir():
            continue
        stats = PackageStats(name=pkg_dir.name)
        for test_file in sorted(tests_dir.rglob("test_*.py")):
            stats.test_files += 1
            stats.test_functions += count_test_functions(test_file)
            stats.files.append(test_file)
            ts = last_commit_ts(test_file)
            if ts is not None and ts < stale_cutoff:
                stats.stale_files += 1
            rel = test_file.relative_to(REPO_ROOT).as_posix()
            if rel in flaky_set:
                stats.flaky_files += 1
        out.append(stats)
    return out


def format_report(packages: list[PackageStats]) -> str:
    if not packages:
        return "Legacy test lane is empty — nothing to audit.\n"
    lines: list[str] = []
    lines.append("Legacy test lane audit — `packages/<pkg>/tests/`")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"{'Package':<16} {'Files':>6} {'Tests':>6} {'Stale':>6} {'Flaky':>6}"
    )
    lines.append("-" * 60)
    totals = [0, 0, 0, 0]
    for stats in packages:
        lines.append(
            f"{stats.name:<16} {stats.test_files:>6} "
            f"{stats.test_functions:>6} {stats.stale_files:>6} "
            f"{stats.flaky_files:>6}"
        )
        totals[0] += stats.test_files
        totals[1] += stats.test_functions
        totals[2] += stats.stale_files
        totals[3] += stats.flaky_files
    lines.append("-" * 60)
    lines.append(
        f"{'TOTAL':<16} {totals[0]:>6} {totals[1]:>6} "
        f"{totals[2]:>6} {totals[3]:>6}"
    )
    lines.append("")
    lines.append("Legend:")
    lines.append("  Stale = file not touched in git for ≥365 days; prune candidate.")
    lines.append("  Flaky = file appears in tests/.telemetry/flake_board.json.")
    lines.append("")
    lines.append(
        "Exit criteria per ADR 2026-05-11: archive the legacy lane when "
        "the TOTAL `Tests` column drops below 100, OR when 12-month "
        "rolling stability hits 100% (no real failures in a year)."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the legacy `packages/<pkg>/tests/` lane."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of the human report.",
    )
    args = parser.parse_args()
    packages = gather()
    if args.json:
        payload = [
            {
                "name": p.name,
                "test_files": p.test_files,
                "test_functions": p.test_functions,
                "stale_files": p.stale_files,
                "flaky_files": p.flaky_files,
            }
            for p in packages
        ]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    sys.stdout.write(format_report(packages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
