#!/usr/bin/env python3
"""Compare today's mutation kill rates against the committed baseline.

Inputs:
  - scripts/mutation/ratchet_baseline.json  (committed)
  - latest report (default: most recent file under reports/)

Rules:
  - If a baselined module's kill rate drops by more than --tolerance (default
    5 percentage points), exit 1.
  - If a baselined module's kill rate IMPROVES by more than --improvement
    threshold (default 2 pp), suggest updating the baseline (exit 0 but
    print a notice).
  - New modules (in latest but not in baseline) are noted but never fail.
  - Missing modules (in baseline but not in latest) are warnings.

Usage:
  python scripts/mutation/check_ratchet.py
  python scripts/mutation/check_ratchet.py --report path/to/report.json
  python scripts/mutation/check_ratchet.py --update-baseline    # rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE = REPO_ROOT / "scripts" / "mutation" / "ratchet_baseline.json"
REPORTS_DIR = REPO_ROOT / "scripts" / "mutation" / "reports"


def latest_report() -> Path:
    candidates = sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.stderr.write(f"No reports found under {REPORTS_DIR}\n")
        sys.exit(2)
    return candidates[0]


def _load(path: Path) -> dict:
    if not path.exists():
        return {"files": []}
    return json.loads(path.read_text())


def _index(report: dict) -> dict[str, float]:
    """Map {path -> effective_kill_rate} for files that executed mutations."""
    out: dict[str, float] = {}
    for f in report.get("files", []):
        if f.get("mutations_executed", 0) > 0:
            out[f["path"]] = f.get("effective_kill_rate", 0.0)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="Path to today's report JSON")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Max allowed kill-rate drop (default 0.05)")
    parser.add_argument(
        "--improvement",
        type=float,
        default=0.02,
        help="Min improvement that triggers a baseline-update suggestion (default 0.02)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite the baseline file with today's rates (use in a separate PR).",
    )
    args = parser.parse_args(argv)

    report_path = args.report or latest_report()
    today = _load(report_path)
    today_idx = _index(today)
    base = _load(BASELINE).get("files", {}) if BASELINE.exists() else {}

    if args.update_baseline:
        try:
            src_path = str(report_path.relative_to(REPO_ROOT))
        except ValueError:
            src_path = str(report_path)
        payload = {
            "tolerance": args.tolerance,
            "files": today_idx,
            "source_report": src_path,
        }
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(payload, indent=2))
        print(f"Baseline written: {BASELINE} ({len(today_idx)} entries)")
        return 0

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(f"Comparing {_rel(report_path)} vs {_rel(BASELINE)}")
    print(f"Tolerance: {args.tolerance:.2%}")

    failures: list[str] = []
    improvements: list[str] = []
    new_files: list[str] = []
    missing: list[str] = []

    for path, today_rate in today_idx.items():
        base_rate = base.get(path)
        if base_rate is None:
            new_files.append(f"{path}: new (today={today_rate:.2%})")
            continue
        drop = base_rate - today_rate
        if drop > args.tolerance:
            failures.append(
                f"{path}: today={today_rate:.2%} baseline={base_rate:.2%} drop={drop:.2%} > tolerance"
            )
        elif drop < -args.improvement:
            improvements.append(
                f"{path}: today={today_rate:.2%} baseline={base_rate:.2%} improved={-drop:.2%}"
            )

    for path in base:
        if path not in today_idx:
            missing.append(path)

    if failures:
        print("\nRATCHET VIOLATIONS:")
        for m in failures:
            print(f"  - {m}")
    if improvements:
        print("\nRatchet improvements (consider --update-baseline):")
        for m in improvements:
            print(f"  - {m}")
    if new_files:
        print("\nNew files (not in baseline yet):")
        for m in new_files:
            print(f"  - {m}")
    if missing:
        print("\nMissing from today's report (in baseline only):")
        for p in missing:
            print(f"  - {p}")

    if failures:
        return 1
    print("\nRatchet check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
