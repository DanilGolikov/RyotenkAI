#!/usr/bin/env python3
"""Auto-flip ``.mutation-hotspots.yml`` from ``mode: advisory`` to ``blocking``.

Reads recent nightly + per-PR mutation-testing run history from GitHub
Actions and patches the mode field once the stability bar is met. Used
weekly by ``.github/workflows/mutation-flip-check.yml``; can also be
invoked locally for inspection.

Flip criteria (all must hold):
- ``mode: advisory`` in ``.mutation-hotspots.yml`` (already-blocking is a no-op)
- Last 4 nightly runs (``mutation-nightly.yml``) finished with ``conclusion=success``
- Last 14 days of per-PR runs (``mutation-pr.yml``) report zero ``failure``
  outcomes attributable to the mutation gate (i.e. ``conclusion=success``
  or ``cancelled``)
- ``gh`` CLI available + authenticated

When the bar is met the script:
1. Patches ``.mutation-hotspots.yml`` (single-line replacement).
2. Stages the change.
3. Opens a PR via ``gh pr create``.
4. Returns exit 0.

When the bar is NOT met the script prints a one-line status (which
condition failed) and returns 0 (no error — the check ran, conditions
just weren't met). Return 2 only on unexpected errors (gh missing,
malformed config, etc).

Always reversible — never deletes anything. Worst case: the auto-PR is
closed; the workflow tries again next week.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOTSPOTS_PATH = REPO_ROOT / ".mutation-hotspots.yml"

NIGHTLY_WORKFLOW = "mutation-nightly.yml"
PR_WORKFLOW = "mutation-pr.yml"

NIGHTLY_REQUIRED = 4
PR_WINDOW_DAYS = 14


@dataclass
class RunSummary:
    workflow: str
    conclusion: str  # success / failure / cancelled / etc
    created_at: str

    @property
    def created(self) -> datetime:
        return datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))


def _have_gh() -> bool:
    return shutil.which("gh") is not None


def _gh_runs(workflow: str, limit: int = 50) -> list[RunSummary]:
    cmd = [
        "gh",
        "run",
        "list",
        "--workflow",
        workflow,
        "--limit",
        str(limit),
        "--json",
        "conclusion,createdAt",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if res.returncode != 0:
        sys.stderr.write(f"gh run list failed for {workflow}: {res.stderr.strip()}\n")
        return []
    try:
        payload = json.loads(res.stdout)
    except json.JSONDecodeError:
        return []
    return [
        RunSummary(workflow=workflow, conclusion=item.get("conclusion") or "", created_at=item.get("createdAt") or "")
        for item in payload
        if item.get("conclusion")  # skip in-progress (conclusion=None)
    ]


def _current_mode() -> str:
    if not HOTSPOTS_PATH.exists():
        sys.stderr.write(f"{HOTSPOTS_PATH} not found\n")
        sys.exit(2)
    for ln in HOTSPOTS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = ln.strip()
        if stripped.startswith("mode:"):
            return stripped.split(":", 1)[1].strip()
    sys.stderr.write(f"`mode:` key not found in {HOTSPOTS_PATH}\n")
    sys.exit(2)


def _patch_to_blocking() -> None:
    """Replace ``mode: advisory`` with ``mode: blocking`` in place."""
    text = HOTSPOTS_PATH.read_text(encoding="utf-8")
    new_lines: list[str] = []
    patched = False
    for ln in text.splitlines(keepends=True):
        if ln.strip().startswith("mode:") and "advisory" in ln and not patched:
            indent = ln[: len(ln) - len(ln.lstrip())]
            new_lines.append(f"{indent}mode: blocking  # auto-flipped by maybe_flip_to_blocking.py\n")
            patched = True
        else:
            new_lines.append(ln)
    HOTSPOTS_PATH.write_text("".join(new_lines), encoding="utf-8")


def _gh_pr_create(summary: str) -> int:
    branch_name = f"mutation/flip-blocking-{datetime.now(UTC):%Y%m%d}"
    cmds = [
        ["git", "switch", "-c", branch_name],
        ["git", "add", str(HOTSPOTS_PATH.relative_to(REPO_ROOT))],
        ["git", "commit", "-m", "chore(mutation): auto-flip hotspots mode advisory → blocking\n\n" + summary],
        ["git", "push", "-u", "origin", branch_name],
        ["gh", "pr", "create", "--title", "Auto-flip mutation hotspots to blocking", "--body", summary],
    ]
    for cmd in cmds:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if res.returncode != 0:
            sys.stderr.write(f"FAILED: {' '.join(cmd)}\n{res.stderr}\n")
            return res.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print diagnostic info; do not patch or PR")
    args = parser.parse_args(argv)

    current = _current_mode()
    if current == "blocking":
        print("Already blocking. No action.")
        return 0
    if current != "advisory":
        print(f"Unexpected mode {current!r}; no action.")
        return 0

    if not _have_gh():
        print("gh CLI not available; cannot read run history. No action.")
        return 0

    nightly_runs = _gh_runs(NIGHTLY_WORKFLOW, limit=10)
    recent_nightlies = nightly_runs[:NIGHTLY_REQUIRED]
    if len(recent_nightlies) < NIGHTLY_REQUIRED:
        print(
            f"Found only {len(recent_nightlies)} recent nightly runs (need {NIGHTLY_REQUIRED}). "
            f"Wait for more nightly history."
        )
        return 0

    bad_nightlies = [r for r in recent_nightlies if r.conclusion != "success"]
    if bad_nightlies:
        print(
            f"{len(bad_nightlies)} of last {NIGHTLY_REQUIRED} nightlies are not green "
            f"({[r.conclusion for r in bad_nightlies]}); not flipping."
        )
        return 0

    cutoff = datetime.now(UTC) - timedelta(days=PR_WINDOW_DAYS)
    pr_runs = [r for r in _gh_runs(PR_WORKFLOW, limit=100) if r.created >= cutoff]
    pr_failures = [r for r in pr_runs if r.conclusion == "failure"]
    if pr_failures:
        print(
            f"{len(pr_failures)} per-PR mutation runs failed in the last {PR_WINDOW_DAYS} days. "
            f"Not flipping until all clean."
        )
        return 0

    summary = (
        f"Last {NIGHTLY_REQUIRED} nightly runs of `{NIGHTLY_WORKFLOW}` are green; "
        f"zero failed per-PR runs of `{PR_WORKFLOW}` in the last {PR_WINDOW_DAYS} days. "
        f"Promoting `.mutation-hotspots.yml` from `advisory` → `blocking`.\n\n"
        f"Per the policy in `docs/testing/mutation_testing.md`, this means hotspot "
        f"kill-rate floors are now enforced on every PR."
    )

    if args.dry_run:
        print("DRY RUN — would flip to blocking.")
        print("\nSummary:\n" + summary)
        return 0

    print("Conditions met. Flipping mode → blocking and opening PR...")
    _patch_to_blocking()
    if os.environ.get("GITHUB_ACTIONS"):
        return _gh_pr_create(summary)
    print("Local run — patched the file but not opening PR (set GITHUB_ACTIONS=1 to enable).")
    print("Summary:\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
