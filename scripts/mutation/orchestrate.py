#!/usr/bin/env python3
"""Orchestrate Cosmic Ray on a set of changed production files.

This is the core of the per-PR mutation testing gate (tier 1 in
`docs/testing/mutation_testing.md`). It:

1. Loads `.mutation-hotspots.yml` for thresholds + test targets.
2. Takes a list of files (either `--files a.py b.py` or `--diff base_ref`).
3. Filters to production files (`packages/*/src/*.py`).
4. For each file:
     a. Rewrites a temp cosmic-ray.toml that points at the file + its
        unit test target.
     b. Runs `cosmic-ray init` to populate the session DB.
     c. Runs `cr-filter-operators` to drop the PEP 604 BitOr noise.
     d. Runs `cr-filter-pragma` to honour `# pragma: no mutate`.
     e. Estimates wall-clock cost; if budget exceeded, EMITS a clear
        message and skips (does NOT fail the build).
     f. Runs `cosmic-ray exec`.
     g. Reads kill rate via direct SQL (cosmic-ray dump crashes on
        SKIPPED jobs in 8.4.6).
5. Aggregates results to a structured JSON report.
6. Exit code 1 if any HOTSPOT is below threshold and mode == blocking;
   else 0 (advisory: print warnings but exit 0).

Designed to be safe to invoke from CI or as a manual pre-flight check.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in dev deps
    sys.stderr.write("error: PyYAML required. `uv sync --extra dev`.\n")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HOTSPOT_CONFIG = REPO_ROOT / ".mutation-hotspots.yml"
SESSIONS_DIR = REPO_ROOT / "scripts" / "mutation" / "sessions"
REPORTS_DIR = REPO_ROOT / "scripts" / "mutation" / "reports"
VENV_BIN = REPO_ROOT / ".venv" / "bin"


@dataclass
class FileResult:
    """Outcome of running mutation testing on one file."""

    path: str
    is_hotspot: bool
    tests: str
    threshold: float
    status: str  # ok | below_threshold | skipped_budget | skipped_no_tests | error
    mutations_total: int = 0
    mutations_skipped: int = 0  # filtered out (BitOr / pragma)
    mutations_executed: int = 0
    survived: int = 0
    killed: int = 0
    incompetent: int = 0
    timeout: int = 0
    kill_rate: float = 0.0
    effective_kill_rate: float = 0.0  # killed / (executed - incompetent)
    wall_clock_s: float = 0.0
    note: str = ""


@dataclass
class Report:
    base_ref: str
    mode: str
    files: list[FileResult] = field(default_factory=list)
    hotspots_below_threshold: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    overall_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_ref": self.base_ref,
            "mode": self.mode,
            "files": [asdict(f) for f in self.files],
            "hotspots_below_threshold": self.hotspots_below_threshold,
            "advisories": self.advisories,
            "overall_status": self.overall_status,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, decoding stdout/stderr as text."""
    kwargs.setdefault("text", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("cwd", REPO_ROOT)
    return subprocess.run(cmd, **kwargs)


def load_hotspots() -> dict[str, Any]:
    if not HOTSPOT_CONFIG.exists():
        sys.stderr.write(f"error: {HOTSPOT_CONFIG} not found\n")
        sys.exit(2)
    with HOTSPOT_CONFIG.open() as fh:
        return yaml.safe_load(fh)


def changed_python_files(base_ref: str) -> list[str]:
    """Files changed in the current branch vs base_ref, restricted to
    `packages/*/src/*.py`.
    """
    res = _run(["git", "diff", "--name-only", "--diff-filter=ACM", f"{base_ref}...HEAD"])
    if res.returncode != 0:
        sys.stderr.write(f"git diff failed:\n{res.stderr}\n")
        return []
    out: list[str] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.endswith(".py"):
            continue
        # Production files only: packages/<pkg>/src/...
        if not (line.startswith("packages/") and "/src/" in line):
            continue
        # Skip __init__.py — nothing meaningful to mutate.
        if line.endswith("/__init__.py"):
            continue
        if (REPO_ROOT / line).exists():
            out.append(line)
    return out


def hotspot_for(path: str, hotspots: list[dict[str, Any]]) -> dict[str, Any] | None:
    for h in hotspots:
        if h["path"] == path:
            return h
    return None


def _tests_exist(tests_spec: str) -> bool:
    """Verify the pytest target(s) exist on disk.

    `tests_spec` is a space-separated list (orchestrate-friendly) of
    file or directory paths.
    """
    return all((REPO_ROOT / t).exists() for t in tests_spec.split())


def _infer_tests(path: str) -> str | None:
    """Heuristic test path for a non-hotspot file.

    Maps `packages/<pkg>/src/ryotenkai_<pkg>/<rest>/<name>.py` to
    `tests/unit/<pkg>/<rest>/test_<name>.py` if that exists.
    """
    p = Path(path)
    parts = p.parts
    # packages/<pkg>/src/ryotenkai_<pkg>/...
    if len(parts) < 4 or parts[0] != "packages" or parts[2] != "src":
        return None
    pkg = parts[1]
    rest = parts[4:]  # past ryotenkai_<pkg>
    if not rest:
        return None
    name = rest[-1].replace(".py", "")
    candidates = [
        Path("tests/unit") / pkg / Path(*rest[:-1]) / f"test_{name}.py",
        Path("tests/unit") / pkg / f"test_{name}.py",
        Path("tests/unit") / pkg / Path(*rest[:-1]) / name,  # directory
    ]
    for c in candidates:
        if (REPO_ROOT / c).exists():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# Cosmic Ray driver
# ---------------------------------------------------------------------------


def _write_config(toml_path: Path, module_path: str, tests: str, timeout: float = 30.0) -> None:
    toml_path.write_text(
        f"""[cosmic-ray]
module-path = "{module_path}"
timeout = {timeout}
excluded-modules = []
test-command = ".venv/bin/python -m pytest -c tests/pytest.ini {tests} -x --no-header -q --disable-warnings"

[cosmic-ray.distributor]
name = "local"

[cosmic-ray.filters.operators-filter]
exclude-operators = [
    "core/ReplaceBinaryOperator_BitOr_.*",
    "core/ReplaceBinaryOperator_.*_BitOr",
]
"""
    )


def _count_pending(session_db: Path) -> int:
    """Count work_items without a result (i.e., still pending)."""
    with sqlite3.connect(session_db) as conn:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM work_items wi
            LEFT JOIN work_results wr ON wi.job_id = wr.job_id
            WHERE wr.job_id IS NULL
            """
        )
        return cur.fetchone()[0]


def _count_total(session_db: Path) -> int:
    with sqlite3.connect(session_db) as conn:
        return conn.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]


def _outcome_counts(session_db: Path) -> dict[str, int]:
    """Counts grouped by worker_outcome + test_outcome."""
    with sqlite3.connect(session_db) as conn:
        cur = conn.execute(
            "SELECT worker_outcome, test_outcome, COUNT(*) FROM work_results GROUP BY worker_outcome, test_outcome"
        )
        out: dict[str, int] = {}
        for worker, test, count in cur.fetchall():
            out[f"{worker}:{test}"] = count
        return out


def run_cosmic_ray_for_file(
    file_path: str,
    tests: str,
    threshold: float,
    is_hotspot: bool,
    budget_minutes: float,
    projected_mpm: float,
    timeout_per_mutation: float = 30.0,
    keep_session: bool = True,
) -> FileResult:
    """End-to-end mutation testing for a single file."""
    result = FileResult(
        path=file_path,
        is_hotspot=is_hotspot,
        tests=tests,
        threshold=threshold,
        status="error",
    )

    if not _tests_exist(tests):
        result.status = "skipped_no_tests"
        result.note = f"no test files found at: {tests}"
        return result

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Stable session name keyed by file path (slugified)
    slug = file_path.replace("/", "__").replace(".py", "")
    session_db = SESSIONS_DIR / f"{slug}.sqlite"
    if session_db.exists():
        session_db.unlink()

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tf:
        toml_path = Path(tf.name)
    _write_config(toml_path, file_path, tests, timeout=timeout_per_mutation)

    t0 = time.monotonic()
    try:
        # 1. init
        res = _run([str(VENV_BIN / "cosmic-ray"), "init", str(toml_path), str(session_db)])
        if res.returncode != 0:
            result.note = f"cosmic-ray init failed: {res.stderr[:500]}"
            return result

        total_before_filter = _count_total(session_db)

        # 2. filter operators (PEP 604 BitOr noise)
        res = _run([str(VENV_BIN / "cr-filter-operators"), str(session_db), str(toml_path)])
        if res.returncode != 0:
            # Non-fatal — log and continue
            result.note += f"[warn] cr-filter-operators failed: {res.stderr[:200]} "

        # 3. pragma filter (# pragma: no mutate) — best effort.
        # cr-filter-pragma takes ONLY a session arg (no config).
        if (VENV_BIN / "cr-filter-pragma").exists():
            res_pragma = _run([str(VENV_BIN / "cr-filter-pragma"), str(session_db)])
            if res_pragma.returncode != 0:
                result.note += f"[warn] cr-filter-pragma failed: {res_pragma.stderr[:200]} "

        pending = _count_pending(session_db)
        result.mutations_total = total_before_filter
        result.mutations_skipped = total_before_filter - pending

        # 4. Budget check
        projected_minutes = pending / projected_mpm if projected_mpm > 0 else 0
        if projected_minutes > budget_minutes:
            result.status = "skipped_budget"
            result.note = (
                f"projected {projected_minutes:.1f} min for {pending} mutations exceeds "
                f"budget {budget_minutes:.1f} min. Run nightly job instead, or split the PR."
            )
            return result

        # 5. exec
        res = _run(
            [str(VENV_BIN / "cosmic-ray"), "exec", str(toml_path), str(session_db)],
            timeout=int(budget_minutes * 60 * 1.5),
        )
        # exec can have non-zero exit even on success (one mutation timed out etc)
        if res.returncode not in (0, 1) and "Aborted" not in res.stderr:
            result.note += f"[warn] cosmic-ray exec rc={res.returncode}: {res.stderr[:200]} "

        result.wall_clock_s = time.monotonic() - t0

        # 6. tally
        counts = _outcome_counts(session_db)
        # worker_outcome values: NORMAL / SKIPPED / ABNORMAL / TIMEOUT / EXCEPTION
        # test_outcome values: KILLED / SURVIVED / INCOMPETENT
        result.killed = sum(v for k, v in counts.items() if k.endswith(":KILLED"))
        result.survived = sum(v for k, v in counts.items() if k.endswith(":SURVIVED"))
        result.incompetent = sum(v for k, v in counts.items() if k.endswith(":INCOMPETENT"))
        result.timeout = counts.get("TIMEOUT:None", 0)
        result.mutations_executed = result.killed + result.survived + result.incompetent + result.timeout

        if result.mutations_executed > 0:
            # `kill_rate` = killed / executed (no normalisation)
            result.kill_rate = result.killed / result.mutations_executed
            # `effective_kill_rate` excludes incompetent (mutations the
            # interpreter couldn't compile — Cosmic Ray treats them as
            # neither killed nor survived).
            denom = result.mutations_executed - result.incompetent
            result.effective_kill_rate = result.killed / denom if denom > 0 else 0.0
        else:
            result.kill_rate = 0.0
            result.effective_kill_rate = 0.0

        if result.effective_kill_rate >= threshold:
            result.status = "ok"
        else:
            result.status = "below_threshold"

        return result
    except subprocess.TimeoutExpired:
        result.status = "skipped_budget"
        result.note = "cosmic-ray exec timed out at hard budget"
        result.wall_clock_s = time.monotonic() - t0
        return result
    finally:
        with contextlib.suppress(OSError):
            toml_path.unlink()
        if not keep_session and session_db.exists():
            session_db.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--diff", metavar="BASE_REF", help="Run on files changed vs BASE_REF")
    src.add_argument("--files", nargs="+", help="Run on these explicit files")
    src.add_argument("--all-hotspots", action="store_true", help="Run on every hotspot in config")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "latest.json",
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat advisory mode as blocking (override hotspot config)",
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=None,
        help="Override per-PR budget (default: from .mutation-hotspots.yml)",
    )
    parser.add_argument(
        "--projected-mpm",
        type=float,
        default=None,
        help="Override projected mutations/minute (default: from config)",
    )
    args = parser.parse_args(argv)

    cfg = load_hotspots()
    hotspots: list[dict[str, Any]] = cfg.get("hotspots", [])
    mode: str = cfg.get("mode", "advisory")
    if args.strict:
        mode = "blocking"
    default_warn: float = cfg.get("default_kill_rate_warning", 0.5)
    budget_minutes: float = args.budget_minutes or cfg.get("budget_minutes_per_pr", 25)
    projected_mpm: float = args.projected_mpm or cfg.get("projected_mutations_per_minute", 25)

    # Resolve input files
    base_ref = "(explicit)"
    if args.all_hotspots:
        files = [h["path"] for h in hotspots]
        base_ref = "(all hotspots)"
    elif args.diff:
        base_ref = args.diff
        files = changed_python_files(args.diff)
    else:
        files = [f for f in args.files if (REPO_ROOT / f).exists()]

    if not files:
        print("No production files to mutate. Skipping.")
        report = Report(base_ref=base_ref, mode=mode, overall_status="ok")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2))
        return 0

    print(f"Mode: {mode}")
    print(f"Files to mutate ({len(files)}):")
    for f in files:
        print(f"  - {f}")

    report = Report(base_ref=base_ref, mode=mode)
    total_budget = budget_minutes  # Hard cap on aggregate per-PR cost
    elapsed = 0.0

    for fpath in files:
        h = hotspot_for(fpath, hotspots)
        if h is not None:
            tests = h["tests"]
            threshold = h["min_kill_rate"]
            is_hotspot = True
        else:
            inferred = _infer_tests(fpath)
            if inferred is None:
                fr = FileResult(
                    path=fpath,
                    is_hotspot=False,
                    tests="",
                    threshold=default_warn,
                    status="skipped_no_tests",
                    note="no inferred test path",
                )
                report.files.append(fr)
                print(f"  [SKIP] {fpath} — no inferred test path")
                continue
            tests = inferred
            threshold = default_warn
            is_hotspot = False

        remaining_budget = max(total_budget - elapsed, 1.0)
        print(f"\n>>> {fpath} (hotspot={is_hotspot}, threshold={threshold:.2f}, budget_left={remaining_budget:.1f}m)")
        fr = run_cosmic_ray_for_file(
            fpath,
            tests,
            threshold,
            is_hotspot,
            remaining_budget,
            projected_mpm,
        )
        elapsed += fr.wall_clock_s / 60.0
        report.files.append(fr)
        print(
            f"    status={fr.status} executed={fr.mutations_executed} "
            f"killed={fr.killed} survived={fr.survived} "
            f"kill_rate={fr.effective_kill_rate:.2%} ({fr.wall_clock_s:.1f}s)"
        )
        if fr.note:
            print(f"    note: {fr.note.strip()}")

        if fr.status == "below_threshold":
            msg = f"{fpath}: kill_rate {fr.effective_kill_rate:.2%} < threshold {threshold:.2%}"
            if is_hotspot:
                report.hotspots_below_threshold.append(msg)
            else:
                report.advisories.append(msg)

    # Decide overall status
    if report.hotspots_below_threshold and mode == "blocking":
        report.overall_status = "fail"
    elif report.hotspots_below_threshold:
        report.overall_status = "advisory"
    else:
        report.overall_status = "ok"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nReport: {args.report}")

    if report.hotspots_below_threshold:
        print("\nHotspots BELOW threshold:")
        for m in report.hotspots_below_threshold:
            print(f"  - {m}")
    if report.advisories:
        print("\nAdvisories (non-blocking):")
        for m in report.advisories:
            print(f"  - {m}")

    if report.overall_status == "fail":
        print("\nFAIL: one or more hotspots below threshold in blocking mode.")
        return 1
    if report.overall_status == "advisory":
        print("\nADVISORY: hotspots below threshold, but mode=advisory (PR not blocked).")
        return 0
    print("\nAll mutation gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
