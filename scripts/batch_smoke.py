#!/usr/bin/env python3
"""Batch smoke-test runner.

Discovers all *.yaml configs under a given directory (recursively),
launches each as an independent ``python -m src.main train`` subprocess
in parallel, and produces a Markdown summary report.

All runs are grouped into ``runs/smoke_<id>/`` so they don't pollute
the main runs directory.

Usage::

    python scripts/batch_smoke.py /path/to/FUNCS_CHECKS
    python scripts/batch_smoke.py /path/to/FUNCS_CHECKS --workers 4 --timeout 900
    python scripts/batch_smoke.py /path/to/FUNCS_CHECKS --dry-run
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_WORKERS = 4
_DEFAULT_TIMEOUT_S = 900  # 15 min per config
_REPORT_FILENAME = "smoke_report.md"
_SMOKE_ID_LEN = 5


def _generate_smoke_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=_SMOKE_ID_LEN))


@dataclass
class RunResult:
    config_path: Path
    config_rel: str
    exit_code: int | None = None
    duration_s: float = 0.0
    error_summary: str = ""
    run_dir: str = ""
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


def _discover_configs(root: Path) -> list[Path]:
    configs = sorted(root.rglob("*.yaml"))
    configs.extend(sorted(root.rglob("*.yml")))
    return configs


def _find_run_dir(output: str) -> str:
    """Extract the run directory path created by the orchestrator."""
    for line in output.splitlines():
        if "runs/run_" in line:
            for token in line.split():
                if "runs/run_" in token:
                    return token.strip()
    return ""


def _extract_error(stderr_text: str, stdout_text: str) -> str:
    """Extract the most useful error line from output."""
    combined = stdout_text + "\n" + stderr_text
    for line in reversed(combined.splitlines()):
        lower = line.lower()
        if "pipeline failed:" in lower or "error:" in lower:
            cleaned = line.strip()
            if "  " in cleaned and cleaned[0].isdigit():
                parts = cleaned.split("  ", 2)
                if len(parts) >= 3:
                    return parts[-1].strip()
            return cleaned
    return ""


def _run_single(config: Path, root: Path, timeout: int, project_root: Path) -> RunResult:
    rel = str(config.relative_to(root))
    result = RunResult(config_path=config, config_rel=rel)

    cmd = [
        sys.executable, "-m", "src.main", "train",
        "--config", str(config),
    ]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
            env=None,
        )
        result.exit_code = proc.returncode
        result.duration_s = time.monotonic() - t0
        if proc.returncode != 0:
            result.error_summary = _extract_error(proc.stderr, proc.stdout)
        result.run_dir = _find_run_dir(proc.stderr + proc.stdout)
    except subprocess.TimeoutExpired:
        result.duration_s = time.monotonic() - t0
        result.exit_code = -1
        result.timed_out = True
        result.error_summary = f"Timed out after {timeout}s"

    return result


def _move_run_to_smoke_dir(result: RunResult, smoke_dir: Path, project_root: Path) -> None:
    """Move the run directory into the smoke group folder."""
    if not result.run_dir:
        return
    # run_dir might be absolute or relative
    run_path = Path(result.run_dir)
    if not run_path.is_absolute():
        run_path = project_root / run_path
    run_path = run_path.resolve()
    if not run_path.is_dir():
        return

    dest = smoke_dir / run_path.name
    try:
        shutil.move(str(run_path), str(dest))
        result.run_dir = str(dest.relative_to(project_root))
    except OSError:
        pass  # keep original path if move fails


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _build_report(results: list[RunResult], root: Path, elapsed: float, smoke_dir: Path) -> str:
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Smoke Test Report")
    lines.append("")
    lines.append(f"**Date**: {ts}  ")
    lines.append(f"**Configs root**: `{root}`  ")
    lines.append(f"**Runs dir**: `{smoke_dir}`  ")
    lines.append(f"**Total time**: {_fmt_duration(elapsed)}  ")
    lines.append(f"**Configs**: {len(results)} | **Passed**: {passed} | **Failed**: {failed}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| # | Config | Status | Duration |")
    lines.append("|---|--------|--------|----------|")
    for i, r in enumerate(results, 1):
        status = "PASSED" if r.passed else "**FAILED**"
        if r.timed_out:
            status = "**TIMEOUT**"
        dur = _fmt_duration(r.duration_s)
        lines.append(f"| {i} | `{r.config_rel}` | {status} | {dur} |")
    lines.append("")

    failed_results = [r for r in results if not r.passed]
    if failed_results:
        lines.append("## Failed Runs")
        lines.append("")
        for r in failed_results:
            lines.append(f"### `{r.config_rel}`")
            lines.append("")
            if r.run_dir:
                lines.append(f"**Run dir**: `{r.run_dir}`  ")
            if r.timed_out:
                lines.append(f"**Timeout**: process killed after {_DEFAULT_TIMEOUT_S}s  ")
            if r.error_summary:
                lines.append(f"```\n{r.error_summary}\n```")
            else:
                lines.append("_No error details captured. Check pipeline.log in run directory._")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch smoke-test runner for pipeline configs")
    parser.add_argument("config_dir", type=Path, help="Directory with *.yaml configs (searched recursively)")
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS, help=f"Max parallel runs (default: {_DEFAULT_WORKERS})")
    parser.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_S, help=f"Per-config timeout in seconds (default: {_DEFAULT_TIMEOUT_S})")
    parser.add_argument("--report-dir", type=Path, default=None, help="Where to save the report (default: smoke dir)")
    parser.add_argument("--dry-run", action="store_true", help="List discovered configs without running them")
    args = parser.parse_args()

    config_dir: Path = args.config_dir.resolve()
    if not config_dir.is_dir():
        print(f"Error: {config_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[1]
    configs = _discover_configs(config_dir)
    if not configs:
        print(f"No *.yaml / *.yml configs found in {config_dir}", file=sys.stderr)
        sys.exit(1)

    smoke_id = _generate_smoke_id()
    smoke_dir = (project_root / "runs" / f"smoke_{smoke_id}").resolve()

    print(f"Found {len(configs)} configs in {config_dir}")
    print(f"Workers: {args.workers}, timeout: {args.timeout}s per config")
    print(f"Smoke dir: {smoke_dir}")
    print()

    if args.dry_run:
        for i, c in enumerate(configs, 1):
            print(f"  {i}. {c.relative_to(config_dir)}")
        print(f"\nDry run: {len(configs)} configs would be launched into {smoke_dir}")
        sys.exit(0)

    smoke_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_single, cfg, config_dir, args.timeout, project_root): cfg
            for cfg in configs
        }
        for future in as_completed(futures):
            r = future.result()
            status = "PASSED" if r.passed else "FAILED"
            print(f"  [{status}] {r.config_rel} ({_fmt_duration(r.duration_s)})")
            results.append(r)

    elapsed = time.monotonic() - t_start

    # Move all created run dirs into the smoke group folder
    for r in results:
        _move_run_to_smoke_dir(r, smoke_dir, project_root)

    # Sort by original config order for stable report
    config_order = {str(c): i for i, c in enumerate(configs)}
    results.sort(key=lambda r: config_order.get(str(r.config_path), 999))

    report = _build_report(results, config_dir, elapsed, smoke_dir)

    report_dir = (args.report_dir or smoke_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / _REPORT_FILENAME
    report_path.write_text(report)

    print()
    print(report)
    print(f"\nReport saved to: {report_path}")

    failed_count = sum(1 for r in results if not r.passed)
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
