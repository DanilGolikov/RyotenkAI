#!/usr/bin/env python3
"""Batch smoke-test runner.

Discovers all *.yaml configs under a given directory (recursively),
launches each as an independent ``python -m src.main train`` subprocess
in parallel, and produces a Markdown summary report.

All runs are grouped into ``runs/smoke_<id>/`` via the
``RYOTENKAI_RUNS_DIR`` env var — the orchestrator creates run
directories directly inside the smoke folder.

Liveness detection replaces a hard per-run timeout: while the
subprocess keeps producing stdout/stderr output it is considered
alive.  When no output arrives for ``--idle-timeout`` seconds the
runner triggers a graceful shutdown sequence:

    SIGINT → wait 60 s → SIGINT → wait 60 s → SIGKILL

Usage::

    python scripts/batch_smoke.py /path/to/FUNCS_CHECKS
    python scripts/batch_smoke.py /path/to/FUNCS_CHECKS --workers 4 --idle-timeout 600
    python scripts/batch_smoke.py /path/to/FUNCS_CHECKS --workers -1  # 1 worker per config
    python scripts/batch_smoke.py /path/to/FUNCS_CHECKS --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import string
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_WORKERS = 4
_DEFAULT_IDLE_TIMEOUT_S = 1200  # 20 min without output → dead
_DEFAULT_STAGGER_S = 5  # delay between launching successive runs
_GRACEFUL_WAIT_S = 60  # wait after each SIGINT
_LIVENESS_POLL_S = 5  # check interval
_REPORT_FILENAME = "smoke_report.md"
_SMOKE_ID_LEN = 5
_RUNS_BASE_ENV = "RYOTENKAI_RUNS_DIR"


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


# ── Error extraction ─────────────────────────────────────────────────────────

_LOG_PREFIX_RE = re.compile(r"^(?:ERROR|WARNING|INFO)\s*-\s*")
_STAGE_FAILED_RE = re.compile(r"Stage '[^']+' failed:\s*")
_ERROR_CODE_RE = re.compile(r"\[[A-Z_]+\]\s*")


def _clean_error_message(raw: str) -> str:
    """Strip pipeline wrapper noise, keep the meaningful error."""
    msg = _LOG_PREFIX_RE.sub("", raw)
    for prefix in ("Pipeline failed: ", "pipeline failed: "):
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
    msg = _STAGE_FAILED_RE.sub("", msg, count=1)
    msg = _ERROR_CODE_RE.sub("", msg)
    return msg.strip()


def _extract_error(stderr_text: str, stdout_text: str) -> str:
    """Extract the most useful error from process output.

    Handles multi-line errors (e.g. JSON payloads after "Pipeline failed:").
    """
    combined = stdout_text + "\n" + stderr_text
    lines = combined.splitlines()

    for idx in range(len(lines) - 1, -1, -1):
        lower = lines[idx].lower()
        if "pipeline failed:" not in lower and "configuration error:" not in lower:
            continue

        raw = lines[idx].strip()
        if "  " in raw and raw[0].isdigit():
            parts = raw.split("  ", 2)
            if len(parts) >= 3:
                raw = parts[-1].strip()

        brace_depth = raw.count("{") - raw.count("}")
        collected = [raw]
        j = idx + 1
        while brace_depth > 0 and j < len(lines):
            cont = lines[j].strip()
            brace_depth += cont.count("{") - cont.count("}")
            collected.append(cont)
            j += 1

        full_msg = "\n".join(collected)
        return _clean_error_message(full_msg)

    for line in reversed(lines):
        if "error:" in line.lower():
            raw = line.strip()
            if "  " in raw and raw[0].isdigit():
                parts = raw.split("  ", 2)
                if len(parts) >= 3:
                    return _clean_error_message(parts[-1].strip())
            return _clean_error_message(raw)

    return ""


# ── Run dir discovery ────────────────────────────────────────────────────────

def _discover_run_dirs(smoke_dir: Path) -> dict[str, Path]:
    """Map config_path → run directory by reading pipeline_state.json files."""
    result: dict[str, Path] = {}
    if not smoke_dir.is_dir():
        return result
    for entry in smoke_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("run_"):
            continue
        state_file = entry / "pipeline_state.json"
        if not state_file.exists():
            continue
        try:
            data = json.loads(state_file.read_text())
            cfg = data.get("config_path", "")
            if cfg:
                result[cfg] = entry
        except (json.JSONDecodeError, OSError):
            continue
    return result


def _assign_run_dirs(results: list[RunResult], smoke_dir: Path) -> None:
    """Match each RunResult to its run directory inside smoke_dir."""
    mapping = _discover_run_dirs(smoke_dir)
    for r in results:
        cfg_str = str(r.config_path.resolve())
        if cfg_str in mapping:
            r.run_dir = str(mapping[cfg_str])
            continue
        if str(r.config_path) in mapping:
            r.run_dir = str(mapping[str(r.config_path)])


# ── Graceful shutdown ────────────────────────────────────────────────────────

def _graceful_stop(proc: subprocess.Popen[str], config_rel: str) -> None:
    """SIGINT → 60 s → SIGINT → 60 s → SIGKILL."""
    try:
        proc.send_signal(signal.SIGINT)
        print(f"    [{config_rel}] sent SIGINT (1/2), waiting {_GRACEFUL_WAIT_S}s…")
    except OSError:
        return
    try:
        proc.wait(timeout=_GRACEFUL_WAIT_S)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.send_signal(signal.SIGINT)
        print(f"    [{config_rel}] sent SIGINT (2/2), waiting {_GRACEFUL_WAIT_S}s…")
    except OSError:
        return
    try:
        proc.wait(timeout=_GRACEFUL_WAIT_S)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.kill()
        print(f"    [{config_rel}] sent SIGKILL")
        proc.wait(timeout=10)
    except OSError:
        pass


# ── Stream reader ────────────────────────────────────────────────────────────

class _LivenessReader:
    """Reads a subprocess stream in a background thread and tracks last output time."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.last_output_at: float = time.monotonic()
        self._lock = threading.Lock()

    def feed(self, stream) -> None:
        """Read *stream* line-by-line until EOF, updating liveness timestamp."""
        for line in stream:
            self.lines.append(line)
            with self._lock:
                self.last_output_at = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self.last_output_at


# ── Single run ───────────────────────────────────────────────────────────────

def _run_single(
    config: Path,
    root: Path,
    idle_timeout: int,
    project_root: Path,
    env: dict[str, str],
) -> RunResult:
    rel = str(config.relative_to(root))
    result = RunResult(config_path=config, config_rel=rel)

    cmd = [
        sys.executable, "-m", "src.main", "train",
        "--config", str(config),
    ]

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(project_root),
        env=env,
        start_new_session=True,
    )

    stdout_reader = _LivenessReader()
    stderr_reader = _LivenessReader()

    t_out = threading.Thread(target=stdout_reader.feed, args=(proc.stdout,), daemon=True)
    t_err = threading.Thread(target=stderr_reader.feed, args=(proc.stderr,), daemon=True)
    t_out.start()
    t_err.start()

    idle_triggered = False
    while proc.poll() is None:
        time.sleep(_LIVENESS_POLL_S)
        idle = min(stdout_reader.idle_seconds, stderr_reader.idle_seconds)
        if idle >= idle_timeout:
            print(f"    [{rel}] no output for {int(idle)}s — starting graceful shutdown")
            _graceful_stop(proc, rel)
            idle_triggered = True
            break

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    result.exit_code = proc.returncode
    result.duration_s = time.monotonic() - t0

    stdout_text = "".join(stdout_reader.lines)
    stderr_text = "".join(stderr_reader.lines)

    if idle_triggered:
        result.timed_out = True
        result.error_summary = f"No output for {idle_timeout}s — terminated via graceful shutdown"
    elif result.exit_code != 0:
        result.error_summary = _extract_error(stderr_text, stdout_text)

    return result


# ── Report ───────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _build_report(
    results: list[RunResult],
    root: Path,
    elapsed: float,
    smoke_dir: Path,
    idle_timeout: int,
) -> str:
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Smoke Test Report")
    lines.append("")
    lines.append(f"**Date**: {ts}  ")
    lines.append(f"**Configs root**: `{root}`  ")
    lines.append(f"**Runs dir**: `{smoke_dir}`  ")
    lines.append(f"**Total time**: {_fmt_duration(elapsed)}  ")
    lines.append(f"**Idle timeout**: {idle_timeout}s  ")
    lines.append(f"**Configs**: {len(results)} | **Passed**: {passed} | **Failed**: {failed}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| # | Config | Status | Duration | Run dir |")
    lines.append("|---|--------|--------|----------|---------|")
    for i, r in enumerate(results, 1):
        status = "PASSED" if r.passed else "**FAILED**"
        if r.timed_out:
            status = "**IDLE**"
        dur = _fmt_duration(r.duration_s)
        run_name = Path(r.run_dir).name if r.run_dir else "—"
        lines.append(f"| {i} | `{r.config_rel}` | {status} | {dur} | `{run_name}` |")
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
                lines.append(f"**Idle timeout**: no output for {idle_timeout}s — graceful shutdown  ")
            if r.error_summary:
                lines.append(f"```\n{r.error_summary}\n```")
            else:
                lines.append("_No error details captured. Check pipeline.log in run directory._")
            lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch smoke-test runner for pipeline configs")
    parser.add_argument("config_dir", type=Path, help="Directory with *.yaml configs (searched recursively)")
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS, help=f"Max parallel runs (default: {_DEFAULT_WORKERS}). Use -1 for unlimited (1 worker per config).")
    parser.add_argument("--idle-timeout", type=int, default=_DEFAULT_IDLE_TIMEOUT_S, help=f"Seconds without output before graceful shutdown (default: {_DEFAULT_IDLE_TIMEOUT_S})")
    parser.add_argument("--stagger", type=int, default=_DEFAULT_STAGGER_S, help=f"Seconds between launching successive runs (default: {_DEFAULT_STAGGER_S}). Use 0 to start all at once.")
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
    idle_timeout: int = args.idle_timeout
    workers: int = len(configs) if args.workers == -1 else args.workers

    print(f"Found {len(configs)} configs in {config_dir}")
    stagger_s: int = args.stagger
    print(f"Workers: {workers}{' (unlimited)' if args.workers == -1 else ''}, idle timeout: {idle_timeout}s, stagger: {stagger_s}s")
    print(f"Smoke dir: {smoke_dir}")
    print()

    if args.dry_run:
        for i, c in enumerate(configs, 1):
            print(f"  {i}. {c.relative_to(config_dir)}")
        print(f"\nDry run: {len(configs)} configs would be launched into {smoke_dir}")
        sys.exit(0)

    smoke_dir.mkdir(parents=True, exist_ok=True)

    child_env = {**os.environ, _RUNS_BASE_ENV: str(smoke_dir)}

    results: list[RunResult] = []
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[threading.Future, Path] = {}  # type: ignore[type-arg]
        for i, cfg in enumerate(configs):
            futures[pool.submit(_run_single, cfg, config_dir, idle_timeout, project_root, child_env)] = cfg
            if stagger_s > 0 and i < len(configs) - 1:
                time.sleep(stagger_s)
        for future in as_completed(futures):
            r = future.result()
            _assign_run_dirs([r], smoke_dir)
            status = "PASSED" if r.passed else "FAILED"
            if r.timed_out:
                status = "IDLE"
            run_name = Path(r.run_dir).name if r.run_dir else "—"
            print(f"  [{status}] {r.config_rel} ({_fmt_duration(r.duration_s)}) → {run_name}")
            results.append(r)

    elapsed = time.monotonic() - t_start

    config_order = {str(c): i for i, c in enumerate(configs)}
    results.sort(key=lambda r: config_order.get(str(r.config_path), 999))

    report = _build_report(results, config_dir, elapsed, smoke_dir, idle_timeout)

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
