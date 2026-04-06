from __future__ import annotations

import os
from pathlib import Path

from src.tui.adapters.state import get_running_attempt_no, latest_attempt_no, predict_next_attempt_no


def read_run_lock_pid(run_dir: Path) -> int | None:
    lock_path = run_dir.expanduser().resolve() / "run.lock"
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    first_line = raw.splitlines()[0].strip()
    candidate = first_line.split("=", maxsplit=1)[-1].strip()
    try:
        return int(candidate)
    except ValueError:
        return None


def resolve_interrupt_pid(run_dir: Path, launch_pid: int | None) -> int | None:
    if launch_pid is not None:
        return launch_pid
    pid = read_run_lock_pid(run_dir)
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return None
    return pid


def predict_attempt_for_launch(run_dir: Path) -> int:
    return predict_next_attempt_no(run_dir)


def resolve_attempt_to_open(run_dir: Path, attempt_no: int | None = None) -> int:
    if attempt_no is not None:
        return attempt_no
    latest = latest_attempt_no(run_dir)
    return latest or 1


def running_attempt_for_run(run_dir: Path) -> int | None:
    return get_running_attempt_no(run_dir)
