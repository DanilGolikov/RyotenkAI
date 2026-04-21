"""
Detached pipeline launch and interrupt mechanics.

This module owns the subprocess lifecycle for starting a pipeline run out-of-process.
Launched processes use start_new_session=True so that they survive the launcher
(TUI or web backend) dying, which preserves the Shared-State architecture: the
source of truth is on disk (pipeline_state.json + run.lock), not in-memory.

Previously lived in src/tui/launch.py. That path now exposes a shim for backward
compatibility with existing TUI imports.
"""

from __future__ import annotations

import collections
import contextlib
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.pipeline.launch_queries import (
    RestartPointOption,
)
from src.pipeline.launch_queries import (
    load_restart_point_options as _load_restart_point_options,
)
from src.pipeline.launch_queries import (
    pick_default_launch_mode as _pick_default_launch_mode,
)
from src.pipeline.launch_queries import (
    resolve_config_path_for_run as _resolve_config_path_for_run,
)
from src.pipeline.launch_queries import (
    validate_resume_run as _validate_resume_run,
)

if TYPE_CHECKING:
    from collections.abc import Callable

LaunchMode = Literal["new_run", "fresh", "resume", "restart"]
LaunchLogLevel = Literal["INFO", "DEBUG"]
LaunchStatus = Literal["launching", "running", "stopping", "completed", "failed", "interrupted"]

MODE_NEW_RUN: LaunchMode = "new_run"
MODE_FRESH: LaunchMode = "fresh"
MODE_RESUME: LaunchMode = "resume"
MODE_RESTART: LaunchMode = "restart"

_OUTPUT_TAIL_MAX_LINES = 40
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _launcher_log_path(run_dir: Path) -> Path:
    return run_dir.expanduser().resolve() / "tui_launch.log"


def _read_launch_log_tail(path: Path, *, max_lines: int = _OUTPUT_TAIL_MAX_LINES) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    tail = [line.strip() for line in lines[-max_lines:] if line.strip()]
    return tuple(tail)


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    mode: LaunchMode
    run_dir: Path
    config_path: Path | None = None
    restart_from_stage: str | None = None
    log_level: LaunchLogLevel = "INFO"

    def normalized(self) -> LaunchRequest:
        config_path = self.config_path.expanduser().resolve() if self.config_path is not None else None
        normalized_log_level = str(self.log_level).upper()
        return LaunchRequest(
            mode=self.mode,
            run_dir=self.run_dir.expanduser().resolve(),
            config_path=config_path,
            restart_from_stage=(self.restart_from_stage or "").strip() or None,
            log_level=normalized_log_level,  # type: ignore[arg-type]
        )

    def validate(self) -> LaunchRequest:
        normalized = self.normalized()
        if normalized.mode in {MODE_NEW_RUN, MODE_FRESH} and normalized.config_path is None:
            raise ValueError("New run and fresh attempt require a config path.")
        if normalized.mode == MODE_RESTART and not normalized.restart_from_stage:
            raise ValueError("Restart launch requires a stage name.")
        if normalized.mode != MODE_RESTART and normalized.restart_from_stage:
            raise ValueError("Restart stage can be set only for restart launches.")
        if normalized.log_level not in {"INFO", "DEBUG"}:
            raise ValueError("Launch log level must be INFO or DEBUG.")
        return normalized


@dataclass(frozen=True, slots=True)
class LaunchResult:
    request: LaunchRequest
    command: tuple[str, ...]
    return_code: int
    output_tail: tuple[str, ...]
    pid: int | None
    started_at: str
    finished_at: str

    @property
    def is_success(self) -> bool:
        return self.return_code == 0


@dataclass(slots=True)
class ActiveLaunch:
    request: LaunchRequest
    command: tuple[str, ...]
    status: LaunchStatus = "launching"
    pid: int | None = None
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    return_code: int | None = None
    output_tail: list[str] = field(default_factory=list)


def interrupt_launch_process(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        pgid = os.getpgid(pid) if hasattr(os, "getpgid") else None
    except ProcessLookupError:
        return False
    except OSError:
        return False

    try:
        if hasattr(os, "killpg") and pgid == pid:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(pid, signal.SIGCONT)
            os.killpg(pid, signal.SIGINT)
        else:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(pid, signal.SIGCONT)
            os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def is_process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_lock_pid(run_dir: Path) -> int | None:
    lock_path = run_dir.expanduser().resolve() / "run.lock"
    try:
        content = lock_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("pid="):
            with contextlib.suppress(ValueError):
                return int(line.split("=", 1)[1])
    return None


def build_train_command(request: LaunchRequest, *, python_executable: str | None = None) -> list[str]:
    normalized = request.validate()
    command = [python_executable or sys.executable, "-m", "src.main", "train", "--run-dir", str(normalized.run_dir)]
    if normalized.config_path is not None:
        command.extend(["--config", str(normalized.config_path)])
    if normalized.mode == MODE_RESUME:
        command.append("--resume")
    elif normalized.mode == MODE_RESTART:
        command.extend(["--restart-from-stage", normalized.restart_from_stage or ""])
    return command


def resolve_config_path_for_run(run_dir: Path, config_path: Path | None = None) -> Path:
    return _resolve_config_path_for_run(run_dir, config_path)


def load_restart_point_options(run_dir: Path, config_path: Path | None = None) -> tuple[Path, list[RestartPointOption]]:
    return _load_restart_point_options(run_dir, config_path)


def pick_default_launch_mode(run_dir: Path) -> str:
    return _pick_default_launch_mode(run_dir)


def validate_resume_run(run_dir: Path, config_path: Path | None = None) -> tuple[Path, str]:
    return _validate_resume_run(run_dir, config_path)


def execute_launch_subprocess(
    request: LaunchRequest,
    *,
    python_executable: str | None = None,
    on_started: Callable[[int | None], None] | None = None,
) -> LaunchResult:
    normalized = request.validate()
    command = tuple(build_train_command(normalized, python_executable=python_executable))
    process_env = os.environ.copy()
    process_env["LOG_LEVEL"] = normalized.log_level
    output_tail: collections.deque[str] = collections.deque(maxlen=_OUTPUT_TAIL_MAX_LINES)
    started_at = _utc_now_iso()
    pid: int | None = None
    normalized.run_dir.mkdir(parents=True, exist_ok=True)
    launcher_log_path = _launcher_log_path(normalized.run_dir)
    launcher_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with launcher_log_path.open("a", encoding="utf-8", buffering=1) as launch_log:
            launch_log.write(f"[{_utc_now_iso()}] Starting detached launch\n")
            launch_log.write(f"Command: {' '.join(command)}\n")
            launch_log.write(f"LOG_LEVEL: {normalized.log_level}\n")
            launch_log.flush()
            process = subprocess.Popen(
                command,
                cwd=str(_PROJECT_ROOT),
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=launch_log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
            pid = process.pid
            if on_started is not None:
                on_started(pid)
            output_tail.append(f"launcher_log={launcher_log_path}")
            return_code = process.wait()
            launch_log.flush()
    except Exception as exc:
        output_tail.append(str(exc))
        return_code = 1
    for line in _read_launch_log_tail(launcher_log_path):
        output_tail.append(line)
    return LaunchResult(
        request=normalized,
        command=command,
        return_code=return_code,
        output_tail=tuple(output_tail),
        pid=pid,
        started_at=started_at,
        finished_at=_utc_now_iso(),
    )


def spawn_launch_detached(
    request: LaunchRequest,
    *,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, tuple[str, ...], Path]:
    """Spawn the training subprocess without waiting.

    Returns (pid, command, launcher_log_path). Unlike execute_launch_subprocess,
    this never blocks — use it from HTTP handlers that just want to kick off a
    pipeline and return immediately. The subprocess remains attached to the
    detached session it was started in, so the caller process can die without
    orphaning it in a bad way (the orchestrator owns run.lock).

    ``extra_env`` (e.g. project-scoped env.json overrides) is merged on top of
    the parent process env — project values win over server-wide defaults so
    users can override ambient creds per-experiment.
    """
    normalized = request.validate()
    command = tuple(build_train_command(normalized, python_executable=python_executable))
    process_env = os.environ.copy()
    process_env["LOG_LEVEL"] = normalized.log_level
    if extra_env:
        for k, v in extra_env.items():
            if v != "":
                process_env[k] = v
    normalized.run_dir.mkdir(parents=True, exist_ok=True)
    launcher_log_path = _launcher_log_path(normalized.run_dir)
    launcher_log_path.parent.mkdir(parents=True, exist_ok=True)

    launch_log = launcher_log_path.open("a", encoding="utf-8", buffering=1)
    launch_log.write(f"[{_utc_now_iso()}] Starting detached launch\n")
    launch_log.write(f"Command: {' '.join(command)}\n")
    launch_log.write(f"LOG_LEVEL: {normalized.log_level}\n")
    launch_log.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_PROJECT_ROOT),
            env=process_env,
            stdin=subprocess.DEVNULL,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except Exception:
        launch_log.close()
        raise
    return process.pid, command, launcher_log_path


__all__ = [
    "MODE_FRESH",
    "MODE_NEW_RUN",
    "MODE_RESTART",
    "MODE_RESUME",
    "ActiveLaunch",
    "LaunchLogLevel",
    "LaunchMode",
    "LaunchRequest",
    "LaunchResult",
    "LaunchStatus",
    "RestartPointOption",
    "build_train_command",
    "execute_launch_subprocess",
    "interrupt_launch_process",
    "is_process_alive",
    "load_restart_point_options",
    "pick_default_launch_mode",
    "read_lock_pid",
    "resolve_config_path_for_run",
    "spawn_launch_detached",
    "validate_resume_run",
]
