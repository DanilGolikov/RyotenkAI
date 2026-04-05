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

from src.pipeline.restart_points import compute_config_hashes, list_restart_points
from src.pipeline.state import PipelineStateStore, StageRunState
from src.utils.config import load_config

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
class RestartPointOption:
    stage: str
    available: bool
    mode: str
    reason: str


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
        # TUI-launched runs use start_new_session=True, so pid == pgid and we can
        # gracefully interrupt the entire process group. For externally launched
        # runs (e.g. smoke runner), fall back to signalling the process itself.
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
    if config_path is not None:
        return config_path.expanduser().resolve()

    state = PipelineStateStore(run_dir.expanduser().resolve()).load()
    if not state.config_path:
        raise ValueError("Existing run has no config_path in pipeline_state.json")
    return Path(state.config_path).expanduser().resolve()


def load_restart_point_options(run_dir: Path, config_path: Path | None = None) -> tuple[Path, list[RestartPointOption]]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_config = resolve_config_path_for_run(resolved_run_dir, config_path)
    config = load_config(resolved_config)
    points = list_restart_points(resolved_run_dir, config)
    return resolved_config, [RestartPointOption(**point) for point in points]


def _derive_resume_stage(state) -> str | None:
    if not state.attempts:
        return None
    latest = state.attempts[-1]
    for stage_name in latest.enabled_stage_names or latest.stage_runs:
        stage_state = latest.stage_runs.get(stage_name)
        if stage_state is None:
            return stage_name
        if stage_state.status in {
            StageRunState.STATUS_FAILED,
            StageRunState.STATUS_INTERRUPTED,
            StageRunState.STATUS_PENDING,
            StageRunState.STATUS_RUNNING,
            StageRunState.STATUS_STALE,
        }:
            return stage_name
    return None


def pick_default_launch_mode(run_dir: Path) -> str:
    """Return 'resume' if the run has a failed/interrupted stage, otherwise 'restart'.

    Does not check config hashes — just inspects stage statuses. Safe to call
    with potentially stale Python module state (no model_dump involved).
    """
    try:
        state = PipelineStateStore(run_dir.expanduser().resolve()).load()
    except Exception:
        return MODE_RESTART
    resume_stage = _derive_resume_stage(state)
    return MODE_RESUME if resume_stage is not None else MODE_RESTART


def validate_resume_run(run_dir: Path, config_path: Path | None = None) -> tuple[Path, str]:
    resolved_run_dir = run_dir.expanduser().resolve()
    state = PipelineStateStore(resolved_run_dir).load()
    resolved_config = resolve_config_path_for_run(resolved_run_dir, config_path)
    config = load_config(resolved_config)
    config_hashes = compute_config_hashes(config)

    # Fine-grained check: if model_dataset_config_hash is stored, provider changes are allowed.
    # Legacy fallback for states without model_dataset_config_hash.
    if state.model_dataset_config_hash:
        if state.model_dataset_config_hash != config_hashes["model_dataset"]:
            raise ValueError("training_critical config changed for existing logical run; resume is blocked")
    elif state.training_critical_config_hash != config_hashes["training_critical"]:
        raise ValueError("training_critical config changed for existing logical run; resume is blocked")

    if state.late_stage_config_hash != config_hashes["late_stage"]:
        raise ValueError(
            "late_stage config changed; resume is blocked. Use manual restart from Inference Deployer or Model Evaluator"
        )

    start_stage = _derive_resume_stage(state)
    if start_stage is None:
        latest_status = state.attempts[-1].status if state.attempts else state.pipeline_status
        raise ValueError(f"Nothing to resume: latest attempt is already {latest_status}. Use restart or new run.")
    return resolved_config, start_stage


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
