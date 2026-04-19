"""Query helpers over PipelineState / PipelineStateStore.

Thin wrappers used by CLI, TUI and web backend to answer common questions
about a run directory without poking at PipelineStateStore directly. Moved
from src/tui/adapters/state.py so the web backend does not depend on the
TUI package.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.state.models import PipelineAttemptState, PipelineState, StageRunState
from src.pipeline.state.store import PipelineStateStore


def state_store(run_dir: Path) -> PipelineStateStore:
    return PipelineStateStore(run_dir.expanduser().resolve())


def run_state_exists(run_dir: Path) -> bool:
    return state_store(run_dir).exists()


def load_pipeline_state(run_dir: Path) -> PipelineState:
    return state_store(run_dir).load()


def resolve_config_path_from_state(run_dir: Path, config_path: Path | None = None) -> Path:
    if config_path is not None:
        return config_path.expanduser().resolve()
    state = load_pipeline_state(run_dir)
    if not state.config_path:
        raise ValueError("Existing run has no config_path in pipeline_state.json")
    return Path(state.config_path).expanduser().resolve()


def predict_next_attempt_no(run_dir: Path) -> int:
    try:
        state = load_pipeline_state(run_dir)
    except Exception:
        return 1
    if not state.attempts:
        return 1
    return max(attempt.attempt_no for attempt in state.attempts) + 1


def find_running_attempt_no(state: PipelineState) -> int | None:
    if state.active_attempt_id:
        for attempt in state.attempts:
            if attempt.attempt_id == state.active_attempt_id and attempt.status == StageRunState.STATUS_RUNNING:
                return attempt.attempt_no
    for attempt in reversed(state.attempts):
        if attempt.status == StageRunState.STATUS_RUNNING:
            return attempt.attempt_no
    return None


def latest_attempt_no(run_dir: Path) -> int | None:
    try:
        state = load_pipeline_state(run_dir)
    except Exception:
        return None
    if not state.attempts:
        return None
    return state.attempts[-1].attempt_no


def get_running_attempt_no(run_dir: Path) -> int | None:
    try:
        state = load_pipeline_state(run_dir)
    except Exception:
        return None
    return find_running_attempt_no(state)


def get_attempt_by_no(state: PipelineState, attempt_no: int) -> PipelineAttemptState | None:
    for attempt in state.attempts:
        if attempt.attempt_no == attempt_no:
            return attempt
    return None


def discover_run_dirs(target: Path) -> tuple[Path, ...]:
    resolved_target = target.expanduser().resolve()
    if not resolved_target.exists():
        return ()
    if (resolved_target / "pipeline_state.json").exists():
        return (resolved_target,)

    run_dirs: list[Path] = []
    for current in sorted(
        resolved_target.rglob("*"),
        key=lambda path: (len(path.relative_to(resolved_target).parts), path.as_posix()),
    ):
        if current.is_dir() and (current / "pipeline_state.json").exists():
            run_dirs.append(current)
    return tuple(run_dirs)


__all__ = [
    "discover_run_dirs",
    "find_running_attempt_no",
    "get_attempt_by_no",
    "get_running_attempt_no",
    "latest_attempt_no",
    "load_pipeline_state",
    "predict_next_attempt_no",
    "resolve_config_path_from_state",
    "run_state_exists",
    "state_store",
]
