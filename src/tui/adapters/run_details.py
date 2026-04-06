from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.state import PipelineState
from src.tui.adapters.state import load_pipeline_state, state_store

_LOG_TAIL_LINES = 30


@dataclass(frozen=True, slots=True)
class RunInspectionData:
    run_dir: Path
    state: PipelineState
    log_tails: dict[int, list[str]]


def load_run_inspection(run_dir: Path, *, include_logs: bool = False) -> RunInspectionData:
    resolved_run_dir = run_dir.expanduser().resolve()
    store = state_store(resolved_run_dir)
    state = store.load()
    log_tails: dict[int, list[str]] = {}
    if include_logs:
        for attempt in state.attempts:
            log_tails[attempt.attempt_no] = tail_lines(store.next_attempt_dir(attempt.attempt_no) / "pipeline.log")
    return RunInspectionData(run_dir=resolved_run_dir, state=state, log_tails=log_tails)


def tail_lines(path: Path, limit: int = _LOG_TAIL_LINES) -> list[str]:
    try:
        queue: collections.deque[str] = collections.deque(maxlen=limit)
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                queue.append(line.rstrip())
        return list(queue)
    except OSError:
        return []


def diff_attempts(state: PipelineState, attempt_a: int, attempt_b: int) -> dict[str, Any]:
    by_no = {attempt.attempt_no: attempt for attempt in state.attempts}
    left = by_no.get(attempt_a)
    right = by_no.get(attempt_b)
    result: dict[str, Any] = {
        "attempt_a": attempt_a,
        "attempt_b": attempt_b,
        "found_a": left is not None,
        "found_b": right is not None,
        "training_critical_changed": False,
        "late_stage_changed": False,
        "hash_a_critical": "",
        "hash_b_critical": "",
        "hash_a_late": "",
        "hash_b_late": "",
    }
    if left is None or right is None:
        return result
    result["hash_a_critical"] = left.training_critical_config_hash
    result["hash_b_critical"] = right.training_critical_config_hash
    result["hash_a_late"] = left.late_stage_config_hash
    result["hash_b_late"] = right.late_stage_config_hash
    result["training_critical_changed"] = left.training_critical_config_hash != right.training_critical_config_hash
    result["late_stage_changed"] = left.late_stage_config_hash != right.late_stage_config_hash
    return result


def resolve_run_config_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()
