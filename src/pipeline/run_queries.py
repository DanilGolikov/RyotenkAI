from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline.state import PipelineState, PipelineStateLoadError, PipelineStateStore
from src.pipeline.state.cache import load_state_snapshot

_LOG_TAIL_LINES = 30
ROOT_GROUP = "(root)"


@dataclass(frozen=True, slots=True)
class RunInspectionData:
    run_dir: Path
    state: PipelineState
    log_tails: dict[int, list[str]]


@dataclass(frozen=True, slots=True)
class RunSummaryRow:
    run_id: str
    run_dir: Path
    created_at: str
    created_ts: float
    status: str
    attempts: int
    config_name: str
    mlflow_run_id: str | None
    started_at: str | None
    completed_at: str | None
    error: str | None = None
    group: str = ROOT_GROUP

    def __getitem__(self, key: str) -> Any:
        return self._aliases()[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._aliases()

    def _aliases(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "created_at": self.created_at,
            "created_ts": self.created_ts,
            "status": self.status,
            "attempts": self.attempts,
            "config": self.config_name,
            "config_name": self.config_name,
            "mlflow_run_id": self.mlflow_run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "group": self.group,
        }


class RunInspector:
    """Loads persisted run state and optional log tails from a run directory."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir.expanduser().resolve()
        self._store = PipelineStateStore(self._run_dir)

    def load(self, *, include_logs: bool = False) -> RunInspectionData:
        # Cached load — the API polls this hundreds of times per minute across
        # open clients. Cache key (state_path, mtime_ns) guarantees freshness.
        snapshot = load_state_snapshot(self._run_dir)
        state = snapshot.state
        log_tails: dict[int, list[str]] = {}
        if include_logs:
            for attempt in state.attempts:
                log_tails[attempt.attempt_no] = tail_lines(
                    self._store.next_attempt_dir(attempt.attempt_no) / "pipeline.log",
                )
        return RunInspectionData(run_dir=self._run_dir, state=state, log_tails=log_tails)


def tail_lines(path: Path, limit: int = _LOG_TAIL_LINES) -> list[str]:
    try:
        queue: collections.deque[str] = collections.deque(maxlen=limit)
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                queue.append(line.rstrip())
        return list(queue)
    except OSError:
        return []


def effective_pipeline_status(state: PipelineState) -> str:
    if state.attempts:
        latest_status = state.attempts[-1].status
        if latest_status:
            return latest_status
    return state.pipeline_status


def scan_runs_dir(runs_dir: Path) -> list[RunSummaryRow]:
    rows: list[RunSummaryRow] = []
    if not runs_dir.is_dir():
        return rows
    for entry in sorted(runs_dir.iterdir(), key=lambda path: path.name, reverse=True):
        if entry.is_dir() and (entry / "pipeline_state.json").exists():
            rows.append(build_run_summary_row(entry))
    return rows


def scan_runs_dir_grouped(runs_dir: Path) -> dict[str, list[RunSummaryRow]]:
    groups: dict[str, list[RunSummaryRow]] = {}
    if not runs_dir.is_dir():
        return groups
    _scan_recursive(runs_dir, runs_dir, groups)
    return groups


def build_run_summary_row(entry: Path, *, group: str = ROOT_GROUP) -> RunSummaryRow:
    stat = entry.stat()
    created_ts = float(getattr(stat, "st_birthtime", None) or stat.st_ctime)
    created_at = _format_created_at(created_ts)

    try:
        # Cached: ``list_runs`` fans out to every run on disk — without this
        # a 50-run workspace costs 50 JSON parses per HTTP GET.
        state = load_state_snapshot(entry).state
    except (PipelineStateLoadError, Exception) as exc:
        return RunSummaryRow(
            run_id=entry.name,
            run_dir=entry,
            created_at=created_at,
            created_ts=created_ts,
            status="unknown",
            attempts=0,
            config_name="—",
            mlflow_run_id=None,
            started_at=None,
            completed_at=None,
            error=str(exc),
            group=group,
        )

    first_start = state.attempts[0].started_at if state.attempts else None
    last_end = state.attempts[-1].completed_at if state.attempts else None
    return RunSummaryRow(
        run_id=entry.name,
        run_dir=entry,
        created_at=created_at,
        created_ts=created_ts,
        status=effective_pipeline_status(state),
        attempts=len(state.attempts),
        config_name=Path(state.config_path).name if state.config_path else "—",
        mlflow_run_id=state.root_mlflow_run_id,
        started_at=first_start,
        completed_at=last_end,
        error=None,
        group=group,
    )


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


def _scan_recursive(current: Path, root: Path, groups: dict[str, list[RunSummaryRow]]) -> None:
    try:
        children = sorted(current.iterdir(), key=lambda path: path.name, reverse=True)
    except OSError:
        return

    for entry in children:
        if not entry.is_dir():
            continue
        if (entry / "pipeline_state.json").exists():
            group_name = ROOT_GROUP if current == root else str(current.relative_to(root))
            groups.setdefault(group_name, []).append(build_run_summary_row(entry, group=group_name))
            continue
        _scan_recursive(entry, root, groups)


def _format_created_at(created_ts: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M")
