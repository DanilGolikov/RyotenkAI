from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.pipeline.state import PipelineStateLoadError
from src.tui.adapters.presentation import effective_pipeline_status, format_duration
from src.tui.adapters.state import load_pipeline_state

ROOT_GROUP = "(root)"


def build_suggested_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.expanduser().resolve() / f"run_{timestamp}"


def scan_runs_dir(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return rows
    for entry in sorted(runs_dir.iterdir(), key=lambda path: path.name, reverse=True):
        if entry.is_dir() and (entry / "pipeline_state.json").exists():
            rows.append(_build_row_dict(entry))
    return rows


def scan_runs_dir_grouped(runs_dir: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    if not runs_dir.is_dir():
        return groups
    _scan_recursive(runs_dir, runs_dir, groups)
    return groups


def _scan_recursive(current: Path, root: Path, groups: dict[str, list[dict[str, Any]]]) -> None:
    try:
        children = sorted(current.iterdir(), key=lambda path: path.name, reverse=True)
    except OSError:
        return

    for entry in children:
        if not entry.is_dir():
            continue
        if (entry / "pipeline_state.json").exists():
            group_name = ROOT_GROUP if current == root else str(current.relative_to(root))
            row = _build_row_dict(entry)
            row["group"] = group_name
            groups.setdefault(group_name, []).append(row)
            continue
        _scan_recursive(entry, root, groups)


def _build_row_dict(entry: Path) -> dict[str, Any]:
    stat = entry.stat()
    created_ts = getattr(stat, "st_birthtime", None) or stat.st_ctime
    created_at = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M")

    row: dict[str, Any] = {
        "run_id": entry.name,
        "run_dir": entry,
        "created_at": created_at,
        "created_ts": created_ts,
        "error": None,
    }
    try:
        state = load_pipeline_state(entry)
        row["status"] = effective_pipeline_status(state)
        row["attempts"] = len(state.attempts)
        row["config"] = Path(state.config_path).name if state.config_path else "—"
        row["mlflow_run_id"] = state.root_mlflow_run_id
        first_start = state.attempts[0].started_at if state.attempts else None
        last_end = state.attempts[-1].completed_at if state.attempts else None
        row["duration"] = format_duration(first_start, last_end)
        row["started_at"] = first_start
    except (PipelineStateLoadError, Exception) as exc:
        row["status"] = "unknown"
        row["attempts"] = 0
        row["config"] = "—"
        row["mlflow_run_id"] = None
        row["duration"] = ""
        row["started_at"] = None
        row["error"] = str(exc)
    return row
