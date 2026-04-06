from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.pipeline.run_queries import ROOT_GROUP, RunSummaryRow, scan_runs_dir as scan_run_rows, scan_runs_dir_grouped as scan_run_rows_grouped
from src.tui.adapters.presentation import format_duration


def build_suggested_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.expanduser().resolve() / f"run_{timestamp}"


def scan_runs_dir(runs_dir: Path) -> list[dict[str, object]]:
    return [_to_row_dict(row) for row in scan_run_rows(runs_dir)]


def scan_runs_dir_grouped(runs_dir: Path) -> dict[str, list[dict[str, object]]]:
    return {
        group: [_to_row_dict(row) for row in rows]
        for group, rows in scan_run_rows_grouped(runs_dir).items()
    }


def _to_row_dict(row: RunSummaryRow) -> dict[str, object]:
    return {
        "run_id": row.run_id,
        "run_dir": row.run_dir,
        "created_at": row.created_at,
        "created_ts": row.created_ts,
        "status": row.status,
        "attempts": row.attempts,
        "config": row.config_name,
        "mlflow_run_id": row.mlflow_run_id,
        "duration": format_duration(row.started_at, row.completed_at),
        "started_at": row.started_at,
        "error": row.error,
        "group": row.group,
    }
