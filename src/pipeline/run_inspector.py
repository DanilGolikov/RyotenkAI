"""
Lightweight run inspection — reads pipeline_state.json and log files.
No orchestrator, no stage init, no training stack.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final

from src.pipeline.state import PipelineState, PipelineStateLoadError, PipelineStateStore, StageRunState

if TYPE_CHECKING:
    from pathlib import Path

_LOG_TAIL_LINES: Final = 30
_SECS_PER_HOUR: Final = 3600
_SECS_PER_MINUTE: Final = 60
_MLFLOW_ID_SHORT: Final = 12
_ERROR_TRUNCATE: Final = 120
_ATTEMPT_ERROR_TRUNCATE: Final = 160
_MAX_OUTPUT_FIELDS: Final = 8
_MAX_OUTPUT_VALUE_LEN: Final = 60

# Column widths for stage table
_COL_ICON_W: Final = 3
_COL_NAME_W: Final = 22
_COL_STATUS_W: Final = 13
_COL_MODE_W: Final = 10
_COL_DURATION_W: Final = 8
_TIMESTAMP_DISPLAY_LEN: Final = 19  # len("YYYY-MM-DDTHH:MM:SS")

_STATUS_ICONS: Final[dict[str, str]] = {  # noqa: WPS407
    StageRunState.STATUS_COMPLETED: "✅",
    StageRunState.STATUS_FAILED: "❌",
    StageRunState.STATUS_RUNNING: "⟳",
    StageRunState.STATUS_INTERRUPTED: "⚡",
    StageRunState.STATUS_STALE: "~",
    StageRunState.STATUS_SKIPPED: "↩",
    StageRunState.STATUS_PENDING: "—",
}
_STATUS_COLORS: Final[dict[str, str]] = {  # noqa: WPS407
    StageRunState.STATUS_COMPLETED: "green",
    StageRunState.STATUS_FAILED: "red",
    StageRunState.STATUS_RUNNING: "cyan",
    StageRunState.STATUS_INTERRUPTED: "yellow",
    StageRunState.STATUS_STALE: "dim",
    StageRunState.STATUS_SKIPPED: "blue",
    StageRunState.STATUS_PENDING: "dim",
}


def effective_pipeline_status(state: PipelineState) -> str:
    """Return the best-available top-level status for a logical run.

    Root ``pipeline_status`` can be stale in historical states. For UI / inspection
    purposes, the latest attempt status is the most truthful summary whenever
    attempts exist.
    """

    if state.attempts:
        latest_status = state.attempts[-1].status
        if latest_status:
            return latest_status
    return state.pipeline_status


def _fmt_duration(started_at: str | None, completed_at: str | None) -> str:
    if not started_at:
        return ""
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at) if completed_at else datetime.now(timezone.utc)
        # Normalize: make both tz-aware or both tz-naive to avoid TypeError
        if start.tzinfo is None and end.tzinfo is not None:
            start = start.replace(tzinfo=timezone.utc)
        elif start.tzinfo is not None and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        delta = int((end - start).total_seconds())
        if delta < 0:
            return ""
        hours, remainder = divmod(delta, _SECS_PER_HOUR)
        minutes, seconds = divmod(remainder, _SECS_PER_MINUTE)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    except (ValueError, TypeError):
        return ""


def _tail_lines(path: Path, n: int = _LOG_TAIL_LINES) -> list[str]:
    try:
        dq: collections.deque[str] = collections.deque(maxlen=n)
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                dq.append(line.rstrip())
        return list(dq)
    except OSError:
        return []


class RunInspectionData:
    """Holds all data loaded by RunInspector for a single run directory."""

    def __init__(self, run_dir: Path, state: PipelineState, log_tails: dict[int, list[str]]) -> None:
        self.run_dir = run_dir
        self.state = state
        self.log_tails = log_tails  # attempt_no → last N lines


class RunInspector:
    """Loads PipelineState and optional log tails from a run directory."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir.expanduser().resolve()
        self._store = PipelineStateStore(self._run_dir)

    def load(self, *, include_logs: bool = False) -> RunInspectionData:
        state = self._store.load()
        log_tails: dict[int, list[str]] = {}
        if include_logs:
            for attempt in state.attempts:
                attempt_dir = self._store.next_attempt_dir(attempt.attempt_no)
                log_file = attempt_dir / "pipeline.log"
                lines = _tail_lines(log_file)
                log_tails[attempt.attempt_no] = lines
        return RunInspectionData(run_dir=self._run_dir, state=state, log_tails=log_tails)


# =============================================================================
# Rich Renderer
# =============================================================================


class RunInspectionRenderer:
    """Renders RunInspectionData to stdout as plain text."""

    def render(
        self,
        data: RunInspectionData,
        *,
        verbose: bool = False,
        include_logs: bool = False,
    ) -> None:
        state = data.state
        pipeline_status = effective_pipeline_status(state)

        pipeline_duration = _fmt_duration(
            state.attempts[0].started_at if state.attempts else None,
            state.attempts[-1].completed_at if state.attempts else None,
        )
        config_name = state.config_path.split("/")[-1] if state.config_path else "-"
        mlflow_short = (state.root_mlflow_run_id or "-")[:_MLFLOW_ID_SHORT] + "..." if state.root_mlflow_run_id else "-"

        print(f"Run: {data.run_dir.name}")
        print(f"  Status  : {pipeline_status.upper()}")
        print(f"  Config  : {config_name}")
        if state.attempts:
            print(f"  Started : {state.attempts[0].started_at[:_TIMESTAMP_DISPLAY_LEN].replace('T', ' ')}")
        print(f"  Duration: {pipeline_duration or '-'}")
        print(f"  MLflow  : {mlflow_short}")
        print(f"  Attempts: {len(state.attempts)}")
        print(f"  Run ID  : {state.logical_run_id}")
        print()

        for attempt in state.attempts:
            attempt_duration = _fmt_duration(attempt.started_at, attempt.completed_at)
            action_label = attempt.restart_from_stage or attempt.effective_action
            sep = "-" * 70  # noqa: WPS432
            print(f"Attempt {attempt.attempt_no}  {action_label}  {attempt.status}  {attempt_duration}")
            print(sep)

            fmt = "  {:3} {:<28} {:<13} {:<14} {}"
            for stage_name in attempt.enabled_stage_names or list(attempt.stage_runs):
                sr = attempt.stage_runs.get(stage_name)
                if sr is None:
                    print(fmt.format("-", stage_name, "pending", "-", ""))
                    continue

                icon = _STATUS_ICONS.get(sr.status, "?")
                duration = _fmt_duration(sr.started_at, sr.completed_at)
                mode_label = _mode_label(sr)
                print(fmt.format(icon, sr.stage_name, sr.status, mode_label, duration))

                if sr.error:
                    print(f"      Error: {sr.error[:_ERROR_TRUNCATE]}")

                if verbose and sr.outputs:
                    for key, val in list(sr.outputs.items())[:_MAX_OUTPUT_FIELDS]:
                        val_str = str(val)[:_MAX_OUTPUT_VALUE_LEN] if val is not None else "-"
                        print(f"      {key} = {val_str}")

            if attempt.error and attempt.status not in (StageRunState.STATUS_COMPLETED,):
                print(f"  Pipeline error: {attempt.error[:_ATTEMPT_ERROR_TRUNCATE]}")
            print()

        if include_logs:
            for attempt in state.attempts:
                lines = data.log_tails.get(attempt.attempt_no, [])
                print(f"--- Attempt {attempt.attempt_no} / pipeline.log ---")
                if not lines:
                    print("  (no log file)")
                else:
                    print(f"  (last {len(lines)} lines)")
                    print("\n".join(lines))
                print()


def _mode_label(sr: StageRunState) -> str:
    if sr.execution_mode == StageRunState.MODE_REUSED and sr.reuse_from:
        attempt_id = sr.reuse_from.get("attempt_id", "?")
        suffix = str(attempt_id).split(":")[-1] if ":" in str(attempt_id) else str(attempt_id)
        return f"reused ({suffix})"
    return sr.execution_mode or "—"


# =============================================================================
# runs-list helpers
# =============================================================================


def scan_runs_dir(runs_dir: Path) -> list[dict[str, Any]]:
    """
    Scan runs_dir for subdirectories containing pipeline_state.json.
    Returns list of dicts with summary info, sorted newest first.
    """
    rows: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return rows
    for entry in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not entry.is_dir():
            continue
        state_file = entry / "pipeline_state.json"
        if not state_file.exists():
            continue
        stat = entry.stat()
        created_ts = getattr(stat, "st_birthtime", None) or stat.st_ctime
        created_at = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M")

        row: dict[str, Any] = {"run_id": entry.name, "run_dir": entry, "created_at": created_at, "error": None}
        try:
            store = PipelineStateStore(entry)
            state = store.load()
            row["status"] = effective_pipeline_status(state)
            row["attempts"] = len(state.attempts)
            row["config"] = state.config_path.split("/")[-1] if state.config_path else "—"
            row["mlflow_run_id"] = state.root_mlflow_run_id
            first_start = state.attempts[0].started_at if state.attempts else None
            last_end = state.attempts[-1].completed_at if state.attempts else None
            row["duration"] = _fmt_duration(first_start, last_end)
            row["started_at"] = first_start
        except (PipelineStateLoadError, Exception) as exc:
            row["status"] = "unknown"
            row["attempts"] = 0
            row["config"] = "—"
            row["mlflow_run_id"] = None
            row["duration"] = ""
            row["started_at"] = None
            row["error"] = str(exc)
        rows.append(row)
    return rows


# =============================================================================
# run-diff helpers
# =============================================================================


def diff_attempts(state: PipelineState, attempt_a: int, attempt_b: int) -> dict[str, Any]:
    """
    Compare config hashes between two attempts.
    Returns a dict with comparison info.
    """
    by_no = {a.attempt_no: a for a in state.attempts}
    a = by_no.get(attempt_a)
    b = by_no.get(attempt_b)
    result: dict[str, Any] = {
        "attempt_a": attempt_a,
        "attempt_b": attempt_b,
        "found_a": a is not None,
        "found_b": b is not None,
        "training_critical_changed": False,
        "late_stage_changed": False,
        "hash_a_critical": "",
        "hash_b_critical": "",
        "hash_a_late": "",
        "hash_b_late": "",
    }
    if a is None or b is None:
        return result
    result["hash_a_critical"] = a.training_critical_config_hash
    result["hash_b_critical"] = b.training_critical_config_hash
    result["hash_a_late"] = a.late_stage_config_hash
    result["hash_b_late"] = b.late_stage_config_hash
    result["training_critical_changed"] = a.training_critical_config_hash != b.training_critical_config_hash
    result["late_stage_changed"] = a.late_stage_config_hash != b.late_stage_config_hash
    return result
