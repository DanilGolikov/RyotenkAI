from __future__ import annotations

from datetime import datetime, timezone

from src.pipeline.state import PipelineState, StageRunState

_SECS_PER_HOUR = 3600
_SECS_PER_MINUTE = 60

STATUS_ICONS: dict[str, str] = {
    StageRunState.STATUS_COMPLETED: "◉",
    StageRunState.STATUS_FAILED: "◉",
    StageRunState.STATUS_RUNNING: "▸",
    StageRunState.STATUS_INTERRUPTED: "◈",
    StageRunState.STATUS_STALE: "◌",
    StageRunState.STATUS_SKIPPED: "◇",
    StageRunState.STATUS_PENDING: "○",
}

STATUS_COLORS: dict[str, str] = {
    StageRunState.STATUS_COMPLETED: "green",
    StageRunState.STATUS_FAILED: "red",
    StageRunState.STATUS_RUNNING: "cyan",
    StageRunState.STATUS_INTERRUPTED: "yellow",
    StageRunState.STATUS_STALE: "dim",
    StageRunState.STATUS_SKIPPED: "blue",
    StageRunState.STATUS_PENDING: "dim",
}


def effective_pipeline_status(state: PipelineState) -> str:
    if state.attempts:
        latest_status = state.attempts[-1].status
        if latest_status:
            return latest_status
    return state.pipeline_status


def format_duration(started_at: str | None, completed_at: str | None) -> str:
    if not started_at:
        return ""
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at) if completed_at else datetime.now(timezone.utc)
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
    except (TypeError, ValueError):
        return ""


def format_mode_label(stage_run: StageRunState) -> str:
    if stage_run.execution_mode == StageRunState.MODE_REUSED and stage_run.reuse_from:
        attempt_id = stage_run.reuse_from.get("attempt_id", "?")
        suffix = str(attempt_id).split(":")[-1] if ":" in str(attempt_id) else str(attempt_id)
        return f"reused ({suffix})"
    return stage_run.execution_mode or "—"
