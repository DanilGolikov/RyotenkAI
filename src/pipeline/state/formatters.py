"""Human-friendly formatters for stage / pipeline state values.

UI-framework-agnostic — used by both the web API enrichment and CLI
rendering. The functions here read from ``StageRunState`` fields
(durations, execution mode, reuse refs) and turn them into display
strings; they never mutate state.

Kept separate from icons.py because the two concerns evolve
independently: glyphs are a stable visual vocabulary, formatters churn
when product copy changes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.pipeline.state.models import StageRunState

_SECS_PER_HOUR = 3600
_SECS_PER_MINUTE = 60


def format_duration(started_at: str | None, completed_at: str | None) -> str:
    """Return ``Hh Mm Ss`` / ``Mm Ss`` / ``Ss`` depending on duration size.

    Returns "" for missing or invalid timestamps so callers can use
    ``str.format(...) or "—"`` without try/except.
    """
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
    """Render ``execution_mode`` for the UI.

    Reused stages get an extra ``(<attempt-suffix>)`` hint so users can
    see at a glance which prior attempt the artifact came from.
    """
    if stage_run.execution_mode == StageRunState.MODE_REUSED and stage_run.reuse_from:
        attempt_id = stage_run.reuse_from.get("attempt_id", "?")
        suffix = str(attempt_id).split(":")[-1] if ":" in str(attempt_id) else str(attempt_id)
        return f"reused ({suffix})"
    return stage_run.execution_mode or "—"


__all__ = ["format_duration", "format_mode_label"]
