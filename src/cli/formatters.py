"""Small, dependency-free formatting helpers shared by text and JSON renderers.

Kept separate from ``style.py`` / ``renderer.py`` so they're safe to import
from anywhere in the CLI layer without pulling Rich as a hard dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime

_SECS_PER_HOUR = 3600
_SECS_PER_MINUTE = 60


def duration_seconds(started_at: str | None, completed_at: str | None) -> int | None:
    """Elapsed seconds between two ISO timestamps. ``None`` if not computable."""
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at) if completed_at else datetime.now(UTC)
        if start.tzinfo is None and end.tzinfo is not None:
            start = start.replace(tzinfo=UTC)
        elif start.tzinfo is not None and end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        delta = int((end - start).total_seconds())
    except (TypeError, ValueError):
        return None
    return max(delta, 0)


def format_duration(started_at: str | None, completed_at: str | None) -> str:
    """Human-friendly duration string: ``2h 14m 8s`` / ``41m 22s`` / ``17s``.

    Empty string when input timestamps are missing or malformed — the
    ``-`` placeholder is the caller's responsibility.
    """
    seconds = duration_seconds(started_at, completed_at)
    if seconds is None:
        return ""
    hours, remainder = divmod(seconds, _SECS_PER_HOUR)
    minutes, secs = divmod(remainder, _SECS_PER_MINUTE)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


__all__ = ["duration_seconds", "format_duration"]
