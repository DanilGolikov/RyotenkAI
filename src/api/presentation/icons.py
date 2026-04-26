"""UI-agnostic glyph + colour mapping for stage / pipeline statuses.

Both the web API (api/services/run_service.py) and any CLI renderer that
shows stage status read these dicts and pass the values through to their
own front-end. Keeping the mapping next to ``StageRunState`` (its only
source of truth for status strings) makes drift impossible: rename a
status, the IDE catches the matching key here.

The dicts are intentionally a Mapping[str, str] rather than enum-keyed —
JSON responses pass plain status strings, and adding an enum lookup
indirection would buy nothing for callers that already string-compare.
"""

from __future__ import annotations

from src.pipeline.state.models import StageRunState

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


__all__ = ["STATUS_COLORS", "STATUS_ICONS"]
