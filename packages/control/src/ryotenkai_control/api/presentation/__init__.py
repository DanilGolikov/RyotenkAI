"""View-layer helpers for the run inspection API.

Lives next to ``api/services`` rather than under ``pipeline/state``
because these modules are pure presentation: they exist to colour and
format StageRunState values for HTTP responses, never to mutate state.
The state package owns the source of truth (StageRunState statuses,
execution modes); this package translates those values for the UI.
"""

from src.api.presentation.formatters import format_duration, format_mode_label
from src.api.presentation.icons import STATUS_COLORS, STATUS_ICONS

__all__ = [
    "STATUS_COLORS",
    "STATUS_ICONS",
    "format_duration",
    "format_mode_label",
]
