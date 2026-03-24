from __future__ import annotations

from textual.widgets import Static


class DockedHotkeyBar(Static):
    DEFAULT_CSS = """
    DockedHotkeyBar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
        color: $text-muted;
        width: 100%;
    }
    """
