"""HelpModal — centered popup with key-binding reference."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class HelpModal(ModalScreen[None]):
    """Centred popup that lists all bindings passed to it."""

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }
    HelpModal #help-dialog {
        background: $surface;
        border: thick $primary;
        padding: 1 3;
        width: 56;
        height: auto;
        max-height: 80vh;
    }
    HelpModal #help-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    HelpModal #help-footer {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape,q,question_mark,enter,space", "dismiss_modal", show=False),
    ]

    def __init__(self, bindings: list[Binding]) -> None:
        super().__init__()
        self._screen_bindings = [b for b in bindings if b.show]

    def compose(self) -> ComposeResult:
        from rich.table import Table
        from textual.containers import Container

        tbl = Table(box=None, show_header=False, padding=(0, 2), expand=True)
        tbl.add_column("key", style="bold yellow", no_wrap=True)
        tbl.add_column("description", style="white")

        for b in self._screen_bindings:
            key_display = b.key_display or b.key.replace(",", " / ")
            tbl.add_row(key_display, b.description)

        with Container(id="help-dialog"):
            yield Static("Key Bindings", id="help-title")
            yield Static(tbl)
            yield Static("Press any key to close", id="help-footer")

    def on_key(self) -> None:
        self.dismiss()

    def action_dismiss_modal(self) -> None:
        self.dismiss()
