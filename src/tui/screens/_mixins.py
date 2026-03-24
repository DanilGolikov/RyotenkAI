"""Shared screen mixins for TUI screens."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from pathlib import Path

    from textual.events import Key


class _HelpMixin:
    """Adds ``action_show_help`` to any Screen that declares BINDINGS."""

    def action_show_help(self) -> None:
        from src.tui.screens._help import HelpModal

        self.app.push_screen(HelpModal(self.BINDINGS))  # type: ignore[attr-defined]


class _InterruptConfirmModal(ModalScreen[bool]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter,y", "confirm", "Confirm", show=True),
        Binding("escape,n", "cancel", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    _InterruptConfirmModal {
        align: center middle;
    }
    _InterruptConfirmModal > Static {
        width: 64;
        height: auto;
        padding: 1 2;
        border: solid $warning;
        background: $surface;
    }
    """

    def __init__(self, run_name: str) -> None:
        super().__init__()
        self._run_name = run_name

    def compose(self):
        yield Static(
            "[bold]Stop run?[/bold]\n\n"
            f"[dim]{self._run_name}[/dim]\n\n"
            "[yellow]A graceful interrupt (SIGINT) will be sent to the pipeline process.[/yellow]\n\n"
            "[dim]Enter / Y — confirm    Esc / N — cancel[/dim]",
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class _InterruptConfirmMixin:
    def _confirm_interrupt_run(self, run_dir: Path) -> None:
        resolved_run_dir = run_dir.expanduser().resolve()

        def _after_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            if self.app.interrupt_active_launch_for_run(resolved_run_dir):  # type: ignore[attr-defined]
                with contextlib.suppress(Exception):
                    self.refresh_bindings()  # type: ignore[attr-defined]

        self.app.push_screen(_InterruptConfirmModal(resolved_run_dir.name), _after_confirm)  # type: ignore[attr-defined]


class _TabbedScreenMixin:
    """Helpers for Screens that use TabbedContent.

    Provides:
    - ``_focus_tabs_after_mount()``: call at end of ``on_mount`` to park
      initial focus on the Tabs bar (k9s / lazygit UX pattern).
    - ``on_key``: Down / Enter from focused Tabs bar → ``focus_next()``,
      moving focus into the active tab's first focusable widget.
    """

    def _focus_tabs_after_mount(self) -> None:
        from textual.widgets import Tabs

        self.call_after_refresh(lambda: self.query_one(Tabs).focus())  # type: ignore[attr-defined]

    def _activate_tab_by_index(self, tab_index: int) -> bool:
        from textual.widgets import TabPane, Tabs

        panes = [pane.id for pane in self.query(TabPane) if pane.id]  # type: ignore[attr-defined]
        if not 0 <= tab_index < len(panes):
            return False

        tabs = self.query_one(Tabs)  # type: ignore[attr-defined]
        tabs.focus()
        self.action_show_tab(panes[tab_index])  # type: ignore[attr-defined]
        self.call_after_refresh(tabs.focus)  # type: ignore[attr-defined]
        return True

    def on_key(self, event: Key) -> None:  # type: ignore[override]
        from textual.widgets import Tabs

        if event.key in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
            with contextlib.suppress(Exception):
                if self._activate_tab_by_index(int(event.key) - 1):
                    event.stop()
                    return

        if isinstance(self.focused, Tabs) and event.key in ("down", "enter"):  # type: ignore[attr-defined]
            self.focus_next()  # type: ignore[attr-defined]
            event.stop()
