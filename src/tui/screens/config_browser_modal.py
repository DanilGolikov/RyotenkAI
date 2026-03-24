from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text

if TYPE_CHECKING:
    from pathlib import Path

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, RichLog, Static

from src.tui.config_browser_state import ConfigBrowserItem, ConfigBrowserState
from src.tui.screens._hotkey_bar import DockedHotkeyBar
from src.tui.screens._mixins import _HelpMixin

if TYPE_CHECKING:
    from textual.app import ComposeResult


class StructuredConfigBrowser(_HelpMixin, ModalScreen[None]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "escape_or_close", "Close", show=True, priority=True),
        Binding("question_mark", "show_help", "Help", key_display="?", show=True),
        Binding("enter", "open_item", "Open", show=True, priority=True),
        Binding("right", "move_right", "Items", show=True, priority=True),
        Binding("backspace,left", "navigate_up", "Up", show=True, priority=True),
        Binding("r", "open_raw_yaml", "Raw", show=True, priority=True),
        Binding("w", "toggle_wrap", "Wrap", show=True, priority=True),
    ]

    DEFAULT_CSS = """
    StructuredConfigBrowser {
        align: center middle;
    }
    #config-browser-dialog {
        width: 136;
        height: 80vh;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    #config-browser-title {
        color: $primary;
        text-style: bold;
    }
    #config-browser-path {
        color: $text-muted;
        padding-bottom: 1;
    }
    #config-browser-breadcrumbs {
        color: $text-muted;
        padding-bottom: 1;
    }
    #config-browser-body {
        height: 1fr;
    }
    .config-browser-pane {
        height: 1fr;
        border: solid $panel;
        margin-right: 1;
        padding: 0 1;
    }
    .config-browser-pane:last-child {
        margin-right: 0;
    }
    .config-browser-pane-title {
        color: $accent;
        text-style: bold;
        padding: 0;
    }
    #config-browser-sections-pane {
        width: 28;
    }
    #config-browser-items-pane {
        width: 40;
    }
    #config-browser-detail-pane {
        width: 1fr;
    }
    #config-browser-sections {
        height: 1fr;
    }
    #config-browser-items {
        height: 1fr;
    }
    #config-browser-detail {
        height: 1fr;
        border: none;
    }
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path.expanduser().resolve()
        self._browser_state: ConfigBrowserState | None = None
        self._sections: tuple[ConfigBrowserItem, ...] = ()
        self._current_items: tuple[ConfigBrowserItem, ...] = ()
        self._current_path: tuple[str | int, ...] = ()
        self._detail_path: tuple[str | int, ...] = ()
        self._suspend_item_highlight_detail = False
        self._detail_wrap_enabled = False

    def compose(self) -> ComposeResult:
        with Vertical(id="config-browser-dialog"):
            yield Static("Structured config browser", id="config-browser-title")
            yield Static(str(self._path), id="config-browser-path")
            yield Static("", id="config-browser-breadcrumbs")
            with Horizontal(id="config-browser-body"):
                with Vertical(id="config-browser-sections-pane", classes="config-browser-pane"):
                    yield Static("Sections", classes="config-browser-pane-title")
                    yield OptionList(id="config-browser-sections", markup=True, compact=True)
                with Vertical(id="config-browser-items-pane", classes="config-browser-pane"):
                    yield Static("Items", classes="config-browser-pane-title")
                    yield OptionList(id="config-browser-items", markup=True, compact=True)
                with Vertical(id="config-browser-detail-pane", classes="config-browser-pane"):
                    yield Static("Details", classes="config-browser-pane-title")
                    yield RichLog(
                        id="config-browser-detail",
                        highlight=False,
                        markup=False,
                        wrap=self._detail_wrap_enabled,
                        auto_scroll=False,
                    )
            yield DockedHotkeyBar("", id="config-browser-hotkey-bar")

    def on_mount(self) -> None:
        detail = self.query_one("#config-browser-detail", RichLog)
        try:
            self._browser_state = ConfigBrowserState.load(self._path)
        except Exception as exc:
            detail.write(f"Cannot load config: {exc}")
            return
        self._sections = self._browser_state.section_items()
        self._set_section_options()
        if self._sections:
            self._activate_section(0)
        self._refresh_hotkeys()
        self.call_after_refresh(lambda: self.query_one("#config-browser-sections", OptionList).focus())

    def _set_section_options(self) -> None:
        section_list = self.query_one("#config-browser-sections", OptionList)
        section_list.set_options([_item_prompt(item) for item in self._sections])
        if self._sections:
            section_list.highlighted = 0

    def _activate_section(self, index: int) -> None:
        if not (0 <= index < len(self._sections)):
            return
        self._current_path = self._sections[index].path
        self._refresh_items(detail_path=self._current_path)

    def _refresh_items(
        self,
        *,
        highlighted_path: tuple[str | int, ...] | None = None,
        detail_path: tuple[str | int, ...] | None = None,
    ) -> None:
        if self._browser_state is None:
            return
        item_list = self.query_one("#config-browser-items", OptionList)
        self._current_items = self._browser_state.list_children(self._current_path)
        item_list.set_options([_item_prompt(item) for item in self._current_items])
        self._detail_path = detail_path or self._current_path
        self._refresh_breadcrumbs()
        self._refresh_hotkeys()
        if self._current_items:
            highlighted_index = 0
            if highlighted_path is not None:
                for index, item in enumerate(self._current_items):
                    if item.path == highlighted_path:
                        highlighted_index = index
                        break
            self._suspend_item_highlight_detail = True
            item_list.highlighted = highlighted_index
            self._update_detail(self._detail_path)
            return
        self._update_detail(self._detail_path)

    def _refresh_breadcrumbs(self) -> None:
        path = self._detail_path or self._current_path
        if not path:
            breadcrumbs = "Root"
        else:
            breadcrumbs = " > ".join(_format_breadcrumb_part(part) for part in path)
        self.query_one("#config-browser-breadcrumbs", Static).update(f"Location: {breadcrumbs}")

    def _browser_nodes_ready(self) -> bool:
        with contextlib.suppress(Exception):
            self.query_one("#config-browser-breadcrumbs", Static)
            self.query_one("#config-browser-detail", RichLog)
            return True
        return False

    def _sync_detail_to_focus(self) -> None:
        if not self._browser_nodes_ready():
            return
        focused = self.focused
        if isinstance(focused, OptionList) and focused.id == "config-browser-items":
            highlighted = focused.highlighted
            if highlighted is not None and 0 <= highlighted < len(self._current_items):
                self._detail_path = self._current_items[highlighted].path
            else:
                self._detail_path = self._current_path
        else:
            self._detail_path = self._current_path
        self._refresh_breadcrumbs()
        self._update_detail(self._detail_path)

    def _update_detail(self, path: tuple[str | int, ...]) -> None:
        detail = self.query_one("#config-browser-detail", RichLog)
        detail.wrap = self._detail_wrap_enabled
        detail.clear()
        if self._browser_state is None:
            return
        lines = self._browser_state.describe(path)
        yaml_start = next((index for index, line in enumerate(lines) if line == "YAML:"), None)
        summary_lines = lines if yaml_start is None else lines[:yaml_start]
        yaml_lines = () if yaml_start is None else lines[yaml_start + 1 :]

        for line in summary_lines:
            if not line:
                detail.write("")
                continue
            detail.write(_styled_detail_line(line))

        if yaml_lines:
            if summary_lines:
                detail.write("")
            detail.write(Text("YAML", style="bold cyan"))
            detail.write(
                Syntax(
                    "\n".join(line[2:] if line.startswith("  ") else line for line in yaml_lines),
                    "yaml",
                    theme="monokai",
                    word_wrap=self._detail_wrap_enabled,
                    line_numbers=False,
                )
            )

    def action_open_item(self) -> None:
        focused = self.focused
        if isinstance(focused, OptionList) and focused.id == "config-browser-sections":
            if focused.highlighted is not None:
                self._activate_section(focused.highlighted)
                self.query_one("#config-browser-items", OptionList).focus()
            return
        if not isinstance(focused, OptionList) or focused.id != "config-browser-items":
            return
        highlighted = focused.highlighted
        if highlighted is None or not (0 <= highlighted < len(self._current_items)):
            return
        selected = self._current_items[highlighted]
        if selected.has_children:
            self._current_path = selected.path
            self._refresh_items(detail_path=self._current_path)

    def action_navigate_up(self) -> None:
        if len(self._current_path) > 1:
            previous_path = self._current_path
            self._current_path = self._current_path[:-1]
            self._refresh_items(highlighted_path=previous_path, detail_path=previous_path)
            self.query_one("#config-browser-items", OptionList).focus()
            return
        with contextlib.suppress(Exception):
            self.query_one("#config-browser-sections", OptionList).focus()

    def action_open_raw_yaml(self) -> None:
        from src.tui.screens.file_preview_modal import FilePreviewModal

        self.app.push_screen(FilePreviewModal(self._path))

    def action_close_browser(self) -> None:
        self.dismiss(None)

    def action_escape_or_close(self) -> None:
        if len(self._current_path) > 1:
            self.action_navigate_up()
            return
        self.action_close_browser()

    def action_toggle_wrap(self) -> None:
        self._detail_wrap_enabled = not self._detail_wrap_enabled
        self._refresh_hotkeys()
        self._update_detail(self._detail_path or self._current_path)

    def _refresh_hotkeys(self) -> None:
        footer = self.query_one("#config-browser-hotkey-bar", DockedHotkeyBar)
        escape_label = "Up" if len(self._current_path) > 1 else "Close"
        footer.update(
            "   ".join(
                [
                    f"[bold]Esc[/bold] {escape_label}",
                    "[bold]?[/bold] Help",
                    "[bold]←↑↓→[/bold] Move",
                    "[bold]R[/bold] Raw",
                    "[bold]W[/bold] Wrap",
                ]
            )
        )

    @on(OptionList.OptionHighlighted, "#config-browser-sections")
    def _section_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._activate_section(event.option_index)

    @on(OptionList.OptionSelected, "#config-browser-sections")
    def _section_selected(self, event: OptionList.OptionSelected) -> None:
        self._activate_section(event.option_index)
        self.query_one("#config-browser-items", OptionList).focus()

    @on(OptionList.OptionHighlighted, "#config-browser-items")
    def _item_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if not (0 <= event.option_index < len(self._current_items)):
            return
        if self._suspend_item_highlight_detail:
            self._suspend_item_highlight_detail = False
            return
        self._detail_path = self._current_items[event.option_index].path
        self._refresh_breadcrumbs()
        self._update_detail(self._detail_path)

    @on(OptionList.OptionSelected, "#config-browser-items")
    def _item_selected(self, event: OptionList.OptionSelected) -> None:
        if not (0 <= event.option_index < len(self._current_items)):
            return
        selected = self._current_items[event.option_index]
        if selected.has_children:
            self._current_path = selected.path
            self._refresh_items(detail_path=self._current_path)

    def action_move_right(self) -> None:
        focused = self.focused
        if isinstance(focused, OptionList) and focused.id == "config-browser-sections":
            if focused.highlighted is not None:
                self._activate_section(focused.highlighted)
            self.query_one("#config-browser-items", OptionList).focus()
            return
        if isinstance(focused, OptionList) and focused.id == "config-browser-items":
            self.action_open_item()

    def _watch_focused(self) -> None:
        super()._watch_focused()
        self.call_after_refresh(self._sync_detail_to_focus)


def _item_prompt(item: ConfigBrowserItem) -> str:
    label = f"[bold cyan]{escape(item.label)}[/bold cyan]"
    if item.is_section:
        return label
    if item.value_text is not None:
        return f"{label} [dim]-[/dim] [green]{escape(item.value_text)}[/green]"
    if item.item_count is not None:
        noun = "item" if item.item_count == 1 else "items"
        return f"{label} [dim]-[/dim] [magenta]{item.item_count} {noun}[/magenta]"
    if item.field_count is not None:
        noun = "field" if item.field_count == 1 else "fields"
        return f"{label} [dim]-[/dim] [yellow]{item.field_count} {noun}[/yellow]"
    if item.subtitle:
        return f"{label} [dim]-[/dim] [green]{escape(item.subtitle)}[/green]"
    return label


def _styled_detail_line(line: str) -> Text:
    if ":" not in line:
        return Text(line, style="white")
    key, raw_value = line.split(":", 1)
    value = raw_value.strip()
    text = Text()
    text.append(f"{key}:", style="bold cyan")
    text.append(" ")
    count_style = _detail_count_style(key, value)
    if count_style is not None:
        text.append(value, style=count_style)
    else:
        text.append(value, style="green")
    return text


def _detail_count_style(key: str, value: str) -> str | None:
    normalized_key = key.lower()
    normalized_value = value.lower()
    if any(token in normalized_key for token in ("field", "threshold fields")):
        return "yellow"
    if any(token in normalized_key for token in ("items", "plugins")):
        return "magenta"
    if normalized_value.endswith((" item", " items", " plugin", " plugins")):
        return "magenta"
    if normalized_value.endswith((" field", " fields")):
        return "yellow"
    return None


def _format_breadcrumb_part(part: str | int) -> str:
    if isinstance(part, int):
        return f"[{part}]"
    return str(part)


__all__ = ["StructuredConfigBrowser"]
