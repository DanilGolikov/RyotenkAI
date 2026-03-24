from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
from typing import TYPE_CHECKING, ClassVar

import yaml
from rich.syntax import Syntax
from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import RichLog, Static, TextArea

if TYPE_CHECKING:
    from pathlib import Path

from src.tui.screens._hotkey_bar import DockedHotkeyBar


def _editor_command_for_path(path: Path) -> list[str]:
    editor = os.environ.get("VISUAL", "").strip() or os.environ.get("EDITOR", "").strip()
    if editor:
        return [*shlex.split(editor), str(path)]
    return ["cursor", "-r", str(path)]


def _supports_textarea_syntax() -> bool:
    return bool(importlib.util.find_spec("tree_sitter"))


class _DiscardChangesModal(ModalScreen[bool]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter,y", "confirm", "Discard", show=True),
        Binding("escape,n", "cancel", "Keep editing", show=True),
    ]

    DEFAULT_CSS = """
    _DiscardChangesModal {
        align: center middle;
    }
    #discard-dialog {
        width: 62;
        height: auto;
        padding: 1 2;
        border: solid $warning;
        background: $surface;
    }
    """

    def compose(self):
        yield Static(
            "[bold]Discard unsaved changes?[/bold]\n\n" "[dim]Enter / Y — discard   Esc / N — continue editing[/dim]",
            id="discard-dialog",
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class FilePreviewModal(ModalScreen[None]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel_or_close", "Close", show=True, priority=True),
        Binding("e", "start_edit", "Edit", show=True, priority=True),
        Binding("ctrl+s", "save_changes", "Save", key_display="⌃s", show=True, priority=True),
        Binding("ctrl+o", "open_external_editor", "Open Editor", key_display="⌃o", show=True, priority=True),
        Binding("q", "close_modal", "Close", show=True),
    ]

    DEFAULT_CSS = """
    FilePreviewModal {
        align: center middle;
    }
    #file-preview-dialog {
        width: 104;
        height: 74vh;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    #file-preview-body {
        width: 100%;
        height: auto;
    }
    #file-preview-title {
        color: $primary;
        text-style: bold;
        padding: 0;
    }
    #file-preview-path {
        color: $text-muted;
        padding-bottom: 1;
    }
    #file-preview-error {
        color: $text-muted;
        height: auto;
        padding-bottom: 1;
    }
    .status-error {
        color: $error;
    }
    .status-success {
        color: $success;
    }
    .status-warning {
        color: $warning;
    }
    #file-preview-log {
        height: 1fr;
        border: solid $panel;
    }
    #file-preview-editor {
        height: 1fr;
        border: solid $panel;
    }
    .is-hidden {
        display: none;
    }
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path.expanduser().resolve()
        self._original_content = ""
        self._is_dirty = False
        self._is_editing = False
        self._supports_syntax = _supports_textarea_syntax()

    def compose(self):
        with Vertical(id="file-preview-dialog"):
            with Vertical(id="file-preview-body"):
                yield Static("Config preview", id="file-preview-title")
                yield Static(str(self._path), id="file-preview-path")
                yield Static("", id="file-preview-error")
                yield RichLog(id="file-preview-log", highlight=False, markup=False, wrap=False, auto_scroll=False)
                yield TextArea(
                    "",
                    id="file-preview-editor",
                    language="yaml"
                    if self._supports_syntax and self._path.suffix.lower() in {".yaml", ".yml"}
                    else None,
                    show_line_numbers=True,
                    soft_wrap=False,
                    tab_behavior="indent",
                    classes="is-hidden",
                )
            yield DockedHotkeyBar("", id="file-preview-hotkey-bar")

    def on_mount(self) -> None:
        if not self._path.exists():
            self.query_one("#file-preview-log", RichLog).write("File not found.")
            self._refresh_title()
            self._refresh_hotkeys()
            return
        try:
            self._original_content = self._path.read_text(encoding="utf-8", errors="replace")
            self._render_preview()
            self.query_one("#file-preview-editor", TextArea).text = self._original_content
        except OSError as exc:
            self.query_one("#file-preview-log", RichLog).write(f"Cannot read file: {exc}")
        self._refresh_title()
        self._refresh_hotkeys()

    def _refresh_title(self) -> None:
        title = "Config edit" if self._is_editing else "Config preview"
        if self._is_dirty:
            title += " [unsaved]"
        self.query_one("#file-preview-title", Static).update(title)

    def _refresh_hotkeys(self) -> None:
        footer = self.query_one("#file-preview-hotkey-bar", DockedHotkeyBar)
        parts: list[str] = []
        if self._is_editing:
            parts.append("[bold]Esc[/bold] Discard")
            parts.append("[bold]⌃s[/bold] Save")
            parts.append("[bold]⌃z[/bold] Undo")
            parts.append("[bold]⌃y[/bold] Redo")
            parts.append("[bold]⌃o[/bold] Open Editor")
            if self._is_dirty:
                parts.append("[bold yellow]*[/bold yellow] Unsaved")
        else:
            parts.append("[bold]Esc[/bold] Close")
            parts.append("[bold]E[/bold] Edit")
        footer.update("   ".join(parts))

    def _set_status(self, message: str, level: str = "muted") -> None:
        status = self.query_one("#file-preview-error", Static)
        status.remove_class("status-error")
        status.remove_class("status-success")
        status.remove_class("status-warning")
        if level == "error":
            status.add_class("status-error")
        elif level == "success":
            status.add_class("status-success")
        elif level == "warning":
            status.add_class("status-warning")
        status.update(message)

    def _render_preview(self) -> None:
        log = self.query_one("#file-preview-log", RichLog)
        log.clear()
        if self._path.suffix.lower() in {".yaml", ".yml"}:
            log.write(
                Syntax(
                    self._original_content,
                    "yaml",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=False,
                ),
            )
            return
        for line in self._original_content.splitlines():
            log.write(line.rstrip())

    def _set_edit_mode(self, editing: bool) -> None:
        self._is_editing = editing
        log = self.query_one("#file-preview-log", RichLog)
        editor = self.query_one("#file-preview-editor", TextArea)
        if editing:
            log.add_class("is-hidden")
            editor.remove_class("is-hidden")
            editor.focus()
        else:
            editor.add_class("is-hidden")
            log.remove_class("is-hidden")
            log.focus()
        self._set_status("")
        self._refresh_title()
        self._refresh_hotkeys()

    def _discard_changes(self) -> None:
        editor = self.query_one("#file-preview-editor", TextArea)
        editor.text = self._original_content
        self._is_dirty = False
        self._set_edit_mode(False)

    def action_start_edit(self) -> None:
        if not self._path.exists() or self._is_editing:
            return
        self._set_edit_mode(True)

    def action_save_changes(self) -> None:
        if not self._is_editing:
            return
        editor = self.query_one("#file-preview-editor", TextArea)
        content = editor.text
        if self._path.suffix.lower() in {".yaml", ".yml"}:
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as exc:
                self._set_status(f"YAML validation failed: {exc}", level="error")
                return
        try:
            self._path.write_text(content, encoding="utf-8")
        except OSError as exc:
            self._set_status(f"Cannot save file: {exc}", level="error")
            return
        self._original_content = content
        self._is_dirty = False
        self._render_preview()
        self._set_edit_mode(False)
        self._set_status(f"Saved: {self._path.name}", level="success")
        self.notify(f"Saved: {self._path.name}", severity="information")

    def action_open_external_editor(self) -> None:
        try:
            if self._is_dirty:
                self.notify("Unsaved TUI changes are not opened externally", severity="warning")
            command = _editor_command_for_path(self._path)
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._set_status(f"Opened in editor: {self._path.name}", level="success")
            self.notify(f"Opened in editor: {self._path.name}", severity="information")
        except OSError as exc:
            self._set_status(f"Cannot open editor: {exc}", level="error")
            self.notify(f"Cannot open editor: {exc}", severity="error")

    def action_cancel_or_close(self) -> None:
        if not self._is_editing:
            self.dismiss(None)
            return
        if not self._is_dirty:
            self._set_edit_mode(False)
            return

        def _after_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self._discard_changes()

        self.app.push_screen(_DiscardChangesModal(), _after_confirm)

    def action_close_modal(self) -> None:
        self.action_cancel_or_close()

    @on(TextArea.Changed, "#file-preview-editor")
    def _editor_changed(self, event: TextArea.Changed) -> None:
        self._is_dirty = event.text_area.text != self._original_content
        self._set_status("")
        self._refresh_title()
        self._refresh_hotkeys()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "start_edit":
            return self._path.exists() and not self._is_editing
        if action == "save_changes":
            return self._is_editing
        if action == "open_external_editor":
            return self._is_editing
        if action == "close_modal":
            return not self._is_editing
        return True
