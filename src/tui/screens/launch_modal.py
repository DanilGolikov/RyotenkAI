from __future__ import annotations

import contextlib
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Select, Static

from src.pipeline.domain import build_run_directory
from src.tui.launch import LaunchRequest, RestartPointOption
from src.tui.launch_state import build_submittable_launch_request, mode_label, prepare_launch_mode_state
from src.tui.screens._hotkey_bar import DockedHotkeyBar

_MODE_OPTIONS = [
    ("New Run", "new_run"),
    ("Fresh attempt", "fresh"),
    ("Resume", "resume"),
    ("Start from stage", "restart"),
]

_LOG_LEVEL_OPTIONS = [
    ("INFO", "INFO"),
    ("DEBUG", "DEBUG"),
]


def _run_dir_value_for_mode(mode: str, *, context_run_dir: Path | None, new_run_dir: Path) -> str:
    if mode == "new_run":
        return str(new_run_dir)
    return str(context_run_dir) if context_run_dir is not None else ""


def _restart_section_visible(mode: str) -> bool:
    return mode == "restart"


def _update_plain_text(widget: Static, message: str, *, style: str | None = None) -> None:
    text = Text(message)
    if style:
        text.stylize(style)
    widget.update(text)


class _FocusSink(Static):
    can_focus = True


class _LaunchConfirmModal(ModalScreen[bool]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter,y", "confirm", "Confirm", show=True),
        Binding("escape,n", "cancel", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    _LaunchConfirmModal {
        align: center middle;
    }
    #launch-confirm-dialog {
        width: 68;
        height: auto;
        padding: 1 2;
        border: solid $warning;
        background: $surface;
    }
    #launch-confirm-title {
        color: $warning;
        text-style: bold;
        padding-bottom: 1;
    }
    #launch-confirm-body {
        color: $text;
    }
    """

    def __init__(self, request: LaunchRequest) -> None:
        super().__init__()
        self._request = request

    def compose(self):
        config_text = str(self._request.config_path) if self._request.config_path is not None else "auto"
        stage_text = self._request.restart_from_stage or "auto"
        mode_text = mode_label(self._request.mode)
        lines = [
            f"[bold]Mode:[/bold] {mode_text}",
            f"[bold]Config:[/bold] {config_text}",
            f"[bold]Log level:[/bold] {self._request.log_level}",
        ]
        if self._request.mode != "new_run":
            lines.insert(0, f"[bold]Run dir:[/bold] {self._request.run_dir}")
        if self._request.mode == "restart":
            lines.append(f"[bold]Stage:[/bold] {stage_text}")
        body = "\n".join(lines) + "\n\n[dim]Enter / Y — confirm    Esc / N — cancel[/dim]"
        with Vertical(id="launch-confirm-dialog"):
            yield Static("[bold]Confirm launch[/bold]", id="launch-confirm-title")
            yield Static(body, id="launch-confirm-body")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class LaunchModal(ModalScreen[LaunchRequest | None]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Close", show=True, priority=True),
        Binding("enter", "submit", "Launch", show=True),
        Binding("ctrl+b", "browse_config", "Config", key_display="⌃b", show=True, priority=True),
        Binding("ctrl+o", "preview_config", "Preview", key_display="⌃o", show=True, priority=True),
        Binding("p", "reload_points", "Reload", show=True, priority=True),
        Binding("ctrl+z", "undo_input", "Undo", key_display="⌃z", show=True, priority=True),
    ]

    DEFAULT_CSS = """
    LaunchModal {
        align: center middle;
    }
    #launch-dialog {
        width: 84;
        height: auto;
        padding: 1;
        border: solid $accent;
        background: $surface;
    }
    #launch-body {
        width: 100%;
        height: auto;
        max-height: 80vh;
        overflow-y: auto;
    }
    #launch-title {
        color: $accent;
        text-style: bold;
        padding: 0 0 1 0;
    }
    #launch-focus-sink {
        height: 0;
        width: 0;
        margin: 0;
        padding: 0;
    }
    .is-hidden {
        display: none;
    }
    .launch-section {
        border: solid $panel;
        background: $boost;
        padding: 0 1;
        margin-top: 1;
        height: auto;
    }
    .launch-section-title {
        color: $primary;
        text-style: bold;
        padding: 0;
    }
    .launch-label {
        color: $text-muted;
        padding-top: 0;
    }
    .launch-field {
        margin-top: 0;
    }
    .launch-readonly {
        background: $surface-lighten-1;
        color: $text-muted;
    }
    #restart-points {
        height: auto;
        min-height: 5;
        max-height: 9;
        overflow-y: auto;
        border: solid $panel;
        padding: 0 1;
        margin-top: 0;
    }
    #launch-error {
        padding-top: 1;
        color: $error;
    }
    """

    def __init__(
        self,
        *,
        default_mode: str = "new_run",
        default_run_dir: Path | None = None,
        default_config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._default_mode = default_mode
        self._default_config_path = default_config_path
        generated_run_dir, _created_at = build_run_directory(base_dir=Path("runs"))
        self._context_run_dir = default_run_dir if default_mode != "new_run" else None
        self._new_run_dir = (
            default_run_dir if default_mode == "new_run" and default_run_dir is not None else generated_run_dir
        )
        self._restart_points: list[RestartPointOption] = []
        self._history_lock = False
        self._input_history: dict[str, list[str]] = {}
        self._input_history_index: dict[str, int] = {}

    def compose(self):
        run_dir = _run_dir_value_for_mode(
            self._default_mode,
            context_run_dir=self._context_run_dir,
            new_run_dir=self._new_run_dir,
        )
        config_path = str(self._default_config_path) if self._default_config_path is not None else ""
        with Vertical(id="launch-dialog"):
            with Vertical(id="launch-body"):
                yield _FocusSink("", id="launch-focus-sink")
                yield Static("[bold]Pipeline launch[/bold]", id="launch-title")
                with Vertical(id="launch-target-section", classes="launch-section"):
                    yield Static("Target", classes="launch-section-title")
                    yield Static("Run directory", classes="launch-label")
                    yield Input(
                        value=str(run_dir), placeholder="runs/my_run", id="launch-run-dir", classes="launch-field"
                    )
                with Vertical(classes="launch-section"):
                    yield Static("Launch mode", classes="launch-section-title")
                    yield Static("Mode", classes="launch-label")
                    yield Select(_MODE_OPTIONS, value=self._default_mode, id="launch-mode", classes="launch-field")
                    with Vertical(id="launch-restart-section"):
                        yield Static("Restart stage", classes="launch-label")
                        yield Select(
                            [], id="launch-restart-stage", prompt="Select restart stage…", classes="launch-field"
                        )
                        yield Static("", id="restart-points")
                with Vertical(classes="launch-section"):
                    yield Static("Config", classes="launch-section-title")
                    yield Static("Config path", classes="launch-label")
                    yield Input(
                        value=config_path,
                        placeholder="config/pipeline.yaml",
                        id="launch-config-path",
                        classes="launch-field",
                    )
                    yield Static("Log level", classes="launch-label")
                    yield Select(_LOG_LEVEL_OPTIONS, value="INFO", id="launch-log-level", classes="launch-field")
                yield Static("", id="launch-error")
            yield DockedHotkeyBar("", id="launch-hotkey-bar")

    def on_mount(self) -> None:
        run_dir_input = self.query_one("#launch-run-dir", Input)
        run_dir_input.disabled = True
        run_dir_input.add_class("launch-readonly")
        config_input = self.query_one("#launch-config-path", Input)
        self._seed_input_history("launch-config-path", config_input.value)
        self._sync_mode_ui()
        self._refresh_restart_points()
        self._refresh_footer()
        self.call_after_refresh(self._focus_initial_field)

    def _focus_initial_field(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#launch-mode", Select).focus()
        self._refresh_footer()

    def _selected_mode(self) -> str:
        value = self.query_one("#launch-mode", Select).value
        return str(value) if value is not Select.BLANK else self._default_mode

    def _sync_mode_ui(self) -> None:
        mode = self._selected_mode()
        target_section = self.query_one("#launch-target-section", Vertical)
        restart_section = self.query_one("#launch-restart-section", Vertical)
        run_dir_input = self.query_one("#launch-run-dir", Input)
        if mode == "new_run":
            target_section.add_class("is-hidden")
            run_dir_input.value = _run_dir_value_for_mode(
                mode,
                context_run_dir=self._context_run_dir,
                new_run_dir=self._new_run_dir,
            )
        else:
            target_section.remove_class("is-hidden")
            run_dir_input.value = _run_dir_value_for_mode(
                mode,
                context_run_dir=self._context_run_dir,
                new_run_dir=self._new_run_dir,
            )
        if _restart_section_visible(mode):
            restart_section.remove_class("is-hidden")
        else:
            restart_section.add_class("is-hidden")

    def _read_run_dir(self) -> Path | None:
        raw = self.query_one("#launch-run-dir", Input).value.strip()
        return Path(raw).expanduser() if raw else None

    def _read_config_path(self) -> Path | None:
        raw = self.query_one("#launch-config-path", Input).value.strip()
        return Path(raw).expanduser() if raw else None

    def _selected_log_level(self) -> str:
        value = self.query_one("#launch-log-level", Select).value
        return str(value) if value is not Select.BLANK else "INFO"

    def _refresh_restart_points(self) -> None:
        info = self.query_one("#restart-points", Static)
        selector = self.query_one("#launch-restart-stage", Select)
        mode_selector = self.query_one("#launch-mode", Select)
        error = self.query_one("#launch-error", Static)
        _update_plain_text(error, "")

        mode = self._selected_mode()
        run_dir = self._read_run_dir()
        config_path = self._read_config_path()
        try:
            state = prepare_launch_mode_state(mode, run_dir, config_path)
        except Exception as exc:
            self._restart_points = []
            selector.set_options([])
            selector.clear()
            selector.disabled = True
            error_prefix = "Cannot prepare resume" if mode == "resume" else "Cannot load restart points"
            _update_plain_text(info, f"{error_prefix}: {exc}", style="red")
            self._refresh_footer()
            return

        self._restart_points = list(state.restart_points)
        if config_path is None and state.resolved_config_path is not None:
            self.query_one("#launch-config-path", Input).value = str(state.resolved_config_path)

        selector.set_options(state.restart_options)
        selector.disabled = state.restart_selector_disabled
        if state.selected_restart_stage is not None:
            selector.value = state.selected_restart_stage
        else:
            selector.clear()
        mode_selector.border_title = state.mode_title
        info.update(state.info_markup)
        self._refresh_footer()

    def _refresh_footer(self) -> None:
        with contextlib.suppress(Exception):
            footer = self.query_one("#launch-hotkey-bar", DockedHotkeyBar)
            mode = self._selected_mode()
            focused = self.focused
            parts: list[str] = []
            if not isinstance(focused, OptionList):
                parts.append("[bold]Esc[/bold] Unfocus" if isinstance(focused, Input) else "[bold]Esc[/bold] Close")
            if isinstance(focused, OptionList):
                parts = ["[bold]Esc[/bold] Close list", "[bold]Enter[/bold] Choose"]
            elif isinstance(focused, Select):
                parts.append("[bold]Enter[/bold] Open list")
            else:
                parts.append("[bold]Enter[/bold] Launch")
            if mode == "restart":
                parts.append("[bold]P[/bold] Reload")
            if isinstance(focused, Input) and focused.id == "launch-config-path":
                parts.append("[bold]⌃z[/bold] Undo")
                parts.append("[bold]⌃b[/bold] Config")
                parts.append("[bold]⌃o[/bold] Preview")

            footer.update("   ".join(parts))

    def action_reload_points(self) -> None:
        self._release_focus()
        self._refresh_restart_points()

    def action_preview_config(self) -> None:
        from src.tui.screens.file_preview_modal import FilePreviewModal

        error = self.query_one("#launch-error", Static)
        _update_plain_text(error, "")
        focused = self.focused
        if not isinstance(focused, Input) or focused.id != "launch-config-path":
            self._refresh_footer()
            return
        config_path = self._read_config_path()
        if config_path is None:
            _update_plain_text(error, "Config path is empty.", style="red")
            self._refresh_footer()
            return
        self._release_focus()
        self.app.push_screen(FilePreviewModal(config_path))

    def action_browse_config(self) -> None:
        from src.tui.screens.config_browser_modal import StructuredConfigBrowser

        error = self.query_one("#launch-error", Static)
        _update_plain_text(error, "")
        config_path = self._read_config_path()
        if config_path is None:
            _update_plain_text(error, "Config path is empty.", style="red")
            self._refresh_footer()
            return
        self._release_focus()
        self.app.push_screen(StructuredConfigBrowser(config_path))

    def action_submit(self) -> None:
        error = self.query_one("#launch-error", Static)
        mode = self._selected_mode()
        run_dir = self._read_run_dir()
        config_path = self._read_config_path()
        restart_stage_value = self.query_one("#launch-restart-stage", Select).value
        log_level = self._selected_log_level()
        restart_stage = None if mode != "restart" or restart_stage_value is Select.BLANK else str(restart_stage_value)

        if run_dir is None:
            _update_plain_text(error, "Run directory is required.", style="red")
            return

        try:
            request, resolved_config = build_submittable_launch_request(
                mode=mode,
                run_dir=run_dir,
                config_path=config_path,
                restart_stage=restart_stage,
                log_level=log_level,
            )
        except ValueError as exc:
            _update_plain_text(error, str(exc), style="red")
            return
        if config_path is None and resolved_config is not None:
            self.query_one("#launch-config-path", Input).value = str(resolved_config)

        def _after_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self.dismiss(request)

        self.app.push_screen(_LaunchConfirmModal(request), _after_confirm)

    def action_cancel(self) -> None:
        if self._release_focus():
            self._refresh_footer()
            return
        self.dismiss(None)

    def _release_focus(self) -> bool:
        focused = self.focused
        focus_sink = self.query_one("#launch-focus-sink", _FocusSink)
        if focused is None or focused is focus_sink:
            return False
        with contextlib.suppress(Exception):
            focus_sink.focus()
            return True
        return False

    def _seed_input_history(self, input_id: str, value: str) -> None:
        self._input_history[input_id] = [value]
        self._input_history_index[input_id] = 0

    def _remember_input_state(self, input_id: str, value: str) -> None:
        history = self._input_history.setdefault(input_id, [])
        current_idx = self._input_history_index.setdefault(input_id, -1)
        if current_idx >= 0 and history[current_idx] == value:
            return
        del history[current_idx + 1 :]
        history.append(value)
        self._input_history_index[input_id] = len(history) - 1

    def action_undo_input(self) -> None:
        focused = self.focused
        if not isinstance(focused, Input) or focused.id is None:
            return
        input_id = focused.id
        if input_id not in self._input_history:
            self._seed_input_history(input_id, focused.value)
            return
        current_idx = self._input_history_index.get(input_id, 0)
        if current_idx <= 0:
            return
        next_idx = current_idx - 1
        self._input_history_index[input_id] = next_idx
        previous_value = self._input_history[input_id][next_idx]
        self._history_lock = True
        focused.value = previous_value
        focused.cursor_position = len(previous_value)
        self._history_lock = False

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "submit":
            return not isinstance(self.focused, Select | OptionList)
        if action == "reload_points":
            return self._selected_mode() == "restart"
        if action == "undo_input":
            return isinstance(self.focused, Input) and getattr(self.focused, "id", None) != "launch-run-dir"
        if action == "browse_config":
            return self._read_config_path() is not None
        if action == "preview_config":
            return isinstance(self.focused, Input) and getattr(self.focused, "id", None) == "launch-config-path"
        return True

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "launch-mode":
            self._sync_mode_ui()
            self._refresh_restart_points()
        else:
            self._refresh_footer()

    @on(Input.Submitted)
    def _input_submitted(self) -> None:
        self.action_submit()

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        if self._history_lock or event.input.id is None:
            return
        if event.input.id == "launch-run-dir":
            return
        self._remember_input_state(event.input.id, event.value)
        self._refresh_footer()

    def _watch_focused(self) -> None:
        super()._watch_focused()
        self.call_after_refresh(self._refresh_footer)
