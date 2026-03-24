"""RunDetailScreen — pipeline overview + attempts navigation."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Label, Static, TabbedContent, TabPane, Tabs

from src.tui.screens._mixins import _HelpMixin, _InterruptConfirmMixin, _TabbedScreenMixin
from src.tui.table_utils import duration_sort_seconds, integer_sort_key, plain_sort_key

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from src.pipeline.state import PipelineState
    from src.tui.apps import RyotenkaiApp

_TIMESTAMP_LEN = 16
_HASH_SHORT = 16
_ATTEMPT_SORT_COLS: dict[str, str] = {
    "no": "#",
    "status": "Status",
    "action": "Action",
    "duration": "Duration",
    "started": "Started",
}
_SORT_ICON = {"asc": " ↑", "desc": " ↓"}


def _resolve_run_config_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()


def _default_attempt_display_order(attempts: list[Any]) -> list[Any]:
    """Default attempts order is always newest first."""
    return list(reversed(attempts))


def _default_attempt_no_sort_dir() -> str:
    """First explicit sort by attempt number should invert the default order."""
    return "asc"


def _next_attempt_no_sort_dir(current_dir: str | None) -> str | None:
    """Cycle attempt-number sorting between default and the opposite order only."""
    explicit_dir = _default_attempt_no_sort_dir()
    if current_dir is None:
        return explicit_dir
    if current_dir == explicit_dir:
        return None
    return explicit_dir


class _AttemptSortModal(ModalScreen[tuple[str, str] | None]):
    """Quick sort-column picker for attempts table."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("1", "pick_no", "#", show=False),
        Binding("2", "pick_status", "Status", show=False),
        Binding("3", "pick_action", "Action", show=False),
        Binding("4", "pick_duration", "Duration", show=False),
        Binding("5", "pick_started", "Started", show=False),
        Binding("0", "reset", "Reset", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    _AttemptSortModal {
        align: center middle;
    }
    _AttemptSortModal > Static {
        width: 44;
        padding: 1 3;
        border: solid $accent;
        background: $surface;
    }
    """

    _OPTIONS: ClassVar[list[tuple[str, str, str]]] = [
        ("1", "no", "#"),
        ("2", "status", "Status"),
        ("3", "action", "Action"),
        ("4", "duration", "Duration"),
        ("5", "started", "Started"),
    ]

    def __init__(self, sort_col: str | None, sort_dir: str | None) -> None:
        super().__init__()
        self._sort_col = sort_col
        self._sort_dir = sort_dir

    def compose(self) -> ComposeResult:
        lines = ["[bold]Sort attempts by:[/bold]\n"]
        for key, col, label in self._OPTIONS:
            indicator = _SORT_ICON.get(self._sort_dir or "", "") if self._sort_col == col else ""
            lines.append(f"  \\[[bold]{key}[/bold]] {label}{indicator}")
        lines.append("\n  \\[[bold]0[/bold]] Reset sort    \\[[bold]Esc[/bold]] Cancel")
        yield Static("\n".join(lines))

    def _pick(self, col: str) -> None:
        if self._sort_col is None and col == "no":
            self.dismiss((col, _next_attempt_no_sort_dir(None)))  # type: ignore[arg-type]
        elif self._sort_col == col and col == "no":
            next_dir = _next_attempt_no_sort_dir(self._sort_dir)
            self.dismiss(None if next_dir is None else (col, next_dir))
        elif self._sort_col == col and self._sort_dir == "asc":
            self.dismiss((col, "desc"))
        elif self._sort_col == col and self._sort_dir == "desc":
            self.dismiss(None)
        else:
            self.dismiss((col, "asc"))

    def action_pick_no(self) -> None:
        self._pick("no")

    def action_pick_status(self) -> None:
        self._pick("status")

    def action_pick_action(self) -> None:
        self._pick("action")

    def action_pick_duration(self) -> None:
        self._pick("duration")

    def action_pick_started(self) -> None:
        self._pick("started")

    def action_reset(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None if self._sort_col is None else (self._sort_col, self._sort_dir))  # type: ignore[arg-type]


class _AttemptsTable(DataTable):
    """DataTable that returns focus to the nearest Tabs bar when Up is pressed at row 0.

    Guard: if the widget is hidden (inactive tab pane), skip interception entirely —
    Textual may still route key events to a focused-but-hidden widget, which would
    cause the invisible DataTable to steal focus back to the Tabs and inadvertently
    make the Overview tab re-activate.
    """

    def action_cursor_up(self) -> None:
        if not self.display:
            return
        if self.cursor_row == 0 and self.row_count > 0:
            with contextlib.suppress(Exception):
                self.screen.query_one(Tabs).focus()
        else:
            super().action_cursor_up()


class RunDetailScreen(_HelpMixin, _InterruptConfirmMixin, _TabbedScreenMixin, Screen):
    """Shows pipeline Overview (info + attempts) / Diff tabs for a single run."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape,q", "go_back", "Back", show=True),
        Binding("question_mark", "show_help", "Help", key_display="?", show=True),
        Binding("c", "browse_config", "Config", show=True, priority=True),
        Binding("l", "relaunch", "Launch", show=True),
        Binding("m", "monitor", "Current", show=True),
        Binding("s", "sort_attempts", "Sort", show=True),
        Binding("ctrl+e", "stop_run", "Stop", key_display="⌃e", show=True, priority=True),
        Binding("1", "show_tab('overview')", "Overview", show=False),
        Binding("2", "show_tab('diff')", "Diff", show=False),
    ]

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self._run_dir = run_dir
        self._config_path: Path | None = None
        self._sort_col: str | None = None
        self._sort_dir: str | None = None

    def _ryotenkai_app(self) -> RyotenkaiApp:
        from typing import cast

        return cast("RyotenkaiApp", self.app)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="overview"):
            with TabPane("Overview [1]", id="overview"):
                yield Static(id="overview-content")
                yield Label("[dim]─── Attempts ─────────────────────────────────[/dim]", id="attempts-sep")
                yield _AttemptsTable(id="attempts-table", cursor_type="row")
            with TabPane("Diff [2]", id="diff"):
                yield Static(id="diff-content", expand=True)
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = self._run_dir.name
        self._setup_attempts_columns()
        self._refresh_run_detail()
        self.set_interval(1.0, self._auto_refresh_overview)
        self._focus_tabs_after_mount()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _setup_attempts_columns(self) -> None:
        table = self.query_one("#attempts-table", _AttemptsTable)
        table.add_column("", width=3, key="icon")
        table.add_column("#", width=4, key="no")
        table.add_column("Status", width=14, key="status")
        table.add_column("Action", width=22, key="action")
        table.add_column("Duration", width=10, key="duration")
        table.add_column("Started", width=20, key="started")

    # ── Data population ────────────────────────────────────────────────────────

    def _populate_overview(self, state: PipelineState) -> None:
        from src.pipeline.run_inspector import _STATUS_COLORS, _STATUS_ICONS, _fmt_duration, effective_pipeline_status

        pipeline_status = effective_pipeline_status(state)
        color = _STATUS_COLORS.get(pipeline_status, "white")
        icon = _STATUS_ICONS.get(pipeline_status, "?")
        pipeline_dur = _fmt_duration(
            state.attempts[0].started_at if state.attempts else None,
            state.attempts[-1].completed_at if state.attempts else None,
        )
        config_name = state.config_path.split("/")[-1] if state.config_path else "—"
        mlflow_id = (state.root_mlflow_run_id or "—")[:20]

        self.query_one("#overview-content", Static).update(
            "\n".join(
                [
                    f"[bold cyan]{self._run_dir.name}[/bold cyan]",
                    f"  Status   : [{color}]{icon} {pipeline_status.upper()}[/{color}]",
                    f"  Config   : [dim]{config_name}[/dim]",
                    f"  MLflow   : [dim]{mlflow_id}[/dim]",
                    f"  Duration : [dim]{pipeline_dur or '—'}[/dim]",
                    f"  Attempts : [dim]{len(state.attempts)}[/dim]",
                ]
            ),
        )

    def _populate_attempts(self, state: PipelineState) -> None:
        from src.pipeline.run_inspector import _STATUS_COLORS, _STATUS_ICONS, _fmt_duration

        table = self.query_one("#attempts-table", _AttemptsTable)
        table.clear()

        for attempt in _default_attempt_display_order(state.attempts):
            att_color = _STATUS_COLORS.get(attempt.status, "white")
            icon = _STATUS_ICONS.get(attempt.status, "?")
            dur = _fmt_duration(attempt.started_at, attempt.completed_at)
            started = (attempt.started_at or "")[:_TIMESTAMP_LEN].replace("T", " ")
            action = attempt.restart_from_stage or attempt.effective_action or "fresh"

            table.add_row(
                icon,
                str(attempt.attempt_no),
                f"[{att_color}]{attempt.status}[/{att_color}]",
                f"[dim]{action}[/dim]",
                dur or "—",
                started,
                key=str(attempt.attempt_no),
            )

        if self._sort_col is not None:
            self._apply_attempt_sort(table)
        self._refresh_attempt_header_labels()

    def _refresh_run_detail(self, *, preserve_cursor: bool = False) -> None:
        from src.pipeline.run_inspector import RunInspector

        selected_attempt_no: int | None = None
        if preserve_cursor:
            selected_attempt_no = self._current_attempt_no()

        try:
            data = RunInspector(self._run_dir).load(include_logs=False)
        except Exception as exc:
            self._config_path = None
            self.query_one("#overview-content", Static).update(
                f"[bold red]Cannot load state:[/bold red] {exc}",
            )
            return

        self._config_path = _resolve_run_config_path(data.state.config_path)
        self._populate_overview(data.state)
        self._populate_attempts(data.state)
        self._populate_diff(data.state)
        self.refresh_bindings()

        if selected_attempt_no is not None:
            self._restore_attempt_cursor(selected_attempt_no)

    def _restore_attempt_cursor(self, attempt_no: int) -> None:
        table = self.query_one("#attempts-table", _AttemptsTable)
        if table.row_count <= 0:
            return
        from textual.coordinate import Coordinate

        for row_index in range(table.row_count):
            cell_key = table.coordinate_to_cell_key(Coordinate(row_index, 0))
            if str(cell_key.row_key.value) == str(attempt_no):
                table.move_cursor(row=row_index, animate=False)
                return

    def _current_attempt_no(self) -> int | None:
        table = self.query_one("#attempts-table", _AttemptsTable)
        if table.row_count <= 0 or table.cursor_row < 0 or table.cursor_row >= table.row_count:
            return None
        from textual.coordinate import Coordinate

        cell_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0))
        try:
            return int(str(cell_key.row_key.value))
        except ValueError:
            return None

    def _refresh_attempt_header_labels(self) -> None:
        from rich.text import Text

        table = self.query_one("#attempts-table", _AttemptsTable)
        for col_key, base_label in _ATTEMPT_SORT_COLS.items():
            indicator = _SORT_ICON.get(self._sort_dir or "", "") if self._sort_col == col_key else ""
            table.columns[col_key].label = Text(base_label + indicator)  # type: ignore[index]

    def _apply_attempt_sort(self, table: _AttemptsTable) -> None:
        if self._sort_col is None:
            return
        if self._sort_col == "no":
            table.sort("no", key=integer_sort_key, reverse=(self._sort_dir == "desc"))
            return
        if self._sort_col == "duration":
            table.sort("duration", key=duration_sort_seconds, reverse=(self._sort_dir == "desc"))
            return
        table.sort(self._sort_col, key=plain_sort_key, reverse=(self._sort_dir == "desc"))

    def _auto_refresh_overview(self) -> None:
        tabs = self.query_one(TabbedContent)
        if tabs.active != "overview":
            return
        self._refresh_run_detail(preserve_cursor=True)

    def _populate_diff(self, state: PipelineState) -> None:
        from src.pipeline.run_inspector import diff_attempts

        widget = self.query_one("#diff-content", Static)

        if len(state.attempts) < 2:  # noqa: WPS432
            widget.update("[dim]Only one attempt — nothing to diff.[/dim]")
            return

        attempt_nos = sorted(a.attempt_no for a in state.attempts)
        first, last = attempt_nos[0], attempt_nos[-1]
        result = diff_attempts(state, first, last)

        lines: list[str] = [f"[bold cyan]Config diff: attempt {first} → {last}[/bold cyan]", ""]

        if result["training_critical_changed"]:
            lines.append("[bold red]Training-critical config CHANGED[/bold red]")
            lines.append(f"  Attempt {first}: [dim]{result['hash_a_critical'][:_HASH_SHORT]}...[/dim]")
            lines.append(f"  Attempt {last}: [dim]{result['hash_b_critical'][:_HASH_SHORT]}...[/dim]")
        else:
            lines.append("[green]Training-critical config unchanged[/green]")

        lines.append("")

        if result["late_stage_changed"]:
            lines.append("[bold yellow]Late-stage config CHANGED[/bold yellow]")
            lines.append(f"  Attempt {first}: [dim]{result['hash_a_late'][:_HASH_SHORT]}...[/dim]")
            lines.append(f"  Attempt {last}: [dim]{result['hash_b_late'][:_HASH_SHORT]}...[/dim]")
        else:
            lines.append("[green]Late-stage config unchanged[/green]")

        if len(attempt_nos) > 2:  # noqa: WPS432
            lines.append("")
            lines.append(f"[dim]All attempts: {', '.join(str(n) for n in attempt_nos)}[/dim]")

        widget.update("\n".join(lines))

    # ── Navigation ────────────────────────────────────────────────────────────

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        if event.data_table.id != "attempts-table":
            return
        col_key = str(event.column_key)
        if col_key not in _ATTEMPT_SORT_COLS:
            return

        table = self.query_one("#attempts-table", _AttemptsTable)
        if self._sort_col != col_key:
            self._sort_col = col_key
            self._sort_dir = _next_attempt_no_sort_dir(None) if col_key == "no" else "asc"
            self._apply_attempt_sort(table)
        elif col_key == "no":
            next_dir = _next_attempt_no_sort_dir(self._sort_dir)
            if next_dir is None:
                self._sort_col = None
                self._sort_dir = None
                self._refresh_run_detail(preserve_cursor=True)
                return
            self._sort_dir = next_dir
            self._apply_attempt_sort(table)
        elif self._sort_dir == "asc":
            self._sort_dir = "desc"
            self._apply_attempt_sort(table)
        else:
            self._sort_col = None
            self._sort_dir = None
            self._refresh_run_detail(preserve_cursor=True)
            return
        self._refresh_attempt_header_labels()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "attempts-table":
            return
        try:
            attempt_no = int(str(event.row_key.value))
        except ValueError:
            return
        from src.tui.screens.attempt_detail import AttemptDetailScreen

        self.app.push_screen(AttemptDetailScreen(self._run_dir, attempt_no))

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_go_back(self) -> None:
        self.app.sub_title = ""
        self.app.pop_screen()

    def action_monitor(self) -> None:
        self._ryotenkai_app().open_running_attempt_for_run(self._run_dir)

    def action_relaunch(self) -> None:
        from src.tui.screens.launch_modal import LaunchModal

        def _on_submit(result) -> None:
            if result is not None:
                self._ryotenkai_app().start_launch(result)

        self.app.push_screen(
            LaunchModal(default_mode="restart", default_run_dir=self._run_dir),
            _on_submit,
        )

    def action_browse_config(self) -> None:
        from src.tui.screens.config_browser_modal import StructuredConfigBrowser

        if self._config_path is None:
            self.notify("Config path is unavailable for this run", severity="warning")
            return
        self.app.push_screen(StructuredConfigBrowser(self._config_path))

    def action_sort_attempts(self) -> None:
        def _apply(result: tuple[str, str] | None) -> None:
            if result is None:
                self._sort_col = None
                self._sort_dir = None
                self._refresh_run_detail(preserve_cursor=True)
                return
            col, direction = result
            table = self.query_one("#attempts-table", _AttemptsTable)
            self._sort_col = col
            self._sort_dir = direction
            self._apply_attempt_sort(table)
            self._refresh_attempt_header_labels()

        self.app.push_screen(_AttemptSortModal(self._sort_col, self._sort_dir), _apply)

    def action_stop_run(self) -> None:
        self._confirm_interrupt_run(self._run_dir)

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "stop_run":
            return self._ryotenkai_app().can_interrupt_run(self._run_dir)
        if action == "monitor":
            return self._ryotenkai_app().can_open_running_attempt_for_run(self._run_dir)
        if action == "browse_config":
            return self._config_path is not None
        return True
