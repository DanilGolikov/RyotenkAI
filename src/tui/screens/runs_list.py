"""RunsListScreen — browse all pipeline runs from a directory."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Label, Static

from src.pipeline.domain import build_run_directory
from src.tui.screens._mixins import _HelpMixin, _InterruptConfirmMixin
from src.tui.table_utils import created_timestamp_sort_key, duration_sort_seconds, plain_sort_key

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult

    from src.tui.apps import RyotenkaiApp

# Bold-variant styles for the runs table; icons come from run_inspector._STATUS_ICONS.
_STATUS_STYLE: dict[str, str] = {  # noqa: WPS407
    "completed": "bold green",
    "failed": "bold red",
    "running": "bold cyan",
    "interrupted": "bold yellow",
    "stale": "dim",
    "skipped": "blue",
    "pending": "dim",
    "unknown": "dim red",
}

# Columns available for sorting: column key → default label.
_SORT_COLS: dict[str, str] = {  # noqa: WPS407
    "status": "Status",
    "config": "Config",
    "duration": "Duration",
    "created": "Created",
}
_SORT_ICON = {"asc": " ↑", "desc": " ↓"}


def _default_runs_display_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Default runs order is newest first by creation timestamp."""
    return sorted(rows, key=lambda row: float(row.get("created_ts", 0) or 0), reverse=True)


def _default_created_sort_dir() -> str:
    """First explicit sort on Created should invert the default order."""
    return "asc"


def _next_created_sort_dir(current_dir: str | None) -> str | None:
    """Cycle Created sorting between default and the opposite order only."""
    explicit_dir = _default_created_sort_dir()
    if current_dir is None:
        return explicit_dir
    if current_dir == explicit_dir:
        return None
    return explicit_dir


class _FocusSink(Static):
    can_focus = True


class _SortModal(ModalScreen[tuple[str, str] | None]):
    """Quick sort-column picker (k9s style).

    Returns (col_key, direction) on selection, or None to reset.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("1", "pick_status", "Status", show=False),
        Binding("2", "pick_config", "Config", show=False),
        Binding("3", "pick_duration", "Duration", show=False),
        Binding("4", "pick_created", "Created", show=False),
        Binding("0", "reset", "Reset", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    _SortModal {
        align: center middle;
    }
    _SortModal > Static {
        width: 44;
        padding: 1 3;
        border: solid $accent;
        background: $surface;
    }
    """

    _OPTIONS: ClassVar[list[tuple[str, str, str]]] = [
        ("1", "status", "Status"),
        ("2", "config", "Config"),
        ("3", "duration", "Duration"),
        ("4", "created", "Created"),
    ]

    def __init__(self, sort_col: str | None, sort_dir: str | None) -> None:
        super().__init__()
        self._sort_col = sort_col
        self._sort_dir = sort_dir

    def compose(self) -> ComposeResult:
        lines = ["[bold]Sort by column:[/bold]\n"]
        for key, col, label in self._OPTIONS:
            indicator = _SORT_ICON.get(self._sort_dir or "", "") if self._sort_col == col else ""
            lines.append(f"  \\[[bold]{key}[/bold]] {label}{indicator}")
        lines.append("\n  \\[[bold]0[/bold]] Reset sort    \\[[bold]Esc[/bold]] Cancel")
        yield Static("\n".join(lines))

    def _pick(self, col: str) -> None:
        if self._sort_col is None and col == "created":
            self.dismiss((col, _next_created_sort_dir(None)))  # type: ignore[arg-type]
        elif self._sort_col == col and col == "created":
            next_dir = _next_created_sort_dir(self._sort_dir)
            self.dismiss(None if next_dir is None else (col, next_dir))
        elif self._sort_col == col and self._sort_dir == "asc":
            self.dismiss((col, "desc"))
        elif self._sort_col == col and self._sort_dir == "desc":
            self.dismiss(None)  # third press → reset
        else:
            self.dismiss((col, "asc"))

    def action_pick_status(self) -> None:
        self._pick("status")

    def action_pick_config(self) -> None:
        self._pick("config")

    def action_pick_duration(self) -> None:
        self._pick("duration")

    def action_pick_created(self) -> None:
        self._pick("created")

    def action_reset(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None if self._sort_col is None else (self._sort_col, self._sort_dir))  # type: ignore[arg-type]


class _DeleteConfirmModal(ModalScreen[bool]):
    """Confirmation dialog before deleting a run directory."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "confirm", "Yes — delete", show=True),
        Binding("n", "cancel", "No — cancel", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    _DeleteConfirmModal {
        align: center middle;
    }
    _DeleteConfirmModal > Static {
        width: 60;
        padding: 2 4;
        border: solid $error;
        background: $surface;
    }
    """

    def __init__(self, run_name: str) -> None:
        super().__init__()
        self._run_name = run_name

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold red]Delete run?[/bold red]\n\n"
            f"[dim]{self._run_name}[/dim]\n\n"
            f"[yellow]This will permanently remove all files.[/yellow]\n\n"
            f"  [bold]Y[/bold] — confirm delete    [bold]N / Esc[/bold] — cancel",
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class _ExitConfirmModal(ModalScreen[bool]):
    """Confirmation dialog before exiting the TUI from the main screen."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "confirm", "Exit", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    DEFAULT_CSS = """
    _ExitConfirmModal {
        align: center middle;
    }
    _ExitConfirmModal > Static {
        width: 56;
        height: auto;
        padding: 1 2;
        border: solid $warning;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]Exit ryotenkai?[/bold]\n\n"
            "[dim]Press Enter to quit the TUI.[/dim]\n"
            "[dim]Press Esc to stay in the main menu.[/dim]",
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class RunsListScreen(_HelpMixin, _InterruptConfirmMixin, Screen):
    """Fullscreen browsable table of all pipeline runs."""

    DEFAULT_CSS = """
    RunsListScreen #runs-focus-sink {
        height: 0;
        width: 0;
        margin: 0;
        padding: 0;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape,q", "quit", "Quit", show=True),
        Binding("question_mark", "show_help", "Help", key_display="?", show=True),
        Binding("l", "launch_selected", "Launch", show=True),
        Binding("ctrl+e", "stop_selected_run", "Stop", key_display="⌃e", show=True, priority=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("s", "sort", "Sort", show=True),
        Binding("d", "delete_run", "Delete", show=True),
    ]

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self._runs_dir = runs_dir
        self._rows: list[dict] = []
        self._sort_col: str | None = None
        self._sort_dir: str | None = None  # "asc" | "desc" | None

    def _ryotenkai_app(self) -> RyotenkaiApp:
        from typing import cast

        return cast("RyotenkaiApp", self.app)

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield _FocusSink("", id="runs-focus-sink")
        yield DataTable(id="runs-table", cursor_type="row")
        yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_column("", width=3, key="icon")
        table.add_column("Run ID", width=32, key="run_id")
        table.add_column("Status", width=13, key="status")
        table.add_column("Config", width=28, key="config")
        table.add_column("Att", width=4, key="attempts")
        table.add_column("Duration", width=10, key="duration")
        table.add_column("Created", width=17, key="created")
        self._load_rows()
        self.set_interval(1.0, self._auto_refresh_rows)
        self.call_after_refresh(table.focus)

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _load_rows(self, *, preserve_cursor: bool = False) -> None:
        """Reload run list from disk, update DataTable rows, re-apply active sort."""

        table = self.query_one("#runs-table", DataTable)
        current_run_id: str | None = None
        if preserve_cursor:
            current_run_dir = self._current_run_dir()
            current_run_id = current_run_dir.name if current_run_dir is not None else None
        previous_rows = self._rows
        next_rows = self._scan_rows()
        self._sync_table_rows(table, previous_rows, next_rows)
        self._rows = next_rows

        if not self._rows:
            self.query_one("#status-bar", Label).update(
                f"[dim]No runs found in {self._runs_dir}[/dim]",
            )
            return

        self._apply_current_order(table)

        if current_run_id is not None:
            self._restore_cursor_to_run_id(current_run_id)
        self._refresh_header_labels()
        self._refresh_status_bar()
        self.refresh_bindings()

    def _scan_rows(self) -> list[dict[str, Any]]:
        from src.pipeline.run_inspector import scan_runs_dir

        return _default_runs_display_order(scan_runs_dir(self._runs_dir))

    def _render_row_cells(self, row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
        from src.pipeline.run_inspector import _STATUS_ICONS

        status = row.get("status", "unknown")
        icon = _STATUS_ICONS.get(status, "?")
        style = _STATUS_STYLE.get(status, "")
        created = row.get("created_at", "—")
        error_suffix = " ⚠" if row.get("error") else ""
        return (
            icon,
            f"[{style}]{row['run_id']}[/{style}]" if style else row["run_id"],
            f"[{style}]{status}[/{style}]" if style else status,
            row.get("config", "—"),
            str(row.get("attempts", 0)),
            row.get("duration", "—") or "—",
            created + error_suffix,
        )

    def _sync_table_rows(
        self,
        table: DataTable,
        previous_rows: list[dict[str, Any]],
        next_rows: list[dict[str, Any]],
    ) -> None:
        existing_run_ids = {existing["run_id"] for existing in previous_rows}
        next_run_ids = {row["run_id"] for row in next_rows}

        for run_id in existing_run_ids - next_run_ids:
            with contextlib.suppress(Exception):
                table.remove_row(run_id)

        for row in next_rows:
            rendered_cells = self._render_row_cells(row)
            if row["run_id"] not in existing_run_ids:
                table.add_row(*rendered_cells, key=row["run_id"])
                continue
            for column_key, value in zip(
                ("icon", "run_id", "status", "config", "attempts", "duration", "created"),
                rendered_cells,
                strict=True,
            ):
                table.update_cell(row["run_id"], column_key, value)

    def _restore_cursor_to_run_id(self, run_id: str) -> None:
        table = self.query_one("#runs-table", DataTable)
        if table.row_count <= 0:
            return
        from textual.coordinate import Coordinate

        for row_index in range(table.row_count):
            cell_key = table.coordinate_to_cell_key(Coordinate(row_index, 0))
            if str(cell_key.row_key.value) == run_id:
                table.move_cursor(row=row_index, animate=False, scroll=False)
                return

    def _auto_refresh_rows(self) -> None:
        self._load_rows(preserve_cursor=True)

    def _release_table_focus(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#runs-focus-sink", _FocusSink).focus()

    def _selected_run_can_be_interrupted(self) -> bool:
        run_dir = self._current_run_dir()
        if run_dir is None:
            return False
        return self._ryotenkai_app().can_interrupt_run(run_dir)

    def _run_dir_for_row_key(self, run_id: str) -> Path | None:
        """Look up run_dir by run_id — stable even after DataTable.sort() reorders rows."""
        for row in self._rows:
            if row["run_id"] == run_id:
                return row["run_dir"]
        return None

    def _current_run_dir(self) -> Path | None:
        """Return run_dir for the highlighted row, resolved via row key (sort-safe)."""
        table = self.query_one("#runs-table", DataTable)
        if not self._rows or table.cursor_row < 0 or table.cursor_row >= table.row_count:
            return None
        from textual.coordinate import Coordinate

        cell_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0))
        return self._run_dir_for_row_key(str(cell_key.row_key.value))

    # ── UI refresh helpers ────────────────────────────────────────────────────

    def _refresh_header_labels(self) -> None:
        """Update column header labels: add ↑/↓ to active sort column, clear others."""
        from rich.text import Text

        table = self.query_one("#runs-table", DataTable)
        for col_key, base_label in _SORT_COLS.items():
            indicator = _SORT_ICON.get(self._sort_dir or "", "") if self._sort_col == col_key else ""
            table.columns[col_key].label = Text(base_label + indicator)  # type: ignore[index]

    def _refresh_status_bar(self) -> None:
        sort_hint = f" — sorted by {self._sort_col} {self._sort_dir}" if self._sort_col else ""
        self.query_one("#status-bar", Label).update(
            f"[dim]{len(self._rows)} run(s) — {self._runs_dir}{sort_hint}[/dim]",
        )

    def _apply_sort(self, table: DataTable) -> None:
        if self._sort_col is None:
            return
        if self._sort_col == "duration":
            table.sort("duration", key=duration_sort_seconds, reverse=(self._sort_dir == "desc"))
            return
        if self._sort_col == "created":
            table.sort("created", key=created_timestamp_sort_key, reverse=(self._sort_dir == "desc"))
            return
        table.sort(self._sort_col, key=plain_sort_key, reverse=(self._sort_dir == "desc"))

    def _apply_current_order(self, table: DataTable) -> None:
        if self._sort_col is None:
            table.sort("created", key=created_timestamp_sort_key, reverse=True)
            return
        self._apply_sort(table)

    # ── Events ────────────────────────────────────────────────────────────────

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Cycle sort state on header click."""
        col_key = str(event.column_key)
        if col_key not in _SORT_COLS:
            return

        table = self.query_one("#runs-table", DataTable)

        if self._sort_col != col_key:
            self._sort_col = col_key
            self._sort_dir = _next_created_sort_dir(None) if col_key == "created" else "asc"
            self._apply_sort(table)
        elif col_key == "created":
            next_dir = _next_created_sort_dir(self._sort_dir)
            if next_dir is None:
                self._sort_col = None
                self._sort_dir = None
                self._load_rows()
                return
            self._sort_dir = next_dir
            self._apply_sort(table)
        elif self._sort_dir == "asc":
            self._sort_dir = "desc"
            self._apply_sort(table)
        else:
            self._sort_col = None
            self._sort_dir = None
            self._load_rows()
            return

        self._refresh_header_labels()
        self._refresh_status_bar()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        run_dir = self._run_dir_for_row_key(str(event.row_key.value))
        if run_dir is None:
            return
        self._open_run_detail(run_dir)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        del event
        self.refresh_bindings()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_new_run(self) -> None:
        from src.tui.screens.launch_modal import LaunchModal

        suggested_run_dir, _created_at = build_run_directory(base_dir=self._runs_dir)

        def _on_submit(result) -> None:
            if result is not None:
                self._ryotenkai_app().start_launch(result)

        self._release_table_focus()
        self.app.push_screen(
            LaunchModal(default_mode="new_run", default_run_dir=suggested_run_dir),
            _on_submit,
        )

    def _open_run_detail(self, run_dir: Path) -> None:
        from src.tui.screens.run_detail import RunDetailScreen

        self.app.push_screen(RunDetailScreen(run_dir))

    def action_open_selected_run(self) -> None:
        run_dir = self._current_run_dir()
        if run_dir is None:
            return
        self._open_run_detail(run_dir)

    def action_launch_selected(self) -> None:
        run_dir = self._current_run_dir()
        if run_dir is None:
            if not self._rows:
                self.action_new_run()
            return
        from src.tui.screens.launch_modal import LaunchModal

        def _on_submit(result) -> None:
            if result is not None:
                self._ryotenkai_app().start_launch(result)

        self._release_table_focus()
        self.app.push_screen(
            LaunchModal(default_mode="restart", default_run_dir=run_dir),
            _on_submit,
        )

    def action_sort(self) -> None:
        def _apply(result: tuple[str, str] | None) -> None:
            if result is None:
                # Reset
                self._sort_col = None
                self._sort_dir = None
                self._load_rows()
                return
            col, direction = result
            table = self.query_one("#runs-table", DataTable)
            self._sort_col = col
            self._sort_dir = direction
            self._apply_sort(table)
            self._refresh_header_labels()
            self._refresh_status_bar()

        self.app.push_screen(_SortModal(self._sort_col, self._sort_dir), _apply)

    def action_refresh(self) -> None:
        self._load_rows()

    def action_monitor_run(self) -> None:
        run_dir = self._current_run_dir()
        if run_dir is None:
            return
        self._ryotenkai_app().open_running_attempt_for_run(run_dir)

    def action_stop_selected_run(self) -> None:
        run_dir = self._current_run_dir()
        if run_dir is None:
            return
        self._confirm_interrupt_run(run_dir)

    def action_delete_run(self) -> None:
        run_dir = self._current_run_dir()
        if run_dir is None:
            return

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            import shutil

            try:
                shutil.rmtree(run_dir)
            except Exception as exc:
                self.query_one("#status-bar", Label).update(f"[red]Delete failed: {exc}[/red]")
                return
            self._load_rows()

        self.app.push_screen(_DeleteConfirmModal(run_dir.name), _on_confirm)

    def action_quit(self) -> None:
        def _on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self.app.exit()

        self.app.push_screen(_ExitConfirmModal(), _on_confirm)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "open_selected_run":
            return self._current_run_dir() is not None
        if action == "monitor_run":
            run_dir = self._current_run_dir()
            return run_dir is not None and self._ryotenkai_app().can_open_running_attempt_for_run(run_dir)
        if action == "stop_selected_run":
            return self._selected_run_can_be_interrupted()
        return True
