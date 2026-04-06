"""RunsListScreen — browse all pipeline runs from a directory."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Label, Static

from src.pipeline.domain import build_run_directory
from src.pipeline.run_inspector import ROOT_GROUP
from src.pipeline.run_deletion import RunDeletionMode
from src.tui.run_deletion_flow import DeleteAction, DeleteRequest, TuiDeleteController, format_delete_completion_message
from src.tui.screens._mixins import _HelpMixin, _InterruptConfirmMixin
from src.tui.table_utils import created_timestamp_sort_key, duration_sort_seconds, plain_sort_key

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from src.pipeline.run_deletion import RunDeletionIssue, RunDeletionResult
    from src.tui.apps import RyotenkaiApp

_TUI_LOG = logging.getLogger("ryotenkai.tui.runs_list")

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

_SORT_COLS: dict[str, str] = {  # noqa: WPS407
    "status": "Status",
    "config": "Config",
    "duration": "Duration",
    "created": "Created",
}
_SORT_ICON = {"asc": " ↑", "desc": " ↓"}

_FOLDER_KEY_PREFIX = "__folder__"
_RUN_KEY_SEP = "::"
_ICON_EXPANDED = "▼"
_ICON_COLLAPSED = "▶"
_INDENT = "  "

_AUTO_COLLAPSE_THRESHOLD = 15
_MARK_HOLD_GAP = 0.35
_MARK_ACCEL_AFTER = 3.0
_MARK_ACCEL_STRIDE = 5


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

@dataclass
class _FolderNode:
    name: str
    path: str
    runs: list[dict[str, Any]] = field(default_factory=list)
    children: dict[str, _FolderNode] = field(default_factory=dict)


def _build_folder_tree(groups: dict[str, list[dict[str, Any]]]) -> _FolderNode:
    root = _FolderNode(name="", path="")
    for group_path, runs in groups.items():
        if group_path == ROOT_GROUP:
            root.runs = runs
            continue
        parts = group_path.split("/")
        node = root
        for i, part in enumerate(parts):
            if part not in node.children:
                node.children[part] = _FolderNode(name=part, path="/".join(parts[: i + 1]))
            node = node.children[part]
        node.runs = runs
    return root


def _collect_descendant_runs(node: _FolderNode) -> list[dict[str, Any]]:
    all_runs = list(node.runs)
    for child in node.children.values():
        all_runs.extend(_collect_descendant_runs(child))
    return all_runs


def _tree_newest_ts(node: _FolderNode) -> float:
    return _newest_created_ts(_collect_descendant_runs(node))


def _count_all_folders(node: _FolderNode) -> int:
    total = len(node.children)
    for child in node.children.values():
        total += _count_all_folders(child)
    return total


def _flatten_display(
    node: _FolderNode,
    depth: int,
    collapsed: set[str],
    sort_col: str | None,
    sort_dir: str | None,
    *,
    is_root: bool = False,
) -> list[dict[str, Any]]:
    """DFS flatten of tree into ordered display entries (folders-first, then runs)."""
    entries: list[dict[str, Any]] = []

    sorted_children = sorted(node.children.values(), key=lambda c: -_tree_newest_ts(c))
    for child in sorted_children:
        desc_runs = _collect_descendant_runs(child)
        newest = max((r.get("created_at", "") for r in desc_runs), default="")
        entries.append({
            "kind": "folder",
            "key": f"{_FOLDER_KEY_PREFIX}{child.path}",
            "depth": depth,
            "name": child.name,
            "path": child.path,
            "run_count": len(desc_runs),
            "newest": newest,
        })
        if child.path not in collapsed:
            entries.extend(_flatten_display(child, depth + 1, collapsed, sort_col, sort_dir))
            for run in _sort_rows(child.runs, sort_col, sort_dir):
                entries.append({"kind": "run", "key": _run_row_key(run), "depth": depth + 1, "row": run})

    if is_root:
        for run in _sort_rows(node.runs, sort_col, sort_dir):
            entries.append({"kind": "run", "key": _run_row_key(run), "depth": 0, "row": run})

    return entries


# ---------------------------------------------------------------------------
# Sort / display helpers
# ---------------------------------------------------------------------------

def _default_runs_display_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get("created_ts", 0) or 0), reverse=True)


def _default_created_sort_dir() -> str:
    return "asc"


def _next_created_sort_dir(current_dir: str | None) -> str | None:
    explicit_dir = _default_created_sort_dir()
    if current_dir is None:
        return explicit_dir
    if current_dir == explicit_dir:
        return None
    return explicit_dir


def _newest_created_ts(rows: list[dict[str, Any]]) -> float:
    return max((float(r.get("created_ts", 0) or 0) for r in rows), default=0)


def _run_row_key(row: dict[str, Any]) -> str:
    return f"{row.get('group', ROOT_GROUP)}{_RUN_KEY_SEP}{row['run_id']}"


def _run_row_key_by_dir(run_dir: Any, rows: list[dict[str, Any]]) -> str:
    """Find the row key for a given run_dir path. Returns empty string if not found."""
    for row in rows:
        if row.get("run_dir") == run_dir:
            return _run_row_key(row)
    return ""


def _sort_rows(
    rows: list[dict[str, Any]],
    sort_col: str | None,
    sort_dir: str | None,
) -> list[dict[str, Any]]:
    reverse = sort_dir == "desc"
    if sort_col is None:
        return sorted(rows, key=lambda r: float(r.get("created_ts", 0) or 0), reverse=True)
    if sort_col == "created":
        return sorted(rows, key=lambda r: r.get("created_at", ""), reverse=reverse)
    if sort_col == "duration":
        return sorted(rows, key=lambda r: _duration_seconds(r.get("duration", "")), reverse=reverse)
    return sorted(rows, key=lambda r: str(r.get(sort_col, "")), reverse=reverse)


def _duration_seconds(text: str) -> int:
    import re
    total = 0
    for amount, unit in re.findall(r"(\d+)([hms])", text or ""):
        if unit == "h":
            total += int(amount) * 3600
        elif unit == "m":
            total += int(amount) * 60
        else:
            total += int(amount)
    return total if total else -1


# ---------------------------------------------------------------------------
# Modal dialogs
# ---------------------------------------------------------------------------

class _FocusSink(Static):
    can_focus = True


class _SortModal(ModalScreen[tuple[str, str] | None]):
    """Quick sort-column picker (k9s style)."""

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
            self.dismiss(None)
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


class _DeleteConfirmModal(ModalScreen[DeleteAction]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "delete_all", "Delete folder + MLflow", show=True),
        Binding("y", "delete_local_only", "Delete folder only", show=True),
        Binding("n", "cancel", "Cancel", show=True),
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
            f"[yellow]Choose how to delete this item.[/yellow]\n\n"
            f"  [bold]Enter[/bold] — delete folder and MLflow run\n"
            f"  [bold]Y[/bold] — delete folder only\n"
            f"  [bold]N / Esc[/bold] — cancel",
        )

    def action_delete_all(self) -> None:
        self.dismiss(DeleteAction.DELETE_ALL)

    def action_delete_local_only(self) -> None:
        self.dismiss(DeleteAction.DELETE_LOCAL_ONLY)

    def action_cancel(self) -> None:
        self.dismiss(DeleteAction.CANCEL)


class _ExitConfirmModal(ModalScreen[bool]):
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


# ---------------------------------------------------------------------------
# Main screen
# ---------------------------------------------------------------------------

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
        Binding("space", "toggle_group", "Fold", show=True),
        Binding("x", "toggle_mark", "Mark", show=True),
    ]

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self._runs_dir = runs_dir
        self._rows: list[dict] = []
        self._sort_col: str | None = None
        self._sort_dir: str | None = None

        self._tree_root: _FolderNode = _FolderNode(name="", path="")
        self._display_entries: list[dict[str, Any]] = []
        self._collapsed_folders: set[str] = set()
        self._explicitly_toggled: set[str] = set()
        self._has_subfolders: bool = False
        self._marked_keys: set[str] = set()
        self._mark_hold_start: float = 0.0
        self._mark_last_press: float = 0.0
        self._delete_controller = TuiDeleteController(
            self,
            service_factory=self._run_deletion_service,
            on_pending=self._handle_delete_pending,
            on_success=self._handle_delete_success,
            on_error=self._handle_delete_error,
        )

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
        table.add_column("Run", width=38, key="name")
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
        table = self.query_one("#runs-table", DataTable)
        current_row_key: str | None = None
        if preserve_cursor:
            current_row_key = self._current_row_key()

        self._scan_and_build_tree()

        if preserve_cursor and self._try_incremental_update(table):
            self._refresh_status_bar()
            self.refresh_bindings()
            return

        self._rebuild_table(table)

        if not self._rows:
            self.query_one("#status-bar", Label).update(
                f"[dim]No runs found in {self._runs_dir}[/dim]",
            )
            return

        if current_row_key is not None:
            self._restore_cursor(current_row_key)
        self._refresh_header_labels()
        self._refresh_status_bar()
        self.refresh_bindings()

    def _scan_and_build_tree(self) -> None:
        from src.pipeline.run_inspector import scan_runs_dir_grouped

        groups = scan_runs_dir_grouped(self._runs_dir)
        self._tree_root = _build_folder_tree(groups)
        self._has_subfolders = len(self._tree_root.children) > 0

        self._apply_auto_collapse(self._tree_root)

        self._display_entries = _flatten_display(
            self._tree_root, 0, self._collapsed_folders,
            self._sort_col, self._sort_dir, is_root=True,
        )
        self._rows = [e["row"] for e in self._display_entries if e["kind"] == "run"]
        existing_keys = {e["key"] for e in self._display_entries if e["kind"] == "run"}
        self._marked_keys &= existing_keys

    def _apply_auto_collapse(self, node: _FolderNode, *, top_level: bool = True) -> None:
        """Auto-collapse folders with many runs, unless explicitly toggled or single top-level."""
        if top_level and len(node.children) == 1:
            only = next(iter(node.children.values()))
            self._collapsed_folders.discard(only.path)
            self._apply_auto_collapse(only, top_level=False)
            return

        for child in node.children.values():
            total = len(_collect_descendant_runs(child))
            if total > _AUTO_COLLAPSE_THRESHOLD and child.path not in self._explicitly_toggled:
                self._collapsed_folders.add(child.path)
            self._apply_auto_collapse(child, top_level=False)

    # ── Table rendering ──────────────────────────────────────────────────────

    def _expected_row_keys(self) -> list[str]:
        return [e["key"] for e in self._display_entries]

    def _try_incremental_update(self, table: DataTable) -> bool:
        """Update cell values in-place when the row structure hasn't changed."""
        expected = self._expected_row_keys()
        if table.row_count != len(expected):
            return False

        from textual.coordinate import Coordinate

        for i, exp_key in enumerate(expected):
            cell_key = table.coordinate_to_cell_key(Coordinate(i, 0))
            if str(cell_key.row_key.value) != exp_key:
                return False

        col_keys = ("name", "status", "config", "attempts", "duration", "created")
        for i, entry in enumerate(self._display_entries):
            cells = self._render_entry(entry)
            row_key = expected[i]
            for col, val in zip(col_keys, cells, strict=True):
                table.update_cell(row_key, col, val)
        return True

    def _rebuild_table(self, table: DataTable) -> None:
        table.clear()
        for entry in self._display_entries:
            cells = self._render_entry(entry)
            table.add_row(*cells, key=entry["key"])

    def _render_entry(self, entry: dict[str, Any]) -> tuple:
        if entry["kind"] == "folder":
            return self._render_folder(entry)
        return self._render_run(entry)

    def _render_folder(self, entry: dict[str, Any]) -> tuple:
        from rich.text import Text

        depth = entry["depth"]
        collapsed = entry["path"] in self._collapsed_folders
        icon = _ICON_COLLAPSED if collapsed else _ICON_EXPANDED
        indent = _INDENT * depth
        name_cell = Text(f"{indent}{icon} {entry['name']}", style="bold")
        count = entry["run_count"]
        count_label = f"{count} run{'s' if count != 1 else ''}"
        return (
            name_cell,
            "",
            f"[dim]{count_label}[/dim]",
            "",
            "",
            f"[dim]{entry['newest']}[/dim]",
        )

    def _render_run(self, entry: dict[str, Any]) -> tuple:
        from rich.text import Text

        from src.pipeline.run_inspector import _STATUS_ICONS

        row = entry["row"]
        depth = entry["depth"]
        status = row.get("status", "unknown")
        icon = _STATUS_ICONS.get(status, "?")
        style = _STATUS_STYLE.get(status, "")
        marked = entry["key"] in self._marked_keys
        mark = "│ " if marked else ""
        indent = _INDENT * depth
        run_style = f"{style} underline" if marked else style
        name_cell = Text(f"{mark}{indent}{icon} {row['run_id']}", style=run_style)
        created = row.get("created_at", "—")
        error_suffix = " ⚠" if row.get("error") else ""
        return (
            name_cell,
            f"[{run_style}]{status}[/{run_style}]" if run_style else status,
            row.get("config", "—"),
            str(row.get("attempts", 0)),
            row.get("duration", "—") or "—",
            created + error_suffix,
        )

    # ── Cursor / lookup ──────────────────────────────────────────────────────

    def _restore_cursor(self, row_key: str) -> None:
        table = self.query_one("#runs-table", DataTable)
        if table.row_count <= 0:
            return
        from textual.coordinate import Coordinate

        for row_index in range(table.row_count):
            cell_key = table.coordinate_to_cell_key(Coordinate(row_index, 0))
            if str(cell_key.row_key.value) == row_key:
                table.move_cursor(row=row_index, animate=False, scroll=False)
                return

    _restore_cursor_to_run_id = _restore_cursor

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

    def _run_dir_for_row_key(self, key: str) -> Path | None:
        for row in self._rows:
            if _run_row_key(row) == key:
                return row["run_dir"]
        return None

    def _current_row_key(self) -> str | None:
        table = self.query_one("#runs-table", DataTable)
        if table.row_count <= 0 or table.cursor_row < 0 or table.cursor_row >= table.row_count:
            return None
        from textual.coordinate import Coordinate

        cell_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0))
        return str(cell_key.row_key.value)

    def _current_run_dir(self) -> Path | None:
        key = self._current_row_key()
        if key is None or key.startswith(_FOLDER_KEY_PREFIX):
            return None
        return self._run_dir_for_row_key(key)

    def _current_folder_path(self) -> str | None:
        key = self._current_row_key()
        if key is not None and key.startswith(_FOLDER_KEY_PREFIX):
            return key[len(_FOLDER_KEY_PREFIX):]
        return None

    # ── UI refresh helpers ────────────────────────────────────────────────────

    def _refresh_header_labels(self) -> None:
        from rich.text import Text

        table = self.query_one("#runs-table", DataTable)
        for col_key, base_label in _SORT_COLS.items():
            indicator = _SORT_ICON.get(self._sort_dir or "", "") if self._sort_col == col_key else ""
            table.columns[col_key].label = Text(base_label + indicator)  # type: ignore[index]

    def _refresh_status_bar(self) -> None:
        total_runs = len(self._rows)
        n_folders = _count_all_folders(self._tree_root)
        n_marked = len(self._marked_keys)
        sort_hint = f" — sorted by {self._sort_col} {self._sort_dir}" if self._sort_col else ""
        folder_hint = f", {n_folders} folder{'s' if n_folders != 1 else ''}" if n_folders else ""
        mark_hint = f" — [bold yellow]{n_marked} marked[/bold yellow]" if n_marked else ""
        self.query_one("#status-bar", Label).update(
            f"[dim]{total_runs} run(s){folder_hint} — {self._runs_dir}{sort_hint}[/dim]{mark_hint}",
        )

    def _apply_sort(self, table: DataTable) -> None:
        self._rebuild_table(table)

    def _apply_current_order(self, table: DataTable) -> None:
        self._rebuild_table(table)

    # ── Events ────────────────────────────────────────────────────────────────

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        col_key = str(event.column_key)
        if col_key not in _SORT_COLS:
            return

        table = self.query_one("#runs-table", DataTable)

        if self._sort_col != col_key:
            self._sort_col = col_key
            self._sort_dir = _next_created_sort_dir(None) if col_key == "created" else "asc"
        elif col_key == "created":
            next_dir = _next_created_sort_dir(self._sort_dir)
            if next_dir is None:
                self._sort_col = None
                self._sort_dir = None
                self._load_rows()
                return
            self._sort_dir = next_dir
        elif self._sort_dir == "asc":
            self._sort_dir = "desc"
        else:
            self._sort_col = None
            self._sort_dir = None
            self._load_rows()
            return

        self._scan_and_build_tree()
        self._rebuild_table(table)
        self._refresh_header_labels()
        self._refresh_status_bar()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key.value)
        if row_key.startswith(_FOLDER_KEY_PREFIX):
            self._toggle_folder(row_key[len(_FOLDER_KEY_PREFIX):])
            return
        run_dir = self._run_dir_for_row_key(row_key)
        if run_dir is None:
            return
        self._open_run_detail(run_dir)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        del event
        self.refresh_bindings()

    # ── Folder toggle ────────────────────────────────────────────────────────

    def _toggle_folder(self, folder_path: str) -> None:
        self._explicitly_toggled.add(folder_path)
        if folder_path in self._collapsed_folders:
            self._collapsed_folders.discard(folder_path)
        else:
            self._collapsed_folders.add(folder_path)
        table = self.query_one("#runs-table", DataTable)
        cursor_key = f"{_FOLDER_KEY_PREFIX}{folder_path}"
        self._scan_and_build_tree()
        self._rebuild_table(table)
        self._restore_cursor(cursor_key)

    def action_toggle_group(self) -> None:
        folder_path = self._current_folder_path()
        if folder_path is not None:
            self._toggle_folder(folder_path)
            return
        key = self._current_row_key()
        if key is None:
            return
        for entry in self._display_entries:
            if entry["kind"] == "run" and entry["key"] == key:
                grp = entry["row"].get("group", ROOT_GROUP)
                if grp != ROOT_GROUP:
                    self._toggle_folder(grp)
                return

    # ── Mark / multi-select ────────────────────────────────────────────────────

    def action_toggle_mark(self) -> None:
        import time

        now = time.monotonic()
        if now - self._mark_last_press > _MARK_HOLD_GAP:
            self._mark_hold_start = now
        self._mark_last_press = now

        holding_for = now - self._mark_hold_start
        stride = _MARK_ACCEL_STRIDE if holding_for >= _MARK_ACCEL_AFTER else 1

        table = self.query_one("#runs-table", DataTable)
        for _ in range(stride):
            key = self._current_row_key()
            if key is None or key.startswith(_FOLDER_KEY_PREFIX):
                break
            if key in self._marked_keys:
                self._marked_keys.discard(key)
            else:
                self._marked_keys.add(key)
            self._update_run_cells(table, key)
            table.action_cursor_down()
        self._refresh_status_bar()

    def _update_run_cells(self, table: DataTable, key: str) -> None:
        """Re-render a single run row in the table (e.g. after mark toggle)."""
        col_keys = ("name", "status", "config", "attempts", "duration", "created")
        for entry in self._display_entries:
            if entry["key"] == key:
                cells = self._render_run(entry)
                for col, val in zip(col_keys, cells, strict=True):
                    table.update_cell(key, col, val)
                return

    def _marked_run_dirs(self) -> list[Path]:
        """Return run_dir for every marked run, preserving display order."""
        dirs: list[Path] = []
        for entry in self._display_entries:
            if entry["kind"] == "run" and entry["key"] in self._marked_keys:
                run_dir = entry["row"].get("run_dir")
                if run_dir is not None:
                    dirs.append(run_dir)
        return dirs

    def _run_deletion_service(self):
        from src.pipeline.run_deletion import RunDeletionService

        return RunDeletionService()

    def _start_delete(self, targets: list[Path], *, mode: RunDeletionMode) -> None:
        _TUI_LOG.info(
            "Delete requested for %s target(s) in mode=%s: %s",
            len(targets),
            mode,
            [str(target) for target in targets],
        )
        if not self._delete_controller.start_delete(targets, mode=mode):
            self.notify("Delete is already in progress", severity="warning")

    def _handle_delete_pending(self, request: DeleteRequest) -> None:
        message = f"[yellow]Deleting {len(request.targets)} item(s)...[/yellow]"
        if request.mode == RunDeletionMode.LOCAL_ONLY:
            message = f"[yellow]Deleting {len(request.targets)} folder(s) only...[/yellow]"
        self.query_one("#status-bar", Label).update(message)
        self.refresh_bindings()

    def _handle_delete_success(self, request: DeleteRequest, results: list["RunDeletionResult"]) -> None:
        for result in results:
            _TUI_LOG.info(
                "Delete finished for %s (mode=%s, local_deleted=%s, run_dirs=%s, deleted_mlflow_run_ids=%s, issues=%s)",
                result.target,
                request.mode,
                result.local_deleted,
                [str(run_dir) for run_dir in result.run_dirs],
                list(result.deleted_mlflow_run_ids),
                [self._format_delete_issue(issue) for issue in result.issues],
            )
        self.refresh_bindings()
        self._apply_delete_results(list(request.targets), results)
        if not any(result.issues for result in results):
            self.notify(format_delete_completion_message(request, results), severity="information")

    def _handle_delete_error(self, request: DeleteRequest, error: Exception) -> None:
        _TUI_LOG.exception(
            "Delete worker failed for targets=%s mode=%s",
            [str(target) for target in request.targets],
            request.mode,
            exc_info=error,
        )
        self.refresh_bindings()
        self.query_one("#status-bar", Label).update(f"[red]Delete failed: {error}[/red]")

    @staticmethod
    def _format_delete_issue(issue: "RunDeletionIssue") -> str:
        return f"{issue.run_dir.name}: {issue.phase}: {issue.message}"

    def _apply_delete_results(self, targets: list[Path], results: list["RunDeletionResult"]) -> None:
        self._marked_keys.clear()
        for target in targets:
            self._marked_keys.discard(_run_row_key_by_dir(target, self._rows))
        self._load_rows()

        issues = [issue for result in results for issue in result.issues]
        if not issues:
            return

        first_issue = self._format_delete_issue(issues[0])
        _TUI_LOG.warning(
            "Delete completed with %s issue(s); first issue: %s",
            len(issues),
            first_issue,
        )
        self.query_one("#status-bar", Label).update(
            f"[red]Delete failed for {len(issues)} item(s): {first_issue}[/red]",
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    def _refocus_table(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#runs-table", DataTable).focus()

    def action_new_run(self) -> None:
        from src.tui.launch import MODE_NEW_RUN
        from src.tui.screens.launch_modal import LaunchModal

        suggested_run_dir, _created_at = build_run_directory(base_dir=self._runs_dir)

        def _on_submit(result) -> None:
            if result is not None:
                self._ryotenkai_app().start_launch(result)
            self._refocus_table()

        self._release_table_focus()
        self.app.push_screen(
            LaunchModal(default_mode=MODE_NEW_RUN, default_run_dir=suggested_run_dir),
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
        from src.tui.launch import pick_default_launch_mode
        from src.tui.screens.launch_modal import LaunchModal

        def _on_submit(result) -> None:
            if result is not None:
                self._ryotenkai_app().start_launch(result)
            self._refocus_table()

        self._release_table_focus()
        default_mode = pick_default_launch_mode(run_dir)
        self.app.push_screen(
            LaunchModal(default_mode=default_mode, default_run_dir=run_dir),
            _on_submit,
        )

    def action_sort(self) -> None:
        def _apply(result: tuple[str, str] | None) -> None:
            if result is None:
                self._sort_col = None
                self._sort_dir = None
                self._load_rows()
                return
            col, direction = result
            self._sort_col = col
            self._sort_dir = direction
            table = self.query_one("#runs-table", DataTable)
            self._scan_and_build_tree()
            self._rebuild_table(table)
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
        marked_dirs = self._marked_run_dirs()
        if marked_dirs:
            self._confirm_delete_many(marked_dirs)
            return

        run_dir = self._current_run_dir()
        if run_dir is not None:
            self._confirm_delete_path(run_dir)
            return

        folder_path = self._current_folder_path()
        if folder_path is None:
            return
        folder_dir = self._runs_dir / folder_path
        if not folder_dir.is_dir():
            return
        self._confirm_delete_path(folder_dir)

    def _confirm_delete_many(self, targets: list[Path]) -> None:
        label = f"{len(targets)} marked run{'s' if len(targets) != 1 else ''}"

        def _on_confirm(action: DeleteAction | None) -> None:
            if action is None or action == DeleteAction.CANCEL:
                return
            mode = action.to_mode()
            if mode is None:
                return
            self._start_delete(targets, mode=mode)

        self.app.push_screen(_DeleteConfirmModal(label), _on_confirm)

    def _confirm_delete_path(self, target: "Path") -> None:
        def _on_confirm(action: DeleteAction | None) -> None:
            if action is None or action == DeleteAction.CANCEL:
                return
            mode = action.to_mode()
            if mode is None:
                return
            self._start_delete([target], mode=mode)

        self.app.push_screen(_DeleteConfirmModal(target.name), _on_confirm)

    def on_worker_state_changed(self, event) -> None:
        self._delete_controller.handle_worker_state_changed(event)

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
        if action == "delete_run":
            if self._delete_controller.is_busy:
                return False
            if self._current_run_dir() is not None:
                return True
            return self._current_folder_path() is not None
        if action == "toggle_group":
            return self._has_subfolders
        return True
