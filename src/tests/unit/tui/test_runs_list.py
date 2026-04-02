import pytest
from types import SimpleNamespace

from src.tui.screens.runs_list import (
    RunsListScreen,
    _ExitConfirmModal,
    _default_created_sort_dir,
    _default_runs_display_order,
    _next_created_sort_dir,
)


def test_default_runs_order_is_newest_first_by_created_timestamp() -> None:
    rows = [
        {"run_id": "run_old", "created_ts": 100.0},
        {"run_id": "run_new", "created_ts": 300.0},
        {"run_id": "run_mid", "created_ts": 200.0},
    ]

    ordered = _default_runs_display_order(rows)

    assert [row["run_id"] for row in ordered] == ["run_new", "run_mid", "run_old"]


def test_default_created_sort_direction_is_ascending() -> None:
    assert _default_created_sort_dir() == "asc"


def test_created_sort_cycle_is_default_asc_default() -> None:
    assert _next_created_sort_dir(None) == "asc"
    assert _next_created_sort_dir("asc") is None


def test_runs_list_hides_redundant_open_and_current_bindings() -> None:
    binding_actions = {binding.action for binding in RunsListScreen.BINDINGS}

    assert "open_selected_run" not in binding_actions
    assert "monitor_run" not in binding_actions


def test_exit_confirm_modal_actions_dismiss_expected_values(monkeypatch) -> None:
    modal = _ExitConfirmModal()
    dismissed: list[bool] = []
    monkeypatch.setattr(modal, "dismiss", lambda value: dismissed.append(value))

    modal.action_cancel()
    modal.action_confirm()

    assert dismissed == [False, True]


def test_runs_list_quit_opens_exit_confirmation(monkeypatch, tmp_path) -> None:
    pytest.importorskip("textual")
    from src.tui.screens.runs_list import RunsListScreen

    screen = RunsListScreen(tmp_path)
    pushed: list[object] = []
    exit_calls: list[bool] = []

    class DummyApp:
        def push_screen(self, screen_obj, callback):
            pushed.append(screen_obj)
            callback(False)

        def exit(self):
            exit_calls.append(True)

    monkeypatch.setattr(RunsListScreen, "app", property(lambda self: DummyApp()))

    screen.action_quit()

    assert len(pushed) == 1
    assert isinstance(pushed[0], _ExitConfirmModal)
    assert exit_calls == []


def test_runs_list_launch_selected_opens_new_run_when_no_runs(monkeypatch, tmp_path) -> None:
    pytest.importorskip("textual")
    from src.tui.screens.launch_modal import LaunchModal
    from src.tui.screens.runs_list import RunsListScreen

    screen = RunsListScreen(tmp_path)
    pushed: list[object] = []

    class DummyApp:
        def push_screen(self, screen_obj, callback=None):
            pushed.append(screen_obj)

    class DummyTable:
        def add_column(self, *args, **kwargs):
            return None

        def focus(self):
            return None

    class DummyFocusSink:
        def focus(self):
            return None

    def query_one(selector, *_args, **_kwargs):
        if selector == "#runs-table":
            return DummyTable()
        if selector == "#runs-focus-sink":
            return DummyFocusSink()
        raise AssertionError(selector)

    monkeypatch.setattr(RunsListScreen, "app", property(lambda self: DummyApp()))
    monkeypatch.setattr(screen, "query_one", query_one)
    screen._rows = []

    screen.action_launch_selected()

    assert len(pushed) == 1
    assert isinstance(pushed[0], LaunchModal)


def test_restore_cursor_to_run_id_does_not_scroll_viewport(monkeypatch, tmp_path) -> None:
    pytest.importorskip("textual")

    screen = RunsListScreen(tmp_path)

    class DummyTable:
        def __init__(self):
            self.row_count = 2
            self.move_cursor_calls: list[tuple[int, bool, bool]] = []

        def coordinate_to_cell_key(self, coordinate):
            row_keys = ["run_a", "run_b"]
            return SimpleNamespace(row_key=SimpleNamespace(value=row_keys[coordinate.row]))

        def move_cursor(self, *, row, animate=True, scroll=True):
            self.move_cursor_calls.append((row, animate, scroll))

    table = DummyTable()

    def query_one(selector, *_args, **_kwargs):
        if selector == "#runs-table":
            return table
        raise AssertionError(selector)

    monkeypatch.setattr(screen, "query_one", query_one)
    screen._restore_cursor_to_run_id("run_b")

    assert table.move_cursor_calls == [(1, False, False)]


def test_load_rows_updates_table_incrementally(monkeypatch, tmp_path) -> None:
    pytest.importorskip("textual")

    screen = RunsListScreen(tmp_path)
    screen._rows = [
        {"run_id": "run_a", "run_dir": tmp_path / "run_a", "status": "completed", "created_ts": 100.0},
        {"run_id": "run_b", "run_dir": tmp_path / "run_b", "status": "running", "created_ts": 200.0},
    ]

    class DummyLabel:
        def update(self, _value):
            return None

    class DummyTable:
        def __init__(self):
            self.row_keys = ["run_a", "run_b"]
            self.row_count = len(self.row_keys)
            self.cursor_row = 1
            self.added: list[str] = []
            self.removed: list[str] = []
            self.updated: list[tuple[str, str, object]] = []
            self.sorted: list[tuple[tuple[object, ...], bool]] = []
            self.move_cursor_calls: list[tuple[int, bool, bool]] = []

        def add_row(self, *_values, key=None, **_kwargs):
            self.added.append(key)
            self.row_keys.append(key)
            self.row_count = len(self.row_keys)

        def remove_row(self, row_key):
            self.removed.append(row_key)
            self.row_keys.remove(row_key)
            self.row_count = len(self.row_keys)

        def update_cell(self, row_key, column_key, value):
            self.updated.append((row_key, column_key, value))

        def sort(self, *columns, **kwargs):
            self.sorted.append((columns, kwargs.get("reverse", False)))

        def coordinate_to_cell_key(self, coordinate):
            return SimpleNamespace(row_key=SimpleNamespace(value=self.row_keys[coordinate.row]))

        def move_cursor(self, *, row, animate=True, scroll=True):
            self.move_cursor_calls.append((row, animate, scroll))

    table = DummyTable()
    label = DummyLabel()

    def query_one(selector, *_args, **_kwargs):
        if selector == "#runs-table":
            return table
        if selector == "#status-bar":
            return label
        raise AssertionError(selector)

    monkeypatch.setattr(screen, "query_one", query_one)
    monkeypatch.setattr(screen, "_refresh_header_labels", lambda: None)
    monkeypatch.setattr(screen, "_refresh_status_bar", lambda: None)
    monkeypatch.setattr(screen, "refresh_bindings", lambda: None)
    monkeypatch.setattr(
        "src.pipeline.run_inspector.scan_runs_dir",
        lambda _runs_dir: [
            {"run_id": "run_b", "run_dir": tmp_path / "run_b", "status": "completed", "created_ts": 250.0},
            {"run_id": "run_c", "run_dir": tmp_path / "run_c", "status": "running", "created_ts": 300.0},
        ],
    )

    screen._load_rows(preserve_cursor=True)

    assert table.removed == ["run_a"]
    assert table.added == ["run_c"]
    assert any(update[0] == "run_b" and update[1] == "status" for update in table.updated)
    assert table.sorted == [(("created",), True)]
    assert table.move_cursor_calls == [(0, False, False)]
