import pytest
from types import SimpleNamespace

from src.tui.screens.runs_list import (
    RunsListScreen,
    _ExitConfirmModal,
    _FOLDER_KEY_PREFIX,
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
        row_count = 0
        cursor_row = -1

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


def test_load_rows_updates_table_with_tree_data(monkeypatch, tmp_path) -> None:
    """Verify _load_rows builds tree and populates the table with folder + run entries."""
    pytest.importorskip("textual")

    from src.pipeline.run_inspector import ROOT_GROUP

    screen = RunsListScreen(tmp_path)
    screen._rows = []

    class DummyLabel:
        def update(self, _value):
            return None

    class DummyTable:
        def __init__(self):
            self.row_keys: list[str] = []
            self.row_count = 0
            self.cursor_row = 0
            self.added: list[str] = []
            self.columns = {
                "status": SimpleNamespace(label="Status"),
                "config": SimpleNamespace(label="Config"),
                "duration": SimpleNamespace(label="Duration"),
                "created": SimpleNamespace(label="Created"),
            }

        def clear(self):
            self.row_keys.clear()
            self.row_count = 0
            self.added.clear()

        def add_row(self, *_values, key=None, **_kwargs):
            self.added.append(key)
            self.row_keys.append(key)
            self.row_count = len(self.row_keys)

        def coordinate_to_cell_key(self, coordinate):
            return SimpleNamespace(row_key=SimpleNamespace(value=self.row_keys[coordinate.row]))

        def move_cursor(self, *, row, animate=True, scroll=True):
            pass

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
        "src.pipeline.run_inspector.scan_runs_dir_grouped",
        lambda _runs_dir: {
            ROOT_GROUP: [
                {"run_id": "run_b", "run_dir": tmp_path / "run_b", "status": "completed",
                 "created_ts": 250.0, "created_at": "2026-03-30 14:20", "group": ROOT_GROUP,
                 "config": "cfg.yaml", "attempts": 1, "duration": "5m", "error": None},
            ],
            "smoke_abc": [
                {"run_id": "run_c", "run_dir": tmp_path / "smoke_abc" / "run_c", "status": "running",
                 "created_ts": 300.0, "created_at": "2026-03-30 15:00", "group": "smoke_abc",
                 "config": "check.yaml", "attempts": 1, "duration": "2m", "error": None},
            ],
        },
    )

    screen._load_rows()

    assert screen._has_subfolders is True
    folder_keys = [k for k in table.added if k.startswith(_FOLDER_KEY_PREFIX)]
    assert len(folder_keys) == 1
    assert f"{_FOLDER_KEY_PREFIX}smoke_abc" in folder_keys
    run_keys = [k for k in table.added if not k.startswith(_FOLDER_KEY_PREFIX)]
    assert "(root)::run_b" in run_keys
    assert "smoke_abc::run_c" in run_keys


def test_load_rows_single_group_no_headers(monkeypatch, tmp_path) -> None:
    """When only root runs exist, no folder headers are inserted."""
    pytest.importorskip("textual")

    from src.pipeline.run_inspector import ROOT_GROUP

    screen = RunsListScreen(tmp_path)
    screen._rows = []

    class DummyLabel:
        def update(self, _value):
            return None

    class DummyTable:
        def __init__(self):
            self.row_keys: list[str] = []
            self.row_count = 0
            self.cursor_row = 0
            self.columns = {
                "status": SimpleNamespace(label="Status"),
                "config": SimpleNamespace(label="Config"),
                "duration": SimpleNamespace(label="Duration"),
                "created": SimpleNamespace(label="Created"),
            }

        def clear(self):
            self.row_keys.clear()
            self.row_count = 0

        def add_row(self, *_values, key=None, **_kwargs):
            self.row_keys.append(key)
            self.row_count = len(self.row_keys)

        def coordinate_to_cell_key(self, coordinate):
            return SimpleNamespace(row_key=SimpleNamespace(value=self.row_keys[coordinate.row]))

        def move_cursor(self, *, row, animate=True, scroll=True):
            pass

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
        "src.pipeline.run_inspector.scan_runs_dir_grouped",
        lambda _runs_dir: {
            ROOT_GROUP: [
                {"run_id": "run_a", "run_dir": tmp_path / "run_a", "status": "completed",
                 "created_ts": 100.0, "created_at": "2026-03-30 10:00", "group": ROOT_GROUP,
                 "config": "cfg.yaml", "attempts": 1, "duration": "5m", "error": None},
            ],
        },
    )

    screen._load_rows()

    assert screen._has_subfolders is False
    assert not any(k.startswith(_FOLDER_KEY_PREFIX) for k in table.row_keys)
    assert "(root)::run_a" in table.row_keys


def test_nested_folders_appear_as_tree(monkeypatch, tmp_path) -> None:
    """Verify nested groups like smoke_a/smoke_b become a tree, not flat siblings."""
    pytest.importorskip("textual")

    from src.pipeline.run_inspector import ROOT_GROUP

    screen = RunsListScreen(tmp_path)

    class DummyLabel:
        def update(self, _value):
            return None

    class DummyTable:
        def __init__(self):
            self.row_keys: list[str] = []
            self.row_count = 0
            self.cursor_row = 0
            self.columns = {
                "status": SimpleNamespace(label="Status"),
                "config": SimpleNamespace(label="Config"),
                "duration": SimpleNamespace(label="Duration"),
                "created": SimpleNamespace(label="Created"),
            }

        def clear(self):
            self.row_keys.clear()
            self.row_count = 0

        def add_row(self, *_values, key=None, **_kwargs):
            self.row_keys.append(key)
            self.row_count = len(self.row_keys)

        def coordinate_to_cell_key(self, coordinate):
            return SimpleNamespace(row_key=SimpleNamespace(value=self.row_keys[coordinate.row]))

        def move_cursor(self, *, row, animate=True, scroll=True):
            pass

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
        "src.pipeline.run_inspector.scan_runs_dir_grouped",
        lambda _runs_dir: {
            "smoke_a": [
                {"run_id": "run_1", "run_dir": tmp_path / "smoke_a" / "run_1", "status": "completed",
                 "created_ts": 100.0, "created_at": "2026-03-30 10:00", "group": "smoke_a",
                 "config": "cfg.yaml", "attempts": 1, "duration": "5m", "error": None},
            ],
            "smoke_a/smoke_b": [
                {"run_id": "run_2", "run_dir": tmp_path / "smoke_a" / "smoke_b" / "run_2",
                 "status": "failed", "created_ts": 200.0, "created_at": "2026-03-30 12:00",
                 "group": "smoke_a/smoke_b", "config": "check.yaml", "attempts": 1,
                 "duration": "2m", "error": None},
            ],
        },
    )

    screen._load_rows()

    assert screen._has_subfolders is True
    # smoke_a folder header, then nested smoke_b folder header, then run_2, then run_1
    assert f"{_FOLDER_KEY_PREFIX}smoke_a" in table.row_keys
    assert f"{_FOLDER_KEY_PREFIX}smoke_a/smoke_b" in table.row_keys
    # smoke_b appears AFTER smoke_a (nested inside)
    idx_a = table.row_keys.index(f"{_FOLDER_KEY_PREFIX}smoke_a")
    idx_b = table.row_keys.index(f"{_FOLDER_KEY_PREFIX}smoke_a/smoke_b")
    assert idx_b > idx_a
    # run_2 (inside smoke_b) appears after smoke_b header
    idx_run2 = table.row_keys.index("smoke_a/smoke_b::run_2")
    assert idx_run2 > idx_b
    # run_1 (direct child of smoke_a) appears after nested folder's content
    idx_run1 = table.row_keys.index("smoke_a::run_1")
    assert idx_run1 > idx_run2
