import pytest

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
