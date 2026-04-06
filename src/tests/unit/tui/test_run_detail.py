from types import SimpleNamespace

from src.tui.screens.config_browser_modal import StructuredConfigBrowser
from src.tui.screens.run_detail import (
    RunDetailScreen,
    _default_attempt_display_order,
    _default_attempt_no_sort_dir,
    _next_attempt_no_sort_dir,
)


def test_default_attempt_order_stays_natural_for_short_histories() -> None:
    attempts = [SimpleNamespace(attempt_no=no) for no in range(1, 11)]

    ordered = _default_attempt_display_order(attempts)

    assert [attempt.attempt_no for attempt in ordered] == list(range(10, 0, -1))


def test_default_attempt_order_is_latest_first_for_long_histories() -> None:
    attempts = [SimpleNamespace(attempt_no=no) for no in range(1, 12)]

    ordered = _default_attempt_display_order(attempts)

    assert [attempt.attempt_no for attempt in ordered] == list(range(11, 0, -1))


def test_default_attempt_number_sort_direction_is_ascending() -> None:
    assert _default_attempt_no_sort_dir() == "asc"


def test_attempt_number_sort_cycle_is_default_asc_default() -> None:
    assert _next_attempt_no_sort_dir(None) == "asc"
    assert _next_attempt_no_sort_dir("asc") is None


def test_run_detail_uses_c_for_structured_config() -> None:
    bindings = {binding.action: binding.key for binding in RunDetailScreen.BINDINGS}

    assert "preview_config" not in bindings
    assert bindings["browse_config"] == "c"


def test_run_detail_enables_config_action_when_config_path_is_known(tmp_path) -> None:
    screen = RunDetailScreen(tmp_path / "runs" / "run_1")

    screen._config_path = tmp_path / "config.yaml"
    assert screen.check_action("browse_config", ()) is True

    screen._config_path = None
    assert screen.check_action("browse_config", ()) is False


def test_run_detail_browse_config_opens_structured_browser(monkeypatch, tmp_path) -> None:
    screen = RunDetailScreen(tmp_path / "runs" / "run_1")
    screen._config_path = tmp_path / "config.yaml"
    pushed: list[object] = []

    class DummyApp:
        def push_screen(self, screen_obj):
            pushed.append(screen_obj)

    monkeypatch.setattr(RunDetailScreen, "app", property(lambda self: DummyApp()))

    screen.action_browse_config()

    assert len(pushed) == 1
    assert isinstance(pushed[0], StructuredConfigBrowser)


def test_refresh_run_detail_resolves_config_path_from_loaded_state(monkeypatch, tmp_path) -> None:
    screen = RunDetailScreen(tmp_path / "runs" / "run_1")
    config_path = tmp_path / "config.yaml"
    state = SimpleNamespace(config_path=str(config_path), attempts=[])

    monkeypatch.setattr(
        "src.tui.screens.run_detail.load_run_inspection",
        lambda run_dir, include_logs=False: SimpleNamespace(state=state),
    )
    monkeypatch.setattr(screen, "_populate_overview", lambda _state: None)
    monkeypatch.setattr(screen, "_populate_attempts", lambda _state: None)
    monkeypatch.setattr(screen, "_populate_diff", lambda _state: None)
    monkeypatch.setattr(screen, "refresh_bindings", lambda: None)

    screen._refresh_run_detail()

    assert screen._config_path == config_path.resolve()
