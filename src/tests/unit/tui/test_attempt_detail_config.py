from __future__ import annotations

from pathlib import Path

from src.tui.screens.attempt_detail import AttemptDetailScreen, _resolve_attempt_config_path
from src.tui.screens.config_browser_modal import StructuredConfigBrowser


def test_attempt_detail_uses_c_for_structured_config() -> None:
    bindings = {binding.action: binding.key for binding in AttemptDetailScreen.BINDINGS}

    assert "preview_config" not in bindings
    assert bindings["browse_config"] == "c"
    assert bindings["relaunch"] == "l"


def test_resolve_attempt_config_path_normalizes_existing_value(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "pipeline.yaml"

    assert _resolve_attempt_config_path(str(config_path)) == config_path.resolve()


def test_resolve_attempt_config_path_returns_none_for_empty_value() -> None:
    assert _resolve_attempt_config_path("") is None
    assert _resolve_attempt_config_path(None) is None


def test_attempt_detail_enables_browse_action_when_config_path_is_known(tmp_path: Path) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)

    screen._config_path = tmp_path / "config.yaml"
    assert screen.check_action("browse_config", ()) is True

    screen._config_path = None
    assert screen.check_action("browse_config", ()) is False


def test_attempt_detail_browse_config_opens_structured_browser(monkeypatch, tmp_path: Path) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    screen._config_path = tmp_path / "config.yaml"
    pushed: list[object] = []

    class DummyApp:
        def push_screen(self, screen_obj):
            pushed.append(screen_obj)

    monkeypatch.setattr(AttemptDetailScreen, "app", property(lambda self: DummyApp()))

    screen.action_browse_config()

    assert len(pushed) == 1
    assert isinstance(pushed[0], StructuredConfigBrowser)


def test_attempt_detail_relaunch_opens_launch_modal(monkeypatch, tmp_path: Path) -> None:
    from src.tui.screens.launch_modal import LaunchModal

    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 2)
    pushed: list[object] = []

    class DummyApp:
        def push_screen(self, screen_obj, callback=None):
            pushed.append(screen_obj)

    monkeypatch.setattr(AttemptDetailScreen, "app", property(lambda self: DummyApp()))
    monkeypatch.setattr("src.tui.launch.pick_default_launch_mode", lambda _run_dir: "restart")

    screen.action_relaunch()

    assert len(pushed) == 1
    assert isinstance(pushed[0], LaunchModal)
