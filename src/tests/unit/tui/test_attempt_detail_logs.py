from rich.text import Text
from textual.geometry import Offset
from textual.selection import Selection

from src.tui.screens.attempt_detail import (
    AttemptDetailScreen,
    _LiveLog,
    _PipelineLogHighlighter,
    _normalize_markdown_for_viewer,
)


def test_pipeline_log_highlighter_marks_log_levels_and_tracebacks() -> None:
    text = Text("2026-03-22T12:00:00 INFO step started\nTraceback (most recent call last):\nValueError: bad config")

    _PipelineLogHighlighter().highlight(text)

    styles = {span.style for span in text.spans}
    assert "green" in styles
    assert "bold red" in styles
    assert "magenta" in styles


def test_attempt_detail_enables_word_wrap_by_default(tmp_path) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)

    assert screen._log_word_wrap_enabled is True


def test_attempt_detail_wrap_action_is_contextual_to_logs_tab(tmp_path, monkeypatch) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    screen._log_files = {"Pipeline": tmp_path / "pipeline.log"}

    class DummyTabs:
        active = "logs"

    monkeypatch.setattr(screen, "query_one", lambda *args, **kwargs: DummyTabs())
    assert screen.check_action("toggle_log_wrap", ()) is True

    DummyTabs.active = "details"
    assert screen.check_action("toggle_log_wrap", ()) is False


def test_reload_current_log_preserves_viewport_when_not_following(tmp_path, monkeypatch) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    log_path = tmp_path / "pipeline.log"
    log_path.write_text("line 1\nline 2\n", encoding="utf-8")
    screen._log_files = {"Pipeline": log_path}
    screen._log_auto_follow = False
    screen._log_word_wrap_enabled = True

    class DummySelect:
        value = "Pipeline"

    class DummyWidget:
        scroll_offset = (12, 34)
        is_vertical_scroll_end = False

        def scroll_to(self, *, x=None, y=None, animate=True, immediate=False):
            self.scrolled_to = (x, y, animate, immediate)

    widget = DummyWidget()
    monkeypatch.setattr(
        screen,
        "query_one",
        lambda selector, *_args, **_kwargs: DummySelect() if selector == "#log-selector" else widget,
    )
    monkeypatch.setattr(screen, "_load_log_file", lambda path, widget_arg, reset_follow: None)
    monkeypatch.setattr(screen, "call_after_refresh", lambda callback: callback())

    screen._reload_current_log(reset_follow=False, preserve_viewport=True)

    assert widget.scrolled_to == (0, 34, False, True)


def test_stop_live_updates_stops_timer_and_disables_polling(tmp_path) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    stopped = False

    class DummyTimer:
        def stop(self):
            nonlocal stopped
            stopped = True

    screen._live_updates_active = True
    screen._live_updates_timer = DummyTimer()

    screen._stop_live_updates()

    assert stopped is True
    assert screen._live_updates_active is False
    assert screen._live_updates_timer is None


def test_poll_live_updates_returns_immediately_when_screen_is_inactive(tmp_path) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    screen._live_updates_active = False

    screen._poll_live_updates()


def test_live_log_get_selection_reads_visible_lines() -> None:
    widget = _LiveLog()
    widget._lines = ["alpha beta", "gamma"]

    text, ending = widget.get_selection(Selection(Offset(6, 0), Offset(3, 1)))

    assert text == "beta\ngam"
    assert ending == "\n"


def test_normalize_markdown_for_viewer_promotes_single_newlines_to_hard_breaks() -> None:
    normalized = _normalize_markdown_for_viewer("alpha\nbeta\n\ngamma\r\ndelta")

    assert normalized == "alpha  \nbeta\n\ngamma  \ndelta"
