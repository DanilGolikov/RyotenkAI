from rich.text import Text
from textual.geometry import Offset
from textual.selection import Selection

from src.tui.screens.attempt_detail import (
    AttemptDetailScreen,
    _LOGS_END_SEPARATOR_ID,
    _LOGS_START_SEPARATOR_ID,
    _LiveLog,
    _PipelineLogHighlighter,
    _format_numbered_tab_title,
    _make_log_tab_id,
    _make_log_widget_id,
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
    log_tab_id = _make_log_tab_id("pipeline.log")
    screen._log_files = {log_tab_id: tmp_path / "pipeline.log"}

    class DummyTabs:
        active = log_tab_id

    monkeypatch.setattr(screen, "query_one", lambda *args, **kwargs: DummyTabs())
    assert screen.check_action("toggle_log_wrap", ()) is True

    DummyTabs.active = "details"
    assert screen.check_action("toggle_log_wrap", ()) is False


def test_reload_current_log_preserves_viewport_when_not_following(tmp_path, monkeypatch) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    log_path = tmp_path / "pipeline.log"
    log_path.write_text("line 1\nline 2\n", encoding="utf-8")
    log_tab_id = _make_log_tab_id("pipeline.log")
    screen._log_files = {log_tab_id: log_path}
    screen._log_auto_follow = False
    screen._log_word_wrap_enabled = True

    class DummyWidget:
        scroll_offset = (12, 34)
        is_vertical_scroll_end = False

        def scroll_to(self, *, x=None, y=None, animate=True, immediate=False):
            self.scrolled_to = (x, y, animate, immediate)

    widget = DummyWidget()
    monkeypatch.setattr(screen, "_active_log_tab_id", lambda: log_tab_id)
    monkeypatch.setattr(
        screen,
        "query_one",
        lambda selector, *_args, **_kwargs: widget if selector == f"#{_make_log_widget_id(log_tab_id)}" else None,
    )
    monkeypatch.setattr(screen, "_load_log_file", lambda path, widget_arg, reset_follow: None)
    monkeypatch.setattr(screen, "call_after_refresh", lambda callback: callback())

    screen._reload_current_log(reset_follow=False, preserve_viewport=True)

    assert widget.scrolled_to == (0, 34, False, True)


def test_discover_log_files_uses_filenames_and_keeps_known_logs_first(tmp_path) -> None:
    attempt_dir = tmp_path / "runs" / "run_1" / "attempts" / "attempt_1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "training.log").write_text("", encoding="utf-8")
    (attempt_dir / "pipeline.log").write_text("", encoding="utf-8")
    (attempt_dir / "custom_metrics.log").write_text("", encoding="utf-8")

    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)

    assert list(screen._discover_log_files()) == ["pipeline.log", "training.log", "custom_metrics.log"]


def test_format_numbered_tab_title_appends_index() -> None:
    assert _format_numbered_tab_title("pipeline.log", 2) == "pipeline.log [2]"


def test_sync_log_group_separators_adds_left_separator_for_logs_only(tmp_path, monkeypatch) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    screen._log_files = {_make_log_tab_id("pipeline.log"): tmp_path / "pipeline.log"}
    added: list[tuple[str, str | None, str | None]] = []

    class DummyTabbedContent:
        def add_pane(self, pane, *, before=None, after=None):
            added.append((pane.id, before, after))

        def remove_pane(self, pane_id):
            raise AssertionError(f"unexpected remove_pane({pane_id})")

        def get_pane(self, pane_id):
            raise LookupError(pane_id)

    monkeypatch.setattr(screen, "query_one", lambda *_args, **_kwargs: DummyTabbedContent())

    screen._sync_log_group_separators()

    assert (_LOGS_START_SEPARATOR_ID, None, "details") in added


def test_sync_log_group_separators_adds_right_separator_before_artifact_group(tmp_path, monkeypatch) -> None:
    screen = AttemptDetailScreen(tmp_path / "runs" / "run_1", 1)
    last_log_tab_id = _make_log_tab_id("training.log")
    screen._log_files = {
        _make_log_tab_id("pipeline.log"): tmp_path / "pipeline.log",
        last_log_tab_id: tmp_path / "training.log",
    }
    screen._inference_tab_added = True
    added: list[tuple[str, str | None, str | None]] = []

    class DummyTabbedContent:
        def add_pane(self, pane, *, before=None, after=None):
            added.append((pane.id, before, after))

        def remove_pane(self, pane_id):
            pass

    monkeypatch.setattr(screen, "query_one", lambda *_args, **_kwargs: DummyTabbedContent())

    screen._sync_log_group_separators()

    assert (_LOGS_END_SEPARATOR_ID, None, last_log_tab_id) in added


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
