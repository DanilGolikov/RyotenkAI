from src.tui.config_browser_state import ConfigBrowserItem
from textual.widgets import OptionList

from src.tui.screens.config_browser_modal import (
    StructuredConfigBrowser,
    _detail_count_style,
    _item_prompt,
    _styled_detail_line,
)
from src.tui.screens.file_preview_modal import FilePreviewModal


def test_item_prompt_formats_sections_without_extra_summary() -> None:
    item = ConfigBrowserItem(label="Model", path=("model",), has_children=True, is_section=True)

    assert _item_prompt(item) == "[bold cyan]Model[/bold cyan]"


def test_item_prompt_formats_values_and_counts_with_different_colors() -> None:
    scalar_item = ConfigBrowserItem(label="Name", path=("model", "name"), has_children=False, value_text="Qwen/Test")
    list_item = ConfigBrowserItem(label="Plugins", path=("evaluation", "plugins"), has_children=True, item_count=3)
    mapping_item = ConfigBrowserItem(label="Runpod", path=("providers", "runpod"), has_children=True, field_count=2)

    assert _item_prompt(scalar_item) == "[bold cyan]Name[/bold cyan] [dim]-[/dim] [green]Qwen/Test[/green]"
    assert _item_prompt(list_item) == "[bold cyan]Plugins[/bold cyan] [dim]-[/dim] [magenta]3 items[/magenta]"
    assert _item_prompt(mapping_item) == "[bold cyan]Runpod[/bold cyan] [dim]-[/dim] [yellow]2 fields[/yellow]"


def test_move_right_from_sections_focuses_items(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)

    class DummySections(OptionList):
        @property
        def highlighted(self):
            return 0

    sections = DummySections(id="config-browser-sections")
    items = OptionList(id="config-browser-items")
    calls: list[str] = []

    monkeypatch.setattr(StructuredConfigBrowser, "focused", property(lambda self: sections))
    monkeypatch.setattr(modal, "_activate_section", lambda index: calls.append(f"activate:{index}"))
    monkeypatch.setattr(modal, "query_one", lambda selector, *_args, **_kwargs: items if selector == "#config-browser-items" else None)
    monkeypatch.setattr(items, "focus", lambda: calls.append("focus-items"))

    modal.action_move_right()

    assert calls == ["activate:0", "focus-items"]


def test_set_section_options_highlights_first_section(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._sections = (
        ConfigBrowserItem(label="Model", path=("model",), has_children=True, is_section=True),
        ConfigBrowserItem(label="Training", path=("training",), has_children=True, is_section=True),
    )

    class DummyOptionList:
        def __init__(self) -> None:
            self.options = None
            self.highlighted = None

        def set_options(self, options):
            self.options = options

    sections = DummyOptionList()
    monkeypatch.setattr(modal, "query_one", lambda *_args, **_kwargs: sections)

    modal._set_section_options()

    assert sections.highlighted == 0
    assert sections.options == ["[bold cyan]Model[/bold cyan]", "[bold cyan]Training[/bold cyan]"]


def test_refresh_items_keeps_container_details_on_initial_open(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._browser_state = None
    modal._current_path = ("datasets", "default", "validations", "plugins")
    modal._detail_path = ()
    modal._current_items = ()
    modal._suspend_item_highlight_detail = False

    items = [
        ConfigBrowserItem(label="alpha", path=("datasets", "default", "validations", "plugins", 0), has_children=True),
    ]

    class DummyBrowserState:
        def list_children(self, path):
            return tuple(items)

    class DummyOptionList:
        def __init__(self) -> None:
            self.highlighted = None

        def set_options(self, options):
            self.options = options

    modal._browser_state = DummyBrowserState()
    item_list = DummyOptionList()
    detail_paths: list[tuple[str | int, ...]] = []

    monkeypatch.setattr(modal, "query_one", lambda *_args, **_kwargs: item_list)
    monkeypatch.setattr(modal, "_refresh_breadcrumbs", lambda: None)
    monkeypatch.setattr(modal, "_refresh_hotkeys", lambda: None)
    monkeypatch.setattr(modal, "_update_detail", lambda path: detail_paths.append(path))

    modal._refresh_items(detail_path=modal._current_path)

    assert item_list.highlighted == 0
    assert detail_paths == [("datasets", "default", "validations", "plugins")]


def test_escape_or_close_navigates_up_when_inside_nested_items(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._current_path = ("datasets", "default", "validations")
    calls: list[str] = []
    monkeypatch.setattr(modal, "action_navigate_up", lambda: calls.append("up"))
    monkeypatch.setattr(modal, "action_close_browser", lambda: calls.append("close"))

    modal.action_escape_or_close()

    assert calls == ["up"]


def test_escape_or_close_closes_browser_at_root_level(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._current_path = ("datasets",)
    calls: list[str] = []
    monkeypatch.setattr(modal, "action_navigate_up", lambda: calls.append("up"))
    monkeypatch.setattr(modal, "action_close_browser", lambda: calls.append("close"))

    modal.action_escape_or_close()

    assert calls == ["close"]


def test_refresh_items_restores_highlight_for_previous_child(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._current_path = ("datasets",)
    modal._detail_path = ()
    modal._current_items = ()
    modal._suspend_item_highlight_detail = False

    items = [
        ConfigBrowserItem(label="alpha", path=("datasets", "alpha"), has_children=True),
        ConfigBrowserItem(label="beta", path=("datasets", "beta"), has_children=True),
    ]

    class DummyBrowserState:
        def list_children(self, path):
            return tuple(items)

    modal._browser_state = DummyBrowserState()

    class DummyOptionList:
        def __init__(self) -> None:
            self.options = None
            self.highlighted = None

        def set_options(self, options):
            self.options = options

    item_list = DummyOptionList()
    detail_paths: list[tuple[str | int, ...]] = []
    monkeypatch.setattr(modal, "query_one", lambda *_args, **_kwargs: item_list)
    monkeypatch.setattr(modal, "_refresh_breadcrumbs", lambda: None)
    monkeypatch.setattr(modal, "_refresh_hotkeys", lambda: None)
    monkeypatch.setattr(modal, "_update_detail", lambda path: detail_paths.append(path))

    modal._refresh_items(highlighted_path=("datasets", "beta"), detail_path=("datasets", "beta"))

    assert item_list.highlighted == 1
    assert detail_paths == [("datasets", "beta")]


def test_navigate_up_restores_focus_to_item_that_was_opened(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._current_path = ("datasets", "default", "validations")
    calls: list[object] = []

    class DummyItems:
        def focus(self):
            calls.append("focus")

    monkeypatch.setattr(modal, "_refresh_items", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(modal, "query_one", lambda *_args, **_kwargs: DummyItems())

    modal.action_navigate_up()

    assert modal._current_path == ("datasets", "default")
    assert calls == [
        {
            "highlighted_path": ("datasets", "default", "validations"),
            "detail_path": ("datasets", "default", "validations"),
        },
        "focus",
    ]


def test_sync_detail_to_focus_uses_current_section_when_sections_focused(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._current_path = ("evaluation",)
    modal._detail_path = ()
    modal._current_items = (
        ConfigBrowserItem(label="plugins", path=("evaluation", "plugins"), has_children=True),
    )

    sections = OptionList(id="config-browser-sections")
    calls: list[tuple[str | int, ...]] = []
    monkeypatch.setattr(StructuredConfigBrowser, "focused", property(lambda self: sections))
    monkeypatch.setattr(modal, "_browser_nodes_ready", lambda: True)
    monkeypatch.setattr(modal, "_refresh_breadcrumbs", lambda: None)
    monkeypatch.setattr(modal, "_update_detail", lambda path: calls.append(path))

    modal._sync_detail_to_focus()

    assert modal._detail_path == ("evaluation",)
    assert calls == [("evaluation",)]


def test_sync_detail_to_focus_uses_highlighted_item_when_items_focused(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._current_path = ("evaluation",)
    modal._detail_path = ()
    modal._current_items = (
        ConfigBrowserItem(label="alpha", path=("evaluation", "plugins", 0), has_children=True),
        ConfigBrowserItem(label="beta", path=("evaluation", "plugins", 1), has_children=True),
    )

    class DummyItems(OptionList):
        @property
        def highlighted(self):
            return 1

    items = DummyItems(id="config-browser-items")
    calls: list[tuple[str | int, ...]] = []
    monkeypatch.setattr(StructuredConfigBrowser, "focused", property(lambda self: items))
    monkeypatch.setattr(modal, "_browser_nodes_ready", lambda: True)
    monkeypatch.setattr(modal, "_refresh_breadcrumbs", lambda: None)
    monkeypatch.setattr(modal, "_update_detail", lambda path: calls.append(path))

    modal._sync_detail_to_focus()

    assert modal._detail_path == ("evaluation", "plugins", 1)
    assert calls == [("evaluation", "plugins", 1)]


def test_sync_detail_to_focus_returns_early_when_nodes_are_not_ready(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._current_path = ("evaluation",)
    modal._detail_path = ()
    modal._current_items = ()
    monkeypatch.setattr(modal, "_browser_nodes_ready", lambda: False)

    modal._sync_detail_to_focus()

    assert modal._detail_path == ()


def test_styled_detail_line_uses_field_and_value_colors() -> None:
    text = _styled_detail_line("Path: datasets.default")

    assert text.plain == "Path: datasets.default"
    assert any(span.style == "bold cyan" for span in text.spans)
    assert any(span.style == "green" for span in text.spans)


def test_detail_count_style_uses_different_colors_for_items_and_fields() -> None:
    assert _detail_count_style("Items", "3") == "magenta"
    assert _detail_count_style("Validation plugins", "2") == "magenta"
    assert _detail_count_style("Fields", "4") == "yellow"
    assert _detail_count_style("Threshold fields", "score, syntax") == "yellow"


def test_structured_config_browser_open_raw_yaml_uses_file_preview_modal(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: test\n", encoding="utf-8")
    modal = StructuredConfigBrowser(config_path)
    pushed: list[object] = []

    class DummyApp:
        def push_screen(self, screen_obj):
            pushed.append(screen_obj)

    monkeypatch.setattr(StructuredConfigBrowser, "app", property(lambda self: DummyApp()))

    modal.action_open_raw_yaml()

    assert len(pushed) == 1
    assert isinstance(pushed[0], FilePreviewModal)


def test_toggle_wrap_reloads_current_detail(monkeypatch) -> None:
    modal = StructuredConfigBrowser.__new__(StructuredConfigBrowser)
    modal._detail_wrap_enabled = False
    modal._detail_path = ("model",)
    modal._current_path = ()
    calls: list[tuple[str | int, ...]] = []
    monkeypatch.setattr(modal, "_refresh_hotkeys", lambda: None)
    monkeypatch.setattr(modal, "_update_detail", lambda path: calls.append(path))

    modal.action_toggle_wrap()

    assert modal._detail_wrap_enabled is True
    assert calls == [("model",)]
