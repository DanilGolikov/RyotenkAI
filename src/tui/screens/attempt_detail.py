"""AttemptDetailScreen — attempt details + logs + inference + eval."""

from __future__ import annotations

import contextlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from rich.cells import cell_len
from rich.highlighter import Highlighter, ISO8601Highlighter, ReprHighlighter
from rich.segment import Segment
from rich.text import Text
from textual import events, on
from textual.binding import Binding
from textual.geometry import Size
from textual.screen import Screen
from textual.strip import Strip
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Log,
    MarkdownViewer,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    Tabs,
)

from src.tui.live_logs import LiveLogTail
from src.tui.screens._mixins import _HelpMixin, _InterruptConfirmMixin, _TabbedScreenMixin

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.timer import Timer

    from src.tui.apps import RyotenkaiApp

_TIMESTAMP_LEN = 19  # "YYYY-MM-DDTHH:MM:SS"

_LOG_CANDIDATES: list[tuple[str, str]] = [
    ("pipeline.log", "Pipeline"),
    ("training.log", "Training"),
    ("inference.log", "Inference"),
    ("eval.log", "Eval"),
]

_ARTIFACT_RENDERERS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".json": "text",
    ".yaml": "text",
    ".yml": "text",
}


def _normalize_markdown_for_viewer(markdown: str) -> str:
    """Make single newlines render as visible line breaks in Textual markdown."""
    normalized = markdown.replace("\r\n", "\n")
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", normalized)


def _resolve_attempt_config_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return Path(raw_path).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class _VisualLogLine:
    text: str
    line_no: int
    is_continuation: bool


class _StagesTable(DataTable):
    """DataTable that returns focus to the nearest Tabs bar when Up is pressed at row 0."""

    def action_cursor_up(self) -> None:
        if not self.display:
            return
        if self.cursor_row == 0 and self.row_count > 0:
            with contextlib.suppress(Exception):
                self.screen.query_one(Tabs).focus()
        else:
            super().action_cursor_up()


class _PipelineLogHighlighter(Highlighter):
    """Lightweight highlighting for live pipeline logs."""

    def __init__(self) -> None:
        self._timestamps = ISO8601Highlighter()
        self._repr = ReprHighlighter()

    def highlight(self, text) -> None:
        self._timestamps.highlight(text)
        self._repr.highlight(text)
        text.highlight_regex(r"\bDEBUG\b", "blue")
        text.highlight_regex(r"\bINFO\b", "green")
        text.highlight_regex(r"\bWARN(?:ING)?\b", "yellow")
        text.highlight_regex(r"\bERROR\b", "bold red")
        text.highlight_regex(r"\bCRITICAL\b", "white on red")
        text.highlight_regex(r"Traceback \(most recent call last\):", "bold red")
        text.highlight_regex(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b", "magenta")


class _LiveLog(Log):
    """Selectable live log with optional runtime word wrap."""

    def __init__(
        self,
        *args,
        log_highlighter: Highlighter | None = None,
        word_wrap_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if log_highlighter is not None:
            self.highlighter = log_highlighter
        self._word_wrap_enabled = word_wrap_enabled
        self._raw_lines: list[str] = []
        self._visual_lines: list[_VisualLogLine] = []

    def clear(self):
        self._raw_lines.clear()
        self._visual_lines.clear()
        return super().clear()

    def write_line(self, line: str, scroll_end: bool | None = None):
        self.write_lines([line], scroll_end=scroll_end)
        return self

    def write_lines(self, lines: list[str], scroll_end: bool | None = None):
        if not lines:
            return self
        self._raw_lines.extend(lines)
        auto_scroll = self.auto_scroll if scroll_end is None else scroll_end
        self._rebuild_from_raw(scroll_end=auto_scroll, preserve_viewport=not auto_scroll)
        return self

    def set_highlighting(self, enabled: bool) -> None:
        self.highlight = enabled
        self._render_line_cache.clear()
        self.refresh()

    def set_word_wrap(self, enabled: bool, *, preserve_viewport: bool = True) -> None:
        if self._word_wrap_enabled == enabled:
            return
        self._word_wrap_enabled = enabled
        self._rebuild_from_raw(scroll_end=self.auto_scroll, preserve_viewport=preserve_viewport)

    def on_resize(self, event: events.Resize) -> None:
        del event
        if self._word_wrap_enabled and self._raw_lines:
            self._rebuild_from_raw(scroll_end=self.auto_scroll, preserve_viewport=True)

    def _rebuild_from_raw(self, *, scroll_end: bool | None, preserve_viewport: bool) -> None:
        scroll_x, scroll_y = self.scroll_offset
        should_stick_to_end = bool(scroll_end) and self.is_vertical_scroll_end
        self._visual_lines = self._wrap_lines(self._raw_lines)
        self._lines = [line.text for line in self._visual_lines]
        self._width = self._measure_virtual_width(self._visual_lines)
        self._updates += 1
        self._render_line_cache.clear()
        self.virtual_size = Size(self._width, self.line_count)
        self.refresh()
        if should_stick_to_end:
            self.scroll_end(animate=False, immediate=True, x_axis=not self._word_wrap_enabled)
            return
        if preserve_viewport:
            target_x = 0 if self._word_wrap_enabled else scroll_x
            self.call_after_refresh(lambda: self.scroll_to(x=target_x, y=scroll_y, animate=False, immediate=True))

    def _wrap_lines(self, lines: list[str]) -> list[_VisualLogLine]:
        gutter_width = self._gutter_width
        if not self._word_wrap_enabled:
            return [
                _VisualLogLine(text=self._process_line(line), line_no=index, is_continuation=False)
                for index, line in enumerate(lines, start=1)
            ]
        width = max((self.scrollable_content_region.width or self.size.width or 1) - gutter_width, 1)
        wrapped_lines: list[_VisualLogLine] = []
        console = self.app.console
        options = console.options.update_width(width)
        for line_no, raw_line in enumerate(lines, start=1):
            text = Text(self._process_line(raw_line), no_wrap=False, overflow="fold")
            segments = console.render(text, options)
            visual_lines = ["".join(segment.text for segment in line) for line in Segment.split_lines(segments)]
            if not visual_lines:
                wrapped_lines.append(_VisualLogLine(text="", line_no=line_no, is_continuation=False))
                continue
            for index, visual_line in enumerate(visual_lines):
                wrapped_lines.append(
                    _VisualLogLine(
                        text=visual_line,
                        line_no=line_no,
                        is_continuation=index > 0,
                    )
                )
        return wrapped_lines

    @property
    def _gutter_digits(self) -> int:
        return max(len(str(max(len(self._raw_lines), 1))), 2)

    @property
    def _gutter_width(self) -> int:
        return self._gutter_digits + 3  # " 42 | "

    def _measure_virtual_width(self, visual_lines: list[_VisualLogLine]) -> int:
        if self._word_wrap_enabled:
            return max(
                self.scrollable_content_region.width or self.size.width or self._gutter_width, self._gutter_width
            )
        content_width = max((cell_len(line.text) for line in visual_lines), default=0)
        return self._gutter_width + content_width

    def render_line(self, y: int):
        scroll_x, scroll_y = self.scroll_offset
        line_index = scroll_y + y
        rich_style = self.rich_style
        width = self.size.width
        if line_index >= len(self._visual_lines):
            return self._render_gutter("", width, rich_style)

        visual_line = self._visual_lines[line_index]
        gutter = (
            f"{visual_line.line_no:>{self._gutter_digits}}"
            if not visual_line.is_continuation
            else " " * self._gutter_digits
        )
        gutter_text = Text(f"{gutter} | ", style="dim", no_wrap=True)

        content_text = Text(visual_line.text, no_wrap=True)
        content_text.stylize(rich_style)
        if self.highlight:
            content_text = self.highlighter(content_text)
        selection = self.text_selection
        if selection is not None and (select_span := selection.get_span(line_index - self._clear_y)) is not None:
            start, end = select_span
            if end == -1:
                end = len(content_text)
            selection_style = self.screen.get_component_rich_style("screen--selection")
            content_text.stylize(selection_style, start, end)

        gutter_strip = Strip(gutter_text.render(self.app.console), cell_len(gutter_text.plain))
        available_width = max(width - self._gutter_width, 0)
        content_strip = Strip(content_text.render(self.app.console), cell_len(content_text.plain))
        content_scroll_x = 0 if self._word_wrap_enabled else scroll_x
        content_strip = content_strip.crop_extend(content_scroll_x, content_scroll_x + available_width, rich_style)
        content_strip = content_strip.apply_offsets(content_scroll_x, line_index)
        line_strip = gutter_strip + content_strip
        return line_strip.crop_extend(0, width, rich_style)

    def _render_gutter(self, gutter: str, width: int, rich_style):
        gutter_text = Text(gutter.ljust(self._gutter_width), style="dim", no_wrap=True)
        gutter_strip = Strip(gutter_text.render(self.app.console), cell_len(gutter_text.plain))
        return gutter_strip.crop_extend(0, width, rich_style)


class AttemptDetailScreen(_HelpMixin, _InterruptConfirmMixin, _TabbedScreenMixin, Screen):
    """Shows Details / Logs / Report / Inference / Eval tabs for a single attempt."""

    DEFAULT_CSS = """
    AttemptDetailScreen #details-header {
        height: auto;
        padding: 0 1;
    }
    AttemptDetailScreen #stages-table {
        height: 1fr;
    }
    AttemptDetailScreen #stage-detail {
        height: 16;
        border-top: solid $panel-darken-1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape,q", "go_back", "Back", show=True),
        Binding("question_mark", "show_help", "Help", key_display="?", show=True),
        Binding("c", "browse_config", "Config", show=True, priority=True),
        Binding("h", "toggle_log_highlight", "Highlight", show=True),
        Binding("w", "toggle_log_wrap", "Wrap", show=True),
        Binding("ctrl+e", "stop_run", "Stop", key_display="⌃e", show=True, priority=True),
        Binding("shift+c", "launch_chat", "Chat", show=False),
        Binding("1", "show_tab('details')", "Details", show=False),
        Binding("2", "show_tab('logs')", "Logs", show=False),
    ]

    def __init__(self, run_dir: Path, attempt_no: int) -> None:
        super().__init__()
        self._run_dir = run_dir
        self._attempt_no = attempt_no
        self._log_files: dict[str, Path] = {}
        self._report_files: dict[str, Path] = {}
        self._eval_files: dict[str, Path] = {}
        self._chat_script: Path | None = None
        self._tab_counter: int = 2  # Details=1, Logs=2 are static; dynamic start at 3
        self._report_tab_added = False
        self._eval_tab_added = False
        self._inference_tab_added = False
        # (stage_name, StageRunState | None) indexed by table row
        self._stage_data: list[tuple[str, Any]] = []
        self._attempt_status: str | None = None
        self._config_path: Path | None = None
        self._live_log_tail = LiveLogTail()
        self._live_updates_timer: Timer | None = None
        self._live_updates_active = False
        self._log_auto_follow = True
        self._log_highlight_enabled = True
        self._log_word_wrap_enabled = True
        self._log_option_labels: tuple[str, ...] = ()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="details"):
            with TabPane("Details [1]", id="details"):
                yield Static(id="details-header")
                yield _StagesTable(id="stages-table", cursor_type="row")
                yield RichLog(id="stage-detail", highlight=False, markup=False, wrap=True, auto_scroll=False)
            with TabPane("Logs [2]", id="logs"):
                yield Select([], id="log-selector", prompt="Select log file…")
                yield _LiveLog(
                    id="log-content",
                    highlight=self._log_highlight_enabled,
                    auto_scroll=True,
                    log_highlighter=_PipelineLogHighlighter(),
                    word_wrap_enabled=self._log_word_wrap_enabled,
                )
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = f"{self._run_dir.name} › attempt {self._attempt_no}"
        self._setup_stages_columns()
        self._attempt_status = self._load_details()
        self._populate_log_selector()
        self._maybe_add_inference_tab()
        self._maybe_add_eval_tab()
        self._maybe_add_report_tab()
        self._start_live_updates()
        self._focus_tabs_after_mount()

    def on_unmount(self) -> None:
        self._stop_live_updates()

    def _start_live_updates(self) -> None:
        self._stop_live_updates()
        self._live_updates_active = True
        self._live_updates_timer = self.set_interval(1.0, self._poll_live_updates)

    def _stop_live_updates(self) -> None:
        self._live_updates_active = False
        if self._live_updates_timer is None:
            return
        with contextlib.suppress(Exception):
            self._live_updates_timer.stop()
        self._live_updates_timer = None

    # ── Details ───────────────────────────────────────────────────────────────

    def _setup_stages_columns(self) -> None:
        table = self.query_one("#stages-table", _StagesTable)
        table.add_column("", width=3, key="icon")
        table.add_column("Stage", width=28, key="stage")
        table.add_column("Status", width=13, key="status")
        table.add_column("Mode", width=14, key="mode")
        table.add_column("Duration", width=10, key="duration")
        table.add_column("Started", width=20, key="started")

    def _load_details(self) -> str | None:
        from src.pipeline.run_inspector import (
            _STATUS_COLORS,
            _STATUS_ICONS,
            RunInspector,
            _fmt_duration,
            _mode_label,
        )
        from src.pipeline.state import PipelineStateLoadError

        header = self.query_one("#details-header", Static)
        table = self.query_one("#stages-table", _StagesTable)
        selected_row = table.cursor_row if table.row_count > 0 else 0

        try:
            data = RunInspector(self._run_dir).load(include_logs=False)
        except (PipelineStateLoadError, Exception) as exc:
            self._config_path = None
            header.update(f"[bold red]Cannot load state:[/bold red] {exc}")
            return None

        self._config_path = _resolve_attempt_config_path(data.state.config_path)

        attempt = next(
            (a for a in data.state.attempts if a.attempt_no == self._attempt_no),
            None,
        )
        if attempt is None:
            header.update(f"[yellow]Attempt {self._attempt_no} is not initialized yet.[/yellow]")
            table.clear()
            self._stage_data = []
            self.query_one("#stage-detail", RichLog).clear()
            return None

        # ── Attempt header ────────────────────────────────────────────────────
        att_color = _STATUS_COLORS.get(attempt.status, "white")
        att_icon = _STATUS_ICONS.get(attempt.status, "?")
        att_dur = _fmt_duration(attempt.started_at, attempt.completed_at)
        started = (attempt.started_at or "")[:_TIMESTAMP_LEN].replace("T", " ")
        completed = (attempt.completed_at or "")[:_TIMESTAMP_LEN].replace("T", " ")
        action = attempt.restart_from_stage or attempt.effective_action or "fresh"
        config_hash = (attempt.training_critical_config_hash or "—")[:16]

        from rich.text import Text

        t = Text()
        t.append(f"Attempt {attempt.attempt_no}", style="bold cyan")
        t.append(f"  {att_icon} {attempt.status.upper()}\n", style=att_color)
        t.append(
            f"  Started : {started or '—'}   Finished : {completed or '—'}   Duration : {att_dur or '—'}\n", style="dim"
        )
        t.append(f"  Action  : {action}   Config # : {config_hash}\n", style="dim")
        if attempt.error and attempt.status != "completed":
            t.append("  Error: ", style="bold red")
            t.append(attempt.error[:200], style="dim")
        header.update(t)

        # ── Stages table ──────────────────────────────────────────────────────
        self._stage_data = []
        table.clear()
        stage_names = attempt.enabled_stage_names or list(attempt.stage_runs)

        for stage_name in stage_names:
            sr = attempt.stage_runs.get(stage_name)
            self._stage_data.append((stage_name, sr))

            if sr is None:
                table.add_row("—", stage_name, "[dim]pending[/dim]", "—", "—", "—", key=stage_name)
                continue

            s_icon = _STATUS_ICONS.get(sr.status, "?")
            s_color = _STATUS_COLORS.get(sr.status, "white")
            s_dur = _fmt_duration(sr.started_at, sr.completed_at)
            mode = _mode_label(sr)
            s_started = (sr.started_at or "")[:_TIMESTAMP_LEN].replace("T", " ")

            table.add_row(
                s_icon,
                f"[{s_color}]{sr.stage_name}[/{s_color}]",
                f"[{s_color}]{sr.status}[/{s_color}]",
                f"[dim]{mode}[/dim]",
                s_dur or "—",
                s_started or "—",
                key=stage_name,
            )

        # Seed the detail panel with the first row
        if self._stage_data:
            selected_row = min(selected_row, len(self._stage_data) - 1)
            table.move_cursor(row=selected_row, animate=False)
            self._update_stage_detail(selected_row)
        else:
            self.query_one("#stage-detail", RichLog).clear()
        return attempt.status

    def _update_stage_detail(self, row_idx: int) -> None:
        """Populate the stage-detail panel for the given table row.

        Uses rich.text.Text (no markup parsing) to safely render arbitrary
        output values that may contain '[', ']', '%' and other Rich markup chars.
        """
        from rich.text import Text

        from src.pipeline.run_inspector import _STATUS_COLORS, _STATUS_ICONS, _fmt_duration, _mode_label

        panel = self.query_one("#stage-detail", RichLog)
        panel.clear()

        if row_idx >= len(self._stage_data):
            return

        stage_name, sr = self._stage_data[row_idx]

        if sr is None:
            t = Text()
            t.append(stage_name, style="dim")
            t.append("  pending — not yet executed")
            panel.write(t)
            return

        s_color = _STATUS_COLORS.get(sr.status, "white")
        s_icon = _STATUS_ICONS.get(sr.status, "?")
        s_dur = _fmt_duration(sr.started_at, sr.completed_at)
        mode = _mode_label(sr)
        started = (sr.started_at or "")[:_TIMESTAMP_LEN].replace("T", " ")
        completed = (sr.completed_at or "")[:_TIMESTAMP_LEN].replace("T", " ")

        header_t = Text()
        header_t.append(f"{s_icon} {sr.stage_name}", style="bold")
        header_t.append(f"  {sr.status}", style=s_color)
        header_t.append(f"  {mode}  {s_dur or '—'}", style="dim")
        panel.write(header_t)

        ts_t = Text()
        ts_t.append(f"  Started : {started or '—'}   Finished : {completed or '—'}", style="dim")
        panel.write(ts_t)

        if sr.outputs:
            import json

            panel.write(Text("  Outputs:", style="dim"))
            for key, val in sr.outputs.items():
                key_t = Text()
                key_t.append(f"    {key}", style="cyan")
                if isinstance(val, dict | list):
                    key_t.append(" :", style="dim")
                    panel.write(key_t)
                    for line in json.dumps(val, indent=2, ensure_ascii=False, default=str).splitlines():
                        panel.write(Text(f"      {line}", style="dim"))
                else:
                    key_t.append(f" : {val}", style="dim")
                    panel.write(key_t)

        if sr.status == "failed" and sr.error:
            err_t = Text()
            err_t.append("  Error: ", style="bold red")
            err_t.append(sr.error, style="dim")
            panel.write(err_t)
        elif sr.skip_reason:
            skip_t = Text()
            skip_t.append("  Skip reason: ", style="yellow")
            skip_t.append(sr.skip_reason, style="dim")
            panel.write(skip_t)

    # ── Logs ──────────────────────────────────────────────────────────────────

    def _populate_log_selector(self) -> None:
        attempt_dir = self._run_dir / "attempts" / f"attempt_{self._attempt_no}"
        log_files: dict[str, Path] = {}

        for filename, label in _LOG_CANDIDATES:
            p = attempt_dir / filename
            if p.exists():
                log_files[label] = p

        known = {fname for fname, _ in _LOG_CANDIDATES}
        if attempt_dir.exists():
            for p in sorted(attempt_dir.glob("*.log")):
                if p.name not in known:
                    log_files[p.stem.capitalize()] = p

        self._log_files = log_files
        selector = self.query_one("#log-selector", Select)
        log_widget = self.query_one("#log-content", _LiveLog)
        current_value = None if selector.value is Select.BLANK else str(selector.value)
        option_labels = tuple(log_files)

        if not log_files:
            self._log_option_labels = ()
            log_widget.clear()
            log_widget.write_line("No log files found.")
            return

        if option_labels != self._log_option_labels:
            selector.set_options([(label, label) for label in log_files])
            self._log_option_labels = option_labels
        selected_label = current_value if current_value in log_files else next(iter(log_files))
        if str(selector.value) != selected_label:
            selector.value = selected_label
        if current_value != selected_label:
            self._load_log_file(self._log_files[selected_label], log_widget, reset_follow=True)

    # ── Inference tab (dynamic) ───────────────────────────────────────────────

    def _maybe_add_report_tab(self) -> None:
        if self._report_tab_added:
            return
        attempt_dir = self._run_dir / "attempts" / f"attempt_{self._attempt_no}"
        report_files: dict[str, Path] = {}

        if attempt_dir.is_dir():
            for path in sorted(attempt_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.parent.name == "evaluation":
                    continue
                if "report" not in path.name.lower():
                    continue
                if path.suffix.lower() not in _ARTIFACT_RENDERERS:
                    continue
                label = path.relative_to(attempt_dir).as_posix()
                report_files[label] = path

        if not report_files:
            return

        self._report_files = report_files
        tab_no = self._next_tab_no()
        pane = TabPane(
            f"Report [{tab_no}]",
            Select([], id="report-selector", prompt="Select report artifact…"),
            MarkdownViewer(
                "",
                id="report-markdown",
                show_table_of_contents=False,
            ),
            RichLog(id="report-text", highlight=True, markup=True, wrap=True, auto_scroll=False),
            id="report",
        )
        self.query_one(TabbedContent).add_pane(pane)
        self._report_tab_added = True
        self.call_after_refresh(self._populate_report_selector)

    def _populate_report_selector(self) -> None:
        try:
            selector = self.query_one("#report-selector", Select)
        except Exception:
            return
        if not self._report_files:
            return
        selector.set_options([(label, label) for label in self._report_files])
        first_label = next(iter(self._report_files))
        selector.value = first_label
        self._load_report_file(first_label)

    def _maybe_add_inference_tab(self) -> None:
        if self._inference_tab_added:
            return
        attempt_dir = self._run_dir / "attempts" / f"attempt_{self._attempt_no}"
        manifest_path = attempt_dir / "inference" / "inference_manifest.json"
        if not manifest_path.exists():
            return

        chat_script = attempt_dir / "inference" / "chat_inference.py"
        self._chat_script = chat_script if chat_script.exists() else None

        tab_no = self._next_tab_no()
        pane = TabPane(f"Inference [{tab_no}]", Static(id="inference-content", expand=True), id="inference")
        self.query_one(TabbedContent).add_pane(pane)
        self._inference_tab_added = True
        self.call_after_refresh(self._load_inference_manifest, manifest_path)

    def _load_inference_manifest(self, manifest_path: Path) -> None:
        import json

        try:
            widget = self.query_one("#inference-content", Static)
        except Exception:
            return

        try:
            with manifest_path.open(encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception as exc:
            widget.update(f"[red]Cannot read manifest: {exc}[/red]")
            return

        provider = m.get("provider", "—")
        engine = m.get("engine", "—")
        model_block = m.get("model", {})
        base_model = model_block.get("base_model_id", "—")
        adapter = model_block.get("adapter_ref", "—")
        vllm_block = m.get("vllm", {})
        serve_image = vllm_block.get("serve_image", "—")
        endpoint_block = m.get("endpoint", {})
        client_url = endpoint_block.get("client_base_url", "—")
        config_hash = m.get("config_hash", "—")

        lines = [
            "[bold cyan]Inference Manifest[/bold cyan]",
            "",
            f"  Provider    : [dim]{provider}[/dim]",
            f"  Engine      : [dim]{engine}[/dim]",
            f"  Base model  : [dim]{base_model}[/dim]",
            f"  Adapter     : [dim]{adapter}[/dim]",
            f"  vLLM image  : [dim]{serve_image}[/dim]",
            f"  Endpoint    : [dim]{client_url}[/dim]",
            f"  Config hash : [dim]{config_hash}[/dim]",
            "",
            "[dim]─────────────────────────────────────────────────[/dim]",
            "",
        ]

        if self._chat_script:
            lines += [
                f"  Script : [dim]{self._chat_script.name}[/dim]",
                "  Press [bold green]c[/bold green] to launch interactive chat.",
                "  [dim](TUI suspends — chat runs in full terminal)[/dim]",
            ]
        else:
            lines.append("  [dim](chat_inference.py not found)[/dim]")

        widget.update("\n".join(lines))

    # ── Eval tab (dynamic) ────────────────────────────────────────────────────

    def _maybe_add_eval_tab(self) -> None:
        if self._eval_tab_added:
            return
        eval_dir = self._run_dir / "attempts" / f"attempt_{self._attempt_no}" / "evaluation"
        if not eval_dir.is_dir():
            return

        md_files = sorted(eval_dir.glob("*.md"))
        if not md_files:
            return

        self._eval_files = {p.stem.replace("_", " ").title(): p for p in md_files}

        tab_no = self._next_tab_no()
        pane = TabPane(
            f"Eval [{tab_no}]",
            Select([], id="eval-selector", prompt="Select report…"),
            MarkdownViewer(
                "",
                id="eval-markdown",
                show_table_of_contents=False,
            ),
            RichLog(id="eval-text", highlight=True, markup=True, wrap=True, auto_scroll=False),
            id="eval",
        )
        self.query_one(TabbedContent).add_pane(pane)
        self._eval_tab_added = True
        self.call_after_refresh(self._populate_eval_selector)

    def _populate_eval_selector(self) -> None:
        try:
            selector = self.query_one("#eval-selector", Select)
        except Exception:
            return
        if not self._eval_files:
            return
        selector.set_options([(label, label) for label in self._eval_files])
        first_label = next(iter(self._eval_files))
        selector.value = first_label
        self._load_eval_file(first_label)

    def _load_eval_file(self, label: str) -> None:
        path = self._eval_files.get(label)
        self._render_artifact_file(
            path=path,
            markdown_id="#eval-markdown",
            text_id="#eval-text",
            missing_message="[dim]File not found.[/dim]",
            error_prefix="Cannot read",
        )

    def _set_artifact_view_mode(self, markdown_id: str, text_id: str, mode: str) -> None:
        markdown_viewer = self.query_one(markdown_id, MarkdownViewer)
        text_viewer = self.query_one(text_id, RichLog)
        markdown_viewer.display = mode == "markdown"
        text_viewer.display = mode != "markdown"

    def _render_artifact_file(
        self,
        *,
        path: Path | None,
        markdown_id: str,
        text_id: str,
        missing_message: str,
        error_prefix: str,
    ) -> None:
        if path is None or not path.exists():
            with contextlib.suppress(Exception):
                self._set_artifact_view_mode(markdown_id, text_id, "text")
                text_viewer = self.query_one(text_id, RichLog)
                text_viewer.clear()
                text_viewer.write(missing_message)
            return

        renderer = _ARTIFACT_RENDERERS.get(path.suffix.lower(), "text")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if renderer == "markdown":
                self._set_artifact_view_mode(markdown_id, text_id, "markdown")
                viewer = self.query_one(markdown_id, MarkdownViewer)
                normalized = _normalize_markdown_for_viewer(content)
                self.call_after_refresh(lambda: viewer.document.update(normalized))
            else:
                self._set_artifact_view_mode(markdown_id, text_id, "text")
                text_viewer = self.query_one(text_id, RichLog)
                text_viewer.clear()
                for line in content.splitlines():
                    text_viewer.write(line)
                if not content:
                    text_viewer.write("[dim]File is empty.[/dim]")
        except Exception as exc:
            self._set_artifact_view_mode(markdown_id, text_id, "text")
            text_viewer = self.query_one(text_id, RichLog)
            text_viewer.clear()
            text_viewer.write(f"[red]{error_prefix}: {exc}[/red]")

    def _load_report_file(self, label: str) -> None:
        path = self._report_files.get(label)
        self._render_artifact_file(
            path=path,
            markdown_id="#report-markdown",
            text_id="#report-text",
            missing_message="[dim]Report file not found.[/dim]",
            error_prefix="Cannot read report",
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _load_log_file(self, path: Path, widget: _LiveLog, *, reset_follow: bool) -> None:
        widget.clear()
        if reset_follow:
            self._log_auto_follow = True
        widget.auto_scroll = self._log_auto_follow
        widget.set_highlighting(self._log_highlight_enabled)
        widget.set_word_wrap(self._log_word_wrap_enabled, preserve_viewport=False)
        try:
            lines = self._live_log_tail.load_full(path)
            if lines:
                widget.write_lines(lines, scroll_end=self._log_auto_follow)
        except OSError as exc:
            widget.write_line(f"Cannot read: {exc}")

    def _reload_current_log(self, *, reset_follow: bool = False, preserve_viewport: bool = False) -> None:
        with contextlib.suppress(Exception):
            selector = self.query_one("#log-selector", Select)
            if selector.value is Select.BLANK:
                return
            path = self._log_files.get(str(selector.value))
            if path is None:
                return
            widget = self.query_one("#log-content", _LiveLog)
            scroll_x, scroll_y = widget.scroll_offset
            should_stick_to_end = self._log_auto_follow and widget.is_vertical_scroll_end
            self._load_log_file(path, widget, reset_follow=reset_follow)
            if preserve_viewport and not should_stick_to_end:
                target_x = 0 if self._log_word_wrap_enabled else scroll_x
                self.call_after_refresh(lambda: widget.scroll_to(x=target_x, y=scroll_y, animate=False, immediate=True))

    def _poll_live_updates(self) -> None:
        if not self._live_updates_active:
            return
        try:
            from typing import cast

            app = cast("RyotenkaiApp", self.app)
            should_refresh_attempt = (
                self._attempt_status == "running" or app.get_active_launch_for_run(self._run_dir) is not None
            )
            if should_refresh_attempt:
                self._attempt_status = self._load_details()
                self._populate_log_selector()
                self._maybe_add_inference_tab()
                self._maybe_add_eval_tab()
                self._maybe_add_report_tab()
                self.refresh_bindings()
            tabs = self.query_one(TabbedContent)
            if tabs.active != "logs":
                return
            selector = self.query_one("#log-selector", Select)
            if selector.value is Select.BLANK:
                return
            path = self._log_files.get(str(selector.value))
            if path is None:
                return
            log_widget = self.query_one("#log-content", _LiveLog)
            if self._log_auto_follow and self.focused is log_widget and not log_widget.is_vertical_scroll_end:
                self._log_auto_follow = False
            elif not self._log_auto_follow and self.focused is log_widget and log_widget.is_vertical_scroll_end:
                self._log_auto_follow = True
            log_widget.auto_scroll = self._log_auto_follow
            new_lines = self._live_log_tail.read_new_lines()
            if new_lines:
                log_widget.write_lines(new_lines)
        except Exception:
            return

    def _next_tab_no(self) -> int:
        self._tab_counter += 1
        return self._tab_counter

    # ── Focus traversal ───────────────────────────────────────────────────────

    def on_key(self, event) -> None:
        """Tabs navigation (via mixin) + g/G scroll shortcuts on scrollable text viewers."""
        super().on_key(event)
        if isinstance(self.focused, RichLog | Log):
            if self.focused.id == "log-content":
                if event.key in {"up", "pageup", "home"}:
                    self._log_auto_follow = False
                elif (event.key in {"down", "pagedown"} and self.focused.is_vertical_scroll_end) or event.key in {
                    "end",
                    "G",
                }:
                    self._log_auto_follow = True
            if event.key == "g":
                self.focused.scroll_home(animate=False)
                event.stop()
            elif event.key == "G":
                self.focused.scroll_end(animate=False)
                event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        del event
        if self.focused is self.query_one("#log-content", _LiveLog):
            self._log_auto_follow = False

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        del event
        log_widget = self.query_one("#log-content", _LiveLog)
        if self.focused is log_widget and log_widget.is_vertical_scroll_end:
            self._log_auto_follow = True

    # ── Events ────────────────────────────────────────────────────────────────

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update stage detail panel as user navigates the stages table."""
        if event.data_table.id == "stages-table":
            self._update_stage_detail(event.cursor_row)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a stage row → focus the detail panel so user can scroll it."""
        if event.data_table.id == "stages-table":
            with contextlib.suppress(Exception):
                self.query_one("#stage-detail", RichLog).focus()
            event.stop()

    @on(Select.Changed, "#log-selector")
    def _log_selector_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            path = self._log_files.get(str(event.value))
            if path:
                log_widget = self.query_one("#log-content", _LiveLog)
                self._load_log_file(path, log_widget, reset_follow=True)
                # Only shift focus when the user manually changed the selector,
                # not during programmatic initialisation (selector not focused then).
                if self.focused is event.control:
                    log_widget.focus()

    @on(Select.Changed, "#eval-selector")
    def _eval_selector_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._load_eval_file(str(event.value))
            with contextlib.suppress(Exception):
                if self.focused is event.control:
                    if self.query_one("#eval-markdown", MarkdownViewer).display:
                        self.query_one("#eval-markdown", MarkdownViewer).focus()
                    else:
                        self.query_one("#eval-text", RichLog).focus()

    @on(Select.Changed, "#report-selector")
    def _report_selector_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._load_report_file(str(event.value))
            with contextlib.suppress(Exception):
                if self.focused is event.control:
                    if self.query_one("#report-markdown", MarkdownViewer).display:
                        self.query_one("#report-markdown", MarkdownViewer).focus()
                    else:
                        self.query_one("#report-text", RichLog).focus()

    @on(TabbedContent.TabActivated)
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        del event
        self.refresh_bindings()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_go_back(self) -> None:
        self.app.sub_title = self._run_dir.name
        self.app.pop_screen()

    def action_launch_chat(self) -> None:
        if self._chat_script is None or not self._chat_script.exists():
            self.notify("chat_inference.py not found for this attempt", severity="warning")
            return
        with self.app.suspend():
            subprocess.run(["python3", str(self._chat_script)], check=False)

    def action_stop_run(self) -> None:
        self._confirm_interrupt_run(self._run_dir)

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id
        self.refresh_bindings()

    def action_toggle_log_highlight(self) -> None:
        self._log_highlight_enabled = not self._log_highlight_enabled
        with contextlib.suppress(Exception):
            self.query_one("#log-content", _LiveLog).set_highlighting(self._log_highlight_enabled)
        state = "on" if self._log_highlight_enabled else "off"
        self.notify(f"Log highlighting {state}")

    def action_toggle_log_wrap(self) -> None:
        self._log_word_wrap_enabled = not self._log_word_wrap_enabled
        with contextlib.suppress(Exception):
            self.query_one("#log-content", _LiveLog).set_word_wrap(self._log_word_wrap_enabled, preserve_viewport=True)
        state = "on" if self._log_word_wrap_enabled else "off"
        self.notify(f"Log word wrap {state}")

    def action_browse_config(self) -> None:
        from src.tui.screens.config_browser_modal import StructuredConfigBrowser

        if self._config_path is None:
            self.notify("Config path is unavailable for this attempt", severity="warning")
            return
        self.app.push_screen(StructuredConfigBrowser(self._config_path))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "stop_run":
            from typing import cast

            return cast("RyotenkaiApp", self.app).can_interrupt_run(self._run_dir)
        if action == "browse_config":
            return self._config_path is not None
        if action == "toggle_log_highlight":
            with contextlib.suppress(Exception):
                tabs = self.query_one(TabbedContent)
                return tabs.active == "logs" and bool(self._log_files)
            return False
        if action == "toggle_log_wrap":
            with contextlib.suppress(Exception):
                tabs = self.query_one(TabbedContent)
                return tabs.active == "logs" and bool(self._log_files)
            return False
        return True
