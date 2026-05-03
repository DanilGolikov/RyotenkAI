"""Phase 1 foundation tests — root callback, renderer, errors."""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Root callback — global flags and version
# ---------------------------------------------------------------------------


def test_root_version_flag(runner: CliRunner) -> None:
    from src.main import app

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "ryotenkai " in result.output
    assert "python " in result.output
    assert "platform " in result.output


def test_root_short_version_flag(runner: CliRunner) -> None:
    from src.main import app

    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0, result.output
    assert "ryotenkai " in result.output


def test_version_command_matches_flag(runner: CliRunner) -> None:
    """`ryotenkai version` and `ryotenkai --version` print the same first line."""
    from src.main import app

    a = runner.invoke(app, ["version"])
    b = runner.invoke(app, ["--version"])
    assert a.exit_code == b.exit_code == 0
    assert a.output.strip().splitlines()[0] == b.output.strip().splitlines()[0]


def test_version_command_json(runner: CliRunner) -> None:
    from src.main import app

    result = runner.invoke(app, ["-o", "json", "version"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "ryotenkai" in payload
    assert "python" in payload
    assert "platform" in payload


def test_help_alias(runner: CliRunner) -> None:
    from src.main import app

    for args in (["help"], ["--help"], ["-h"]):
        out = runner.invoke(app, args)
        assert out.exit_code == 0, f"{args} → {out.output}"
        assert "Usage:" in out.output


def test_invalid_output_mode_gives_clean_error(runner: CliRunner) -> None:
    from src.main import app

    result = runner.invoke(app, ["-o", "xml", "version"])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "invalid --output" in combined


# ---------------------------------------------------------------------------
# Renderer — TextRenderer and JsonRenderer
# ---------------------------------------------------------------------------


def test_text_renderer_prints_table(capsys: pytest.CaptureFixture[str]) -> None:
    from src.cli.renderer import TextRenderer

    renderer = TextRenderer()
    renderer.table(
        headers=["name", "status"],
        rows=[("a", "ok"), ("b", "fail")],
    )
    renderer.flush()
    captured = capsys.readouterr()
    # Rich may not emit ANSI under CliRunner's capture, but the data is there.
    assert "name" in captured.out
    assert "status" in captured.out
    assert "a" in captured.out
    assert "fail" in captured.out


def test_json_renderer_buffers_and_flushes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.cli.renderer import JsonRenderer

    renderer = JsonRenderer()
    renderer.table(
        headers=["name", "status"],
        rows=[("a", "ok"), ("b", "fail")],
    )
    renderer.flush()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload == [
        {"name": "a", "status": "ok"},
        {"name": "b", "status": "fail"},
    ]


def test_json_renderer_refuses_double_emit() -> None:
    from src.cli.renderer import JsonRenderer

    renderer = JsonRenderer()
    renderer.emit({"a": 1})
    with pytest.raises(RuntimeError):
        renderer.emit({"b": 2})


def test_json_renderer_flush_without_emit_is_noop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src.cli.renderer import JsonRenderer

    renderer = JsonRenderer()
    renderer.flush()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_get_renderer_picks_by_output_mode() -> None:
    from src.cli.context import CLIContext
    from src.cli.renderer import JsonRenderer, TextRenderer, get_renderer

    assert isinstance(get_renderer(CLIContext(output="text")), TextRenderer)
    assert isinstance(get_renderer(CLIContext(output="json")), JsonRenderer)


# ---------------------------------------------------------------------------
# Context — colour / TTY / NO_COLOR interaction
# ---------------------------------------------------------------------------


def test_context_no_color_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.context import CLIContext

    monkeypatch.setenv("NO_COLOR", "1")
    ctx = CLIContext(color=True, stdout_is_tty=True)
    assert ctx.use_color is False


def test_context_color_flag_off_wins_over_tty() -> None:
    from src.cli.context import CLIContext

    ctx = CLIContext(color=False, stdout_is_tty=True)
    assert ctx.use_color is False


def test_context_no_tty_disables_color() -> None:
    from src.cli.context import CLIContext

    ctx = CLIContext(color=True, stdout_is_tty=False)
    assert ctx.use_color is False


# ---------------------------------------------------------------------------
# Errors — die() and suggest_hint()
# ---------------------------------------------------------------------------


def test_die_raises_typer_exit(capsys: pytest.CaptureFixture[str]) -> None:
    from src.cli.errors import die

    with pytest.raises(typer.Exit) as exc:
        die("something wrong", hint="try again")
    assert exc.value.exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "something wrong" in captured.err
    assert "hint:" in captured.err
    assert "try again" in captured.err


def test_suggest_hint_single_match() -> None:
    from src.cli.errors import suggest_hint

    hint = suggest_hint("sycn", ["sync", "scaffold", "pack"])
    assert hint is not None
    assert "sync" in hint


def test_suggest_hint_no_match() -> None:
    from src.cli.errors import suggest_hint

    hint = suggest_hint("zzqqwe", ["sync", "scaffold", "pack"])
    assert hint is None
