"""Smoke gate for the CLI surface (plan B Q-30 + NR-30).

For every ``(noun, verb)`` pair registered on the root Typer, invoking
``--help`` must:

- exit 0,
- print a ``Usage:`` banner,
- not leak a Python traceback (no ``Traceback`` substring).

This catches the most common regression from a refactor of this size:
an import-time error in one of the ``commands/<noun>.py`` modules
silently wedging the whole tree because Typer fails to mount the
group. Running ``pytest src/tests/smoke -q`` is the cheapest possible
"is the CLI alive?" check — it never spins up the orchestrator, never
hits the disk, never imports torch.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from src.cli.app import app


def _all_command_paths() -> list[list[str]]:
    """Discover every ``(noun, verb)`` pair the root Typer knows about.

    Does NOT touch the network / disk — purely walks the in-memory
    Typer tree. Excludes the eager ``--version`` flag and Typer's
    auto-generated ``--install-completion`` / ``--show-completion``
    helpers, which are not real commands.
    """
    paths: list[list[str]] = [[]]  # the root --help itself
    for group in app.registered_groups:
        name = group.name or (
            group.typer_instance.info.name if group.typer_instance else None
        )
        if not name:
            continue
        paths.append([name])
        sub_app = group.typer_instance
        if sub_app is None:
            continue
        for cmd in sub_app.registered_commands:
            cmd_name = cmd.name or cmd.callback.__name__.replace("_", "-")
            if cmd_name == "help":
                continue
            paths.append([name, cmd_name])
    for cmd in app.registered_commands:
        cmd_name = cmd.name or cmd.callback.__name__.replace("_", "-")
        if cmd_name == "help":
            continue
        paths.append([cmd_name])
    return paths


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("path", _all_command_paths(), ids=lambda p: " ".join(p) or "(root)")
def test_help_renders_for(path: list[str], runner: CliRunner) -> None:
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, (
        f"`ryotenkai {' '.join(path)} --help` exited {result.exit_code}\n"
        f"stdout:\n{result.stdout}\n"
    )
    assert "Usage:" in result.stdout
    assert "Traceback" not in result.stdout


def test_root_version_flag_works(runner: CliRunner) -> None:
    """``ryotenkai --version`` short-circuits before the callback."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ryotenkai " in result.stdout
    assert "python " in result.stdout


def test_root_invalid_output_mode(runner: CliRunner) -> None:
    result = runner.invoke(app, ["-o", "xml", "version"])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "invalid --output" in combined


def test_root_remote_flag_is_reserved(runner: CliRunner) -> None:
    """``--remote`` is reserved for v1.2 — must fail fast with NotImplementedError."""
    result = runner.invoke(app, ["--remote", "http://x", "version"])
    assert result.exit_code != 0
    # Typer wraps NotImplementedError in its exception class; the message
    # propagates to stderr / output via Click.
    combined = result.stdout + (result.stderr or "")
    assert "remote" in combined.lower() or result.exception is not None
