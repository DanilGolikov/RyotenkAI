"""``ryotenkai version`` — print version info as text or JSON/YAML.

Mirrors the ``--version`` eager flag, but goes through the renderer so
``-o json`` / ``-o yaml`` produces a parseable document instead of the
human-readable single line.
"""

from __future__ import annotations

import typer

from src.cli.context import CLIContext
from src.cli.renderer import get_renderer
from src.cli.version import collect_version_info


def _version_cmd(ctx: typer.Context) -> None:
    """Show version info (ryotenkai / python / platform / git sha)."""
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    info = collect_version_info()
    if state.is_machine_readable:
        renderer.emit({
            "ryotenkai": info.ryotenkai,
            "python": info.python,
            "platform": info.platform,
            "git_sha": info.git_sha,
        })
    else:
        renderer.text(info.format())
    renderer.flush()


def register(app: typer.Typer) -> None:
    """Mount ``version`` directly on the root Typer."""
    app.command("version")(_version_cmd)


__all__ = ["register"]
