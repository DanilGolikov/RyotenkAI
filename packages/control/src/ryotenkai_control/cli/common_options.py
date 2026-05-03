"""Shared ``Annotated[...]`` option / argument definitions for CLI commands.

Typer's idiomatic way to keep option signatures consistent across many
commands is ``typing.Annotated`` aliases. Without a single source of
truth, every command grows its own subtly-different ``--config -c`` /
``--run-dir`` / ``--kind`` declaration; help texts drift, autocompletion
diverges, and renames touch every file.

This module pins the canonical shape of each option once. Command
modules import the alias and use it as a type annotation:

    from src.cli.common_options import ConfigOpt, RunDirArg

    @runs_app.command("inspect")
    def inspect_cmd(
        run_dir: RunDirArg,
        config: ConfigOpt = None,
    ) -> None:
        ...

Conventions:
- Aliases describing **arguments** end in ``Arg`` (positional, required
  unless their default is set in the call site).
- Aliases describing **options** end in ``Opt`` and accept ``None`` so
  the call-site can leave them unset.
- Where the option needs different help text per command, the alias
  carries the most-common phrasing; per-command commands can override
  by re-declaring locally — the alias is a default, not a chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from src.community.constants import ALL_PLUGIN_KINDS

# ---------------------------------------------------------------------------
# Config / run-dir / project — most frequently shared across commands
# ---------------------------------------------------------------------------

ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c",
        help="Path to pipeline config YAML.",
        exists=False,        # callers want their own existence check / message
        dir_okay=False,
        resolve_path=True,
    ),
]
"""Optional ``--config / -c`` path. Validate existence inside the command
to keep error wording domain-specific (config vs preset vs manifest)."""

RequiredConfigOpt = Annotated[
    Path,
    typer.Option(
        "--config", "-c",
        help="Path to pipeline config YAML.",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]
"""Same as :data:`ConfigOpt` but Typer enforces existence — use it on
write commands (``run start``, ``dataset validate``) where the path is
mandatory."""

RunDirArg = Annotated[
    Path,
    typer.Argument(
        ...,
        help="Path to an existing logical run directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
]
"""Positional run directory — used by every ``runs <verb>`` command and
by ``run resume / restart / interrupt``."""

OptionalRunDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--run-dir",
        help=(
            "Existing logical run directory (for resume / restart). When "
            "omitted, the command operates on a fresh run."
        ),
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
]
"""Optional ``--run-dir`` for ``run start`` (resume / restart paths)."""

ProjectOpt = Annotated[
    str | None,
    typer.Option(
        "--project", "-p",
        envvar="RYOTENKAI_PROJECT",
        help=(
            "Project id. Falls back to the value persisted by "
            "`ryotenkai project use` when omitted."
        ),
    ),
]
"""Project id override. Resolution order is: this flag > ``RYOTENKAI_PROJECT``
env > ``cli_state.context_store.get_current_project()`` > error."""

# ---------------------------------------------------------------------------
# Plugin / preset
# ---------------------------------------------------------------------------

PluginKindOpt = Annotated[
    Literal[*ALL_PLUGIN_KINDS] | None,  # type: ignore[valid-type]
    typer.Option(
        "--kind",
        help=f"Filter by plugin kind: {' | '.join(ALL_PLUGIN_KINDS)}.",
        case_sensitive=False,
    ),
]
"""Optional ``--kind`` filter for ``plugin ls / show``."""

# ---------------------------------------------------------------------------
# Lifecycle: dry-run, force
# ---------------------------------------------------------------------------

ForceOpt = Annotated[
    bool,
    typer.Option(
        "--force", "-f",
        help="Overwrite existing artefact without prompting.",
    ),
]
"""Boolean ``--force`` flag for write commands that protect against
accidental overwrite (``plugin install``, ``plugin pack``, ``preset
apply --write``, ...)."""

DryRunOpt = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Show what would happen without making any changes.",
    ),
]
"""Boolean ``--dry-run`` flag for write commands."""

YesOpt = Annotated[
    bool,
    typer.Option(
        "--yes", "-y",
        help="Skip the confirmation prompt for destructive operations.",
    ),
]
"""Boolean ``--yes`` for `runs rm`, `project rm`, `plugin uninstall`-style
flows where Typer would otherwise prompt."""


__all__ = [
    "ConfigOpt",
    "DryRunOpt",
    "ForceOpt",
    "OptionalRunDirOpt",
    "PluginKindOpt",
    "ProjectOpt",
    "RequiredConfigOpt",
    "RunDirArg",
    "YesOpt",
]
