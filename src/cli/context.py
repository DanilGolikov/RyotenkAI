"""Shared CLI context: output mode, color, verbosity.

Populated once by the root Typer callback and stored on
``typer.Context.obj`` so downstream commands can read global flags
without redeclaring them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal

OutputMode = Literal["text", "json", "yaml"]


@dataclass(slots=True)
class CLIContext:
    """Global CLI state accessible to every command."""

    output: OutputMode = "text"
    color: bool = True
    verbose: int = 0               # -v → 1, -vv → 2
    quiet: bool = False
    log_level: str | None = None   # None = leave logger alone
    project_id: str | None = None  # --project flag / RYOTENKAI_PROJECT / context store
    remote_url: str | None = None  # --remote URL — reserved, NotImplementedError stub

    # Derived at build time, not user-facing:
    stdout_is_tty: bool = field(default_factory=lambda: sys.stdout.isatty())
    stderr_is_tty: bool = field(default_factory=lambda: sys.stderr.isatty())

    @property
    def use_color(self) -> bool:
        """Effective colour flag: user-disable OR pipe OR NO_COLOR env wins."""
        if not self.color:
            return False
        if "NO_COLOR" in os.environ:
            return False
        return self.stdout_is_tty

    @property
    def is_json(self) -> bool:
        return self.output == "json"

    @property
    def is_yaml(self) -> bool:
        return self.output == "yaml"

    @property
    def is_machine_readable(self) -> bool:
        """True when output should be parsed by a tool, not a human."""
        return self.output in ("json", "yaml")


def default_context() -> CLIContext:
    """Context used when commands are invoked without the root callback
    running first — e.g. in unit tests via CliRunner for a sub-command."""
    return CLIContext()


__all__ = ["CLIContext", "OutputMode", "default_context"]
