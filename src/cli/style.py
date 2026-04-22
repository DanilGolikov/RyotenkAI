"""Central styling: one Rich ``Console`` instance, status icons, colour names.

Importing this module should not have side-effects beyond instantiating two
``Console`` singletons; the actual print calls happen from rendering code.
All command modules should use ``console`` / ``err_console`` from here
instead of constructing new ones, so colour / NO_COLOR handling is consistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rich.console import Console

# ---------------------------------------------------------------------------
# Console singletons
# ---------------------------------------------------------------------------

# ``Console`` auto-detects TTY and honours NO_COLOR by default. We still
# pass ``no_color=True`` explicitly when NO_COLOR is set so other libs that
# ignore the env var don't re-enable colour via the same Console.
_NO_COLOR_ENV = "NO_COLOR" in os.environ

console = Console(no_color=_NO_COLOR_ENV)
err_console = Console(stderr=True, no_color=_NO_COLOR_ENV)


def reconfigure(*, color: bool) -> None:
    """Re-init the console singletons after the root callback resolves the
    effective colour policy (CLI flag > env > TTY)."""
    global console, err_console
    console = Console(no_color=not color)
    err_console = Console(stderr=True, no_color=not color)


# ---------------------------------------------------------------------------
# Icons — single source of truth for status markers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Icons:
    ok: str = "✓"
    warn: str = "⚠"
    err: str = "✗"
    dim: str = "·"
    will: str = "~"     # would-change in dry-run
    arrow: str = "→"

    # Pipeline stage statuses (mirrors the previous run_rendering constants)
    stage_completed: str = "◉"
    stage_failed: str = "◉"         # same glyph, coloured red by renderer
    stage_running: str = "▸"
    stage_interrupted: str = "◈"
    stage_stale: str = "◌"
    stage_skipped: str = "◇"
    stage_pending: str = "○"


ICONS = Icons()


# ---------------------------------------------------------------------------
# Colour names (Rich markup style strings)
# ---------------------------------------------------------------------------

COLOR_OK = "green"
COLOR_WARN = "yellow"
COLOR_ERR = "red"
COLOR_DIM = "dim"
COLOR_LABEL = "bold"
COLOR_CHANGE = "cyan"


__all__ = [
    "COLOR_CHANGE",
    "COLOR_DIM",
    "COLOR_ERR",
    "COLOR_LABEL",
    "COLOR_OK",
    "COLOR_WARN",
    "ICONS",
    "Icons",
    "console",
    "err_console",
    "reconfigure",
]
