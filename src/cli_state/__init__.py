"""CLI-local persistent state.

Holds tiny key-value stores that the CLI alone cares about — currently
the "current project" pointer used by ``ryotenkai project use`` so
follow-up commands inherit the active project without an explicit
``--project`` flag every time.

Storage lives under ``${RYOTENKAI_HOME:-~/.ryotenkai}/`` and never
touches the API, the workspace registries, or the Web UI. Web has its
own URL-based context; the two are intentionally independent.
"""

from src.cli_state.context_store import (
    CLIContext as PersistedCLIContext,
)
from src.cli_state.context_store import (
    clear_current_project,
    cli_context_path,
    get_current_project,
    set_current_project,
)

__all__ = [
    "PersistedCLIContext",
    "clear_current_project",
    "cli_context_path",
    "get_current_project",
    "set_current_project",
]
