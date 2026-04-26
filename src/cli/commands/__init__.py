"""Sub-Typer registry for the CLI.

Each noun lives in its own module (``run.py``, ``runs.py``, ...) and
exports a ``typer.Typer`` instance plus the noun name to mount it under.
This module wires them all into the root Typer in :mod:`src.cli.app`.

Adding a new noun:

1. Create ``src/cli/commands/<noun>.py``.
2. Define ``<noun>_app: typer.Typer`` and decorate commands on it.
3. Append ``(<noun>_app, "<noun>")`` to :data:`_REGISTRY` below.

Order matters only for help-screen display order — Typer sorts groups
alphabetically by default; we keep the workflow nouns first
(``run`` / ``runs`` / ``config`` / ...).
"""

from __future__ import annotations

import typer

from src.cli.commands import (
    config as _config_mod,
)
from src.cli.commands import (
    dataset as _dataset_mod,
)
from src.cli.commands import (
    plugin as _plugin_mod,
)
from src.cli.commands import (
    preset as _preset_mod,
)
from src.cli.commands import (
    project as _project_mod,
)
from src.cli.commands import (
    run as _run_mod,
)
from src.cli.commands import (
    runs as _runs_mod,
)
from src.cli.commands import (
    server as _server_mod,
)
from src.cli.commands import (
    smoke as _smoke_mod,
)
from src.cli.commands import (
    version as _version_mod,
)

#: Mounting registry — ``(sub_app, name)`` pairs in user-facing order.
_REGISTRY: list[tuple[typer.Typer, str]] = [
    (_run_mod.run_app,       "run"),
    (_runs_mod.runs_app,     "runs"),
    (_config_mod.config_app, "config"),
    (_dataset_mod.dataset_app, "dataset"),
    (_project_mod.project_app, "project"),
    (_plugin_mod.plugin_app, "plugin"),
    (_preset_mod.preset_app, "preset"),
    (_smoke_mod.smoke_app,   "smoke"),
    (_server_mod.server_app, "server"),
]


def register_all(app: typer.Typer) -> None:
    """Mount every sub-Typer + the standalone ``version`` command."""
    for sub_app, name in _REGISTRY:
        app.add_typer(sub_app, name=name)
    # ``version`` is a single-leaf command (no verbs underneath), so it
    # registers directly on the root rather than as a sub-Typer.
    _version_mod.register(app)


__all__ = ["register_all"]
