"""Persistent CLI context — currently just the active project pointer.

The CLI mirrors kubectl's ``current-context`` idea: ``ryotenkai project
use <id>`` writes a single short JSON file, and every follow-up command
that takes ``--project`` falls back to the stored value when the flag is
absent. The store is intentionally minimal — anything richer (user
preferences, telemetry opt-out, …) lives in separate keys here, never in
implicit env vars.

Storage path:
  ``${RYOTENKAI_HOME:-~/.ryotenkai}/cli-context.json``

Disk shape (illustrative)::

    {
      "current_project_id": "my-experiment",
      "set_at": "2026-04-26T08:30:00Z"
    }

Writes are atomic (temp-file + rename) via :func:`src.utils.atomic_fs.atomic_write_json`,
so a Ctrl-C mid-write cannot leave a half-written or corrupt file —
either the previous value survives or the new value is fully on disk.
Reads tolerate a missing file (fresh install) and a corrupt file
(returns ``None``, leaves the file in place so the user can inspect it).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.atomic_fs import atomic_write_json, utc_now_iso

#: Env var that overrides ``~/.ryotenkai`` as the CLI's home directory.
#: Mirrors what ``RYOTENKAI_RUNS_DIR`` does for runs storage; lets tests
#: point the store at a tmp_path without monkeypatching ``Path.home``.
HOME_ENV: str = "RYOTENKAI_HOME"

#: Filename inside the CLI home dir. Single source of truth — used by
#: tests and by the ``project use`` command.
CONTEXT_FILENAME: str = "cli-context.json"

#: Top-level keys in the JSON document. Centralised so renames don't
#: silently break readers.
KEY_CURRENT_PROJECT: str = "current_project_id"
KEY_SET_AT: str = "set_at"


@dataclass(frozen=True, slots=True)
class CLIContext:
    """In-memory snapshot of the persisted context.

    Returned by :func:`load_context` so callers can read all fields with
    one disk hit. Frozen — mutate via the helper functions in this
    module, which write atomically.
    """

    current_project_id: str | None = None
    set_at: str | None = None


def cli_home() -> Path:
    """Return the CLI's home directory, honouring ``RYOTENKAI_HOME``.

    Does NOT create the directory — write helpers do that lazily so a
    plain read on a fresh install never touches the filesystem.
    """
    override = os.environ.get(HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ryotenkai"


def cli_context_path() -> Path:
    """Full path to ``cli-context.json``."""
    return cli_home() / CONTEXT_FILENAME


def load_context() -> CLIContext:
    """Read the persisted context. Returns an empty :class:`CLIContext`
    when the file is missing, malformed, or unreadable.

    Malformed reads are intentionally non-fatal: the CLI must keep
    working even if the user (or a stray editor) breaks the JSON. The
    file is left in place so ``cat ~/.ryotenkai/cli-context.json``
    still shows the user what's wrong.
    """
    path = cli_context_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return CLIContext()
    except OSError:
        return CLIContext()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return CLIContext()
    if not isinstance(payload, dict):
        return CLIContext()
    return CLIContext(
        current_project_id=_safe_str(payload.get(KEY_CURRENT_PROJECT)),
        set_at=_safe_str(payload.get(KEY_SET_AT)),
    )


def get_current_project() -> str | None:
    """Convenience: read just the active project id."""
    return load_context().current_project_id


def set_current_project(project_id: str) -> None:
    """Persist ``project_id`` as the new active project.

    Writes atomically — interrupted runs keep the previous content. The
    parent directory is created on demand. Empty / whitespace-only ids
    are rejected to avoid creating "ghost" contexts that look set but
    resolve to nothing.
    """
    cleaned = project_id.strip()
    if not cleaned:
        raise ValueError("project_id must not be empty")

    payload = {
        KEY_CURRENT_PROJECT: cleaned,
        KEY_SET_AT: utc_now_iso(),
    }
    atomic_write_json(cli_context_path(), payload)


def clear_current_project() -> None:
    """Remove the persisted context file, if any.

    No-op when the file doesn't exist — matches ``rm -f`` semantics so
    callers don't have to pre-check.
    """
    try:
        cli_context_path().unlink()
    except FileNotFoundError:
        return


def _safe_str(value: object) -> str | None:
    """Return ``value`` as a non-empty string or ``None``."""
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


__all__ = [
    "CONTEXT_FILENAME",
    "HOME_ENV",
    "KEY_CURRENT_PROJECT",
    "KEY_SET_AT",
    "CLIContext",
    "clear_current_project",
    "cli_context_path",
    "cli_home",
    "get_current_project",
    "load_context",
    "set_current_project",
]
