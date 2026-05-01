"""Uniform error surface for the CLI.

``die(msg, hint=...)`` replaces the per-command ``typer.echo(err=True) +
raise typer.Exit(1)`` pattern; ``suggest()`` powers did-you-mean hints on
typos. ``load_config_or_die()`` wraps :func:`load_pipeline_config` and
renders YAML / pydantic errors as clean ``die`` output. All output goes
through ``err_console`` so it stays out of the data stream of ``-o json``
commands.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml
from pydantic import ValidationError

from src.cli.style import COLOR_DIM, COLOR_ERR, err_console

if TYPE_CHECKING:
    from src.config.pipeline.schema import PipelineConfig


def die(message: str, *, hint: str | None = None, code: int = 1) -> typer.Exit:
    """Print a one-line error (and optional hint), then raise ``typer.Exit``.

    Always raises — the return type is there so call-sites can write
    ``raise die(...)`` if they want explicit control flow.
    """
    err_console.print(f"[{COLOR_ERR}]error:[/{COLOR_ERR}] {message}")
    if hint:
        err_console.print(f"  [{COLOR_DIM}]hint:[/{COLOR_DIM}] {hint}")
    raise typer.Exit(code=code)


def suggest(user_input: str, valid: Iterable[str], *, n: int = 3) -> list[str]:
    """Return up to ``n`` close matches for ``user_input`` from ``valid``.

    Uses ``difflib.get_close_matches`` with a slightly looser cutoff than
    the default so single-letter typos surface.
    """
    return difflib.get_close_matches(user_input, list(valid), n=n, cutoff=0.55)


def suggest_hint(user_input: str, valid: Iterable[str]) -> str | None:
    """Build a ``did you mean 'X'?`` string, or ``None`` if no close match."""
    matches = suggest(user_input, valid)
    if not matches:
        return None
    if len(matches) == 1:
        return f"did you mean '{matches[0]}'?"
    quoted = ", ".join(f"'{m}'" for m in matches)
    return f"did you mean one of: {quoted}?"


def load_config_or_die(path: Path | str) -> PipelineConfig:
    """Load a pipeline YAML, rendering loader errors as clean ``die``.

    Raw ``ValidationError`` / ``YAMLError`` tracebacks are useless to
    end-users — they want to see "this field is wrong in this file".
    Wraps :func:`load_pipeline_config` and converts the three expected
    failure modes (missing file, malformed YAML, schema mismatch) into
    one-line ``die()`` errors with field-level detail.
    """
    from src.workspace.integrations.loader import load_pipeline_config

    path_str = str(path)
    try:
        return load_pipeline_config(path)
    except FileNotFoundError:
        raise die(f"config file not found: {path_str}")
    except yaml.YAMLError as exc:
        raise die(f"invalid YAML in {path_str}: {exc}")
    except ValidationError as exc:
        # Embed field-level errors directly in the message so each
        # appears on its own line — die's `hint:` prefix would clobber
        # the alignment if we passed them through there.
        rendered = format_validation_errors(exc)
        raise die(f"invalid pipeline config: {path_str}\n{rendered}")
    except ValueError as exc:
        # Loader's own "must be a mapping at the top level" check, plus
        # any pydantic-adjacent value errors that escape ValidationError.
        raise die(f"invalid pipeline config: {path_str} — {exc}")


def format_validation_errors(exc: ValidationError, *, max_errors: int = 6) -> str:
    """Render a Pydantic ValidationError as compact multi-line text.

    Each line: ``  - <dotted.path>: <message>``. Truncates after
    ``max_errors`` so a totally broken file doesn't flood the terminal.
    Public so callers that wrap their own load path (e.g. ``run start
    --project``, where ``load_project_inputs`` raises ValidationError
    from inside the adapter) can format the same way without
    duplicating logic.
    """
    errors = exc.errors()
    if not errors:
        return "validation failed"
    lines: list[str] = []
    for err in errors[:max_errors]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        msg = err.get("msg", "validation error")
        lines.append(f"  - {loc}: {msg}")
    if len(errors) > max_errors:
        lines.append(f"  … and {len(errors) - max_errors} more")
    return "\n".join(lines)


__all__ = [
    "die",
    "format_validation_errors",
    "load_config_or_die",
    "suggest",
    "suggest_hint",
]
