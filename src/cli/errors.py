"""Uniform error surface for the CLI.

``die(msg, hint=...)`` replaces the per-command ``typer.echo(err=True) +
raise typer.Exit(1)`` pattern; ``suggest()`` powers did-you-mean hints on
typos. All output goes through ``err_console`` so it stays out of the
data stream of ``-o json`` commands.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

import typer

from src.cli.style import COLOR_DIM, COLOR_ERR, err_console


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


__all__ = ["die", "suggest", "suggest_hint"]
