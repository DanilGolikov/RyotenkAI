"""Uniform error surface for the CLI.

``die(msg, hint=...)`` replaces the per-command ``typer.echo(err=True) +
raise typer.Exit(1)`` pattern; ``suggest()`` powers did-you-mean hints on
typos. ``load_config_or_die()`` wraps :func:`load_pipeline_config` and
renders YAML / pydantic errors as clean ``die`` output. All output goes
through ``err_console`` so it stays out of the data stream of ``-o json``
commands.

Phase F (sharded-stargazing-wigderson, 2026-05-16) adds:

* :func:`die_from_ryotenkai` -- kubectl/Terraform-style rendering for
  typed :class:`RyotenkAIError` exceptions, with code/trace/request
  parts on a single line and exit-code mapping (4xx -> 2, 5xx -> 1).
* :func:`wrap_command` -- decorator applied to every Typer command so a
  bare ``raise RyotenkAIError(...)`` inside the command body lands on
  the unified rendering path. Also generates a fresh ``request_id`` and
  stamps it into the :data:`REQUEST_ID` contextvar so error output and
  any subsequent log lines share a correlation id even though the CLI
  has no HTTP middleware.
"""

from __future__ import annotations

import difflib
import functools
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import typer
import yaml
from pydantic import ValidationError
from rich.markup import escape as _rich_escape

from ryotenkai_control.cli.style import COLOR_DIM, COLOR_ERR, err_console
from ryotenkai_shared.api.request_id import (
    REQUEST_ID,
    current_request_id,
    generate_request_id,
)
from ryotenkai_shared.errors.base import RyotenkAIError, TransportError

if TYPE_CHECKING:
    from ryotenkai_shared.config.pipeline.schema import PipelineConfig


F = TypeVar("F", bound=Callable[..., Any])


def die(message: str, *, hint: str | None = None, code: int = 1) -> typer.Exit:
    """Print a one-line error (and optional hint), then raise ``typer.Exit``.

    Always raises -- the return type is there so call-sites can write
    ``raise die(...)`` if they want explicit control flow.
    """
    err_console.print(
        f"[{COLOR_ERR}]error:[/{COLOR_ERR}] {_rich_escape(message)}"
    )
    if hint:
        err_console.print(
            f"  [{COLOR_DIM}]hint:[/{COLOR_DIM}] {_rich_escape(hint)}"
        )
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
    end-users -- they want to see "this field is wrong in this file".
    Wraps :func:`load_pipeline_config` and converts the three expected
    failure modes (missing file, malformed YAML, schema mismatch) into
    one-line ``die()`` errors with field-level detail.
    """
    from ryotenkai_shared.config.loader import load_pipeline_config

    path_str = str(path)
    try:
        return load_pipeline_config(path)
    except FileNotFoundError:
        raise die(f"config file not found: {path_str}")
    except yaml.YAMLError as exc:
        raise die(f"invalid YAML in {path_str}: {exc}")
    except ValidationError as exc:
        # Embed field-level errors directly in the message so each
        # appears on its own line -- die's `hint:` prefix would clobber
        # the alignment if we passed them through there.
        rendered = format_validation_errors(exc)
        raise die(f"invalid pipeline config: {path_str}\n{rendered}")
    except ValueError as exc:
        # Loader's own "must be a mapping at the top level" check, plus
        # any pydantic-adjacent value errors that escape ValidationError.
        raise die(f"invalid pipeline config: {path_str} -- {exc}")


def format_validation_errors(exc: ValidationError, *, max_errors: int = 6) -> str:
    """Render a Pydantic ValidationError as compact multi-line text.

    Each line: ``  - <dotted.path>: <message>``. Truncates after
    ``max_errors`` so a totally broken file doesn't flood the terminal.
    Public so callers that wrap their own load path (e.g. ``run start
    --project``, where the project resolver bubbles a ValidationError
    from the worker subprocess) can format the same way without
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


# ---------------------------------------------------------------------------
# Phase F -- kubectl/Terraform-style rendering for RyotenkAIError
# ---------------------------------------------------------------------------


def _exit_code_for(exc: RyotenkAIError) -> int:
    """Map HTTP-suggested status onto a POSIX-style exit code.

    Convention mirrors kubectl/Terraform:

    * 4xx -> exit 2 -- user error, fixable by changing inputs.
    * everything else (5xx, TransportError, InternalError) -> exit 1.

    :class:`TransportError` carries status 599 so it lands on exit 1
    via the second branch -- no special-case here.
    """
    if 400 <= exc.status < 500:
        return 2
    return 1


def _render_context(context: dict[str, Any]) -> str:
    """Pretty-print exc.context as a compact JSON blob for ``-v`` mode.

    Sorted keys for deterministic test output. ``default=str`` so any
    exotic values (Path, datetime, enum) coerce without raising.
    """
    return json.dumps(context, sort_keys=True, indent=2, default=str)


def die_from_ryotenkai(
    exc: RyotenkAIError,
    *,
    request_id: str | None = None,
    verbose: bool = False,
) -> typer.Exit:
    """Render a :class:`RyotenkAIError` in kubectl/Terraform style and exit.

    Multi-line output (all to stderr via ``err_console``)::

        error: Job not found
          hint: job_id="abc123" is not active
          code: JOB_NOT_FOUND  trace=a3b1c2d4  request=8e7f6c5b4a3d2e1f

    The ``code:`` line aggregates the machine-readable identifier plus
    optional ``trace=`` / ``request=`` / (TransportError only) ``status=``
    parts. When ``verbose`` is true and ``exc.context`` is non-empty an
    extra ``context: <pretty JSON>`` line is appended for debugging.

    Always raises :class:`typer.Exit`; the return type exists so call
    sites can write ``raise die_from_ryotenkai(exc)`` for explicit
    control flow.
    """
    if not isinstance(exc, RyotenkAIError):
        raise TypeError(
            f"die_from_ryotenkai expects RyotenkAIError, got {type(exc).__name__}"
        )

    # User-supplied strings (title from server, detail from raise site)
    # may legitimately contain square brackets that look like Rich
    # markup tags -- escape them so the renderer doesn't choke on
    # ``detail="job_id=[bug]"``.
    err_console.print(
        f"[{COLOR_ERR}]error:[/{COLOR_ERR}] {_rich_escape(exc.title)}"
    )
    if exc.detail:
        err_console.print(
            f"  [{COLOR_DIM}]hint:[/{COLOR_DIM}] {_rich_escape(exc.detail)}"
        )

    parts: list[str] = [exc.code.value]
    if isinstance(exc, TransportError):
        parts.append(f"status={exc.status}")
    if exc.trace_id:
        parts.append(f"trace={exc.trace_id}")
    if request_id:
        parts.append(f"request={request_id}")
    err_console.print(
        f"  [{COLOR_DIM}]code:[/{COLOR_DIM}] {'  '.join(parts)}"
    )

    if verbose and exc.context:
        err_console.print(
            f"  [{COLOR_DIM}]context:[/{COLOR_DIM}] "
            f"{_rich_escape(_render_context(exc.context))}"
        )

    raise typer.Exit(code=_exit_code_for(exc))


def _verbose_from_ctx() -> bool:
    """Best-effort lookup of ``-v`` flag from the current Typer context.

    The root callback in :mod:`ryotenkai_control.cli.app` stamps a
    :class:`CLIContext` onto ``click.Context.obj``. Typer re-exports
    Click's context primitives, so ``click.get_current_context`` is the
    runtime accessor (``typer.get_current_context`` doesn't exist on
    the public surface). Outside of a CLI invocation (e.g. unit tests
    calling functions directly) the lookup returns ``None`` and we
    treat that as ``verbose=False``.
    """
    import click

    try:
        ctx = click.get_current_context(silent=True)
    except RuntimeError:
        return False
    if ctx is None or ctx.obj is None:
        return False
    return bool(getattr(ctx.obj, "verbose", 0))


def wrap_command(fn: F) -> F:
    """Decorator -- catch typed exceptions and render uniformly.

    Wraps a Typer command body. On entry stamps a freshly-generated
    request id into the :data:`REQUEST_ID` contextvar so any log lines
    emitted during the command share a correlation id, then catches
    four error families:

    1. :class:`RyotenkAIError` -- routed through
       :func:`die_from_ryotenkai` for kubectl-style rendering.
    2. :class:`pydantic.ValidationError` -- field-level rendering via
       :func:`format_validation_errors`.
    3. :class:`FileNotFoundError` -- one-line ``die`` with the path.
    4. :class:`yaml.YAMLError` -- one-line ``die`` with the parse error.

    :class:`typer.Exit` and :class:`typer.Abort` are deliberately NOT
    caught -- they're the controlled-exit primitives the rest of the
    CLI uses, including :func:`die`. Same for :class:`KeyboardInterrupt`
    so Ctrl-C still terminates cleanly.

    The decorator is non-introspecting (uses ``functools.wraps``) so
    Typer's option/argument extraction keeps working unchanged.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # If we're already inside a request scope (nested invocation,
        # CLI calls into in-process FastAPI for testing) inherit the
        # existing id rather than minting a fresh one.
        existing = current_request_id()
        token = None
        if existing is None:
            token = REQUEST_ID.set(generate_request_id())
        try:
            return fn(*args, **kwargs)
        except RyotenkAIError as exc:
            raise die_from_ryotenkai(
                exc,
                request_id=current_request_id(),
                verbose=_verbose_from_ctx(),
            )
        except ValidationError as exc:
            raise die(f"invalid input:\n{format_validation_errors(exc)}")
        except FileNotFoundError as exc:
            raise die(f"file not found: {exc}")
        except yaml.YAMLError as exc:
            raise die(f"invalid YAML: {exc}")
        finally:
            if token is not None:
                REQUEST_ID.reset(token)

    return wrapper  # type: ignore[return-value]


__all__ = [
    "die",
    "die_from_ryotenkai",
    "format_validation_errors",
    "load_config_or_die",
    "suggest",
    "suggest_hint",
    "wrap_command",
]
