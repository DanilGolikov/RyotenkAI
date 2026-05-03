"""``ryotenkai`` Typer root.

This module replaces the old monolithic ``src/main.py`` ``@app.command``
soup with a thin assembly: the root callback wires global flags into a
shared :class:`CLIContext`, the :mod:`commands` package registers one
sub-Typer per noun (``run``, ``runs``, ``project``, ``plugin``, ...).

Top-level imports stay deliberately lean — ``ryotenkai --help`` should
render in well under 300 ms. Anything heavy (orchestrator, mlflow,
torch, datasets) lives behind ``import`` statements inside command
bodies so the help screen never pays for them.
"""

from __future__ import annotations

import logging
import warnings

# Hide noisy third-party deprecation warnings that surface at import
# time in the dependency graph (torch's pynvml, etc). Filter must run
# before any heavy import — done here at the very top of the CLI tree.
warnings.filterwarnings(
    "ignore",
    message=".*pynvml package is deprecated.*",
    category=FutureWarning,
)

import typer  # noqa: E402

from src.cli import _signals  # noqa: E402
from src.cli.commands import register_all  # noqa: E402
from src.cli.context import CLIContext  # noqa: E402
from src.cli.errors import die  # noqa: E402
from src.cli.style import reconfigure as _reconfigure_style  # noqa: E402
from src.cli.version import collect_version_info  # noqa: E402

# Register the Ctrl-C / SIGTERM handler exactly once for the lifetime of
# the process — long-running commands hand off their orchestrator via
# ``_signals.set_active_orchestrator``.
_signals.install()


# ---------------------------------------------------------------------------
# Global ``--version`` flag (eager — short-circuits before the callback)
# ---------------------------------------------------------------------------


def _print_version_and_exit(value: bool) -> None:
    if not value:
        return
    typer.echo(collect_version_info().format())
    raise typer.Exit()


# ---------------------------------------------------------------------------
# Root Typer
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ryotenkai",
    help="RyotenkAI — Automated CI/CD for LLM training.",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    # Plain Python tracebacks for unhandled errors; commands that
    # expect user mistakes print a single ``error: ...`` line via
    # ``src.cli.errors.die`` instead.
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)


# ---------------------------------------------------------------------------
# Root callback — populates ``ctx.obj`` with the shared CLIContext
# ---------------------------------------------------------------------------


@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: ARG001 — consumed by is_eager
        False, "-V", "--version",
        is_eager=True, callback=_print_version_and_exit,
        help="Show version info and exit.",
    ),
    output: str = typer.Option(
        "text", "-o", "--output",
        envvar="RYOTENKAI_OUTPUT",
        help="Output format for read commands: text | json | yaml.",
    ),
    color: bool = typer.Option(
        True, "--color/--no-color",
        help="Colored output (honours NO_COLOR env).",
    ),
    verbose: int = typer.Option(
        0, "-v", "--verbose",
        count=True, help="Increase verbosity (-v, -vv).",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet",
        help="Suppress non-essential output.",
    ),
    log_level: str | None = typer.Option(
        None, "--log-level",
        envvar="LOG_LEVEL",
        help="Override log level (DEBUG / INFO / WARNING / ERROR).",
    ),
    project: str | None = typer.Option(
        None, "--project", "-p",
        envvar="RYOTENKAI_PROJECT",
        help=(
            "Project id. Falls back to the value persisted by "
            "`ryotenkai project use` when omitted."
        ),
    ),
    remote: str | None = typer.Option(
        None, "--remote",
        envvar="RYOTENKAI_REMOTE",
        hidden=True,
        help="(Reserved) HTTP URL of a remote ryotenkai API. "
             "Lands in v1.2; raises NotImplementedError today.",
    ),
) -> None:
    """Wire global flags into ``ctx.obj`` before any sub-command runs."""
    if output not in ("text", "json", "yaml"):
        raise die(
            f"invalid --output value: {output!r}",
            hint="choose one of: text, json, yaml",
        )

    state = CLIContext(
        output=output,  # type: ignore[arg-type]
        color=color,
        verbose=verbose,
        quiet=quiet,
        log_level=log_level,
        project_id=project,
        remote_url=remote,
    )
    _reconfigure_style(color=state.use_color)
    _apply_log_level(state)
    _check_remote_stub(state)
    ctx.obj = state


def _apply_log_level(state: CLIContext) -> None:
    """Resolve effective log level from flags + env and apply once.

    Precedence: ``--log-level`` flag (or LOG_LEVEL env via Typer
    envvar) > ``-vv`` (DEBUG) > ``-v`` (INFO) > ``--quiet`` (ERROR) >
    leave the logger alone.
    """
    if state.log_level is not None:
        try:
            logging.getLogger().setLevel(state.log_level.upper())
        except ValueError:
            raise die(
                f"invalid --log-level value: {state.log_level!r}",
                hint="choose one of: DEBUG, INFO, WARNING, ERROR",
            )
        return

    if state.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif state.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif state.quiet:
        logging.getLogger().setLevel(logging.ERROR)


def _check_remote_stub(state: CLIContext) -> None:
    """``--remote`` is reserved for v1.2. Fail fast with a clear message
    instead of letting commands silently ignore it.

    Skip the check during ``--help`` rendering — Typer eagerly invokes
    the callback even for ``--help``, but a stub error there would
    surprise users typing ``ryotenkai --remote x --help`` to discover
    what the flag does.
    """
    import sys
    if state.remote_url and not any(
        tok in sys.argv for tok in ("-h", "--help")
    ):
        raise die(
            "remote mode (--remote) is reserved for v1.2; see roadmap.",
            hint="for now the CLI runs commands locally only — drop --remote.",
        )


# ---------------------------------------------------------------------------
# ``ryotenkai help`` alias (so ``ryotenkai help`` works alongside ``--help``)
# ---------------------------------------------------------------------------


@app.command("help", hidden=True)
def _help_cmd(ctx: typer.Context) -> None:
    """Alias for --help so `ryotenkai help` works alongside `ryotenkai --help`."""
    parent = ctx.parent
    typer.echo(parent.get_help() if parent is not None else ctx.get_help())


# Mount every noun's sub-Typer.
register_all(app)


__all__ = ["app"]
