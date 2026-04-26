"""``ryotenkai smoke <dir>`` — batch smoke-tests for a directory of configs.

Replaces the standalone ``scripts/batch_smoke.py`` argparse script. The
implementation is the same — we just go through Typer for consistent
help / global flags and to delete the orphan script in Phase 3.

The runner discovers ``*.yaml`` / ``*.yml`` configs recursively, launches
each as a fresh ``run start`` subprocess (capped by ``--workers``), and
writes a Markdown summary report to ``runs/smoke_<id>/``. Per-run
liveness gating replaces a hard timeout — see
``scripts/batch_smoke.py`` docstring for the protocol.

ENV contract preserved: callers can override the smoke dir base via
``RYOTENKAI_RUNS_DIR`` exactly as before (NR-04 / Q-10).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

smoke_app = typer.Typer(
    no_args_is_help=True,
    help="Run a directory of configs as a batch smoke suite.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
    invoke_without_command=True,
)


@smoke_app.callback(invoke_without_command=True)
def smoke_cmd(
    config_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory with *.yaml configs (searched recursively).",
            exists=True, file_okay=False, dir_okay=True, resolve_path=True,
        ),
    ],
    workers: Annotated[
        int,
        typer.Option(
            "--workers", help="Max parallel runs (-1 = unlimited, 1 per config).",
        ),
    ] = 4,
    idle_timeout: Annotated[
        int,
        typer.Option(
            "--idle-timeout",
            help="Seconds without subprocess output before graceful shutdown.",
        ),
    ] = 1200,
    stagger: Annotated[
        int,
        typer.Option(
            "--stagger", help="Seconds between launches (0 = all at once).",
        ),
    ] = 5,
    report_dir: Annotated[
        Path | None,
        typer.Option(
            "--report-dir",
            help="Where to save the markdown report (default: smoke dir).",
            file_okay=False, dir_okay=True,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List configs without launching anything."),
    ] = False,
) -> None:
    """Run every YAML config under ``config_dir`` as a parallel smoke run."""
    # Reuse the existing implementation verbatim by patching sys.argv
    # and calling its ``main()``. Once Phase 3 deletes the script the
    # body lives here directly; today we keep the indirection so the
    # behaviour stays bit-identical.
    argv_backup = sys.argv[:]
    new_argv = ["ryotenkai-smoke", str(config_dir),
                "--workers", str(workers),
                "--idle-timeout", str(idle_timeout),
                "--stagger", str(stagger)]
    if report_dir is not None:
        new_argv += ["--report-dir", str(report_dir)]
    if dry_run:
        new_argv += ["--dry-run"]

    sys.argv = new_argv
    try:
        from src.cli._smoke_runner import main as _smoke_main

        _smoke_main()
    finally:
        sys.argv = argv_backup


__all__ = ["smoke_app"]
