"""``ryotenkai server <verb>`` — manage the FastAPI backend.

Today only ``start`` does real work (launches uvicorn in the foreground);
``status`` / ``stop`` are reserved for the daemon-mode landing in
v1.2 (open question 3 in the plan). Stubs return a clear NotImplemented
message instead of silently failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.cli.errors import die

server_app = typer.Typer(
    no_args_is_help=True,
    help="Run the FastAPI web backend (start / status / stop).",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


@server_app.command("start")
def start_cmd(
    host: Annotated[
        str, typer.Option("--host", help="Bind host (use 0.0.0.0 for remote access)."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8000,
    runs_dir: Annotated[
        Path,
        typer.Option(
            "--runs-dir", help="Runs directory served by the API.",
            file_okay=False, dir_okay=True, resolve_path=True,
        ),
    ] = Path("runs"),
    cors_origins: Annotated[
        str,
        typer.Option(
            "--cors-origins",
            help="Comma-separated CORS origins (Vite dev server by default).",
        ),
    ] = "http://localhost:5173",
    reload: Annotated[
        bool, typer.Option("--reload", help="Dev auto-reload."),
    ] = False,
    log_level: Annotated[
        str, typer.Option("--log-level", help="uvicorn log level."),
    ] = "info",
) -> None:
    """Run the FastAPI web backend (foreground uvicorn)."""
    from src.api.cli import run_server

    run_server(
        host=host,
        port=port,
        runs_dir=runs_dir,
        cors_origins=[o.strip() for o in cors_origins.split(",") if o.strip()],
        reload=reload,
        log_level=log_level,
    )


@server_app.command("status")
def status_cmd() -> None:
    """(Reserved) — daemon mode lands in v1.2."""
    raise die(
        "server status is not implemented yet (daemon mode planned for v1.2)",
        hint="run `ryotenkai server start` in a terminal to see live status",
    )


@server_app.command("stop")
def stop_cmd() -> None:
    """(Reserved) — daemon mode lands in v1.2."""
    raise die(
        "server stop is not implemented yet (daemon mode planned for v1.2)",
        hint="press Ctrl+C in the terminal where `ryotenkai server start` runs",
    )


__all__ = ["server_app"]
