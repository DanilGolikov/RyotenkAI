"""``ryotenkai events <verb>`` — inspect the event subsystem (post-Phase-10 TODO #8).

Sub-commands:

- ``metrics``                     summary of every active run's emitter
- ``metrics --run-id <id>``       deep-dive on one run
- ``metrics ... --json``          machine-readable dump of the metrics dataclass

The CLI is a thin HTTP client over the ``GET /api/v1/health/events``
endpoint registered in :mod:`ryotenkai_control.api.routers.health`.
Direct in-process access via :meth:`EventEmitterRegistry.instance` is
*not* viable because the CLI runs in a separate process from the
FastAPI server — each process has its own registry singleton.

Connection target defaults to ``http://127.0.0.1:8000`` (the same
defaults as :class:`ryotenkai_control.api.config.ApiSettings`); operators
can override with ``--api-url`` or ``RYOTENKAI_API_URL``.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import typer

from ryotenkai_control.cli.context import CLIContext
from ryotenkai_control.cli.errors import die, wrap_command
from ryotenkai_control.cli.renderer import get_renderer

events_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect the event-subsystem (emitter / journal / bus / dedup).",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Mirrors :class:`ryotenkai_control.api.config.ApiSettings` defaults.
#: Centralised here so the CLI's "where is the API" answer is one place
#: to grep when the server-side default changes.
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------


def _resolve_api_url(override: str | None) -> str:
    """Pick the API base URL.

    Precedence: explicit ``--api-url`` flag > ``RYOTENKAI_API_URL`` env
    > :data:`DEFAULT_API_URL`. Trailing slashes are stripped so the
    endpoint path concatenation in :func:`_fetch_events_health` is
    always idempotent.
    """
    candidate = override or os.environ.get("RYOTENKAI_API_URL") or DEFAULT_API_URL
    return candidate.rstrip("/")


def _fetch_events_health(
    *,
    api_url: str,
    run_id: str | None,
    timeout: float,
) -> dict[str, Any]:
    """GET ``/api/v1/health/events`` and return the decoded JSON.

    Raises :class:`typer.Exit` (via :func:`die`) on connection or
    parse failure so commands don't need to plumb errors themselves.
    """
    # Lazy import — keeps ``ryotenkai events --help`` snappy.
    import httpx

    url = f"{api_url}/api/v1/health/events"
    params: dict[str, str] = {}
    if run_id is not None:
        params["run_id"] = run_id

    try:
        response = httpx.get(url, params=params, timeout=timeout)
    except httpx.ConnectError as exc:
        raise die(
            f"cannot reach API at {api_url}: {exc}",
            hint=(
                "is the server running? start it with "
                "`ryotenkai server start` or set RYOTENKAI_API_URL"
            ),
        )
    except httpx.TimeoutException as exc:
        raise die(
            f"API request to {url} timed out after {timeout}s: {exc}",
            hint="increase --timeout or check the server is responsive",
        )
    except httpx.HTTPError as exc:
        raise die(f"API request to {url} failed: {exc}")

    if response.status_code >= 400:
        raise die(
            f"API returned HTTP {response.status_code} for {url}",
            hint=response.text[:200] if response.text else None,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise die(f"API returned non-JSON body: {exc}")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _format_int(value: Any, width: int = 0) -> str:
    """Right-align integer with thousands separators. Returns ``-`` on None."""
    if value is None:
        return "-".rjust(width)
    try:
        return f"{int(value):,}".rjust(width)
    except (TypeError, ValueError):
        return str(value).rjust(width)


def _format_bytes(value: Any) -> str:
    """Render a byte count as ``"<raw> (<human>)"`` (e.g. ``18,734,562 (17.9 MiB)``)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "-"
    if n <= 0:
        return "0"
    # Use MiB / KiB / GiB — matches our log format for sizes.
    units = [
        (1024**3, "GiB"),
        (1024**2, "MiB"),
        (1024**1, "KiB"),
    ]
    for divisor, label in units:
        if n >= divisor:
            return f"{n:,} ({n / divisor:.1f} {label})"
    return f"{n:,} B"


def _format_age(seconds: float | None) -> str:
    """Render last-fsync age as ``"0.3s ago"`` / ``"4m ago"`` / ``"never"``."""
    if seconds is None:
        return "never"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if s < 60:
        return f"{s:.1f}s ago"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    return f"{int(s // 3600)}h ago"


def _status_glyph(status: str) -> str:
    """ASCII status indicator. Avoids unicode for CI log compatibility."""
    if status == "healthy":
        return "OK   healthy"
    if status == "degraded":
        return "WARN degraded"
    if status == "no_active_runs":
        return "--   no active runs"
    return status


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_summary(renderer: Any, payload: dict[str, Any]) -> None:
    """Render the ``no run-id`` summary — one block per active run."""
    active = list(payload.get("active_runs") or [])
    per_run = payload.get("per_run") or {}
    status = str(payload.get("status", "unknown"))

    if not active:
        renderer.text("No active runs.")
        renderer.text("")
        renderer.text(f"Health: {_status_glyph(status)}")
        return

    renderer.text(f"Active runs: {len(active)}")
    renderer.text("-" * 60)

    for run_id in active:
        snapshot = per_run.get(run_id) or {}
        emitted = snapshot.get("emitter_events_emitted_total", 0)
        # Aggregate dropped counts across reason buckets so the summary
        # surfaces a single comparable number per run.
        dropped = sum(
            int(v) for v in (snapshot.get("emitter_events_remote_dropped_total") or {}).values()
        ) + int(snapshot.get("bus_dropped_total") or 0)
        journal_bytes = int(snapshot.get("journal_total_bytes_written") or 0)
        fsync_age = snapshot.get("journal_last_fsync_age_seconds")
        dedup_size = snapshot.get("dedup_size", 0)
        bus_depth = snapshot.get("bus_current_depth", 0)
        bus_capacity = snapshot.get("bus_capacity", 0)

        renderer.text(f"{run_id}")
        renderer.text(f"  Events emitted:    {_format_int(emitted, width=12)}")
        renderer.text(f"  Events dropped:    {_format_int(dropped, width=12)}")
        renderer.text(f"  Journal size:      {_format_bytes(journal_bytes)}")
        renderer.text(f"  Last fsync:        {_format_age(fsync_age)}")
        renderer.text(f"  Dedup set:         {_format_int(dedup_size, width=12)} entries")
        renderer.text(
            f"  In-memory bus:     {_format_int(bus_depth, width=12)} / "
            f"{_format_int(bus_capacity)}"
        )
        renderer.text("")

    renderer.text(f"Health: {_status_glyph(status)}")


def _render_detail(renderer: Any, run_id: str, payload: dict[str, Any]) -> None:
    """Render the ``--run-id`` deep-dive — every counter in dataclass order."""
    per_run = payload.get("per_run") or {}
    status = str(payload.get("status", "unknown"))
    snapshot = per_run.get(run_id)
    if snapshot is None:
        renderer.text(f"Run not found in registry: {run_id}")
        renderer.text("")
        renderer.text(f"Health: {_status_glyph(status)}")
        return

    renderer.text(run_id)
    renderer.text("-" * 60)

    def _section(title: str, rows: list[tuple[str, Any]]) -> None:
        renderer.text(f"{title}:")
        label_width = max((len(label) for label, _ in rows), default=0)
        for label, value in rows:
            renderer.text(f"  {label:<{label_width}}  {value}")
        renderer.text("")

    _section(
        "Emitter",
        [
            ("events_emitted_total", _format_int(snapshot.get("emitter_events_emitted_total"))),
            (
                "events_emit_failed_total",
                _format_dict(snapshot.get("emitter_events_emit_failed_total") or {}),
            ),
            (
                "events_remote_accepted_total",
                _format_int(snapshot.get("emitter_events_remote_accepted_total")),
            ),
            (
                "events_remote_dropped_total",
                _format_dict(snapshot.get("emitter_events_remote_dropped_total") or {}),
            ),
            (
                "offset_collisions_detected_total",
                _format_int(snapshot.get("emitter_offset_collisions_detected_total")),
            ),
        ],
    )

    _section(
        "Bus",
        [
            ("published_total", _format_int(snapshot.get("bus_published_total"))),
            ("dropped_total", _format_int(snapshot.get("bus_dropped_total"))),
            ("current_depth", _format_int(snapshot.get("bus_current_depth"))),
            ("capacity", _format_int(snapshot.get("bus_capacity"))),
            ("subscriber_count", _format_int(snapshot.get("bus_subscriber_count"))),
            (
                "dropped_per_consumer",
                _format_dict(snapshot.get("bus_dropped_per_consumer") or {}),
            ),
        ],
    )

    _section(
        "Journal",
        [
            ("events_appended", _format_int(snapshot.get("journal_appended_total"))),
            (
                "total_bytes_written",
                _format_bytes(snapshot.get("journal_total_bytes_written") or 0),
            ),
            ("fsyncs_total", _format_int(snapshot.get("journal_fsync_total"))),
            (
                "fsync_failed_total",
                _format_int(snapshot.get("journal_fsync_failed_total")),
            ),
            (
                "last_fsync_age_seconds",
                _format_age(snapshot.get("journal_last_fsync_age_seconds")),
            ),
            (
                "write_failures_total",
                _format_int(snapshot.get("journal_write_failed_total")),
            ),
        ],
    )

    _section(
        "Dedup",
        [
            ("size", _format_int(snapshot.get("dedup_size"))),
            ("seen_total", _format_int(snapshot.get("dedup_seen_total"))),
            ("dedup_hits_total", _format_int(snapshot.get("dedup_hits_total"))),
            ("evicted_total", _format_int(snapshot.get("dedup_evicted_total"))),
        ],
    )

    renderer.text(f"Health: {_status_glyph(status)}")


def _format_dict(mapping: dict[str, Any]) -> str:
    """Render a small {reason: count} dict on one line — ``{}`` when empty."""
    if not mapping:
        return "{}"
    parts = [f"{k}={v}" for k, v in sorted(mapping.items())]
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@events_app.command("metrics")
@wrap_command
def metrics_cmd(
    ctx: typer.Context,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Restrict the snapshot to a single run id.",
        ),
    ] = None,
    api_url: Annotated[
        str | None,
        typer.Option(
            "--api-url",
            envvar="RYOTENKAI_API_URL",
            help=f"API base URL (default: {DEFAULT_API_URL}).",
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="HTTP timeout in seconds for the metrics request.",
        ),
    ] = DEFAULT_TIMEOUT_SECONDS,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Dump the raw response as JSON (overrides -o text).",
        ),
    ] = False,
) -> None:
    """Show event-subsystem counters for active runs.

    Without ``--run-id``: prints a one-block summary per active run +
    overall health status. With ``--run-id``: prints every counter for
    that run, grouped by collaborator. ``--json`` or ``-o json`` emits
    the raw response dict so other tools can parse it.
    """
    state = ctx.ensure_object(CLIContext)
    resolved_url = _resolve_api_url(api_url)
    payload = _fetch_events_health(
        api_url=resolved_url, run_id=run_id, timeout=timeout,
    )

    # ``--json`` is a shortcut for "machine-readable, no matter the
    # global -o flag" so a one-off scrape doesn't require ``-o json``.
    # We construct the JsonRenderer directly when ``--json`` is set so
    # the dict is dumped as RFC-8259 JSON regardless of the global
    # ``-o`` flag — TextRenderer.emit would render it as a Python
    # repr, which is not parseable.
    if as_json:
        import json
        import sys

        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    renderer = get_renderer(state)
    if state.is_machine_readable:
        renderer.emit(payload)
        renderer.flush()
        return

    if run_id is not None:
        _render_detail(renderer, run_id, payload)
    else:
        _render_summary(renderer, payload)
    renderer.flush()


__all__ = ["events_app"]
