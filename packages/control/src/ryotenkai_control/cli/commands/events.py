"""``ryotenkai events <verb>`` — inspect the event subsystem (post-Phase-10 TODO #8).

Sub-commands:

- ``metrics``                     summary of every active run's emitter
- ``metrics --run-id <id>``       deep-dive on one run
- ``metrics ... --json``          machine-readable dump of the metrics dataclass
- ``history --run-id <id>``       offline stats from ``events.jsonl`` on disk
- ``history ... --json``          machine-readable dump of HistoryStats

``metrics`` is a thin HTTP client over the ``GET /api/v1/health/events``
endpoint registered in :mod:`ryotenkai_control.api.routers.health`.
Direct in-process access via :meth:`EventEmitterRegistry.instance` is
*not* viable because the CLI runs in a separate process from the
FastAPI server — each process has its own registry singleton.

``history`` is a *pure-local* command — it reads the on-disk
``events.jsonl`` journal via :class:`JournalReader`, computes aggregate
statistics with a single streaming pass, and renders them. No HTTP, no
server required (intentional: a completed run is gone from the runtime
registry by the time you want to inspect it, but the journal stays on
disk forever — until you ``rm`` it).

Connection target for ``metrics`` defaults to ``http://127.0.0.1:8000``
(the same defaults as :class:`ryotenkai_control.api.config.ApiSettings`);
operators can override with ``--api-url`` or ``RYOTENKAI_API_URL``.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from ryotenkai_control.cli.context import CLIContext
from ryotenkai_control.cli.errors import die, wrap_command
from ryotenkai_control.cli.renderer import get_renderer
from ryotenkai_shared.events import SEVERITY_ORDER, UNKNOWN_OFFSET, UnknownEvent

if TYPE_CHECKING:
    from ryotenkai_control.events.journal_reader import JournalReader
    from ryotenkai_shared.events import BaseEvent

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


# ---------------------------------------------------------------------------
# ``events history`` — offline stats from on-disk events.jsonl
# ---------------------------------------------------------------------------


#: Width of timeline buckets when aggregating event counts. 15 minutes
#: is a compromise: short enough that a 30-minute run still produces
#: multiple buckets (so the timeline isn't degenerate); long enough
#: that a multi-day fine-tune doesn't generate thousands of rows.
_TIMELINE_BUCKET = timedelta(minutes=15)

#: Max number of full-detail error envelopes to keep in :class:`HistoryStats`.
#: Pretty mode shows fewer (default 20); ``--json`` dumps up to this many
#: with their payload + traceback excerpt. Bounded to keep memory finite
#: when scanning multi-million-event journals.
_MAX_ERRORS_RETAINED = 1000

#: Unicode block characters used for horizontal-bar rendering of the
#: timeline. Indexed by ``round(fractional_count / max * 8)``; index 0
#: is the empty cell (rendered as a space) and the value is one of
#: ``▏▎▍▌▋▊▉█``.
_BAR_BLOCKS = " ▏▎▍▌▋▊▉█"


@dataclass
class HistoryStats:
    """Aggregated, memory-bounded view of an ``events.jsonl`` journal.

    Populated by :func:`_compute_history_stats` in a single streaming
    pass over :meth:`JournalReader.iter_envelopes`. Aggregates only —
    no per-event accumulation except for the bounded ``errors`` list.
    """

    run_id: str
    journal_path: Path
    size_bytes: int
    total_events: int = 0
    first_event: dict[str, Any] | None = None  # {kind, time, offset}
    last_event: dict[str, Any] | None = None
    duration_seconds: float | None = None
    by_source: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)
    by_stage: dict[str, int] = field(default_factory=dict)
    schema_versions_present: list[int] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    unknown_events: int = 0
    timeline_buckets_15min: dict[str, int] = field(default_factory=dict)


def _format_iso(dt: datetime) -> str:
    """Render a tz-aware datetime as a short ISO-8601 ``Z`` string.

    Used for first/last-event timestamps and timeline bucket keys.
    Trailing microseconds are stripped so the output stays compact;
    the ``Z`` suffix is canonical for UTC timestamps in our JSON shapes.
    """
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bucket_start(time: datetime) -> datetime:
    """Snap ``time`` down to the nearest 15-minute boundary, UTC."""
    minute_bucket = (time.minute // 15) * 15
    return time.replace(minute=minute_bucket, second=0, microsecond=0)


def _truncate(value: str, limit: int = 80) -> str:
    """Trim ``value`` to ``limit`` chars, appending ``…`` if cut."""
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _compute_history_stats(
    reader: JournalReader,
    *,
    run_id: str,
    from_offset: int | None = None,
    to_offset: int | None = None,
) -> HistoryStats:
    """Stream the journal once, build a :class:`HistoryStats`.

    Memory is O(distinct_sources + distinct_kinds + distinct_stages +
    distinct_buckets + min(error_count, _MAX_ERRORS_RETAINED)). The
    bounded ``errors`` list caps per-run memory at a few MB even for
    pathological journals where every event is an error.

    ``from_offset`` / ``to_offset`` (when provided) clamp the window
    of events analysed to ``[from_offset, to_offset]`` inclusive.
    Events with :data:`UNKNOWN_OFFSET` are always counted in
    ``unknown_events`` but excluded from the offset-window filter.
    """
    path = reader.path
    try:
        size_bytes = path.stat().st_size if path.exists() else 0
    except OSError as exc:
        raise die(f"cannot read journal: {exc}", code=3)

    stats = HistoryStats(
        run_id=run_id,
        journal_path=path,
        size_bytes=size_bytes,
    )

    if not path.exists():
        # Caller validates existence before we get here; this is a
        # defensive belt-and-suspenders so the empty-journal branch
        # below renders sensibly when a test invokes the helper
        # directly.
        return stats

    schema_versions: set[int] = set()
    by_source: dict[str, int] = defaultdict(int)
    by_severity: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    by_stage: dict[str, int] = defaultdict(int)
    timeline: dict[str, int] = defaultdict(int)

    first_envelope: BaseEvent | None = None
    last_envelope: BaseEvent | None = None

    severity_error_threshold = SEVERITY_ORDER["error"]

    try:
        envelope_iter = reader.iter_envelopes()
    except OSError as exc:
        raise die(f"cannot read journal: {exc}", code=3)

    for envelope in envelope_iter:
        is_unknown = isinstance(envelope, UnknownEvent)
        # Apply offset-window filter only when the envelope carries a
        # usable offset. UnknownEvent torn-write residue has UNKNOWN_OFFSET
        # and is always counted (unknown_events) regardless of window.
        if envelope.offset != UNKNOWN_OFFSET:
            if from_offset is not None and envelope.offset < from_offset:
                continue
            if to_offset is not None and envelope.offset > to_offset:
                continue

        stats.total_events += 1

        if is_unknown:
            stats.unknown_events += 1

        # Track first/last event in *file order* — iter_envelopes yields
        # in file order, so we just remember the first one we saw and
        # update last on every iteration.
        if first_envelope is None:
            first_envelope = envelope
        last_envelope = envelope

        by_source[envelope.source] += 1
        by_severity[envelope.severity] += 1
        by_kind[envelope.kind] += 1
        if envelope.stage_id:
            by_stage[envelope.stage_id] += 1
        schema_versions.add(envelope.schema_version)

        # 15-minute bucket key (UTC, ISO-8601 with Z suffix).
        bucket_key = _format_iso(_bucket_start(envelope.time))
        timeline[bucket_key] += 1

        # Capture errors (and critical) for the bounded errors list.
        sev_rank = SEVERITY_ORDER.get(envelope.severity, 0)
        if sev_rank >= severity_error_threshold and len(stats.errors) < _MAX_ERRORS_RETAINED:
            err_payload: dict[str, Any]
            if is_unknown:
                err_payload = dict(envelope.raw_payload)
            else:
                # ``model_dump(mode='python')`` would leave datetimes as
                # objects, which trips ``json.dumps`` later; ``mode='json'``
                # gives plain strings and ISO timestamps in one call.
                err_payload = envelope.payload.model_dump(mode="json")
            stats.errors.append(
                {
                    "offset": envelope.offset,
                    "time": _format_iso(envelope.time),
                    "kind": envelope.kind,
                    "severity": envelope.severity,
                    "source": envelope.source,
                    "stage_id": envelope.stage_id,
                    "payload": err_payload,
                }
            )

    # Promote defaultdicts to plain dicts so asdict(...) doesn't surface
    # the defaultdict factory as part of the JSON output.
    stats.by_source = dict(by_source)
    stats.by_severity = dict(by_severity)
    stats.by_kind = dict(by_kind)
    stats.by_stage = dict(by_stage)
    stats.schema_versions_present = sorted(schema_versions)
    stats.timeline_buckets_15min = dict(timeline)

    if first_envelope is not None and last_envelope is not None:
        stats.first_event = {
            "kind": first_envelope.kind,
            "time": _format_iso(first_envelope.time),
            "offset": first_envelope.offset,
        }
        stats.last_event = {
            "kind": last_envelope.kind,
            "time": _format_iso(last_envelope.time),
            "offset": last_envelope.offset,
        }
        duration = (last_envelope.time - first_envelope.time).total_seconds()
        # Clamp to >= 0 for the (rare) case where the last event's
        # ``time`` precedes the first due to clock skew across sources.
        stats.duration_seconds = max(duration, 0.0)

    return stats


def _format_duration(seconds: float | None) -> str:
    """Render a duration as ``"2h 34m 12s"`` / ``"3m 5s"`` / ``"42s"``."""
    if seconds is None:
        return "n/a"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_count_with_pct(count: int, total: int) -> str:
    """Render ``"7,800 (84.6%)"`` — share-of-total in parentheses."""
    if total <= 0:
        return f"{count:,}"
    pct = (count / total) * 100.0
    return f"{count:,} ({pct:.1f}%)"


def _render_top_n_table(
    renderer: Any,
    *,
    title: str,
    mapping: dict[str, int],
    total: int,
    n: int = 10,
    show_pct: bool = True,
) -> None:
    """Render the top-``n`` entries of ``mapping`` as a 2-col table.

    Sort is descending by count, with a stable secondary sort on the
    label so identical counts produce deterministic output across runs.
    """
    if not mapping:
        renderer.text(f"{title}: (none)")
        renderer.text("")
        return
    sorted_items = sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))
    top = sorted_items[:n]
    label_width = max(len(label) for label, _ in top)

    renderer.text(f"{title}:")
    for label, count in top:
        if show_pct:
            value = _format_count_with_pct(count, total)
        else:
            value = f"{count:,}"
        renderer.text(f"  {label:<{label_width}}  {value}")
    if len(sorted_items) > n:
        renderer.text(f"  … and {len(sorted_items) - n} more")
    renderer.text("")


def _render_timeline(renderer: Any, buckets: dict[str, int]) -> None:
    """Render a horizontal-bar timeline of event counts per 15-min bucket.

    Bars use Unicode block characters; the longest bucket gets the full
    width, all others scale linearly. We render every bucket between
    ``min(buckets)`` and ``max(buckets)`` even if the count is zero so
    the timeline shows gaps; absent that, a 4-hour journal with a
    1-hour quiet period would visually collapse onto one row.
    """
    if not buckets:
        renderer.text("Timeline: (no events)")
        renderer.text("")
        return

    sorted_keys = sorted(buckets.keys())
    first_key = datetime.fromisoformat(sorted_keys[0].replace("Z", "+00:00"))
    last_key = datetime.fromisoformat(sorted_keys[-1].replace("Z", "+00:00"))

    # Pad missing buckets with zero counts so gaps render visibly.
    full_buckets: dict[str, int] = {}
    cursor = first_key
    while cursor <= last_key:
        key = _format_iso(cursor)
        full_buckets[key] = buckets.get(key, 0)
        cursor = cursor + _TIMELINE_BUCKET

    max_count = max(full_buckets.values()) or 1
    bar_width = 30  # max horizontal characters

    renderer.text("Timeline (events per 15-minute bucket):")
    for key, count in full_buckets.items():
        # Build the bar: scale count to bar_width, pick a final partial
        # block from _BAR_BLOCKS for the fractional remainder.
        if count <= 0:
            bar = ""
        else:
            fractional = (count / max_count) * bar_width
            full_blocks = int(fractional)
            remainder_idx = round((fractional - full_blocks) * 8)
            bar = "█" * full_blocks
            if remainder_idx > 0 and full_blocks < bar_width:
                bar += _BAR_BLOCKS[remainder_idx]
        # Print only HH:MM–HH:MM of the bucket window — full ISO would
        # dwarf the bar visually.
        bucket_dt = datetime.fromisoformat(key.replace("Z", "+00:00"))
        end_dt = bucket_dt + _TIMELINE_BUCKET
        label = f"{bucket_dt.strftime('%H:%M')}–{end_dt.strftime('%H:%M')}"
        renderer.text(f"  {label}  {count:>7,}  {bar}")
    renderer.text("")


def _render_errors(
    renderer: Any,
    errors: list[dict[str, Any]],
    *,
    limit: int,
) -> None:
    """Render up to ``limit`` errors with short payload previews."""
    if not errors:
        return
    renderer.text(f"Errors (severity >= error): {len(errors)} total")
    shown = errors[:limit] if limit > 0 else []
    for err in shown:
        renderer.text(f"  {err['time']}  {err['kind']}")
        # One-line payload preview — full payload is in --json mode.
        payload_preview = _truncate(
            ", ".join(f"{k}={v}" for k, v in sorted((err.get("payload") or {}).items())),
            limit=80,
        )
        if payload_preview:
            renderer.text(f"    {payload_preview}")
    if len(errors) > len(shown):
        renderer.text(f"  … and {len(errors) - len(shown)} more (use --json for full list)")
    renderer.text("")


def _render_history_pretty(
    renderer: Any,
    stats: HistoryStats,
    *,
    limit_errors: int,
    show_timeline: bool,
) -> None:
    """Render :class:`HistoryStats` in human-readable pretty mode."""
    # Header
    renderer.text(
        f"{stats.run_id} (events.jsonl: {_format_bytes(stats.size_bytes)}, "
        f"{stats.total_events:,} events)"
    )
    renderer.text("-" * 60)

    if stats.total_events == 0:
        renderer.text("no events")
        return

    # Duration / first / last
    if stats.first_event and stats.last_event:
        renderer.text(
            f"Duration:        {_format_duration(stats.duration_seconds)} "
            f"({stats.first_event['time']} -> {stats.last_event['time']})"
        )
        renderer.text(
            f"First event:     {stats.first_event['kind']} @ {stats.first_event['time']}"
        )
        renderer.text(
            f"Last event:      {stats.last_event['kind']} @ {stats.last_event['time']}"
        )
    renderer.text("")

    _render_top_n_table(
        renderer,
        title="By source",
        mapping=stats.by_source,
        total=stats.total_events,
        n=10,
    )

    # By-severity: show ALL 5 levels in canonical order so absences
    # are visible ("error  0  (—)" rather than a missing row).
    renderer.text("By severity:")
    sev_labels = ["debug", "info", "warning", "error", "critical"]
    label_width = max(len(label) for label in sev_labels)
    for sev in sev_labels:
        count = stats.by_severity.get(sev, 0)
        value = _format_count_with_pct(count, stats.total_events) if count else "0   (—)"
        renderer.text(f"  {sev:<{label_width}}  {value}")
    renderer.text("")

    _render_top_n_table(
        renderer,
        title="Top 10 event kinds",
        mapping=stats.by_kind,
        total=stats.total_events,
        n=10,
        show_pct=False,
    )

    if stats.by_stage:
        _render_top_n_table(
            renderer,
            title="By stage (where stage_id is set)",
            mapping=stats.by_stage,
            total=stats.total_events,
            n=10,
            show_pct=False,
        )

    if limit_errors != 0 and stats.errors:
        _render_errors(renderer, stats.errors, limit=limit_errors)

    if show_timeline:
        _render_timeline(renderer, stats.timeline_buckets_15min)

    if stats.unknown_events:
        renderer.text(
            f"Unknown / malformed events: {stats.unknown_events:,} "
            "(forward-compat or torn-write residue — see journal_reader.py)"
        )


def _render_history_json(stats: HistoryStats, *, errors_limit_json: int = 50) -> None:
    """Dump :class:`HistoryStats` as JSON on stdout.

    ``errors_limit_json`` caps how many errors land in the JSON output —
    matches the spec ("first 50 errors with full payload"). The retained
    in-memory list is bounded separately by :data:`_MAX_ERRORS_RETAINED`.
    """
    payload = asdict(stats)
    payload["journal_path"] = str(stats.journal_path)
    if errors_limit_json > 0:
        payload["errors"] = stats.errors[:errors_limit_json]
        if len(stats.errors) > errors_limit_json:
            payload["errors_truncated"] = True
            payload["errors_total"] = len(stats.errors)
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _resolve_journal_path(*, run_id: str, runs_dir: str | None) -> Path:
    """Resolve ``runs_dir + run_id + events.jsonl`` per the spec.

    Default ``runs_dir`` is :attr:`RuntimeSettings.runs_base_dir` (read
    via :func:`load_runtime_settings`); ``--runs-dir`` overrides it. We
    accept the override either as a base dir (``.../runs``) — in which
    case we append ``<run_id>/events.jsonl`` — or as a fully qualified
    path to the journal itself, in which case we use it verbatim. The
    latter is convenient for one-off forensics where you've copied a
    journal out of its run directory.
    """
    # Import lazily so ``ryotenkai events --help`` doesn't pay for the
    # settings module import on every CLI invocation.
    from ryotenkai_shared.config.runtime import load_runtime_settings

    if runs_dir is None:
        base_dir = load_runtime_settings().runs_base_dir
    else:
        base_dir = Path(runs_dir).expanduser()

    # If the override points directly at an events.jsonl file, use it
    # verbatim. Otherwise treat it as a runs-base-dir and compose.
    if base_dir.is_file() and base_dir.name == "events.jsonl":
        return base_dir
    return base_dir / run_id / "events.jsonl"


@events_app.command("history")
@wrap_command
def history_cmd(
    ctx: typer.Context,
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            help="Run id whose on-disk events.jsonl to inspect.",
        ),
    ],
    runs_dir: Annotated[
        str | None,
        typer.Option(
            "--runs-dir",
            help=(
                "Override the runs base directory (default: "
                "RuntimeSettings.runs_base_dir). May also point directly "
                "at an events.jsonl file."
            ),
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Dump HistoryStats as machine-readable JSON.",
        ),
    ] = False,
    limit_errors: Annotated[
        int,
        typer.Option(
            "--limit-errors",
            min=0,
            help="Max errors in pretty output (0 = omit errors section).",
        ),
    ] = 20,
    no_timeline: Annotated[
        bool,
        typer.Option(
            "--no-timeline",
            help="Skip the timeline section in pretty output.",
        ),
    ] = False,
    from_offset: Annotated[
        int | None,
        typer.Option(
            "--from-offset",
            help="Only analyse events with offset >= this value.",
        ),
    ] = None,
    to_offset: Annotated[
        int | None,
        typer.Option(
            "--to-offset",
            help="Only analyse events with offset <= this value.",
        ),
    ] = None,
) -> None:
    """Show aggregate statistics from a run's on-disk ``events.jsonl``.

    Unlike ``events metrics`` (which queries the live runtime registry
    over HTTP and is therefore restricted to active runs), this command
    works on completed runs too — the journal file persists on disk
    after the orchestrator releases the registry slot.

    The output covers totals, duration, by-source / by-severity /
    by-kind / by-stage breakdowns, the first 20 errors, and a
    15-minute-bucket timeline of event volume. Use ``--json`` for a
    machine-readable dump including full error payloads.
    """
    journal_path = _resolve_journal_path(run_id=run_id, runs_dir=runs_dir)

    if not journal_path.exists():
        raise die(
            f"run not found: {run_id}; journal {journal_path} does not exist",
            hint=(
                "check --run-id, or pass --runs-dir if your workspace "
                "uses a non-default runs directory"
            ),
            code=2,
        )

    # Lazy import — keeps ``ryotenkai events --help`` snappy.
    from ryotenkai_control.events.journal_reader import JournalReader

    reader = JournalReader(journal_path)
    stats = _compute_history_stats(
        reader,
        run_id=run_id,
        from_offset=from_offset,
        to_offset=to_offset,
    )

    state = ctx.ensure_object(CLIContext)

    if as_json:
        _render_history_json(stats)
        return

    renderer = get_renderer(state)
    if state.is_machine_readable:
        # ``-o json`` should still produce parseable JSON; route through
        # the dedicated dumper rather than the renderer (which would
        # render a Python repr for dataclass-heavy nested structures).
        _render_history_json(stats)
        return

    _render_history_pretty(
        renderer,
        stats,
        limit_errors=limit_errors,
        show_timeline=not no_timeline,
    )
    renderer.flush()


__all__ = ["HistoryStats", "events_app"]
