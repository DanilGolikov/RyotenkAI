"""HTTP replay + SSE streaming for unified events (Phase 6.a).

Two endpoints, both prefixed by ``/runs/{run_id:path}/events``:

* ``GET /runs/{run_id}/events`` — bounded NDJSON replay.

  Query params:

  - ``after_offset`` (int, default -1 → start from offset 0).
  - ``limit`` (int, default 1000, capped at 10_000).
  - ``type_prefix`` (str, optional) — keep only envelopes whose
    ``kind`` starts with the prefix.
  - ``severity`` (csv, optional) — keep only envelopes whose
    ``severity`` is in the set.
  - ``stage_id`` (str, optional) — exact-match filter.
  - ``source`` (str, optional) — exact-match filter on the source URI.

  Response: ``application/x-ndjson`` with one envelope per line.
  ``X-Next-Offset`` response header carries the highest offset emitted
  so the client can resume with ``after_offset=<value>``.

* ``GET /runs/{run_id}/events/stream`` — SSE catchup-then-live.

  Same filter params; the cursor comes from the ``Last-Event-ID``
  header (takes precedence) or ``after_offset`` query (fallback to 0).

  Implementation (closes R-19):

  1. Subscribe to the run's :class:`InMemoryBus` FIRST so live events
     produced during steps 2-3 are not lost.
  2. Capture the bus's newest offset ``M`` at subscription time.
  3. Replay journal ``(start_offset, M]`` filtered by predicates.
  4. Drain the bus from cursor ``M`` (i.e. yield events with
     ``offset > M``), forwarding through the same filter chain.
  5. Send ``: keepalive`` every 15 s so intermediate proxies keep the
     connection open.

The bus is looked up through :class:`EventEmitterRegistry`. If the run
already completed and the orchestrator deregistered its emitter, the
HTTP replay endpoint still works (journal is on disk) but the SSE
stream falls back to "replay-only mode" — it serves the historical
window and closes once the journal is exhausted.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse

from ryotenkai_control.api.dependencies import resolve_run_dir
from ryotenkai_control.events import EventEmitterRegistry, JournalReader, slice_journal
from ryotenkai_shared.errors import InternalError, JobSpecInvalidError
from ryotenkai_shared.events import UNKNOWN_OFFSET, BaseEvent
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable
    from pathlib import Path

    from ryotenkai_control.events.in_memory_bus import InMemoryBus


logger = get_logger(__name__)


__all__ = ["router"]


router = APIRouter(prefix="/runs/{run_id:path}/events", tags=["events"])


DEFAULT_LIMIT = 1000
MAX_LIMIT = 10_000
JOURNAL_FILENAME = "events.jsonl"
KEEPALIVE_INTERVAL_S = 15.0
VALID_SEVERITIES: frozenset[str] = frozenset(
    {"debug", "info", "warning", "error", "critical"},
)


# ---------------------------------------------------------------------------
# Filter predicates (shared between HTTP and SSE)
# ---------------------------------------------------------------------------


def _parse_severity_csv(raw: str | None) -> frozenset[str] | None:
    """Parse ``severity`` query into a normalized set or ``None``.

    Returns ``None`` if the parameter was not supplied (filter inactive).
    Raises :class:`HTTPException(400)` on invalid severity tokens — the
    caller almost certainly typo'd a name and silently ignoring would
    hide the bug.
    """
    if raw is None:
        return None
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None
    invalid = [t for t in tokens if t not in VALID_SEVERITIES]
    if invalid:
        raise JobSpecInvalidError(
            detail=f"invalid severity value(s): {invalid}",
            context={"invalid": invalid},
        )
    return frozenset(tokens)


def _make_filter(
    *,
    type_prefix: str | None,
    severity: frozenset[str] | None,
    stage_id: str | None,
    source: str | None,
) -> Callable[[BaseEvent], bool]:
    """Compose a predicate over :class:`BaseEvent` from the query filters.

    Each filter is short-circuited when its parameter is ``None``;
    callers that want "match everything" pass all ``None`` and get the
    identity predicate.
    """
    def _predicate(event: BaseEvent) -> bool:
        if type_prefix is not None and not event.kind.startswith(type_prefix):
            return False
        if severity is not None and event.severity not in severity:
            return False
        if stage_id is not None and event.stage_id != stage_id:
            return False
        return not (source is not None and event.source != source)
    return _predicate


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _journal_path(run_dir: Path) -> Path:
    return run_dir / JOURNAL_FILENAME


def _envelope_to_jsonline(event: BaseEvent) -> str:
    """Serialize an envelope as a single NDJSON line (no length prefix).

    The wire format we expose to clients is plain JSON-per-line; the
    length prefix is an on-disk concern only.
    """
    return event.model_dump_json() + "\n"


def _resolve_start_offset(
    *,
    last_event_id: str | None,
    after_offset: int | None,
) -> int:
    """Pick the SSE start cursor (Last-Event-ID > after_offset > 0).

    ``Last-Event-ID`` is treated as the offset of the LAST event the
    client received; replay resumes with offsets strictly greater than
    that value. ``after_offset`` from the query has the same exclusive
    semantic.
    """
    if last_event_id is not None:
        stripped = last_event_id.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError as exc:
                raise JobSpecInvalidError(
                    detail=(
                        f"Last-Event-ID must be an integer, "
                        f"got {last_event_id!r}"
                    ),
                ) from exc
    if after_offset is not None:
        return after_offset
    return -1


# ---------------------------------------------------------------------------
# HTTP replay
# ---------------------------------------------------------------------------


@router.get("", response_class=StreamingResponse)
def list_events(
    after_offset: int = Query(default=-1, ge=-1),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    type_prefix: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    stage_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
    run_dir: Path = Depends(resolve_run_dir),
) -> StreamingResponse:
    """Stream a filtered slice of the run's events journal as NDJSON.

    On success the response advertises ``X-Next-Offset`` — the next
    ``after_offset`` cursor for paginated reads. When no events match
    the filter set, the header carries the supplied ``after_offset``
    unchanged.
    """
    journal_path = _journal_path(run_dir)
    if not journal_path.exists():
        # Journal not yet created (run still in prepare phase) — return
        # an empty stream with the unchanged cursor.
        response = StreamingResponse(
            iter(()),
            media_type="application/x-ndjson",
        )
        response.headers["X-Next-Offset"] = str(after_offset)
        return response

    parsed_severity = _parse_severity_csv(severity)
    predicate = _make_filter(
        type_prefix=type_prefix,
        severity=parsed_severity,
        stage_id=stage_id,
        source=source,
    )

    # Materialize on the request thread — fastapi.StreamingResponse can
    # accept any iterable; we keep linear-scan cost in the caller so an
    # exception during read is mapped to a 500 by the exception
    # handlers cleanly rather than half-streaming a body.
    try:
        events = _collect_http_slice(
            journal_path=journal_path,
            after_offset=after_offset,
            limit=limit,
            predicate=predicate,
        )
    except Exception as exc:
        logger.warning(
            "[events] HTTP replay failed for %s: %s: %s",
            run_dir.name,
            type(exc).__name__,
            exc,
        )
        raise InternalError(detail=str(exc), cause=exc) from exc

    next_offset = events[-1].offset if events else after_offset
    body_lines = [_envelope_to_jsonline(e) for e in events]
    response = StreamingResponse(
        iter(body_lines),
        media_type="application/x-ndjson",
    )
    response.headers["X-Next-Offset"] = str(next_offset)
    return response


def _collect_http_slice(
    *,
    journal_path: Path,
    after_offset: int,
    limit: int,
    predicate: Callable[[BaseEvent], bool],
) -> list[BaseEvent]:
    """Read up to ``limit`` filtered envelopes with offset > after_offset."""
    reader = JournalReader(journal_path)
    collected: list[BaseEvent] = []
    # ``slice_journal`` yields ``offset > after_offset`` and filters out
    # UnknownEvent torn-write residue already. We pass ``limit=None``
    # because predicates may reject events — capping the underlying
    # iterator would underflow the requested limit when many events are
    # filtered.
    for envelope in slice_journal(reader, after_offset=after_offset, limit=None):
        if envelope.offset == UNKNOWN_OFFSET:
            continue
        if not predicate(envelope):
            continue
        collected.append(envelope)
        if len(collected) >= limit:
            break
    return collected


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


@router.get("/stream")
async def stream_events(
    request: Request,
    after_offset: int | None = Query(default=None, ge=-1),
    type_prefix: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    stage_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    run_dir: Path = Depends(resolve_run_dir),
) -> StreamingResponse:
    """Subscribe-first SSE stream of the run's events.

    Closes R-19 (subscribe-first ordering): the bus subscription is
    opened BEFORE journal catchup so a live event published between
    catchup and live-tail is delivered exactly once.
    """
    parsed_severity = _parse_severity_csv(severity)
    predicate = _make_filter(
        type_prefix=type_prefix,
        severity=parsed_severity,
        stage_id=stage_id,
        source=source,
    )
    start_offset = _resolve_start_offset(
        last_event_id=last_event_id,
        after_offset=after_offset,
    )

    journal_path = _journal_path(run_dir)
    registry = EventEmitterRegistry.instance()
    emitter = registry.get(run_dir.name)
    bus = emitter.bus if emitter is not None else None

    generator = _sse_event_stream(
        request=request,
        journal_path=journal_path,
        bus=bus,
        start_offset=start_offset,
        predicate=predicate,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            # Avoid intermediate caches buffering the response — Cache-
            # Control: no-cache is also the EventSource spec's
            # recommended hint.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_format(
    *,
    event_id: int,
    kind: str,
    data: str,
) -> str:
    """Format a single SSE frame.

    The trailing blank line is the SSE record terminator; the
    ``data:`` payload is single-line JSON because newlines inside the
    payload would split the record.
    """
    return f"id: {event_id}\nevent: {kind}\ndata: {data}\n\n"


def _format_event_frame(event: BaseEvent) -> str:
    return _sse_format(
        event_id=event.offset,
        kind=event.kind,
        data=event.model_dump_json(),
    )


def _format_error_frame(code: int, message: str) -> str:
    payload = json.dumps({"code": code, "message": message})
    return f"event: error\ndata: {payload}\n\n"


async def _sse_event_stream(
    *,
    request: Request,
    journal_path: Path,
    bus: InMemoryBus | None,
    start_offset: int,
    predicate: Callable[[BaseEvent], bool],
) -> AsyncIterator[str]:
    """Async generator emitting the catchup-then-live SSE stream."""
    consumer_id = f"sse:{id(request):x}:{time.monotonic_ns()}"

    # ---- Step 1+2: Subscribe FIRST and capture bus snapshot ----
    subscription: AsyncIterator[BaseEvent] | None = None
    bus_snapshot_offset: int | None = None
    if bus is not None:
        bus_snapshot_offset = bus.newest_offset
        # Subscribe with after_offset=newest so we tail strictly NEW
        # events; the catchup phase below covers the historical window
        # up to and including ``bus_snapshot_offset``.
        try:
            subscription = bus.subscribe(
                consumer_id,
                after_offset=bus_snapshot_offset,
            ).__aiter__()
        except Exception as exc:
            logger.warning(
                "[events] SSE bus subscribe failed (replay-only fallback): "
                "%s: %s",
                type(exc).__name__,
                exc,
            )
            subscription = None
            bus_snapshot_offset = None

    last_delivered_offset = start_offset

    # ---- Step 3: Catchup from journal up to bus snapshot ----
    try:
        if journal_path.exists():
            reader = JournalReader(journal_path)
            # When the bus is unavailable (run already finished) we
            # replay everything past ``start_offset`` and close. With
            # a bus we bound the replay at the snapshot so no event is
            # delivered twice.
            for envelope in slice_journal(reader, after_offset=start_offset):
                if envelope.offset == UNKNOWN_OFFSET:
                    continue
                if (
                    bus_snapshot_offset is not None
                    and envelope.offset > bus_snapshot_offset
                ):
                    break
                if predicate(envelope):
                    yield _format_event_frame(envelope)
                    last_delivered_offset = envelope.offset
                if await request.is_disconnected():
                    return

        # ---- Step 4: live tail from bus ----
        if subscription is None:
            # No bus → replay-only mode. Emit a courtesy close to give
            # the client a signal that no more events will arrive on
            # this connection.
            yield _format_error_frame(
                code=4410,
                message="run completed; replay-only response exhausted",
            )
            return

        # Keepalive loop. Use a wait_for race between the next
        # subscription element and the keepalive deadline so a quiet
        # run still emits comments and proxies don't time out.
        live_iter = subscription
        while True:
            if await request.is_disconnected():
                return
            try:
                next_event = await asyncio.wait_for(
                    _safe_next(live_iter),
                    timeout=KEEPALIVE_INTERVAL_S,
                )
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if next_event is None:
                # Bus closed cleanly — terminate the stream.
                return
            if next_event.offset <= last_delivered_offset:
                continue
            if not predicate(next_event):
                continue
            yield _format_event_frame(next_event)
            last_delivered_offset = next_event.offset
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "[events] SSE stream failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        yield _format_error_frame(code=500, message=str(exc))


async def _safe_next(it: AsyncIterator[BaseEvent]) -> BaseEvent | None:
    """``anext`` with ``StopAsyncIteration`` translated to ``None``.

    Returning ``None`` rather than re-raising lets the stream loop fall
    through to a clean shutdown without an outer try/except wrapping
    every iteration.
    """
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        return None


# ---------------------------------------------------------------------------
# Re-exports used by tests
# ---------------------------------------------------------------------------


def _materialize_iter(events: Iterable[BaseEvent]) -> list[BaseEvent]:
    """Helper retained for tests that want to inspect an iterator's payload."""
    return list(events)
