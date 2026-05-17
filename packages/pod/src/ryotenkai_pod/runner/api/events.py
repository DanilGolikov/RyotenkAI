"""WebSocket event stream + HTTP replay fallback.

Endpoints:

* ``WS  /api/v1/jobs/{job_id}/events?since=<offset>`` — replay then
  live-stream events to the Mac control plane. Phase 2
  (ethereal-tumbling-patterson) forwards typed envelopes via
  :func:`envelope_to_wire`, which adds a back-compat ``timestamp``
  alias on top of the canonical envelope ``time`` field so existing
  control-side consumers (job_client.py / training_monitor.py) keep
  reading the same JSON shape.
* ``GET /api/v1/jobs/{job_id}/events/replay?after_offset=<int>``
  (NEW Phase 2) — cursor-paginated NDJSON replay that the control side
  hits when its WebSocket reconnect detects a gap older than the
  ring's tail. Pulls from the on-disk journal so the runner can serve
  events even after a 5+ minute Mac sleep.

WS close codes (RFC 6455 application range 4xxx):

* 4000  client cancelled / WebSocketDisconnect
* 4404  job_id not bound to the active FSM
* 4410  buffer truncated past requested ``since`` and not on disk
* 4422  invalid query (negative ``since``, non-integer)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query, Request, Response, WebSocket, WebSocketDisconnect, status

from ryotenkai_pod.runner.api.deps import get_bus, get_fsm
from ryotenkai_pod.runner.event_bus import (
    BufferTruncatedError,
    DiskJournalExhausted,
    EventBus,
    envelope_to_wire,
)
from ryotenkai_shared.errors import JobNotFoundError
from ryotenkai_shared.events import BaseEvent

router = APIRouter(tags=["events"])


# Custom close codes — keep within the 4000-4999 application-private range.
_CLOSE_NOT_FOUND = 4404
_CLOSE_GONE = 4410
_CLOSE_INVALID = 4422

# Cap on the HTTP replay batch size so a single request can't OOM the
# runner. Mirrors the SSOT-side limit documented in the plan.
_REPLAY_LIMIT_DEFAULT = 1000
_REPLAY_LIMIT_MAX = 10_000


async def _subscribe_with_disk_fallback(
    bus: EventBus, since: int,
) -> AsyncIterator[BaseEvent]:
    """Yield typed envelopes from disk first (when needed), then from the ring.

    Phase 2 contract:

    * ``since >= ring_oldest`` (or ring empty) → straight ring subscribe.
    * ``since < ring_oldest`` AND journal attached → drain the journal
      from ``since`` up to (but not including) ``ring_oldest``, then
      hand off to ``bus.subscribe`` at exactly ``ring_oldest`` so the
      live stream picks up where disk replay stopped — no overlap.
    * ``since < ring_oldest`` AND journal absent → fall through to
      ``bus.subscribe`` which raises :class:`BufferTruncatedError`.
    * ``since < disk_oldest`` → :class:`DiskJournalExhausted`.
    """
    journal = bus.journal
    ring_oldest = bus.oldest_offset

    if ring_oldest is None or since >= ring_oldest:
        async for envelope in bus.subscribe(since=since, consumer_id="ws"):
            yield envelope
        return

    if journal is None:
        # Let ``subscribe`` raise BufferTruncatedError as before.
        async for envelope in bus.subscribe(since=since, consumer_id="ws"):
            yield envelope
        return

    disk_oldest = journal.oldest_persisted_offset()
    if disk_oldest is None or since < disk_oldest:
        raise DiskJournalExhausted(
            requested_offset=since,
            oldest_in_ring=ring_oldest,
            oldest_on_disk=disk_oldest,
        )

    # Stage 1 — disk records ``[since, ring_oldest)``.
    for envelope in journal.iter_envelopes(since=since):
        if envelope.offset >= ring_oldest:
            break
        if envelope.offset < since:
            continue
        yield envelope

    # Stage 2 — hand off to ring at exactly ``ring_oldest`` so we
    # neither duplicate records (no overlap) nor leave a gap.
    async for envelope in bus.subscribe(since=ring_oldest, consumer_id="ws"):
        yield envelope


@router.websocket("/jobs/{job_id}/events")
async def stream_events(
    websocket: WebSocket,
    job_id: str,
    since: int = Query(default=0, ge=0),
) -> None:
    fsm = get_fsm(websocket)  # type: ignore[arg-type]
    bus = get_bus(websocket)  # type: ignore[arg-type]

    snap = fsm.current()
    if snap is None or snap.job_id != job_id:
        await websocket.accept()
        await websocket.close(code=_CLOSE_NOT_FOUND, reason="job_not_found")
        return

    await websocket.accept()

    from ryotenkai_pod.runner.api._activity import send_ws_with_activity

    try:
        async for envelope in _subscribe_with_disk_fallback(bus, since=since):
            wire = envelope_to_wire(envelope)
            await send_ws_with_activity(
                websocket, wire, websocket.app.state,
            )
    except DiskJournalExhausted as exc:
        await websocket.close(
            code=_CLOSE_GONE,
            reason=f"disk_exhausted; oldest_on_disk={exc.oldest_on_disk}",
        )
        return
    except BufferTruncatedError as exc:
        await websocket.close(
            code=_CLOSE_GONE,
            reason=f"truncated; oldest_available={exc.oldest_available}",
        )
        return
    except ValueError as exc:
        await websocket.close(code=_CLOSE_INVALID, reason=str(exc)[:120])
        return
    except WebSocketDisconnect:
        return

    if websocket.client_state.name == "CONNECTED":
        await websocket.close(code=status.WS_1001_GOING_AWAY)


# ---------------------------------------------------------------------------
# HTTP replay (Phase 2 fallback)
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/{job_id}/events/replay",
    summary="Cursor-paginated NDJSON replay of pod-side events",
)
def replay_events(
    job_id: str,
    request: Request,
    after_offset: int = Query(default=0, ge=0),
    limit: int = Query(default=_REPLAY_LIMIT_DEFAULT, ge=1, le=_REPLAY_LIMIT_MAX),
    source: str | None = Query(default=None),
) -> Response:
    """Return up to ``limit`` envelopes with ``offset > after_offset``.

    Closes Risk R-01 (pod journal lost on cold WS reconnect): the
    control side hits this endpoint after WS drops past the ring's
    tail to back-fill from the on-disk journal. Response is
    ``application/x-ndjson`` so cursors can be processed without
    JSON-array buffering on either end.

    The response carries ``X-Next-Offset`` so the client can resume
    pagination on the next call (set to the last returned offset, or
    ``after_offset`` if no events matched). The body is line-by-line
    JSON of the same wire shape the WebSocket emits.
    """
    fsm = get_fsm(request)
    bus = get_bus(request)
    snap = fsm.current()
    if snap is None or snap.job_id != job_id:
        raise JobNotFoundError(
            detail=f"job_id={job_id!r} is not the active job",
            context={"job_id": job_id},
        )

    journal = bus.journal

    lines: list[str] = []
    last_offset = after_offset

    # Always serve from the journal when one is attached (covers
    # offsets older than the ring); fall back to the ring when no
    # journal is present.
    iterator = (
        journal.iter_envelopes(since=after_offset + 1)
        if journal is not None
        else bus.iter_buffered_envelopes()
    )
    for envelope in iterator:
        if envelope.offset <= after_offset:
            continue
        if source is not None and envelope.source != source:
            continue
        wire = envelope_to_wire(envelope)
        lines.append(json.dumps(wire, separators=(",", ":")))
        last_offset = envelope.offset
        if len(lines) >= limit:
            break

    body = "\n".join(lines) + ("\n" if lines else "")
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"X-Next-Offset": str(last_offset)},
    )
