"""WebSocket event stream.

Endpoint: ``WS /api/v1/jobs/{job_id}/events?since=<offset>``

Behaviour:
1. Verify the FSM is currently bound to ``job_id``; close 4404 otherwise.
2. Subscribe to the EventBus from ``since`` (default 0).
3. Replay everything ≥ ``since`` that's still in the ring buffer,
   then live-stream new events.
4. If the bus' buffer has truncated past ``since`` AND no disk
   journal is attached (Phase 12.B), close 4410 (Gone) — the client
   falls back to the durable JSONL on disk.
5. Phase 12.B — when an :class:`EventJournal` IS attached, transparently
   replay records older than the ring's tail from disk before
   handing off to the live ring. The subscriber sees a continuous
   monotonic offset stream regardless of whether the underlying
   storage is RAM or disk. ``DiskJournalExhausted`` (offset older
   than even the journal's oldest record) still maps to 4410.
6. Close cleanly when the FSM enters a terminal state — the
   subscriber loop reads the final event then drains.

Close codes (RFC 6455 application range 4xxx):
- 4000  client cancelled / WebSocketDisconnect
- 4404  job_id not bound to the active FSM
- 4410  buffer truncated past requested ``since`` and not on disk
- 4422  invalid query (negative ``since``, non-integer)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from ryotenkai_pod.runner.api.deps import get_bus, get_fsm
from ryotenkai_pod.runner.event_bus import (
    BufferTruncatedError,
    DiskJournalExhausted,
    Event,
    EventBus,
)

if TYPE_CHECKING:
    pass

router = APIRouter(tags=["events"])


# Custom close codes — keep within the 4000-4999 application-private
# range so they don't collide with the IANA reserved set.
_CLOSE_NOT_FOUND = 4404
_CLOSE_GONE = 4410
_CLOSE_INVALID = 4422


async def _subscribe_with_disk_fallback(
    bus: EventBus, since: int,
) -> AsyncIterator[Event]:
    """Yield events from disk first (when needed), then from the ring.

    Phase 12.B contract:
    * ``since >= ring_oldest`` (or ring empty) → straight ring subscribe.
    * ``since < ring_oldest`` AND journal attached → drain the journal
      from ``since`` up to (but not including) ``ring_oldest``, then
      hand off to ``bus.subscribe(since=ring_oldest)`` so the live
      stream picks up where disk replay stopped — no overlap.
    * ``since < ring_oldest`` AND journal absent → fall through to
      ``bus.subscribe`` which raises :class:`BufferTruncatedError`.
    * ``since < disk_oldest`` → :class:`DiskJournalExhausted`.
    """
    journal = bus.journal
    ring_oldest = bus.oldest_offset

    # Fast path: ring covers the request, OR ring is empty (in which
    # case ``subscribe`` will live-stream from publish forward).
    if ring_oldest is None or since >= ring_oldest:
        async for event in bus.subscribe(since=since):
            yield event
        return

    # Slow path — disk replay needed. Verify journal can serve us.
    if journal is None:
        # Let ``subscribe`` raise BufferTruncatedError as before.
        async for event in bus.subscribe(since=since):
            yield event
        return

    disk_oldest = journal.oldest_persisted_offset()
    if disk_oldest is None or since < disk_oldest:
        raise DiskJournalExhausted(
            requested_offset=since,
            oldest_in_ring=ring_oldest,
            oldest_on_disk=disk_oldest,
        )

    # Stage 1 — disk records ``[since, ring_oldest)``.
    for record in journal.iter_records(since=since):
        if record.offset >= ring_oldest:
            break
        yield Event(
            offset=record.offset,
            timestamp=record.ts,
            kind=record.kind,
            payload=dict(record.payload),
        )

    # Stage 2 — hand off to ring at exactly ``ring_oldest`` so we
    # neither duplicate records (no overlap) nor leave a gap.
    async for event in bus.subscribe(since=ring_oldest):
        yield event


@router.websocket("/jobs/{job_id}/events")
async def stream_events(
    websocket: WebSocket,
    job_id: str,
    since: int = Query(default=0, ge=0),
) -> None:
    # Resolve singletons — WS routes don't get FastAPI Depends() on
    # the dependency-injection decorators, so we reach into app.state
    # directly. ``deps.get_fsm`` / ``get_bus`` accept either Request
    # or WebSocket since both expose ``.app`` / ``.app.state``.
    fsm = get_fsm(websocket)  # type: ignore[arg-type]
    bus = get_bus(websocket)  # type: ignore[arg-type]

    # 404 check before accept — WS spec: server may close with a 4xxx
    # code immediately after handshake. Accept first, send the close
    # frame, return.
    snap = fsm.current()
    if snap is None or snap.job_id != job_id:
        await websocket.accept()
        await websocket.close(code=_CLOSE_NOT_FOUND, reason="job_not_found")
        return

    await websocket.accept()

    # Phase 14.E (V3) — heartbeat marking centralized in
    # :mod:`src.runner.api._activity`. Pre-14.E the WS handler
    # inlined the ``getattr(app.state, "heartbeat", ...)`` +
    # ``mark_active`` calls; now :func:`send_ws_with_activity`
    # owns the "send then mark" ordering. When Mac is asleep,
    # ``send_json`` hangs or raises (TCP backpressure / connection
    # loss) ⇒ mark_active never fires ⇒ heartbeat goes stale ⇒
    # correct. PodTerminator reads the ledger on terminal hooks.
    from ryotenkai_pod.runner.api._activity import send_ws_with_activity

    try:
        async for event in _subscribe_with_disk_fallback(bus, since=since):
            await send_ws_with_activity(
                websocket, event.to_dict(), websocket.app.state,
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
        # since=N where N > next_offset — client cursor is corrupt.
        await websocket.close(code=_CLOSE_INVALID, reason=str(exc)[:120])
        return
    except WebSocketDisconnect:
        # Client closed the connection — normal exit.
        return

    # The async iterator can also exit cleanly when the bus closes
    # (lifespan shutdown). Send a final close frame so the client
    # learns this isn't a network error.
    if websocket.client_state.name == "CONNECTED":
        await websocket.close(code=status.WS_1001_GOING_AWAY)
