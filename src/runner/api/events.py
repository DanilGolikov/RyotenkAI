"""WebSocket event stream.

Endpoint: ``WS /api/v1/jobs/{job_id}/events?since=<offset>``

Behaviour:
1. Verify the FSM is currently bound to ``job_id``; close 4404 otherwise.
2. Subscribe to the EventBus from ``since`` (default 0).
3. Replay everything ≥ ``since`` that's still in the ring buffer,
   then live-stream new events.
4. If the bus' buffer has truncated past ``since``, close 4410
   (Gone) — the client falls back to the durable JSONL on disk.
5. Close cleanly when the FSM enters a terminal state — the
   subscriber loop reads the final event then drains.

Close codes (RFC 6455 application range 4xxx):
- 4000  client cancelled / WebSocketDisconnect
- 4404  job_id not bound to the active FSM
- 4410  buffer truncated past requested ``since``
- 4422  invalid query (negative ``since``, non-integer)

Phase 1 ships replay + live streaming. The "close on terminal state"
hook lands in Phase 2 once the supervisor emits the final lifecycle
event; for now the connection stays open until the client closes it
or the bus is shut down (lifespan exit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from src.runner.api.deps import get_bus, get_fsm
from src.runner.event_bus import BufferTruncatedError

if TYPE_CHECKING:
    pass

router = APIRouter(tags=["events"])


# Custom close codes — keep within the 4000-4999 application-private
# range so they don't collide with the IANA reserved set.
_CLOSE_NOT_FOUND = 4404
_CLOSE_GONE = 4410
_CLOSE_INVALID = 4422


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

    try:
        async for event in bus.subscribe(since=since):
            await websocket.send_json(event.to_dict())
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
