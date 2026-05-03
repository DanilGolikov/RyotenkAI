"""Phase 14.E (V3) — heartbeat-marking helper for API handlers.

Pre-14.E both the WebSocket events handler
(:func:`src.runner.api.events.subscribe_events`) and the REST job
GET handler (:func:`src.runner.api.jobs.get_job`) inlined a manual
``heartbeat = getattr(app.state, "heartbeat", None); if heartbeat
is not None: heartbeat.mark_active()`` block — heartbeat-tracking
logic leaked into endpoint code.

This module centralizes the pattern so the API layer doesn't carry
cross-cutting heartbeat knowledge. Phase 14.E § R-2 invariant test
pins the ordering: ``ws.send_json`` happens BEFORE ``mark_active``
so we only count successfully-delivered frames as activity.

The helper is sync (heartbeat.mark_active is sync, no I/O) so it's
safe to call inside async hot loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket


__all__ = ["mark_heartbeat_if_present", "send_ws_with_activity"]


def mark_heartbeat_if_present(app_state: Any) -> None:
    """Pull ``heartbeat`` from ``app_state`` and call ``mark_active()``
    if present.

    Defensive: ``app_state`` being ``None`` or lacking the
    ``heartbeat`` attribute is a no-op (some test contexts
    construct minimal app instances). The Phase 11.B production
    lifespan always wires the heartbeat, so the no-op path only
    fires in tests that explicitly skip it.

    Phase 14.E (V3): centralizes the
    ``getattr(app.state, "heartbeat", None)`` pattern that was
    inlined in API handlers pre-14.E.
    """
    if app_state is None:
        return
    heartbeat = getattr(app_state, "heartbeat", None)
    if heartbeat is not None:
        heartbeat.mark_active()


async def send_ws_with_activity(
    ws: "WebSocket", payload: dict, app_state: Any,
) -> None:
    """Send a WebSocket frame and mark the heartbeat after successful
    delivery.

    Phase 14.E (V3): the WS handler used to inline:

        await ws.send_json(payload)
        if heartbeat is not None:
            heartbeat.mark_active()

    The order matters — only successfully-yielded frames count as
    Mac liveness signal. If ``send_json`` raises, the exception
    propagates (caller handles it) and the heartbeat is NOT marked.
    """
    await ws.send_json(payload)
    mark_heartbeat_if_present(app_state)
