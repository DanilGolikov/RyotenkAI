"""Event stream DTOs + WebSocket close codes."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ._strict import _StrictModel


class EventResponse(_StrictModel):
    """Single event as broadcast over WebSocket / replayed via REST."""

    offset: int
    timestamp: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


# Custom WebSocket close codes — keep within the 4000-4999
# application-private range so they don't collide with the IANA
# reserved set. Mirror the constants in
# ``ryotenkai_pod.runner.api.events`` so the Mac-side client can
# switch on them without importing runner-side code (the pod
# package may not be installed in the Mac dev environment).
WS_CLOSE_NOT_FOUND = 4404
WS_CLOSE_GONE = 4410
WS_CLOSE_INVALID = 4422


__all__ = [
    "EventResponse",
    "WS_CLOSE_GONE",
    "WS_CLOSE_INVALID",
    "WS_CLOSE_NOT_FOUND",
]
