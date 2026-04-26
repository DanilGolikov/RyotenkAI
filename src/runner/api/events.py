"""WebSocket event stream — Phase 0 placeholder.

Final endpoint: ``WS /jobs/{id}/events?since=<offset>`` — replays the
ring buffer from ``offset`` then live-streams new events. Closes on
job terminal state transition. See ``docs/plans/harmonic-rolling-crayon.md``
§ 8 (detach/reattach scenario).

Phase 0 mounts an empty router so the FastAPI app boots cleanly.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["events"])
