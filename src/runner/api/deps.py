"""FastAPI dependency resolvers for runner singletons.

Both the FSM and the EventBus live on ``app.state`` (set up in
:func:`src.runner.main._lifespan`). Endpoint handlers reach them
via :func:`Depends(get_fsm)` / :func:`Depends(get_bus)` rather than
imported globals — this keeps the test harness clean (each
``TestClient`` rebuilds the app and gets its own pair) and matches
the pattern used in :mod:`src.api.dependencies`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from src.runner.event_bus import EventBus
    from src.runner.state import JobLifecycleFSM

__all__ = ["get_bus", "get_fsm"]


def get_fsm(request: Request) -> "JobLifecycleFSM":
    """Return the FSM bound to the live FastAPI app."""
    return request.app.state.fsm  # type: ignore[no-any-return]


def get_bus(request: Request) -> "EventBus":
    """Return the EventBus bound to the live FastAPI app."""
    return request.app.state.bus  # type: ignore[no-any-return]
