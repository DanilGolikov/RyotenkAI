"""FastAPI entry point for the in-pod job server.

Used by ``uvicorn src.runner.main:app`` (the entrypoint.sh inside
the docker image runs exactly this command).

Phase 1 owns:
- ``GET /healthz`` — liveness probe.
- ``GET /readyz`` — readiness probe (true once the FSM has
  ``restore_or_init``-ed).
- ``GET /version`` — pinned image tag for client compatibility checks.
- ``POST /api/v1/jobs`` — multipart submit.
- ``GET  /api/v1/jobs/{job_id}`` — current snapshot.
- ``POST /api/v1/jobs/{job_id}/stop`` — graceful stop request.
- ``WS   /api/v1/jobs/{job_id}/events`` — replay + live event stream.
- ``POST /api/v1/internal/events`` — trainer-side event push (loopback).

Singletons:
The FSM and the EventBus live on ``app.state`` and are constructed
in :func:`_lifespan` so each :class:`fastapi.testclient.TestClient`
spins up its own pair (no cross-test contamination).

The app binds ``127.0.0.1:8080`` per :file:`docker/training/entrypoint.sh`
— never 0.0.0.0. Mac control plane reaches it through ``ssh -L``.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from src.runner.__about__ import RUNTIME_IMAGE
from src.runner.api import events as events_api
from src.runner.api import internal as internal_api
from src.runner.api import jobs as jobs_api
from src.runner.event_bus import EventBus
from src.runner.state import JobLifecycleFSM

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

API_V1_PREFIX = "/api/v1"


def _resolve_workspace() -> Path:
    """Workspace directory the FSM persists state under.

    Defaults to ``/workspace`` inside the docker container — the
    canonical persistent volume mount on RunPod. Override via
    ``RYOTENKAI_WORKSPACE`` for tests / dev runs outside docker.
    """
    raw = os.environ.get("RYOTENKAI_WORKSPACE", "/workspace")
    return Path(raw)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> "AsyncIterator[None]":
    """Wire up FSM + EventBus on ``app.state``.

    Order:
    1. Build FSM at the workspace dir; call ``restore_or_init()``
       so a container restart resumes the previous state (or
       transitions an unsafe state to ``failed`` — see
       :meth:`JobLifecycleFSM.restore_or_init`).
    2. Build EventBus with the env-driven default capacity.
    3. Yield — endpoints serve traffic.
    4. On shutdown, close the bus so subscribers drain cleanly.
    """
    fsm = JobLifecycleFSM(workspace_dir=_resolve_workspace())
    fsm.restore_or_init()
    bus = EventBus()

    app.state.fsm = fsm
    app.state.bus = bus

    try:
        yield
    finally:
        bus.close()


def create_app() -> FastAPI:
    """Build a fresh app instance (factory pattern — see module docstring)."""
    app = FastAPI(
        title="RyotenkAI Runner",
        version=RUNTIME_IMAGE,
        description=(
            "In-pod control plane for remote training. "
            "Loopback-only (127.0.0.1:8080); reached from Mac via "
            "ssh -L tunnel."
        ),
        lifespan=_lifespan,
    )

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        """Liveness — always 200 while the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["meta"])
    def readyz() -> dict[str, object]:
        """Readiness — FSM has restored, bus is open."""
        bus_open = not app.state.bus.is_closed
        return {
            "status": "ready" if bus_open else "draining",
            "bus_open": bus_open,
            "fsm_restored": True,  # restore_or_init() ran in lifespan
        }

    @app.get("/version", tags=["meta"])
    def version() -> dict[str, str]:
        """Pinned image identity for client compatibility checks."""
        return {"image": RUNTIME_IMAGE}

    app.include_router(jobs_api.router, prefix=API_V1_PREFIX)
    app.include_router(internal_api.router, prefix=API_V1_PREFIX)
    app.include_router(events_api.router, prefix=API_V1_PREFIX)

    return app


app = create_app()


__all__ = ["API_V1_PREFIX", "app", "create_app"]
