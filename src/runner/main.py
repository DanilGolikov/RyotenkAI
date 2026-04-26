"""FastAPI entry point for the in-pod job server.

Used by ``uvicorn ryotenkai_runner.main:app`` (the package installs at
``ryotenkai_runner`` via ``pyproject``-managed ``[project.scripts]`` —
see Phase 0 / docker/training/Dockerfile.runtime).

Phase 0 ships:
- ``GET /healthz`` — kubelet/liveness probe.
- ``GET /readyz`` — readiness probe (always ready in Phase 0; Phase 1
  adds "FSM restored" gate).
- ``GET /version`` — package + image version for client compatibility checks.
- Three sub-routers mounted at the canonical paths (`/jobs`, `/internal`,
  events at root). They expose only stub endpoints right now — full
  semantics arrive in Phase 1.

The app listens on ``127.0.0.1:8080`` (loopback only) inside the pod
container; Mac control plane reaches it through ``ssh -L`` tunnel.
Never bind to ``0.0.0.0`` — that would expose the unauthenticated
internal API to anyone with network access to the pod.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from src.runner.__about__ import RUNTIME_IMAGE
from src.runner.api import events as events_api
from src.runner.api import internal as internal_api
from src.runner.api import jobs as jobs_api

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> "AsyncIterator[None]":
    """Phase 0 lifespan: nothing to set up yet.

    Phase 1 will:
    - Restore FSM from ``/workspace/.ryotenkai/state.jsonl``.
    - If last state is ``preparing`` or ``stopping``, transition to
      ``failed`` with reason ``container_restart_during_unsafe_state``.
    - Boot the supervisor / event bus singletons.
    """
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app.

    Factory pattern (rather than a module-level ``app = FastAPI(...)``)
    keeps the test surface clean — each test can spin up its own app
    instance with custom dependencies overridden.
    """
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
        """Liveness probe — always 200 while the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["meta"])
    def readyz() -> dict[str, str]:
        """Readiness probe — Phase 0 always ready; Phase 1 adds FSM-restored gate."""
        return {"status": "ready"}

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
