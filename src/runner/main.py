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
from typing import TYPE_CHECKING, Protocol

from fastapi import FastAPI

from src.runner.__about__ import RUNTIME_IMAGE
from src.runner.api import events as events_api
from src.runner.api import internal as internal_api
from src.runner.api import jobs as jobs_api
from src.runner.event_bus import EventBus
from src.runner.plugin_unpacker import PluginUnpacker
from src.runner.pod_stopper import PodStopper, stop_pod_on_terminal
from src.runner.state import JobLifecycleFSM
from src.runner.supervisor import Supervisor, TerminalHook

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

API_V1_PREFIX = "/api/v1"


class _SupervisorFactory(Protocol):
    """Constructor signature shared by :class:`Supervisor` and any
    test double. Keep narrow — only the boot-time call signature.

    ``terminal_hook`` is optional: production passes the RunPod
    auto-stop callback; tests omit it (the :class:`MockSupervisor`
    fixture takes a 2-arg call shape).
    """

    def __call__(
        self,
        fsm: JobLifecycleFSM,
        bus: EventBus,
        *,
        terminal_hook: TerminalHook | None = ...,
    ) -> Supervisor: ...


def _resolve_workspace() -> Path:
    """Workspace directory the FSM persists state under.

    Defaults to ``/workspace`` inside the docker container — the
    canonical persistent volume mount on RunPod. Override via
    ``RYOTENKAI_WORKSPACE`` for tests / dev runs outside docker.
    """
    raw = os.environ.get("RYOTENKAI_WORKSPACE", "/workspace")
    return Path(raw)


def _make_lifespan(supervisor_factory: _SupervisorFactory):  # type: ignore[no-untyped-def]
    """Build a lifespan that uses ``supervisor_factory`` to build the
    Supervisor.

    Tests pass a mock factory through :func:`create_app`; production
    uses :class:`Supervisor` directly. Wrapping in a closure keeps
    the lifespan ``@asynccontextmanager`` shape FastAPI expects
    while still letting us swap implementations.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> "AsyncIterator[None]":
        """Wire up FSM + EventBus + Supervisor on ``app.state``.

        Order:
        1. Build FSM at the workspace dir; call ``restore_or_init()``
           so a container restart resumes the previous state (or
           transitions an unsafe state to ``failed`` — see
           :meth:`JobLifecycleFSM.restore_or_init`).
        2. Build EventBus with the env-driven default capacity.
        3. Build :class:`PodStopper` and wire :func:`stop_pod_on_terminal`
           as the supervisor's terminal hook so RunPod billing stops
           the moment the FSM lands in ``completed`` / ``failed`` /
           ``cancelled``. The hook is env-driven (see
           :mod:`src.runner.pod_stopper`); when ``RUNPOD_AUTO_STOP``
           is unset / ``false`` it returns ``"disabled"`` and the pod
           keeps running — same contract as the legacy bash wrapper.
        4. Build Supervisor (or test double) bound to (fsm, bus, hook).
        5. Yield — endpoints serve traffic.
        6. On shutdown, stop the trainer first so its SIGTERM-driven
           save can write through, *then* close the bus so the final
           ``trainer_exited`` event reaches subscribers.
        """
        workspace = _resolve_workspace()
        fsm = JobLifecycleFSM(workspace_dir=workspace)
        fsm.restore_or_init()
        bus = EventBus()

        # The plugin unpacker is stateless beyond its workspace path,
        # so we build it once at boot and reuse across requests.
        plugin_unpacker = PluginUnpacker(workspace_dir=workspace)

        pod_stopper = PodStopper()

        async def _terminal_hook(terminal_state: str) -> None:
            await stop_pod_on_terminal(
                pod_stopper,
                terminal_state=terminal_state,
                bus_publish=bus.publish,
            )

        supervisor = supervisor_factory(fsm, bus, terminal_hook=_terminal_hook)

        app.state.fsm = fsm
        app.state.bus = bus
        app.state.pod_stopper = pod_stopper
        app.state.plugin_unpacker = plugin_unpacker
        app.state.supervisor = supervisor

        try:
            yield
        finally:
            await supervisor.shutdown()
            bus.close()

    return _lifespan


def create_app(
    supervisor_factory: _SupervisorFactory | None = None,
) -> FastAPI:
    """Build a fresh app instance.

    ``supervisor_factory`` is a callable ``(fsm, bus) -> Supervisor``
    used to construct the supervisor in the lifespan. Defaults to
    the real :class:`Supervisor`. Tests pass :class:`MockSupervisor`
    (or any other test double matching the protocol) for unit
    coverage of the API layer without paying for real subprocess
    semantics.
    """
    factory: _SupervisorFactory = supervisor_factory or Supervisor
    app = FastAPI(
        title="RyotenkAI Runner",
        version=RUNTIME_IMAGE,
        description=(
            "In-pod control plane for remote training. "
            "Loopback-only (127.0.0.1:8080); reached from Mac via "
            "ssh -L tunnel."
        ),
        lifespan=_make_lifespan(factory),
    )

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        """Liveness — always 200 while the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["meta"])
    def readyz() -> dict[str, object]:
        """Readiness — FSM has restored, bus is open, supervisor wired."""
        bus_open = not app.state.bus.is_closed
        return {
            "status": "ready" if bus_open else "draining",
            "bus_open": bus_open,
            "fsm_restored": True,  # restore_or_init() ran in lifespan
            "supervisor_running": app.state.supervisor.is_running,
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
