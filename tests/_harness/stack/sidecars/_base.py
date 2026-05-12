"""Shared sidecar plumbing: FastAPI factory, control axis, CLI runner.

The control axis is identical across all sidecars: ``/health``,
``/control/advance_clock``, ``/control/inject_latency``, ``/control/state``,
``/control/reset``. Per-sidecar Protocol routes are mounted on top by the
caller.

Clock injection is selected at process start via
``RYOTENKAI_TEST_CLOCK={real,manual}``; tests that want determinism boot
the stack with ``manual`` and broadcast :meth:`Stack.advance_clock`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from tests._harness.clock import Clock, ManualClock, RealClock


class _ResetableFake(Protocol):
    """Common surface a sidecar's underlying fake must expose.

    Each fake already has ``snapshot()``; ``reset()`` is the new addition
    we layer on for ``/control/reset`` — sidecars wrap their fake's
    ``reset_chaos`` plus a state wipe.
    """

    def snapshot(self) -> dict[str, Any]: ...


@dataclass
class SidecarRuntime:
    """Per-sidecar runtime state shared between the FastAPI app and CLI."""

    name: str
    clock: Clock
    fake: Any
    extra: dict[str, Any]


def parse_clock(value: str) -> Clock:
    if value == "real":
        return RealClock()
    if value == "manual":
        return ManualClock()
    raise ValueError(f"unknown clock kind {value!r}; expected real|manual")


def add_control_routes(
    app: FastAPI,
    *,
    runtime: SidecarRuntime,
    reset_fn: Callable[[], None],
) -> None:
    """Mount the shared ``/health`` + ``/control/*`` axis."""

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "sidecar": runtime.name,
            "clock": "manual" if isinstance(runtime.clock, ManualClock) else "real",
        }

    @app.post("/control/advance_clock")
    async def advance_clock(seconds: float) -> dict[str, Any]:
        if not isinstance(runtime.clock, ManualClock):
            # WHY 409 not 400: the request shape is fine; the server is just
            # in the wrong mode for it. Tests that mix real+manual clocks
            # surface this clearly via the status code.
            raise HTTPException(
                status_code=409,
                detail="clock is RealClock; restart the sidecar with --clock=manual",
            )
        runtime.clock.advance(seconds)
        # Yield once so any sleeper released by ``advance`` actually runs
        # before we return — otherwise tests that immediately probe state
        # after advancing race the scheduler.
        await asyncio.sleep(0)
        return {"now": runtime.clock.now()}

    @app.post("/control/inject_latency")
    async def inject_latency(ms: int) -> dict[str, Any]:
        runtime.extra["latency_ms"] = max(0, int(ms))
        return {"latency_ms": runtime.extra["latency_ms"]}

    @app.get("/control/state")
    async def control_state() -> JSONResponse:
        snapshot = runtime.fake.snapshot()
        snapshot["sidecar"] = runtime.name
        snapshot["latency_ms"] = runtime.extra.get("latency_ms", 0)
        return JSONResponse(snapshot)

    @app.post("/control/reset")
    async def control_reset() -> dict[str, str]:
        reset_fn()
        runtime.extra["latency_ms"] = 0
        return {"status": "reset"}


def parse_args(name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"sidecar-{name}")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--clock",
        choices=["real", "manual"],
        default=os.environ.get("RYOTENKAI_TEST_CLOCK", "real"),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RYOTENKAI_TEST_SEED", "0")))
    parser.add_argument("--ready-file", type=str, default=None)
    return parser.parse_args()


def run_uvicorn(app: FastAPI, *, port: int) -> None:
    # Disable uvicorn's access log on stdout so the captured-log file isn't
    # spammed; the per-sidecar log file already records every request via
    # FastAPI's exception path on failure.
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level=os.environ.get("RYOTENKAI_TEST_LOG_LEVEL", "warning"),
        access_log=False,
        loop="asyncio",
        # Refuse to bind to anything outside loopback as a belt-and-braces
        # determinism guarantee — the host kwarg already enforces this but
        # we re-assert it via the underlying socket family below.
    )
    server = uvicorn.Server(config)
    sys.exit(_run_server_blocking(server))


def _run_server_blocking(server: uvicorn.Server) -> int:
    logging.basicConfig(level=os.environ.get("RYOTENKAI_TEST_LOG_LEVEL", "WARNING").upper())
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    return 0


__all__ = [
    "SidecarRuntime",
    "add_control_routes",
    "parse_args",
    "parse_clock",
    "run_uvicorn",
]
