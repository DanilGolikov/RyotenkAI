"""``fake-runpod`` HTTP sidecar.

Wraps :class:`tests._fakes.runpod.FakeRunPodAPI` behind a FastAPI app.
Maps the ``IRunPodAPI`` Protocol surface onto REST routes plus the
standard control axis (``/health``, ``/control/*``). Chaos surfaces
specific to RunPod (``inject_429``, ``inject_5xx``,
``inject_partial_response``, ``set_pod_state``, ``register_pod``) live
under ``/control/``.

CLI::

    python -m tests._harness.stack.sidecars.runpod_server --port=N
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ryotenkai_shared.infrastructure.runpod_api import (
    RunPodInfo,
    RunPodPartialResponseError,
    RunPodRateLimitedError,
    RunPodTransientError,
)
from tests._fakes.runpod import FakeRunPodAPI
from tests._harness.stack.sidecars._base import (
    SidecarRuntime,
    add_control_routes,
    parse_args,
    parse_clock,
    run_uvicorn,
)


def make_app(*, fake: FakeRunPodAPI, runtime: SidecarRuntime) -> FastAPI:
    app = FastAPI(title="fake-runpod-sidecar")

    def _reset() -> None:
        fake.reset_chaos()
        # Wipe registry — sidecar is reused between tests via /control/reset.
        fake._pods.clear()  # type: ignore[attr-defined]
        fake._call_history.clear()  # type: ignore[attr-defined]

    add_control_routes(app, runtime=runtime, reset_fn=_reset)

    # ---- Error → HTTP status mapping ----------------------------------
    # The Protocol's exception hierarchy maps onto RunPod's actual HTTP
    # behaviour. 429 for rate-limit, 503 for transient, 502 for partial
    # — matches the real provider so client retry/backoff logic exercises
    # the same code paths against fake and real alike.

    @app.exception_handler(RunPodRateLimitedError)
    async def _on_429(_: Any, exc: RunPodRateLimitedError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"error": "rate_limited", "message": str(exc)})

    @app.exception_handler(RunPodTransientError)
    async def _on_5xx(_: Any, exc: RunPodTransientError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "transient", "message": str(exc)})

    @app.exception_handler(RunPodPartialResponseError)
    async def _on_partial(_: Any, exc: RunPodPartialResponseError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": "partial_response", "message": str(exc)})

    # ---- Protocol axis: /api/pods --------------------------------------

    @app.get("/api/pods")
    async def list_pods() -> dict[str, Any]:
        pods = await fake.list_pods()
        return {"pods": [_serialize_pod(p) for p in pods]}

    @app.get("/api/pods/{pod_id}")
    async def find_pod(pod_id: str) -> dict[str, Any]:
        info = await fake.find_pod(pod_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"unknown pod {pod_id}")
        return {"pod": _serialize_pod(info)}

    @app.post("/api/pods/{pod_id}/stop")
    async def stop_pod(pod_id: str) -> dict[str, Any]:
        response = await fake.stop_pod(pod_id)
        return {"outcome": response.outcome, "message": response.message}

    @app.post("/api/pods/{pod_id}/terminate")
    async def terminate_pod(pod_id: str) -> dict[str, Any]:
        response = await fake.terminate_pod(pod_id)
        return {"outcome": response.outcome, "message": response.message}

    @app.post("/api/pods/{pod_id}/resume")
    async def resume_pod(pod_id: str) -> dict[str, Any]:
        response = await fake.resume_pod(pod_id)
        return {"outcome": response.outcome, "message": response.message}

    # ---- Control axis: chaos programming -------------------------------

    @app.post("/control/inject_429")
    async def inject_429(count: int = 5) -> dict[str, int]:
        fake.inject_429(count)
        return {"rate_limit_remaining": count}

    @app.post("/control/inject_5xx")
    async def inject_5xx(count: int = 3) -> dict[str, int]:
        fake.inject_5xx(count)
        return {"transient_remaining": count}

    @app.post("/control/inject_partial_response")
    async def inject_partial(count: int = 1) -> dict[str, int]:
        fake.inject_partial_response(count)
        return {"partial_remaining": count}

    @app.post("/control/set_pod_state")
    async def set_pod_state(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
        pod_id = payload.get("pod_id")
        state = payload.get("state", "").upper()
        if not pod_id or not state:
            raise HTTPException(status_code=400, detail="pod_id and state required")
        if state == "HIBERNATED":
            fake.set_hibernation_mode(pod_id)
            return {"pod_id": pod_id, "state": state}
        # Generic state set: upsert pod with desired_status==state.
        fake.upsert_pod(pod_id, desired_status=state)
        return {"pod_id": pod_id, "state": state}

    @app.post("/control/register_pod")
    async def register_pod(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        pod_id = payload.get("pod_id")
        if not pod_id:
            raise HTTPException(status_code=400, detail="pod_id required")
        info = fake.upsert_pod(
            pod_id,
            desired_status=payload.get("desired_status", "RUNNING"),
            ssh_host=payload.get("ssh_host", "127.0.0.1"),
            ssh_port=payload.get("ssh_port", 22000),
            machine_id=payload.get("machine_id"),
            cost_per_hr=payload.get("cost_per_hr", 0.5),
        )
        return {"pod": _serialize_pod(info)}

    return app


def _serialize_pod(info: RunPodInfo) -> dict[str, Any]:
    return {
        "pod_id": info.pod_id,
        "desired_status": info.desired_status,
        "runtime_status": info.runtime_status,
        "ssh_host": info.ssh_host,
        "ssh_port": info.ssh_port,
        "machine_id": info.machine_id,
        "cost_per_hr": info.cost_per_hr,
    }


def main() -> None:
    args = parse_args("runpod")
    clock = parse_clock(args.clock)
    fake = FakeRunPodAPI(clock=clock)
    runtime = SidecarRuntime(name="runpod", clock=clock, fake=fake, extra={"latency_ms": 0})
    app = make_app(fake=fake, runtime=runtime)

    if args.ready_file:
        # WHY: orchestrator polls this file to detect readiness without
        # racing on /health 404s during cold boot.
        from pathlib import Path

        Path(args.ready_file).write_text(str(args.port))

    run_uvicorn(app, port=args.port)


if __name__ == "__main__":
    main()
