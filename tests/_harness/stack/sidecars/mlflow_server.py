"""``fake-mlflow`` HTTP sidecar.

Wraps :class:`tests._fakes.mlflow.FakeMLflowManager` behind a FastAPI app.
Surfaces a *minimal* slice of the MLflow REST API (``/api/2.0/mlflow/``)
plus the standard control axis. Why minimal: the canonical fake exposes
its ``IMLflowManager`` shape (``start_run``, ``log_metrics``, ...) — a
true MLflow-REST surface would need an extra translation layer. We expose
just enough to make a smoke control-plane talk to it; richer endpoints
land in Phase 4.

CLI:

    python -m tests._harness.stack.sidecars.mlflow_server --port=N
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from tests._fakes.mlflow import FakeMLflowManager, MLflowUnavailableError, TransientMLflowError
from tests._harness.stack.sidecars._base import (
    SidecarRuntime,
    add_control_routes,
    parse_args,
    parse_clock,
    run_uvicorn,
)


def make_app(*, fake: FakeMLflowManager, runtime: SidecarRuntime) -> FastAPI:
    app = FastAPI(title="fake-mlflow-sidecar")

    def _reset() -> None:
        fake.reset_chaos()
        fake.cleanup()
        # Wipe state by reassigning the underlying caches; cheaper than
        # rebuilding the manager.
        fake._runs.clear()  # type: ignore[attr-defined]
        fake._experiments.clear()  # type: ignore[attr-defined]
        fake._is_active = False  # type: ignore[attr-defined]
        fake._tracking_uri = None  # type: ignore[attr-defined]

    add_control_routes(app, runtime=runtime, reset_fn=_reset)

    @app.exception_handler(MLflowUnavailableError)
    async def _on_unavailable(_: Any, exc: MLflowUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "unavailable", "message": str(exc)})

    @app.exception_handler(TransientMLflowError)
    async def _on_transient(_: Any, exc: TransientMLflowError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": "transient", "message": str(exc)})

    # ----- Setup / connectivity -----------------------------------------

    @app.post("/api/setup")
    async def setup() -> dict[str, Any]:
        ok = fake.setup()
        return {"is_active": ok, "tracking_uri": fake.tracking_uri}

    @app.get("/api/connectivity")
    async def connectivity() -> dict[str, bool]:
        return {"ok": fake.check_mlflow_connectivity()}

    # ----- Run lifecycle ------------------------------------------------

    @app.post("/api/2.0/mlflow/runs/create")
    async def create_run(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        if not fake.is_active:
            fake.setup()
        run_name = payload.get("run_name") or payload.get("experiment_name")
        handle = fake.start_run(run_name=run_name, description=payload.get("description"))
        return {"run": {"info": {"run_id": handle.info.run_id, "status": "RUNNING"}}}

    @app.post("/api/2.0/mlflow/runs/log-metric")
    async def log_metric(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
        run_id = payload.get("run_id")
        key = payload.get("key")
        value = payload.get("value")
        step = int(payload.get("step", 0))
        if run_id is None or key is None or value is None:
            raise HTTPException(status_code=400, detail="run_id, key, value required")
        fake.adopt_existing_run(run_id)
        fake.log_metrics({key: float(value)}, step=step)
        return {"status": "ok"}

    @app.post("/api/2.0/mlflow/runs/log-parameter")
    async def log_parameter(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
        run_id = payload.get("run_id")
        key = payload.get("key")
        value = payload.get("value")
        if run_id is None or key is None:
            raise HTTPException(status_code=400, detail="run_id and key required")
        fake.adopt_existing_run(run_id)
        fake.log_params({key: value})
        return {"status": "ok"}

    @app.post("/api/2.0/mlflow/runs/update")
    async def update_run(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
        run_id = payload.get("run_id")
        status = payload.get("status", "FINISHED")
        if run_id is None:
            raise HTTPException(status_code=400, detail="run_id required")
        fake.adopt_existing_run(run_id)
        fake.end_run(status=status)
        return {"status": status}

    # ----- Inspection ---------------------------------------------------

    @app.get("/api/2.0/mlflow/runs/get")
    async def get_run(run_id: str) -> dict[str, Any]:
        try:
            record = fake.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "run": {
                "info": {
                    "run_id": record.run_id,
                    "experiment_name": record.experiment_name,
                    "status": record.status,
                },
                "data": {
                    "params": dict(record.params),
                    "tags": dict(record.tags),
                },
            },
        }

    # ----- Control axis (MLflow-specific) -------------------------------

    @app.post("/control/set_unavailable")
    async def set_unavailable(value: bool = True) -> dict[str, bool]:
        fake.set_unavailable(value)
        return {"unavailable": value}

    @app.post("/control/fail_next_n_calls")
    async def fail_next_n(count: int = 1) -> dict[str, int]:
        fake.fail_next_n_calls(count)
        return {"fail_remaining": count}

    # TODO(phase-4): expand MLflow REST coverage (artifacts, search,
    # nested-run wiring, batch log-metric).

    return app


def main() -> None:
    args = parse_args("mlflow")
    clock = parse_clock(args.clock)
    fake = FakeMLflowManager(clock=clock)
    runtime = SidecarRuntime(name="mlflow", clock=clock, fake=fake, extra={"latency_ms": 0})
    app = make_app(fake=fake, runtime=runtime)
    if args.ready_file:
        from pathlib import Path
        Path(args.ready_file).write_text("port=" + str(args.port) + "\n")
    run_uvicorn(app, port=args.port)


if __name__ == "__main__":
    main()
