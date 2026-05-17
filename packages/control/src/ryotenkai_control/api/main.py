from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from ryotenkai_control.api.config import ApiSettings
from ryotenkai_control.api.dependencies import get_settings
from ryotenkai_control.api.routers import (
    attempts,
    datasets,
    events,
    integrations,
    jobs,
    launch,
    logs,
    plugins,
    projects,
    providers,
    reports,
    runs,
)
from ryotenkai_control.api.routers import (
    config as config_router,
)
from ryotenkai_control.api.routers.health import router as health_router
from ryotenkai_control.api.ws.log_stream import router as ws_router
from ryotenkai_shared.api import EXCEPTION_HANDLERS, RequestIDMiddleware

API_V1_PREFIX = "/api/v1"


def _stable_operation_id(route: APIRoute) -> str:
    """Produce stable, readable ``operationId`` values for the OpenAPI spec.

    FastAPI's default ``<func>_<path>_<method>`` is noisy and changes any
    time the function is renamed or the path shifts. We anchor on
    ``<tag>-<func>`` instead, which keeps codegen output stable.
    """
    tag = route.tags[0] if route.tags else "api"
    return f"{tag}-{route.name}"


def configure_app(app: FastAPI, settings: ApiSettings) -> None:
    # Phase C: RequestIDMiddleware sits at the BOTTOM of the
    # middleware stack -- in Starlette, the middleware added LAST
    # is the OUTERMOST wrapper, meaning it sees the request first
    # and the response last. We want CORS to be the outermost wrap
    # (so CORS headers always land on the response, even on errors),
    # and RequestIDMiddleware to wrap closer to the route handler
    # so REQUEST_ID is populated by the time the exception handlers
    # run. ``add_middleware`` calls -> RequestIDMiddleware first,
    # then CORSMiddleware on top, giving the desired ordering:
    # CORS (outer) -> RequestID -> route -> RequestID -> CORS.
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Order matters: runs.router uses /runs/{run_id:path} which would
    # swallow anything deeper — mount specific paths first.
    api_routers = [
        health_router,
        config_router.router,
        attempts.router,
        datasets.router,
        integrations.router,
        logs.router,
        launch.router,
        plugins.router,
        projects.router,
        providers.router,
        reports.router,
        # ``jobs`` carries prefix ``/runs/{run_id:path}/job`` — must be
        # registered BEFORE ``runs.router`` (whose ``/{run_id:path}`` route
        # would otherwise swallow ``/runs/<id>/job/...``).
        jobs.router,
        # ``events`` carries prefix ``/runs/{run_id:path}/events`` — same
        # ordering rationale as ``jobs`` above.
        events.router,
        runs.router,
    ]
    for router in api_routers:
        app.include_router(router, prefix=API_V1_PREFIX)
    app.include_router(ws_router, prefix=API_V1_PREFIX)

    if settings.serve_spa:
        dist_dir = _resolve_dist_dir(settings.web_dist_dir)
        if dist_dir.is_dir():
            app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="spa")


def _resolve_dist_dir(web_dist_dir: Path) -> Path:
    p = web_dist_dir.expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    effective_settings = settings or get_settings()
    # Phase C (sharded-stargazing-wigderson, 2026-05-16): swap the
    # legacy 4-handler ``{detail, code}`` shape for the shared RFC
    # 9457 ``application/problem+json`` wire by passing
    # ``exception_handlers=EXCEPTION_HANDLERS`` at construction time.
    # The synchronous-constructor form mitigates RP20 (handlers must
    # be registered before the first request can race the boot).
    app = FastAPI(
        title="RyotenkAI Web API",
        version="v0.1.0",
        description=(
            "Sibling client to the RyotenkAI file-based pipeline state store. "
            "Equal citizen with the CLI — reads pipeline_state.json directly, "
            "launches pipeline runs as detached subprocesses."
        ),
        generate_unique_id_function=_stable_operation_id,
        exception_handlers=EXCEPTION_HANDLERS,
    )
    app.state.settings = effective_settings
    configure_app(app, effective_settings)
    return app


__all__ = ["API_V1_PREFIX", "configure_app", "create_app"]
