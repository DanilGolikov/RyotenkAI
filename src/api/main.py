from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from src.api.config import ApiSettings
from src.api.dependencies import get_settings
from src.api.exceptions import install_exception_handlers
from src.api.routers import (
    attempts,
    integrations,
    launch,
    logs,
    plugins,
    projects,
    providers,
    reports,
    runs,
)
from src.api.routers import (
    config as config_router,
)
from src.api.routers.health import router as health_router
from src.api.ws.log_stream import router as ws_router

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_exception_handlers(app)

    # Order matters: runs.router uses /runs/{run_id:path} which would
    # swallow anything deeper — mount specific paths first.
    api_routers = [
        health_router,
        config_router.router,
        attempts.router,
        integrations.router,
        logs.router,
        launch.router,
        plugins.router,
        projects.router,
        providers.router,
        reports.router,
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
    app = FastAPI(
        title="RyotenkAI Web API",
        version="v0.1.0",
        description=(
            "Sibling client to the RyotenkAI file-based pipeline state store. "
            "Equal citizen with CLI and TUI — reads pipeline_state.json directly, "
            "launches pipeline runs as detached subprocesses."
        ),
        generate_unique_id_function=_stable_operation_id,
    )
    app.state.settings = effective_settings
    configure_app(app, effective_settings)
    return app


__all__ = ["API_V1_PREFIX", "configure_app", "create_app"]
