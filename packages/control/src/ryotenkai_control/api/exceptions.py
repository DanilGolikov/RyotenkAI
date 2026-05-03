from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.pipeline.state import PipelineStateLoadError, PipelineStateLockError


def _error_body(detail: str, code: str | None = None) -> dict[str, str]:
    body: dict[str, str] = {"detail": detail}
    if code:
        body["code"] = code
    return body


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PipelineStateLoadError)
    async def _on_state_load_error(_request: Request, exc: PipelineStateLoadError) -> JSONResponse:
        return JSONResponse(status_code=404, content=_error_body(str(exc), code="state_load_error"))

    @app.exception_handler(PipelineStateLockError)
    async def _on_state_lock_error(_request: Request, exc: PipelineStateLockError) -> JSONResponse:
        return JSONResponse(status_code=409, content=_error_body(str(exc), code="state_locked"))

    @app.exception_handler(FileNotFoundError)
    async def _on_file_not_found(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=_error_body(str(exc), code="file_not_found"))

    @app.exception_handler(ValueError)
    async def _on_value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_error_body(str(exc), code="validation_error"))
