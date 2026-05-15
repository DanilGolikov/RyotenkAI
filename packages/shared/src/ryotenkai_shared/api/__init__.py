"""Shared FastAPI-facing helpers (handlers, middleware).

This subpackage is the **only** location in ``ryotenkai_shared`` that
is allowed to import ``fastapi``. The sentinel
``tests/_lint/test_no_fastapi_outside_shared_api.py`` enforces this
boundary -- everything else in ``shared`` must remain framework-free
so it can be consumed unchanged on the pod side (which already pulls
FastAPI for the runner) AND on the Mac side (which doesn't need an
HTTP server, but does call the shared error model).

Phase B (sharded-stargazing-wigderson, 2026-05-16) created this
subpackage by lifting the in-pod exception handlers up so the
control API (Phase C) can register the same wire shape without
copy-pasting.
"""

from ryotenkai_shared.api.error_handlers import (
    APIError,
    EXCEPTION_HANDLERS,
    api_error_handler,
    generic_exception_handler,
    http_exception_handler,
    install_exception_handlers,
    ryotenkai_error_handler,
    validation_exception_handler,
)
from ryotenkai_shared.api.request_id import (
    REQUEST_ID,
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
    current_request_id,
    generate_request_id,
)

__all__ = [
    "APIError",
    "EXCEPTION_HANDLERS",
    "REQUEST_ID",
    "REQUEST_ID_HEADER",
    "RequestIDMiddleware",
    "api_error_handler",
    "current_request_id",
    "generate_request_id",
    "generic_exception_handler",
    "http_exception_handler",
    "install_exception_handlers",
    "ryotenkai_error_handler",
    "validation_exception_handler",
]
