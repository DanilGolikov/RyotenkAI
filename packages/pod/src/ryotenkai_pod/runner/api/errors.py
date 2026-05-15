"""Legacy import shim -- handlers moved to ``ryotenkai_shared.api.error_handlers``.

Phase B (sharded-stargazing-wigderson, 2026-05-16) lifted the four
RFC 9457 ``application/problem+json`` exception handlers up to
``ryotenkai_shared.api.error_handlers`` so the control-plane API
(Phase C) can register the same wire shape. The pod runner consumes
the same :data:`EXCEPTION_HANDLERS` dict; this shim exists for
existing route modules (``jobs.py``, ``files.py``, etc.) that import
``APIError`` from here -- migrating their imports is mechanical work
and will happen in Phase F. The internal helper
``_http_exception_to_code`` is re-exported because
``tests/unit/pod/runner/api/test_errors.py`` (now superseded by
``tests/unit/shared/api/test_error_handlers.py``) referenced it
directly; once that test file is removed in Phase F the re-export
can go too.
"""

from __future__ import annotations

from ryotenkai_shared.api.error_handlers import (
    APIError,
    EXCEPTION_HANDLERS,
    _http_exception_to_code,
    _LEGACY_CODE_ALIASES,
    _build_response,
    _new_trace_id,
    api_error_handler,
    generic_exception_handler,
    http_exception_handler,
    install_exception_handlers,
    validation_exception_handler,
)
from ryotenkai_shared.errors._render import _DEFAULT_TITLES

__all__ = [
    "APIError",
    "EXCEPTION_HANDLERS",
    "api_error_handler",
    "generic_exception_handler",
    "http_exception_handler",
    "install_exception_handlers",
    "validation_exception_handler",
]
