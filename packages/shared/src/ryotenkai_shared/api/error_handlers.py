"""RFC 9457 ``application/problem+json`` handlers for shared FastAPI surfaces.

Phase B (sharded-stargazing-wigderson, 2026-05-16) lifted these four
handlers out of ``ryotenkai_pod.runner.api.errors`` so the control-plane
API (Phase C) can register the same wire shape. Covers four exception
types:

1. :class:`APIError` -- typed runner-side exception raised by route
   handlers that have decided to communicate a specific
   :class:`ErrorCode`. (Predates :class:`RyotenkAIError`; kept for
   pod-runner compatibility. Production code that goes through the
   unified hierarchy raises :class:`RyotenkAIError` subclasses instead,
   which are converted in the same handler path because they expose
   ``.as_problem()`` and ``.status`` ClassVars.)
2. ``fastapi.HTTPException`` -- legacy bare-dict raise sites. Adapter
   :func:`http_exception_handler` maps them to ``problem+json`` so
   wire shape is unified.
3. ``fastapi.exceptions.RequestValidationError`` -- Pydantic validation
   on request bodies. Maps to ``JOB_SPEC_INVALID`` (or domain-specific
   422 when the route raises one explicitly first) with field-level
   errors.
4. Generic ``Exception`` -- server bugs. 500 ``INTERNAL_ERROR``,
   traceback logged at ERROR but **never** in the response body
   (security: don't leak internals).

RP20 mitigation: :data:`EXCEPTION_HANDLERS` is a synchronous dict
intended for ``FastAPI(exception_handlers=EXCEPTION_HANDLERS)`` (not
``app.add_exception_handler(...)`` in lifespan startup), so the very
first request after boot already gets a properly-shaped problem+json
response on failure.

The 1-line :func:`install_exception_handlers` helper exists for
callers that prefer the imperative form (mounting on an already-built
app, or composing with other middleware that registers handlers).
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ryotenkai_shared.contracts.problem_details import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    FieldError,
    ProblemDetails,
)
from ryotenkai_shared.errors._render import _DEFAULT_TITLES, default_title_for

logger = logging.getLogger(__name__)

__all__ = [
    "APIError",
    "EXCEPTION_HANDLERS",
    "api_error_handler",
    "generic_exception_handler",
    "http_exception_handler",
    "install_exception_handlers",
    "validation_exception_handler",
]


def _new_trace_id() -> str:
    """Short opaque correlation id (8 hex chars) emitted in both the
    response body and the server log. 8 chars * 16 alphabet = ~32
    bits, plenty for human-friendly grep'ing during a single run.

    Phase B kept this as a private helper (the same primitive is
    re-exported from :mod:`ryotenkai_shared.errors._render` as
    ``new_trace_id``; both delegate to ``secrets.token_hex(4)``).
    """
    return secrets.token_hex(4)


def _build_response(problem: ProblemDetails) -> JSONResponse:
    """Serialise :class:`ProblemDetails` and emit it with the
    RFC-mandated ``application/problem+json`` media type.

    ``mode="json"`` so the :class:`ErrorCode` enum is rendered as its
    string value (``"JOB_NOT_FOUND"``) and not the Python repr.
    ``exclude_none=True`` per RFC 9457 §3.1 -- null fields are stripped
    from the wire to keep responses clean.
    """
    return JSONResponse(
        status_code=problem.status,
        media_type=PROBLEM_JSON_MEDIA_TYPE,
        content=problem.model_dump(mode="json", exclude_none=True),
    )


# ---------------------------------------------------------------------------
# 1. APIError -- typed runner-side exception (pre-RyotenkAIError legacy)
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Typed exception used by pod route handlers to communicate a
    specific :class:`ErrorCode`.

    Construction shape::

        raise APIError(
            ErrorCode.JOB_NOT_FOUND, status=404,
            detail=f"job_id={requested!r} is not active",
        )

    The handler may also pass ``title=...`` to override the default
    from :data:`_DEFAULT_TITLES` (in
    :mod:`ryotenkai_shared.errors._render`) and ``errors=[...]`` for
    field-level failures.

    Phase A1/A2 introduced :class:`RyotenkAIError` as the unified root
    for new raise sites. ``APIError`` remains for the in-pod runner
    handlers that pre-date the unification; Phase F will retire it
    once those sites are migrated to ``RyotenkAIError`` subclasses.
    """

    def __init__(
        self,
        code: ErrorCode,
        *,
        status: int,
        detail: str | None = None,
        title: str | None = None,
        errors: list[FieldError] | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {detail or ''}")
        self.code: ErrorCode = code
        self.status: int = status
        self.detail: str | None = detail
        self.title: str = title or default_title_for(code)
        self.errors: list[FieldError] | None = errors

    def as_problem(
        self,
        *,
        instance: str | None = None,
        trace_id: str | None = None,
    ) -> ProblemDetails:
        return ProblemDetails(
            title=self.title,
            status=self.status,
            detail=self.detail,
            instance=instance,
            code=self.code,
            trace_id=trace_id,
            errors=self.errors,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """``APIError`` -> ``application/problem+json`` (the happy path)."""
    trace_id = _new_trace_id()
    logger.info(
        "[API_ERROR] %s code=%s status=%d detail=%r trace_id=%s",
        request.url.path, exc.code.value, exc.status, exc.detail, trace_id,
    )
    problem = exc.as_problem(instance=request.url.path, trace_id=trace_id)
    return _build_response(problem)


# ---------------------------------------------------------------------------
# 2. HTTPException adapter (universal -- covers legacy raise sites)
# ---------------------------------------------------------------------------


# Legacy code-string → registry mapping for raise sites whose
# lowercase strings don't simply uppercase to the registry name
# (e.g. ``invalid_job_spec`` != ``JOB_SPEC_INVALID``). PR-3.3 will
# delete the raise sites and this table along with them.
_LEGACY_CODE_ALIASES: dict[str, ErrorCode] = {
    "invalid_job_spec": ErrorCode.JOB_SPEC_INVALID,
}


def _http_exception_to_code(exc: HTTPException) -> tuple[ErrorCode, str | None]:
    """Best-effort recovery of an :class:`ErrorCode` from a legacy
    ``HTTPException(detail={"code": "...", ...})`` raise site.

    Existing call sites in jobs.py / internal.py raise::

        HTTPException(
            status_code=409,
            detail={"code": "job_state_invalid", "current_state": ...},
        )

    Notes:

    * Legacy values are ``lowercase_snake_case``. The unified registry
      is UPPER_SNAKE_CASE per RFC convention. We normalise via
      ``.upper()``, then fall back to :data:`_LEGACY_CODE_ALIASES`
      for the few raise sites whose words are reordered between the
      legacy spelling and the registry.
    * On any mismatch (unknown string, dict missing ``code``,
      ``detail`` is a plain string) we fall back to
      :attr:`ErrorCode.INTERNAL_ERROR` so the wire shape stays valid.
    * The ``message`` returned is ``detail["message"]`` if present,
      or the plain detail string, or ``None``.
    """
    detail = exc.detail
    if isinstance(detail, dict):
        raw_code = detail.get("code")
        message = detail.get("message")
        if raw_code is None:
            return ErrorCode.INTERNAL_ERROR, None
        legacy_alias = _LEGACY_CODE_ALIASES.get(str(raw_code).lower())
        if legacy_alias is not None:
            return legacy_alias, message if isinstance(message, str) else None
        try:
            code = ErrorCode(str(raw_code).upper())
        except ValueError:
            return ErrorCode.INTERNAL_ERROR, f"unknown code={raw_code!r}"
        return code, message if isinstance(message, str) else None

    if isinstance(detail, str):
        return ErrorCode.INTERNAL_ERROR, detail

    return ErrorCode.INTERNAL_ERROR, None


async def http_exception_handler(
    request: Request, exc: HTTPException,
) -> JSONResponse:
    """Adapter -- legacy ``HTTPException`` -> ``problem+json``.

    Universally applied so the wire shape is unified for both the pod
    runner (RP17 mitigation) and the control API (Phase C). PR-3.3
    will rewrite the legacy raise sites to use :class:`APIError`
    directly, after which this handler becomes a no-op for our own
    code (still kept for FastAPI's own 400/422/etc. raises).
    """
    code, message = _http_exception_to_code(exc)
    trace_id = _new_trace_id()
    logger.info(
        "[HTTP_EXCEPTION] %s status=%d adapted_code=%s trace_id=%s",
        request.url.path, exc.status_code, code.value, trace_id,
    )
    problem = ProblemDetails(
        title=default_title_for(code),
        status=exc.status_code,
        detail=message,
        instance=request.url.path,
        code=code,
        trace_id=trace_id,
    )
    return _build_response(problem)


# ---------------------------------------------------------------------------
# 3. Pydantic / FastAPI validation errors
# ---------------------------------------------------------------------------


def _coerce_field_error(raw: dict[str, Any]) -> FieldError:
    """FastAPI's ``RequestValidationError`` exposes errors in a shape
    very close to :class:`FieldError` already; this is a defensive
    coercion that drops fields the model forbids and keeps the rest.
    """
    return FieldError(
        loc=[str(p) if not isinstance(p, int) else p for p in raw.get("loc", [])],
        type=str(raw.get("type", "value_error")),
        msg=str(raw.get("msg", "")),
        input=raw.get("input"),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    """Pydantic validation failures -> 422 ``JOB_SPEC_INVALID`` with
    field-level errors.

    The runner currently only validates ``JobSpec`` via Pydantic on
    the public surface (other fields go through json.loads + manual
    validation). Phase 2 added typed bodies for /diagnostics,
    /resources etc.; Phase C extends this same handler to the
    control API surface.
    """
    trace_id = _new_trace_id()
    field_errors = [_coerce_field_error(e) for e in exc.errors()]
    logger.info(
        "[VALIDATION_ERROR] %s fields=%d trace_id=%s",
        request.url.path, len(field_errors), trace_id,
    )
    problem = ProblemDetails(
        title=default_title_for(ErrorCode.JOB_SPEC_INVALID),
        status=422,
        detail="Request body failed validation.",
        instance=request.url.path,
        code=ErrorCode.JOB_SPEC_INVALID,
        trace_id=trace_id,
        errors=field_errors,
    )
    return _build_response(problem)


# ---------------------------------------------------------------------------
# 4. Catch-all -- server bugs
# ---------------------------------------------------------------------------


async def generic_exception_handler(
    request: Request, exc: Exception,
) -> JSONResponse:
    """Catch-all for unhandled exceptions.

    Logs the full traceback at ERROR (devops needs it for
    correlation) but **never** echoes it into the response body --
    that would leak filesystem paths, environment hints, and stack
    layout (security 101). The body just carries the trace_id so the
    operator can grep the server log.
    """
    trace_id = _new_trace_id()
    logger.error(
        "[INTERNAL_ERROR] %s exc=%s trace_id=%s",
        request.url.path, type(exc).__name__, trace_id,
        exc_info=exc,
    )
    problem = ProblemDetails(
        title=default_title_for(ErrorCode.INTERNAL_ERROR),
        status=500,
        detail="An internal error occurred. See server logs for the trace.",
        instance=request.url.path,
        code=ErrorCode.INTERNAL_ERROR,
        trace_id=trace_id,
    )
    return _build_response(problem)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Synchronous mapping for ``FastAPI(exception_handlers=EXCEPTION_HANDLERS)``
# -- registers BEFORE the first request can be served (RP20). Adding
# entries via ``app.add_exception_handler`` after construction also
# works at startup-time but loses static visibility.
#
# Both pod runner (``ryotenkai_pod.runner.main``) and control API
# (Phase C, ``ryotenkai_control.api.main``) consume this dict directly.
EXCEPTION_HANDLERS: dict[Any, Any] = {
    APIError: api_error_handler,
    HTTPException: http_exception_handler,
    RequestValidationError: validation_exception_handler,
    Exception: generic_exception_handler,
}


def install_exception_handlers(app: FastAPI) -> None:
    """Imperative form of :data:`EXCEPTION_HANDLERS` for callers that
    can't (or don't want to) pass ``exception_handlers=...`` at
    construction time.

    Equivalent to::

        for exc_class, handler in EXCEPTION_HANDLERS.items():
            app.add_exception_handler(exc_class, handler)

    NOTE: For new code prefer ``FastAPI(exception_handlers=EXCEPTION_HANDLERS,
    ...)`` -- that path registers SYNCHRONOUSLY at construction time
    (RP20 mitigation: no race with the first request). The imperative
    form here also runs synchronously when called before
    ``app.router.startup()`` fires, but it's easier to accidentally
    defer to lifespan startup; prefer the constructor form for app
    boot.

    Re-exported (and **not** marked private) so the boundary contract
    is discoverable from outside the package.
    """
    for exc_class, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_class, handler)
