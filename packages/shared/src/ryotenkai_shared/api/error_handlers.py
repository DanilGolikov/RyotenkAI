"""RFC 9457 ``application/problem+json`` handlers for shared FastAPI surfaces.

Phase B (sharded-stargazing-wigderson, 2026-05-16) lifted these
handlers out of ``ryotenkai_pod.runner.api.errors`` so the control-plane
API (Phase C) can register the same wire shape. Post-Phase G fix-up
retired the legacy :class:`APIError` class (every pod-runner raise
site now raises a typed :class:`RyotenkAIError` subclass directly).

Covers four exception types:

1. :class:`RyotenkAIError` -- the unified typed root. The
   :func:`ryotenkai_error_handler` renders any subclass via
   :meth:`RyotenkAIError.as_problem` to RFC 9457 problem+json.
2. ``fastapi.HTTPException`` -- legacy bare-dict raise sites kept only
   for FastAPI's own internal 4xx surfaces (e.g. router 405). Adapter
   :func:`http_exception_handler` maps them to ``problem+json`` so wire
   shape is unified. New code MUST raise a typed :class:`RyotenkAIError`
   instead; sentinel ``tests/_lint/test_no_naked_httpexception.py``
   enforces this.
3. ``fastapi.exceptions.RequestValidationError`` -- Pydantic validation
   on request bodies. Maps to ``JOB_SPEC_INVALID`` with field-level
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

from ryotenkai_shared.api.request_id import current_request_id
from ryotenkai_shared.contracts.problem_details import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    FieldError,
    ProblemDetails,
)
from ryotenkai_shared.errors._render import _DEFAULT_TITLES, default_title_for
from ryotenkai_shared.errors.base import RyotenkAIError

logger = logging.getLogger(__name__)

__all__ = [
    "EXCEPTION_HANDLERS",
    "generic_exception_handler",
    "http_exception_handler",
    "install_exception_handlers",
    "ryotenkai_error_handler",
    "validation_exception_handler",
]


def _new_trace_id() -> str:
    """Short opaque correlation id (16 hex chars, 64 bits) emitted in
    both the response body and the server log. 64 bits is safe for
    billions of events (birthday paradox at ~4 billion samples) --
    suitable for long-running pipelines.

    Phase B kept this as a private helper (the same primitive is
    re-exported from :mod:`ryotenkai_shared.errors._render` as
    ``new_trace_id``; both delegate to ``secrets.token_hex(8)``).
    """
    return secrets.token_hex(8)


def _build_response(problem: ProblemDetails) -> JSONResponse:
    """Serialise :class:`ProblemDetails` and emit it with the
    RFC-mandated ``application/problem+json`` media type.

    ``mode="json"`` so the :class:`ErrorCode` enum is rendered as its
    string value (``"JOB_NOT_FOUND"``) and not the Python repr.
    ``exclude_none=True`` per RFC 9457 §3.1 -- null fields are stripped
    from the wire to keep responses clean.

    Phase C: also stamp the ``X-Request-ID`` response header from the
    same :data:`REQUEST_ID` contextvar that feeds
    ``problem.request_id``. The middleware also adds the header on
    the happy path; setting it here covers the catch-all exception
    case where Starlette's :class:`ServerErrorMiddleware` (which sits
    OUTSIDE our middleware in the stack) sends the response using
    its own outer ``send`` -- bypassing our send-wrapper. Doing it on
    the response object is the canonical fix because every middleware
    above us preserves response headers.
    """
    headers: dict[str, str] = {}
    if problem.request_id:
        # REQUEST_ID_HEADER name matches the middleware's spelling so
        # the happy-path and the error path produce identical wire shape.
        from ryotenkai_shared.api.request_id import REQUEST_ID_HEADER

        headers[REQUEST_ID_HEADER] = problem.request_id
    return JSONResponse(
        status_code=problem.status,
        media_type=PROBLEM_JSON_MEDIA_TYPE,
        content=problem.model_dump(mode="json", exclude_none=True),
        headers=headers or None,
    )


# ---------------------------------------------------------------------------
# 1. HTTPException adapter -- kept for FastAPI's own raises (405 etc.)
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
    """Adapter -- ``HTTPException`` -> ``problem+json``.

    With the post-Phase G fix-up, in-repo raise sites no longer use raw
    ``HTTPException`` (sentinel ``tests/_lint/test_no_naked_httpexception.py``
    enforces this). The handler is retained for FastAPI's OWN raises
    (router 405, path-param coercion bypassing the validation handler,
    third-party middleware that hasn't been migrated).
    """
    code, message = _http_exception_to_code(exc)
    trace_id = _new_trace_id()
    request_id = current_request_id()
    logger.info(
        "[HTTP_EXCEPTION] %s status=%d adapted_code=%s trace_id=%s request_id=%s",
        request.url.path, exc.status_code, code.value, trace_id, request_id,
    )
    problem = ProblemDetails(
        title=default_title_for(code),
        status=exc.status_code,
        detail=message,
        instance=request.url.path,
        code=code,
        trace_id=trace_id,
        request_id=request_id,
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
    request_id = current_request_id()
    field_errors = [_coerce_field_error(e) for e in exc.errors()]
    logger.info(
        "[VALIDATION_ERROR] %s fields=%d trace_id=%s request_id=%s",
        request.url.path, len(field_errors), trace_id, request_id,
    )
    problem = ProblemDetails(
        title=default_title_for(ErrorCode.JOB_SPEC_INVALID),
        status=422,
        detail="Request body failed validation.",
        instance=request.url.path,
        code=ErrorCode.JOB_SPEC_INVALID,
        trace_id=trace_id,
        request_id=request_id,
        errors=field_errors,
    )
    return _build_response(problem)


# ---------------------------------------------------------------------------
# 4. RyotenkAIError -- typed unified hierarchy (Phase A/C)
# ---------------------------------------------------------------------------


async def ryotenkai_error_handler(
    request: Request, exc: RyotenkAIError,
) -> JSONResponse:
    """``RyotenkAIError`` subclass -> ``application/problem+json``.

    Phase C: route handlers (control API + pod runner) raise typed
    domain/infrastructure errors directly; this handler picks them up
    via :meth:`RyotenkAIError.as_problem`, attaches the per-request
    trace_id / request_id, and emits the unified wire shape. No
    bespoke ``HTTPException`` adapter needed at raise sites.

    The MRO-lookup in Starlette's exception-handler dispatch picks
    the most specific registered handler -- so Pydantic
    :class:`RequestValidationError` keeps its own handler because
    it's not a :class:`RyotenkAIError`. Everything else under the
    unified hierarchy lands here.
    """
    trace_id = _new_trace_id()
    request_id = current_request_id()
    logger.info(
        "[RYOTENKAI_ERROR] %s class=%s code=%s status=%d detail=%r "
        "trace_id=%s request_id=%s",
        request.url.path, type(exc).__name__, exc.code.value, exc.status,
        exc.detail, trace_id, request_id,
    )
    problem = exc.as_problem(
        instance=request.url.path,
        trace_id=trace_id,
        request_id=request_id,
    )
    return _build_response(problem)


# ---------------------------------------------------------------------------
# 5. Catch-all -- server bugs
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
    request_id = current_request_id()
    logger.error(
        "[INTERNAL_ERROR] %s exc=%s trace_id=%s request_id=%s",
        request.url.path, type(exc).__name__, trace_id, request_id,
        exc_info=exc,
    )
    problem = ProblemDetails(
        title=default_title_for(ErrorCode.INTERNAL_ERROR),
        status=500,
        detail="An internal error occurred. See server logs for the trace.",
        instance=request.url.path,
        code=ErrorCode.INTERNAL_ERROR,
        trace_id=trace_id,
        request_id=request_id,
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
    RyotenkAIError: ryotenkai_error_handler,
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
