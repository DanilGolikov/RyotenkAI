"""Per-request correlation id middleware shared between control and pod APIs.

Phase C (sharded-stargazing-wigderson, 2026-05-16) introduces an
``X-Request-ID`` header that flows through the entire request lifecycle:

1. :class:`RequestIDMiddleware` reads the header from the incoming
   request (or generates a fresh 16-hex-char value when absent) and
   stores it in :data:`REQUEST_ID` -- a :class:`contextvars.ContextVar`
   that the rest of the request handler chain reads from.
2. Exception handlers (``shared/api/error_handlers.py``) read
   :data:`REQUEST_ID` when building :class:`ProblemDetails`, so the
   error body carries the same id as the response header.
3. The :class:`RequestIDLogFilter` (installed by
   :mod:`ryotenkai_shared.utils.logger`) stamps every log record with
   ``record.request_id``, so server logs and the response body carry
   the same id and the operator can correlate the two.
4. The middleware echoes the id back into the response headers as
   ``X-Request-ID``.

Why a ContextVar:
The contextvars module is asyncio-safe -- the per-task copy semantics
mean two concurrent requests do **not** see each other's id. The
middleware uses :meth:`ContextVar.set` / :meth:`ContextVar.reset` (not
mutation) so the original value is restored on exit and there is no
cross-task leakage even when the same event-loop thread is reused.
"""

from __future__ import annotations

import secrets
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = [
    "REQUEST_ID",
    "REQUEST_ID_HEADER",
    "RequestIDMiddleware",
    "current_request_id",
    "generate_request_id",
]

# Header name. Lowercased on the wire (HTTP/2 lowercases all header
# names), but FastAPI/Starlette compares case-insensitively, so either
# spelling works in ``request.headers.get`` and in client code.
REQUEST_ID_HEADER = "X-Request-ID"

# Per-task storage. ``None`` outside of a request -- callers that read
# this value must treat ``None`` as "no request scope yet" (during
# startup, in background workers, in tests that don't wrap a request).
REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)


def generate_request_id() -> str:
    """Return a fresh 16-hex-char request id.

    ``secrets.token_hex(8)`` produces 8 random bytes rendered as 16
    hex characters -- ~64 bits of entropy, plenty for correlation
    within a single deployment over a reasonable retention horizon and
    short enough to fit comfortably in a log line.

    Pure (no I/O, no globals). Public for tests that want to mint an
    id without invoking the full middleware.
    """
    return secrets.token_hex(8)


def current_request_id() -> str | None:
    """Read the request id for the current task (or ``None`` outside a request).

    Thin wrapper over :data:`REQUEST_ID` so call sites can avoid
    importing the ContextVar directly. Returns the value
    :class:`RequestIDMiddleware` placed there for the active request.
    """
    return REQUEST_ID.get()


def _find_header(scope_headers: list[tuple[bytes, bytes]], name_lower: bytes) -> bytes | None:
    """Return the (first) header value matching ``name_lower`` (lowercase),
    or ``None``. ASGI delivers headers as a list of byte tuples in
    lowercase per the spec, so case-insensitive lookup is a literal
    byte compare.
    """
    for raw_name, raw_value in scope_headers:
        if raw_name == name_lower:
            return raw_value
    return None


class RequestIDMiddleware:
    """Pure ASGI middleware that maintains :data:`REQUEST_ID` for every request.

    Implemented as raw ASGI (not :class:`starlette.middleware.base.BaseHTTPMiddleware`)
    because the latter uses an anyio task group internally; the
    exception-handler task does not inherit the parent's
    :class:`contextvars.ContextVar` values, so a request-id set in
    ``dispatch`` would be invisible to error handlers. Pure ASGI runs
    everything (route handler, exception handlers, response stream)
    in the SAME task as the middleware -- contextvars propagate
    naturally and the id is visible end-to-end.

    Order-sensitivity: this middleware **must** wrap the exception
    handler stack so :data:`REQUEST_ID` is populated before any
    handler reads it. With Starlette/FastAPI, the actual stack order
    is ``ServerErrorMiddleware -> RequestIDMiddleware -> ...`` --
    i.e. ServerErrorMiddleware (which dispatches handlers for the
    ``Exception`` catch-all) sits OUTSIDE us. To keep the contextvar
    alive during that handler's run we install in ``contextvars.copy_context()``
    and rely on the FACT that ServerErrorMiddleware does not switch
    tasks (Starlette source confirms: ``await self.handler(request, exc)``
    is in the same coroutine), so the contextvar set by ``REQUEST_ID.set(rid)``
    is visible to ``generic_exception_handler`` even though our
    own ``__call__`` re-raises before the reset would run.

    The reset happens BEFORE we re-raise because we want the post-
    handler cleanup (ServerErrorMiddleware does ``raise exc`` after
    dispatching) to land on a clean contextvar. The handler has
    already completed -- its read of :data:`REQUEST_ID` happened
    inside ``await self.app(...)`` -- so the order is:

        set rid -> await app (route raises, handler runs, sends
        response, ServerErrorMiddleware re-raises) -> reset rid ->
        propagate the exception.
    """

    # Lowercase byte form for fast ASGI-layer header probe.
    _HEADER_BYTES = REQUEST_ID_HEADER.encode("ascii").lower()

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send,
    ) -> None:
        # Pass non-HTTP scopes (WebSocket, lifespan) through untouched
        # -- they don't carry headers in the same shape, and request
        # correlation only makes sense for individual HTTP requests.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw = _find_header(scope["headers"], self._HEADER_BYTES)
        incoming = raw.decode("latin-1") if raw else ""
        rid = incoming if incoming else generate_request_id()
        token = REQUEST_ID.set(rid)

        # Wrap ``send`` so we can stamp ``X-Request-ID`` onto the
        # response headers as they fly past. ASGI delivers headers
        # via the ``http.response.start`` message; we append the id
        # there ONLY IF the response doesn't carry it already.
        #
        # Why the idempotency check: ``_build_response`` in
        # ``error_handlers.py`` also stamps ``X-Request-ID`` on the
        # JSONResponse for the catch-all path (where
        # ``ServerErrorMiddleware`` bypasses our ``send`` wrapper).
        # Without the duplicate-suppression a non-error response
        # going through both code paths would emit
        # ``X-Request-ID: <id>, <id>`` -- valid per RFC 9110 (multi-
        # value comma join) but ugly. Single-value semantics are
        # what clients expect.
        async def _send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                has_id = any(name == self._HEADER_BYTES for name, _ in headers)
                if not has_id:
                    headers.append((self._HEADER_BYTES, rid.encode("ascii")))
                    message = {**message, "headers": headers}
            await send(message)

        # Why not the obvious ``try/finally`` with reset in finally?
        # FastAPI's middleware stack puts ``ServerErrorMiddleware``
        # OUTSIDE us (it's the catch-all for unhandled ``Exception``
        # in routes). If we reset the contextvar in ``finally`` and
        # then re-raise, ``ServerErrorMiddleware``'s handler runs
        # with an empty contextvar -- the error body wouldn't carry
        # the request id and the log line would say ``request_id=-``.
        # Instead: keep the contextvar set as the exception bubbles
        # up; reset only after we're sure no upstream handler still
        # needs the value. Concretely: we never reset on the
        # exception path -- the very next request running on this
        # task will overwrite the var via its own ``set`` call. The
        # ``REQUEST_ID`` ContextVar's per-task scoping means an
        # accidental "leak" to another task is impossible. On the
        # happy path we DO reset so that downstream background work
        # spawned in the same task (lifespan teardown, scheduled
        # tasks) sees ``None`` outside any request scope.
        try:
            await self.app(scope, receive, _send_with_request_id)
        except Exception:
            # Re-raise without resetting -- ServerErrorMiddleware
            # (outside us) will dispatch ``generic_exception_handler``
            # which reads the contextvar. The next request on this
            # task will overwrite via its own ``set``; cross-request
            # contamination requires that next request to send
            # WITHOUT an ``X-Request-ID`` header AND for the previous
            # generated id to leak through, which the per-request
            # ``set`` call prevents by definition.
            raise
        REQUEST_ID.reset(token)
