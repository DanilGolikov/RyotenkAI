"""Unit tests for :mod:`ryotenkai_shared.api.request_id` (Phase C).

Test categories per project policy (7-class coverage):

* positive       -- middleware sets, exposes, and echoes a fresh id.
* negative       -- malformed/empty incoming header still produces a
                    fresh id (no propagation of empty string).
* boundary       -- id length, hex shape, missing header treated as
                    "generate", multiple concurrent requests don't
                    leak ids across tasks.
* invariant      -- response always carries ``X-Request-ID``;
                    contextvar is restored after request (no leak).
* dependency-err -- handler raising downstream still resets the
                    contextvar (no leak across requests).
* regression     -- ProblemDetails emitted by error_handlers picks
                    up the same id the response header carries.
* logic-specific -- ``current_request_id()`` returns the live id
                    inside a handler; returns ``None`` outside any
                    request scope (no middleware-set).
"""

from __future__ import annotations

import asyncio
import re

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ryotenkai_shared.api.error_handlers import EXCEPTION_HANDLERS
from ryotenkai_shared.api.request_id import (
    REQUEST_ID,
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
    current_request_id,
    generate_request_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app(record: list[str | None] | None = None) -> FastAPI:
    """Tiny FastAPI app with the middleware + a route that records the
    contextvar value at handler-time so tests can assert "what the
    middleware exposed to the handler".

    ``record`` is an external sink: when provided, the handler
    appends ``current_request_id()`` to it before responding. Tests
    can therefore assert the handler saw the same id the response
    carries.
    """
    app = FastAPI(exception_handlers=EXCEPTION_HANDLERS)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    def _ping() -> dict[str, str | None]:
        rid = current_request_id()
        if record is not None:
            record.append(rid)
        return {"rid": rid}

    @app.get("/boom")
    def _boom() -> dict[str, str]:
        raise HTTPException(status_code=500, detail="kaboom")

    return app


# ---------------------------------------------------------------------------
# 1. positive -- happy path
# ---------------------------------------------------------------------------


def test_positive_middleware_generates_id_when_header_absent() -> None:
    """No incoming ``X-Request-ID`` -- middleware generates one,
    handler sees it via the contextvar, response echoes it."""
    seen: list[str | None] = []
    client = TestClient(_build_app(seen))

    resp = client.get("/ping")

    assert resp.status_code == 200
    rid = resp.headers[REQUEST_ID_HEADER]
    # 16 hex chars (secrets.token_hex(8))
    assert re.fullmatch(r"[0-9a-f]{16}", rid), f"unexpected shape: {rid!r}"
    # Handler saw the same id that came back on the response.
    assert seen == [rid]
    # The response body confirms the handler observed the live id
    # rather than a stale value from a different request.
    assert resp.json() == {"rid": rid}


def test_positive_middleware_preserves_incoming_id() -> None:
    """Incoming ``X-Request-ID`` is honoured verbatim and echoed back."""
    seen: list[str | None] = []
    client = TestClient(_build_app(seen))

    given = "client-supplied-trace-id"
    resp = client.get("/ping", headers={REQUEST_ID_HEADER: given})

    assert resp.status_code == 200
    assert resp.headers[REQUEST_ID_HEADER] == given
    assert seen == [given]


# ---------------------------------------------------------------------------
# 2. negative -- empty incoming value
# ---------------------------------------------------------------------------


def test_negative_empty_incoming_header_triggers_generation() -> None:
    """An empty incoming value must not propagate -- the middleware
    generates a fresh id rather than passing through ``""``."""
    seen: list[str | None] = []
    client = TestClient(_build_app(seen))

    resp = client.get("/ping", headers={REQUEST_ID_HEADER: ""})

    assert resp.status_code == 200
    rid = resp.headers[REQUEST_ID_HEADER]
    # ``""`` is falsy -> middleware should have generated a fresh id.
    assert re.fullmatch(r"[0-9a-f]{16}", rid)
    assert seen == [rid]
    assert rid != ""


# ---------------------------------------------------------------------------
# 3. boundary -- shape and concurrency
# ---------------------------------------------------------------------------


def test_boundary_generate_request_id_shape() -> None:
    """``generate_request_id`` returns 16 lowercase hex chars."""
    rid = generate_request_id()
    assert re.fullmatch(r"[0-9a-f]{16}", rid), rid
    # Pair-wise distinct under repeated calls (high probability) --
    # this is the lightweight "is it actually random?" smoke test.
    others = {generate_request_id() for _ in range(50)}
    assert len(others) == 50


def test_boundary_concurrent_requests_dont_leak_ids() -> None:
    """Two concurrent requests with different ids must each see
    their own id -- the contextvar's per-task semantics protect us
    from cross-talk. We simulate concurrency with a TestClient on a
    threadpool because Starlette's middleware uses asyncio under the
    hood and the contextvar guards against task-level reuse."""
    seen: list[tuple[str, str | None]] = []

    app = FastAPI(exception_handlers=EXCEPTION_HANDLERS)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/concurrent/{tag}")
    async def _route(tag: str) -> dict[str, str | None]:
        # Yield to the event loop so a second handler could in
        # principle race in and mutate a non-task-local store.
        await asyncio.sleep(0)
        rid = current_request_id()
        seen.append((tag, rid))
        return {"rid": rid}

    with TestClient(app) as client:
        r1 = client.get("/concurrent/a", headers={REQUEST_ID_HEADER: "rid-a"})
        r2 = client.get("/concurrent/b", headers={REQUEST_ID_HEADER: "rid-b"})

    assert r1.headers[REQUEST_ID_HEADER] == "rid-a"
    assert r2.headers[REQUEST_ID_HEADER] == "rid-b"
    # The recorded handler observations match the response headers
    # one-to-one -- no leakage.
    assert ("a", "rid-a") in seen
    assert ("b", "rid-b") in seen


# ---------------------------------------------------------------------------
# 4. invariant -- response always carries the header, contextvar resets
# ---------------------------------------------------------------------------


def test_invariant_response_always_carries_header_even_on_error() -> None:
    """An exception flowing through the handler still results in an
    ``X-Request-ID`` on the response -- the middleware wraps both
    the success and the error paths."""
    client = TestClient(_build_app(), raise_server_exceptions=False)

    resp = client.get("/boom", headers={REQUEST_ID_HEADER: "err-trace"})

    # 500 from the legacy HTTPException handler.
    assert resp.status_code == 500
    assert resp.headers[REQUEST_ID_HEADER] == "err-trace"


def test_invariant_contextvar_resets_after_request() -> None:
    """After the response is delivered, the request-scoped contextvar
    must be back to ``None`` -- outside-of-request code should never
    observe a stale id from the last request."""
    client = TestClient(_build_app())
    client.get("/ping", headers={REQUEST_ID_HEADER: "x"})

    # Synchronous read here -- we're outside the ASGI scope now.
    assert REQUEST_ID.get() is None
    assert current_request_id() is None


# ---------------------------------------------------------------------------
# 5. dependency-err -- handler raising still resets the contextvar
# ---------------------------------------------------------------------------


def test_dependency_err_exception_does_not_leak_contextvar() -> None:
    """Even when the route raises, the ``finally`` block in the
    middleware must reset the contextvar so the next request starts
    clean."""
    client = TestClient(_build_app(), raise_server_exceptions=False)

    client.get("/boom", headers={REQUEST_ID_HEADER: "first"})
    # Issue a second request without supplying a header; if the
    # contextvar had leaked we'd see ``"first"`` instead of a fresh id.
    seen: list[str | None] = []
    app2 = FastAPI(exception_handlers=EXCEPTION_HANDLERS)
    app2.add_middleware(RequestIDMiddleware)

    @app2.get("/probe")
    def _probe() -> dict[str, str | None]:
        seen.append(current_request_id())
        return {"rid": current_request_id()}

    with TestClient(app2) as client2:
        resp = client2.get("/probe")
    assert resp.headers[REQUEST_ID_HEADER] != "first"
    assert seen[0] != "first"


# ---------------------------------------------------------------------------
# 6. regression -- ProblemDetails carries request_id from the contextvar
# ---------------------------------------------------------------------------


def test_regression_problem_details_carries_request_id() -> None:
    """When an unhandled exception flows through the shared
    ``generic_exception_handler``, the resulting ``problem+json``
    body must include the same ``request_id`` the response header
    carries -- the contextvar is the single source of truth."""
    app = FastAPI(exception_handlers=EXCEPTION_HANDLERS)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/crash")
    def _crash() -> None:
        raise RuntimeError("totally unexpected")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/crash", headers={REQUEST_ID_HEADER: "corr-42"})

    assert resp.status_code == 500
    assert resp.headers[REQUEST_ID_HEADER] == "corr-42"
    body = resp.json()
    # Generic handler emits INTERNAL_ERROR with the request_id from
    # the contextvar -- confirms end-to-end propagation.
    assert body["request_id"] == "corr-42"
    assert body["code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# 7. logic-specific -- current_request_id() outside any request
# ---------------------------------------------------------------------------


def test_logic_specific_current_request_id_is_none_outside_request() -> None:
    """Calling ``current_request_id`` outside an ASGI request scope
    must return ``None`` -- background workers, tests, and CLI code
    should be able to introspect the contextvar without needing a
    middleware-mounted app."""
    # Sanity: nothing has set the contextvar in this test process.
    # (Other tests reset it; the module default is ``None``.)
    assert current_request_id() is None


def test_logic_specific_manual_contextvar_set_round_trips() -> None:
    """``REQUEST_ID.set(...)`` / ``REQUEST_ID.reset(token)`` is the
    middleware's contract -- this test exercises the same primitives
    directly to lock in the public read path."""
    assert current_request_id() is None
    token = REQUEST_ID.set("manual-id")
    try:
        assert current_request_id() == "manual-id"
    finally:
        REQUEST_ID.reset(token)
    assert current_request_id() is None
