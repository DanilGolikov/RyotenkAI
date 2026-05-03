"""Shared fixtures for the runner integration suite.

The integration tests in this directory drive a *real* FastAPI runner
app through :class:`JobClient` — same wire format the launcher uses
in production — but inject :class:`MockSupervisor` so we don't fork
a real Python interpreter per test. That lets the suite cover the
full submit → events → terminal flow with deterministic timing while
staying CI-friendly.

Fixtures
--------

- :func:`client_against_runner` — :class:`JobClient` whose HTTP
  transport is the runner app itself via :class:`httpx.ASGITransport`.
  No sockets, no subprocesses. Lifespan runs once per test.
- :func:`runner_app` — the underlying :class:`FastAPI` instance for
  tests that need to reach into ``app.state`` (e.g. drive the mock
  supervisor's ``finish()``).

Pattern follows :file:`src/tests/unit/api/clients/test_job_client_contract.py`
which already wires a :class:`JobClient` through ASGI; this module
generalises that fixture for reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import pytest_asyncio

from src.utils.clients.job_client import JobClient
from src.runner.main import create_app
from src.tests.unit.runner.conftest import MockSupervisor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


@pytest_asyncio.fixture
async def runner_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> "AsyncIterator[tuple[FastAPI, JobClient]]":
    """``(app, client)`` pair sharing the same FastAPI lifespan.

    Tests that need to reach into ``app.state.supervisor`` to drive
    ``finish()`` / ``fail()`` from the test thread receive the app.
    Tests that only consume the JobClient surface can ignore it.
    """
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    app = create_app(supervisor_factory=MockSupervisor)

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(
        transport=transport, base_url="http://runner.local",
    )
    client = JobClient("http://runner.local", http_client=http)

    async with app.router.lifespan_context(app):
        try:
            yield app, client
        finally:
            await client.aclose()


@pytest.fixture
def runner_testclient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):  # type: ignore[no-untyped-def]
    """Synchronous :class:`fastapi.testclient.TestClient`.

    ``httpx.ASGITransport`` doesn't speak WebSocket, so any test that
    exercises ``GET /api/v1/jobs/{id}/events`` (the WS endpoint) needs
    TestClient instead — it knows how to drive ASGI WS in-process.
    Tests that only need HTTP can use either fixture.

    Yields ``(app, client)`` so test code can also reach
    ``app.state.supervisor`` for deterministic ``finish()`` /
    ``fail()`` driving.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    app = create_app(supervisor_factory=MockSupervisor)
    with TestClient(app) as client:
        yield app, client
