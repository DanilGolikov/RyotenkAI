"""Phase 5 contract tests — :class:`JobClient` ↔ real runner app.

The unit tests in :mod:`test_job_client` use ``httpx.MockTransport`` —
they assert on the wire shape the client builds, but a typo in either
side could leave them passing while the actual integration breaks.

These contract tests wire :class:`JobClient` directly to the runner's
FastAPI app via :class:`httpx.ASGITransport` (no sockets, no
subprocess). They verify the round-trip works end-to-end for all
HTTP endpoints — submit, get_status, request_stop. WebSocket flow is
exercised in :mod:`src.tests.unit.runner.test_api_events` through a
TestClient and not duplicated here.

We import the runner's app via :func:`src.runner.main.create_app`
with :class:`MockSupervisor` so we don't fork real subprocesses.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import pytest_asyncio

from ryotenkai_shared.utils.clients.job_client import JobClient, JobNotFoundError
from ryotenkai_pod.runner.main import create_app

# Phase B follow-up: see packages/control/tests/integration/runner/conftest.py
import importlib.util as _ilu
import pathlib as _pathlib

_pod_runner_conftest_path = (
    _pathlib.Path(__file__).resolve().parents[5]
    / "pod" / "tests" / "unit" / "runner" / "conftest.py"
)
_spec = _ilu.spec_from_file_location(
    "_pod_runner_conftest_for_contract_tests", str(_pod_runner_conftest_path),
)
assert _spec is not None and _spec.loader is not None
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MockSupervisor = _mod.MockSupervisor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client_against_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> "AsyncIterator[JobClient]":
    """Build a :class:`JobClient` whose HTTP transport is the runner
    app itself — no sockets, no subprocesses. Lifespan runs once per
    test so the FSM is fresh.

    ``httpx.ASGITransport`` does NOT drive the FastAPI lifespan
    automatically the way :class:`fastapi.testclient.TestClient` does;
    we drive it by hand via ``app.router.lifespan_context``. This is
    the pattern recommended by the httpx docs when you can't pull in
    the ``asgi-lifespan`` helper.
    """
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    app = create_app(supervisor_factory=MockSupervisor)

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(
        transport=transport, base_url="http://contract.local",
    )
    client = JobClient("http://contract.local", http_client=http)
    # ``lifespan_context`` is async-context-managed: entering wires
    # ``app.state.fsm/bus/supervisor``; leaving drives shutdown.
    async with app.router.lifespan_context(app):
        try:
            yield client
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestSubmitContract:
    async def test_submit_returns_job_submitted_response(
        self, client_against_runner: JobClient,
    ) -> None:
        result = await client_against_runner.submit_job(
            {"job_id": "j-contract", "command": ["python", "-c", "pass"]},
        )
        # The runner replies with :class:`JobSubmittedResponse` —
        # ``job_id`` echoes the input, ``sequence`` and ``offset``
        # are the cursor values the WS subscriber needs to start.
        assert result["job_id"] == "j-contract"
        assert "sequence" in result
        assert "offset" in result


class TestGetStatusContract:
    async def test_404_for_unknown_job(
        self, client_against_runner: JobClient,
    ) -> None:
        with pytest.raises(JobNotFoundError):
            await client_against_runner.get_status("never-submitted")

    async def test_round_trip_submit_then_status(
        self, client_against_runner: JobClient,
    ) -> None:
        await client_against_runner.submit_job(
            {"job_id": "j-rt", "command": ["python", "-c", "pass"]},
        )
        snapshot = await client_against_runner.get_status("j-rt")
        assert snapshot["job_id"] == "j-rt"
        assert "state" in snapshot


class TestStopContract:
    async def test_stop_unknown_job_404s(
        self, client_against_runner: JobClient,
    ) -> None:
        with pytest.raises(JobNotFoundError):
            await client_against_runner.request_stop("never-submitted")
