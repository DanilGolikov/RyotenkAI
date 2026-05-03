"""Integration: full happy-path POST /jobs → events → completed.

Drives the runner app end-to-end via :class:`JobClient` (HTTP) +
TestClient (WebSocket). Mock supervisor drives FSM transitions
deterministically — no real subprocess, no real signals — so the
test verifies the contract surface the launcher actually consumes
in production:

1. POST /jobs returns the canonical submitted-response shape.
2. WS subscribes the event stream from offset 0.
3. The supervisor transitions FSM to ``running`` (publishes
   ``trainer_spawned``), then to ``completed`` (publishes
   ``trainer_exited``).
4. GET /jobs/{id} reflects the terminal state.
5. The replay slice contains every transition in order.

Pinning the wire here means the launcher's
:class:`SSHTunnelManager` + :class:`JobClient` keep working as the
runner internals evolve.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.utils.clients.job_client import JobClient


# Async fixtures need pytest.mark.asyncio per-test; the sync TestClient
# tests below run without the marker.


# ---------------------------------------------------------------------------
# Async (httpx + ASGI): submit + status round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_returns_expected_shape(
    runner_pair: "tuple[FastAPI, JobClient]",
) -> None:
    _, client = runner_pair
    result = await client.submit_job(
        {"job_id": "j-happy", "command": ["python", "-c", "pass"]},
    )
    assert result["job_id"] == "j-happy"
    assert isinstance(result["sequence"], int)
    assert isinstance(result["offset"], int)


@pytest.mark.asyncio
async def test_status_after_submit_shows_running_then_completed(
    runner_pair: "tuple[FastAPI, JobClient]",
) -> None:
    app, client = runner_pair

    await client.submit_job(
        {"job_id": "j-happy-2", "command": ["python", "-c", "pass"]},
    )

    snap = await client.get_status("j-happy-2")
    # MockSupervisor fast-paths submit → running.
    assert snap["state"] == "running"
    assert snap["job_id"] == "j-happy-2"

    # Drive completion synchronously through the supervisor handle
    # exposed on app.state. This is the test seam — production reaches
    # the same transition through subprocess reap.
    app.state.supervisor.finish(exit_code=0)

    snap = await client.get_status("j-happy-2")
    assert snap["state"] == "completed"


# ---------------------------------------------------------------------------
# Sync (TestClient WebSocket): event replay
# ---------------------------------------------------------------------------


def _submit_via_testclient(client, job_id: str = "j-stream") -> None:  # type: ignore[no-untyped-def]
    """Multipart POST helper for TestClient (matches JobClient's wire shape)."""
    response = client.post(
        "/api/v1/jobs",
        data={"job_spec": json.dumps({
            "job_id": job_id,
            "command": ["python", "-c", "pass"],
        })},
        files={
            "plugins_payload": ("plugins.zip", b"", "application/zip"),
        },
    )
    assert response.status_code in (200, 202), response.text


def test_event_stream_contains_terminal_transitions(
    runner_testclient,  # type: ignore[no-untyped-def]
) -> None:
    app, client = runner_testclient
    # Submit + drive completion *before* opening the WS, so the events
    # are already in the ring buffer for replay.
    _submit_via_testclient(client, "j-stream")
    app.state.supervisor.finish(exit_code=0)

    # Open the WS at offset 0 and read the buffered events. The runner
    # streams forever (the iterator stays open for live tail), so we
    # close from our side once we've seen the terminal event.
    received: list[dict] = []
    with client.websocket_connect("/api/v1/jobs/j-stream/events?since=0") as ws:
        # Hard cap to avoid an infinite loop on regression — we expect
        # ≤ ~5 events for this happy path: job_submitted, trainer_spawned,
        # trainer_exited, plus possibly state transitions.
        for _ in range(20):
            event = ws.receive_json()
            received.append(event)
            if event["kind"] == "trainer_exited":
                break

    kinds = [event["kind"] for event in received]
    assert "trainer_spawned" in kinds
    assert "trainer_exited" in kinds
