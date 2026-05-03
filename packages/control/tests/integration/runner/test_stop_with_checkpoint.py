"""Integration: graceful stop drives FSM to ``cancelled``.

The launcher's stop path (CLI ``ryotenkai job stop`` or the Web UI's
Stop button) issues ``POST /jobs/{id}/stop``. The runner transitions
the FSM to ``stopping`` synchronously, then SIGTERMs the trainer.
When the trainer exits, the FSM lands in ``cancelled`` (clean exit
that was user-initiated) or ``failed`` (process died with non-zero
rc but cancellation was requested).

Mock supervisor models this: ``request_stop`` flips FSM to ``stopping``,
``finish(cancelled=True)`` lands the terminal transition.
"""

from __future__ import annotations

import json

import pytest

from src.utils.clients.job_client import JobClient


pytestmark = pytest.mark.asyncio


async def test_stop_request_drives_fsm_to_stopping_then_cancelled(
    runner_pair,  # type: ignore[no-untyped-def]
) -> None:
    app, client = runner_pair  # type: tuple[..., JobClient]

    await client.submit_job(
        {"job_id": "j-stop", "command": ["python", "-c", "pass"]},
    )

    snap = await client.get_status("j-stop")
    assert snap["state"] == "running"

    # Issue the stop. Runner transitions FSM to ``stopping`` and
    # publishes ``stop_requested`` synchronously.
    await client.request_stop("j-stop", grace_seconds=0.0)
    snap = await client.get_status("j-stop")
    assert snap["state"] == "stopping"

    # Trainer exits with cancellation flag set — terminal state
    # is ``cancelled``, not ``failed`` (despite non-zero rc) since
    # the user requested it.
    app.state.supervisor.finish(exit_code=130, cancelled=True)
    snap = await client.get_status("j-stop")
    assert snap["state"] == "cancelled"


def test_stop_event_visible_on_websocket(
    runner_testclient,  # type: ignore[no-untyped-def]
) -> None:
    app, client = runner_testclient

    response = client.post(
        "/api/v1/jobs",
        data={"job_spec": json.dumps(
            {"job_id": "j-stop-ws", "command": ["python", "-c", "pass"]},
        )},
        files={"plugins_payload": ("plugins.zip", b"", "application/zip")},
    )
    assert response.status_code in (200, 202)

    # Drive the stop synchronously through the supervisor handle so
    # the events land in the buffer before we open the WS.
    response = client.post(
        "/api/v1/jobs/j-stop-ws/stop?grace_seconds=0",
    )
    assert response.status_code in (200, 202), response.text
    app.state.supervisor.finish(exit_code=130, cancelled=True)

    seen_kinds: list[str] = []
    with client.websocket_connect(
        "/api/v1/jobs/j-stop-ws/events?since=0",
    ) as ws:
        for _ in range(20):
            event = ws.receive_json()
            seen_kinds.append(event["kind"])
            if event["kind"] == "trainer_exited":
                break

    # All three lifecycle markers visible.
    assert "stop_requested" in seen_kinds
    assert "trainer_exited" in seen_kinds
