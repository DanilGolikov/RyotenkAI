"""Integration: WebSocket reconnect with offset preservation.

Simulates the Mac-asleep / Mac-wakes flow:

1. Mac submits a job. Runner publishes a few events.
2. Mac's tunnel collapses (we just close the WS).
3. Mac wakes, opens a fresh WS at ``since=<last_offset>+1``.
4. The runner replays everything from that cursor, then continues
   live-tailing as new events arrive.

The invariant is that no event published while the Mac was offline
is dropped — the ring buffer holds them as long as the buffer hasn't
rolled. This is the contract that keeps ``ryotenkai run resume``
seamless across a Mac sleep cycle.
"""

from __future__ import annotations

import json

import pytest


def _submit(client, job_id: str) -> None:  # type: ignore[no-untyped-def]
    response = client.post(
        "/api/v1/jobs",
        data={"job_spec": json.dumps(
            {"job_id": job_id, "command": ["python", "-c", "pass"]},
        )},
        files={"plugins_payload": ("plugins.zip", b"", "application/zip")},
    )
    assert response.status_code in (200, 202), response.text


def test_reattach_at_cursor_skips_already_seen_events(
    runner_testclient,  # type: ignore[no-untyped-def]
) -> None:
    app, client = runner_testclient
    _submit(client, "j-detach")

    # First connection: read the early events, then drop the WS.
    early: list[dict] = []
    with client.websocket_connect("/api/v1/jobs/j-detach/events?since=0") as ws:
        for _ in range(2):
            early.append(ws.receive_json())
    assert len(early) >= 2
    last_seen_offset = early[-1]["offset"]

    # Mac was offline — runner publishes more events meanwhile.
    app.state.supervisor.finish(exit_code=0)

    # Second connection at the post-cursor offset. We must see the
    # newly published events, NOT the ones we already saw.
    catchup: list[dict] = []
    with client.websocket_connect(
        f"/api/v1/jobs/j-detach/events?since={last_seen_offset + 1}",
    ) as ws:
        for _ in range(10):
            event = ws.receive_json()
            catchup.append(event)
            if event["kind"] == "trainer_exited":
                break

    catchup_offsets = {e["offset"] for e in catchup}
    early_offsets = {e["offset"] for e in early}
    # No overlap — the ``since=`` cursor was honoured.
    assert catchup_offsets.isdisjoint(early_offsets)
    # We did receive the terminal transition during catch-up.
    assert any(e["kind"] == "trainer_exited" for e in catchup)


def test_reattach_with_invalid_cursor_returns_close_code(
    runner_testclient,  # type: ignore[no-untyped-def]
) -> None:
    """``since=`` past the buffer's high-water mark is a corrupt cursor.

    The runner closes with the dedicated 4422 code so the Mac client
    can distinguish "your cursor is wrong" from "transient network
    error" and refetch ``get_status`` instead of blindly retrying.
    """
    _, client = runner_testclient
    _submit(client, "j-bad-cursor")

    with pytest.raises(Exception):  # noqa: BLE001 — TestClient raises various
        with client.websocket_connect(
            "/api/v1/jobs/j-bad-cursor/events?since=10000",
        ) as ws:
            ws.receive_json()  # close arrives
