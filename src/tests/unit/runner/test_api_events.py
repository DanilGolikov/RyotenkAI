"""Phase 1.4 — WebSocket event stream contract.

Covers:
- replay from offset 0 yields buffered events in order
- ``since=N`` resumes from offset N
- truncated buffer closes 4410
- unknown job_id closes 4404
- live events arrive after the connection is open

``fastapi.testclient.TestClient.websocket_connect`` is a sync context
manager; it drives the underlying async server through a synchronous
shim. ``ws.receive_json()`` blocks until the server sends a frame
or closes — perfect for our contract checks.
"""

from __future__ import annotations

import io
import json

import pytest
from starlette.testclient import WebSocketTestSession  # noqa: F401  (used only for typing)
from starlette.websockets import WebSocketDisconnect

from src.runner.main import API_V1_PREFIX
from src.runner.state import JobState

JOBS = f"{API_V1_PREFIX}/jobs"


def _submit(runner_client, job_id: str = "j-1") -> None:  # type: ignore[no-untyped-def]
    kw = {
        "data": {"job_spec": json.dumps({"job_id": job_id})},
        "files": {"plugins_payload": ("p.zip", io.BytesIO(b""), "application/zip")},
    }
    runner_client.post(JOBS, **kw)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestReplay:
    def test_since_zero_yields_buffered_events(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        # The submit endpoint already published one event ("job_submitted")
        # at offset 0. The client should receive it on connect.
        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=0",
        ) as ws:
            event = ws.receive_json()
            assert event["offset"] == 0
            assert event["kind"] == "job_submitted"
            ws.close()

    def test_since_skips_earlier(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        # Publish a couple of in-process events before the client connects.
        bus = runner_client.app.state.bus
        bus.publish("a", {"i": 1})
        bus.publish("b", {"i": 2})

        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=2",
        ) as ws:
            event = ws.receive_json()
            assert event["offset"] == 2
            assert event["kind"] == "b"
            ws.close()


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


class TestLive:
    def test_event_published_after_connect(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=0",
        ) as ws:
            # Drain the buffered submit event.
            ws.receive_json()

            # Now publish a live event — the WS subscriber wakes up.
            runner_client.app.state.bus.publish("step", {"loss": 0.5})
            event = ws.receive_json()
            assert event["kind"] == "step"
            assert event["payload"] == {"loss": 0.5}
            ws.close()


# ---------------------------------------------------------------------------
# Error close codes
# ---------------------------------------------------------------------------


class TestCloseCodes:
    def test_unknown_job_closes_4404(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client, job_id="j-1")
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with runner_client.websocket_connect(
                f"{API_V1_PREFIX}/jobs/other/events?since=0",
            ) as ws:
                ws.receive_json()
        assert exc_info.value.code == 4404

    def test_truncated_closes_4410(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        # Saturate the bus past its capacity. Default capacity is
        # 10000; we can't reasonably bump that here, so manually
        # narrow the bus to make the test fast.
        from src.runner.event_bus import EventBus

        narrow = EventBus(capacity=4)
        runner_client.app.state.bus = narrow
        # Refill events: 10 → buffer holds last 4 (offsets 6..9).
        for _ in range(10):
            narrow.publish("x", {})

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with runner_client.websocket_connect(
                f"{API_V1_PREFIX}/jobs/j-1/events?since=0",  # truncated
            ) as ws:
                ws.receive_json()
        assert exc_info.value.code == 4410


# ---------------------------------------------------------------------------
# State integration
# ---------------------------------------------------------------------------


class TestStateIntegration:
    def test_stop_event_flows_through(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # End-to-end: submit → drive to RUNNING → POST /stop → the
        # WS subscriber sees both the stop_requested event and the
        # FSM state on the snapshot endpoint.
        _submit(runner_client)
        runner_client.app.state.fsm.transition(JobState.RUNNING)

        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=0",
        ) as ws:
            ws.receive_json()  # job_submitted (offset 0)

            r = runner_client.post(f"{JOBS}/j-1/stop")
            assert r.status_code == 202

            # Drain until we see stop_requested. There may be no
            # other events between 0 and stop_requested, so this
            # should be the next frame.
            event = ws.receive_json()
            assert event["kind"] == "stop_requested"
            ws.close()
