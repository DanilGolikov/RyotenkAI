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

from ryotenkai_pod.runner.main import API_V1_PREFIX

JOBS = f"{API_V1_PREFIX}/jobs"


def _submit(runner_client, job_id: str = "j-1") -> None:  # type: ignore[no-untyped-def]
    kw = {
        "data": {"job_spec": json.dumps({"job_id": job_id, "command": ["python", "-c", "pass"]})},
        "files": {"plugins_payload": ("p.zip", io.BytesIO(b""), "application/zip")},
    }
    runner_client.post(JOBS, **kw)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestReplay:
    def test_since_zero_yields_buffered_events(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        # POST /jobs publishes three events in this order:
        # offset 0: plugins_unpacked  — emitted BEFORE the supervisor
        #                                spawn so plugins are on disk
        #                                before the trainer starts
        # offset 1: job_submitted     — emitted by supervisor.submit
        # offset 2: trainer_spawned   — emitted after FSM → running
        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=0",
        ) as ws:
            ev0 = ws.receive_json()
            assert ev0["offset"] == 0
            assert ev0["kind"] == "plugins_unpacked"
            ev1 = ws.receive_json()
            assert ev1["offset"] == 1
            assert ev1["kind"] == "job_submitted"
            ev2 = ws.receive_json()
            assert ev2["offset"] == 2
            assert ev2["kind"] == "trainer_spawned"
            ws.close()

    def test_since_skips_earlier(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        # submit_and_spawn populated offsets 0..2 (job_submitted,
        # plugins_unpacked, trainer_spawned). Add two more then
        # subscribe from 4.
        bus = runner_client.app.state.bus
        bus.publish("a", {"i": 1})
        bus.publish("b", {"i": 2})

        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=4",
        ) as ws:
            event = ws.receive_json()
            assert event["offset"] == 4
            assert event["kind"] == "b"
            ws.close()


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


class TestLive:
    def test_event_published_after_connect(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=3",  # skip job_submitted, plugins_unpacked, trainer_spawned
        ) as ws:
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
        with pytest.raises(WebSocketDisconnect) as exc_info, runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/other/events?since=0",
        ) as ws:
            ws.receive_json()
        assert exc_info.value.code == 4404

    def test_truncated_closes_4410(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        # Saturate the bus past its capacity. Default capacity is
        # 10000; we can't reasonably bump that here, so manually
        # narrow the bus to make the test fast.
        from ryotenkai_pod.runner.event_bus import EventBus

        narrow = EventBus(capacity=4)
        runner_client.app.state.bus = narrow
        # Refill events: 10 → buffer holds last 4 (offsets 6..9).
        for _ in range(10):
            narrow.publish("x", {})

        with pytest.raises(WebSocketDisconnect) as exc_info, runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=0",  # truncated
        ) as ws:
            ws.receive_json()
        assert exc_info.value.code == 4410


# ---------------------------------------------------------------------------
# State integration
# ---------------------------------------------------------------------------


class TestStateIntegration:
    def test_stop_event_flows_through(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # End-to-end: submit (mock supervisor lands FSM in running)
        # → POST /stop → WS subscriber sees stop_requested.
        _submit(runner_client)

        with runner_client.websocket_connect(
            f"{API_V1_PREFIX}/jobs/j-1/events?since=3",  # skip 3 boot events
        ) as ws:
            r = runner_client.post(f"{JOBS}/j-1/stop")
            assert r.status_code == 202

            event = ws.receive_json()
            assert event["kind"] == "stop_requested"
            ws.close()
