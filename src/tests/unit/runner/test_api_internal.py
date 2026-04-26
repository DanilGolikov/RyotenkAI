"""Phase 1.5 — loopback internal events endpoint.

Covers the trainer-side push contract (``POST /internal/events``):
- happy path publishes to the bus and echoes the bus assignment
- request without an active job returns 409
- malformed payload (missing ``kind``) → 422
"""

from __future__ import annotations

import io
import json

from src.runner.main import API_V1_PREFIX

INTERNAL_EVENTS = f"{API_V1_PREFIX}/internal/events"
JOBS = f"{API_V1_PREFIX}/jobs"


def _submit(runner_client, job_id: str = "j-1") -> None:  # type: ignore[no-untyped-def]
    """Drive the FSM through ``preparing → running`` (via MockSupervisor)
    so the trainer-event endpoint has a parent job to attribute to."""
    kw = {
        "data": {
            "job_spec": json.dumps(
                {"job_id": job_id, "command": ["python", "-c", "pass"]},
            ),
        },
        "files": {"plugins_payload": ("p.zip", io.BytesIO(b""), "application/zip")},
    }
    runner_client.post(JOBS, **kw)


class TestPushEvent:
    def test_happy_path(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        r = runner_client.post(
            INTERNAL_EVENTS,
            json={"kind": "step", "payload": {"loss": 0.42, "step": 100}},
        )
        assert r.status_code == 202
        body = r.json()
        assert body["kind"] == "step"
        assert body["payload"] == {"loss": 0.42, "step": 100}
        # The bus has three prior events from submit_and_spawn:
        # job_submitted (0), plugins_unpacked (1), trainer_spawned (2).
        # Our trainer-pushed event is offset 3.
        assert body["offset"] == 3

    def test_no_active_job_returns_409(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # No prior submit → no FSM snapshot.
        r = runner_client.post(
            INTERNAL_EVENTS,
            json={"kind": "step", "payload": {}},
        )
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "no_active_job"

    def test_missing_kind_returns_422(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        r = runner_client.post(INTERNAL_EVENTS, json={"payload": {}})
        assert r.status_code == 422

    def test_extra_field_rejected(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        r = runner_client.post(
            INTERNAL_EVENTS,
            json={"kind": "step", "payload": {}, "extra": "oops"},
        )
        assert r.status_code == 422

    def test_payload_defaults_to_empty(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        r = runner_client.post(INTERNAL_EVENTS, json={"kind": "heartbeat"})
        assert r.status_code == 202
        assert r.json()["payload"] == {}


class TestLoopbackGate:
    def test_testclient_treated_as_loopback(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # TestClient's synthetic ``testclient`` peer is whitelisted —
        # it can never come from a real network so the loopback gate
        # treats it as trusted. See the rationale in
        # :data:`src.runner.api.internal._TRUSTED_HOSTS`.
        _submit(runner_client)
        r = runner_client.post(
            INTERNAL_EVENTS,
            json={"kind": "step", "payload": {}},
        )
        assert r.status_code == 202

    def test_non_trusted_host_returns_403(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # Simulate a request from a non-loopback peer (someone who
        # bypassed the uvicorn 127.0.0.1 binding). We can't easily
        # change TestClient's peer host, so instead we widen the
        # check: monkeypatch the trusted set down to a value
        # TestClient does not match, and confirm the gate fires.
        from src.runner.api import internal as internal_module

        _submit(runner_client)
        original = internal_module._TRUSTED_HOSTS
        internal_module._TRUSTED_HOSTS = frozenset({"127.0.0.1"})  # drop testclient
        try:
            r = runner_client.post(
                INTERNAL_EVENTS,
                json={"kind": "step", "payload": {}},
            )
        finally:
            internal_module._TRUSTED_HOSTS = original
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "loopback_required"
