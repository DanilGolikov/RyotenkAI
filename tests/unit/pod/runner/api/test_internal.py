"""Phase 2 — loopback internal events endpoint over typed envelopes.

Covers the trainer-side push contract (``POST /internal/events``):
- happy path validates a full envelope and assigns the next offset.
- request without an active job returns 409.
- malformed envelope (missing required fields) returns 422.
- non-loopback peer is rejected.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from uuid import uuid4

from ryotenkai_pod.runner.main import API_V1_PREFIX

INTERNAL_EVENTS = f"{API_V1_PREFIX}/internal/events"
JOBS = f"{API_V1_PREFIX}/jobs"


def _make_envelope(kind: str = "ryotenkai.pod.training.step") -> dict:
    """Build a minimal valid envelope JSON for the codec."""
    return {
        "event_id": str(uuid4()),
        "kind": kind,
        "source": "pod://j-1/trainer",
        "time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": "j-1",
        "stage_id": None,
        "offset": -1,
        "schema_version": 1,
        "severity": "debug",
        "payload": {
            "step": 100,
            "loss": 0.42,
            "learning_rate": 0.001,
        },
    }


def _submit(runner_client, job_id: str = "j-1") -> None:  # type: ignore[no-untyped-def]
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
        envelope = _make_envelope()
        r = runner_client.post(INTERNAL_EVENTS, json=envelope)
        assert r.status_code == 202
        body = r.json()
        # Bus assigned offset 3 (prior boot events at offsets 0/1/2).
        assert body["offset"] == 3
        assert body["kind"] == "ryotenkai.pod.training.step"

    def test_no_active_job_returns_409(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        envelope = _make_envelope()
        r = runner_client.post(INTERNAL_EVENTS, json=envelope)
        assert r.status_code == 409

    def test_invalid_envelope_returns_422(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        # Missing required ``payload`` field for the discriminator kind.
        bad = _make_envelope()
        bad.pop("payload")
        r = runner_client.post(INTERNAL_EVENTS, json=bad)
        assert r.status_code == 422


class TestLoopbackGate:
    def test_testclient_treated_as_loopback(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        _submit(runner_client)
        r = runner_client.post(INTERNAL_EVENTS, json=_make_envelope())
        assert r.status_code == 202

    def test_non_trusted_host_returns_403(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        from ryotenkai_pod.runner.api import internal as internal_module

        _submit(runner_client)
        original = internal_module._TRUSTED_HOSTS
        internal_module._TRUSTED_HOSTS = frozenset({"127.0.0.1"})  # drop testclient
        try:
            r = runner_client.post(INTERNAL_EVENTS, json=_make_envelope())
        finally:
            internal_module._TRUSTED_HOSTS = original
        assert r.status_code == 403
        assert r.json()["code"] == "LOOPBACK_REQUIRED"
