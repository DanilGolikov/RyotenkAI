"""Phase 1.3 — REST endpoints under ``/api/v1/jobs``.

Covers:
- POST /jobs              happy path, validation errors, double-submit conflict
- GET  /jobs/{id}         active match, 404 for unknown id
- POST /jobs/{id}/stop    happy path, illegal-from-state conflict, 404
"""

from __future__ import annotations

import io
import json

from src.runner.main import API_V1_PREFIX

JOBS = f"{API_V1_PREFIX}/jobs"


def _multipart(spec: dict, payload: bytes = b"") -> dict:
    """Build the multipart payload TestClient consumes."""
    return {
        "data": {"job_spec": json.dumps(spec)},
        "files": {"plugins_payload": ("plugins.zip", io.BytesIO(payload), "application/zip")},
    }


# ---------------------------------------------------------------------------
# POST /jobs
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_happy_path(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        kw = _multipart({"job_id": "j-1"})
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["job_id"] == "j-1"
        assert body["sequence"] == 0
        assert body["offset"] == 0  # first event on the bus

    def test_invalid_job_spec_returns_422(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # Empty job_id violates min_length=1
        kw = _multipart({"job_id": ""})
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "invalid_job_spec"

    def test_malformed_json_returns_422(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        kw = {
            "data": {"job_spec": "{not valid json"},
            "files": {"plugins_payload": ("p.zip", io.BytesIO(b""), "application/zip")},
        }
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 422

    def test_extra_field_rejected(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # extra="forbid" surfaces typos at parse time.
        kw = _multipart({"job_id": "j-1", "typo_field": "oops"})
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 422

    def test_double_submit_while_running_returns_409(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # Submit + manually advance to RUNNING via the FSM directly,
        # then try to submit a second job — should 409.
        from src.runner.state import JobState

        kw = _multipart({"job_id": "j-1"})
        runner_client.post(JOBS, **kw)
        runner_client.app.state.fsm.transition(JobState.RUNNING)

        kw2 = _multipart({"job_id": "j-2"})
        r = runner_client.post(JOBS, **kw2)
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "job_in_progress"
        assert r.json()["detail"]["current_state"] == "running"

    def test_resubmit_after_terminal_state_succeeds(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        from src.runner.state import JobState

        kw = _multipart({"job_id": "j-1"})
        runner_client.post(JOBS, **kw)
        # Drive to terminal.
        runner_client.app.state.fsm.transition(JobState.RUNNING)
        runner_client.app.state.fsm.transition(JobState.COMPLETED)

        kw2 = _multipart({"job_id": "j-2"})
        r = runner_client.post(JOBS, **kw2)
        assert r.status_code == 202
        assert r.json()["job_id"] == "j-2"


# ---------------------------------------------------------------------------
# GET /jobs/{id}
# ---------------------------------------------------------------------------


class TestGet:
    def test_returns_snapshot_for_active_job(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        kw = _multipart({"job_id": "j-1"})
        runner_client.post(JOBS, **kw)
        r = runner_client.get(f"{JOBS}/j-1")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == "j-1"
        assert body["state"] == "preparing"
        assert body["sequence"] == 0
        assert body["last_event_offset"] == 0  # the synthetic submit event

    def test_404_for_unknown_id(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        kw = _multipart({"job_id": "j-1"})
        runner_client.post(JOBS, **kw)
        r = runner_client.get(f"{JOBS}/j-other")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "job_not_found"

    def test_404_when_no_job_submitted(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        r = runner_client.get(f"{JOBS}/anything")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /jobs/{id}/stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_happy_path_running_to_stopping(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        from src.runner.state import JobState

        runner_client.post(JOBS, **_multipart({"job_id": "j-1"}))
        runner_client.app.state.fsm.transition(JobState.RUNNING)

        r = runner_client.post(f"{JOBS}/j-1/stop")
        assert r.status_code == 202
        body = r.json()
        assert body["state"] == "stopping"
        assert body["sequence"] == 2

    def test_stop_from_preparing_returns_409(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # FSM forbids preparing → stopping.
        runner_client.post(JOBS, **_multipart({"job_id": "j-1"}))
        r = runner_client.post(f"{JOBS}/j-1/stop")
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "stop_not_allowed"
        assert r.json()["detail"]["current_state"] == "preparing"

    def test_stop_unknown_job_returns_404(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        runner_client.post(JOBS, **_multipart({"job_id": "j-1"}))
        r = runner_client.post(f"{JOBS}/other/stop")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Plugins payload handling
# ---------------------------------------------------------------------------


class TestPluginsPayload:
    def test_payload_is_consumed_even_if_unused(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # Phase 1 reads + discards the body. Asserting status 202
        # with non-empty bytes confirms the consume path doesn't
        # raise on real ZIP magic bytes.
        zip_magic = b"PK\x03\x04" + b"\x00" * 100
        kw = _multipart({"job_id": "j-1"}, payload=zip_magic)
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 202
