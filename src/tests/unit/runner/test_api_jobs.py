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
        kw = _multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]})
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["job_id"] == "j-1"
        # MockSupervisor synchronously advances preparing → running,
        # so the snapshot reported back is at sequence=1.
        assert body["sequence"] == 1
        # Offset 0 is ``job_submitted``; the trainer_spawned event
        # follows at offset 1. The endpoint always returns 0 so the
        # client can replay both with since=0.
        assert body["offset"] == 0

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
        # First submit drives FSM through preparing → running via the
        # MockSupervisor. The second one must 409 — the supervisor
        # is busy.
        kw = _multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]})
        runner_client.post(JOBS, **kw)

        kw2 = _multipart({"job_id": "j-2", "command": ["python", "-c", "pass"]})
        r = runner_client.post(JOBS, **kw2)
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "job_in_progress"

    def test_resubmit_after_terminal_state_succeeds(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # Drive the first job to a terminal state via the mock's
        # ``finish`` helper (real supervisor does this through ``_reap``
        # when the trainer subprocess exits).
        kw = _multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]})
        runner_client.post(JOBS, **kw)
        runner_client.app.state.supervisor.finish(exit_code=0)

        kw2 = _multipart({"job_id": "j-2", "command": ["python", "-c", "pass"]})
        r = runner_client.post(JOBS, **kw2)
        assert r.status_code == 202
        assert r.json()["job_id"] == "j-2"


# ---------------------------------------------------------------------------
# GET /jobs/{id}
# ---------------------------------------------------------------------------


class TestGet:
    def test_returns_snapshot_for_active_job(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        kw = _multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]})
        runner_client.post(JOBS, **kw)
        r = runner_client.get(f"{JOBS}/j-1")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == "j-1"
        # MockSupervisor advances FSM to ``running`` synchronously.
        assert body["state"] == "running"
        assert body["sequence"] == 1  # 0 = preparing, 1 = running
        # Three events on the bus after submit:
        # offset 0: job_submitted
        # offset 1: plugins_unpacked  (Phase 6.2 — empty payload still
        #                              publishes a structured event)
        # offset 2: trainer_spawned
        assert body["last_event_offset"] == 2

    def test_404_for_unknown_id(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        kw = _multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]})
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
        # MockSupervisor lands in ``running`` after submit_and_spawn.
        runner_client.post(JOBS, **_multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]}))

        r = runner_client.post(f"{JOBS}/j-1/stop")
        assert r.status_code == 202
        body = r.json()
        assert body["state"] == "stopping"
        # 0 = preparing, 1 = running, 2 = stopping.
        assert body["sequence"] == 2

    def test_stop_from_terminal_returns_409(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # MockSupervisor.finish drives the FSM straight to ``completed``;
        # stop from a terminal state is rejected.
        runner_client.post(JOBS, **_multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]}))
        runner_client.app.state.supervisor.finish(exit_code=0)
        r = runner_client.post(f"{JOBS}/j-1/stop")
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "stop_not_allowed"
        assert r.json()["detail"]["current_state"] == "completed"

    def test_stop_unknown_job_returns_404(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        runner_client.post(JOBS, **_multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]}))
        r = runner_client.post(f"{JOBS}/other/stop")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Plugins payload handling
# ---------------------------------------------------------------------------


class TestPluginsPayload:
    def test_empty_payload_accepted(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # SFT-only jobs ship no reward plugins. The Mac client sends
        # ``b""`` and the runner short-circuits to a no-op unpack.
        kw = _multipart({"job_id": "j-1", "command": ["python", "-c", "pass"]})
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 202

    def test_valid_zip_extracted(self, runner_client, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # Build a minimal valid ZIP with one reward plugin and verify
        # the unpacker landed it on disk + emitted the event.
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("reward/plugin_a/manifest.toml", b"[plugin]\nid='a'\n")
            zf.writestr("reward/plugin_a/plugin.py", b"# code")

        kw = _multipart(
            {"job_id": "j-1", "command": ["python", "-c", "pass"]},
            payload=buf.getvalue(),
        )
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 202, r.text
        # Plugin landed in the runner's workspace community/.
        unpacker = runner_client.app.state.plugin_unpacker
        plugin_path = (
            unpacker.community_root / "reward" / "plugin_a" / "manifest.toml"
        )
        assert plugin_path.exists()

    def test_invalid_zip_returns_422(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        # Garbage bytes that aren't a valid ZIP must be rejected before
        # the trainer is spawned — no half-running jobs against a
        # missing community/ folder.
        kw = _multipart(
            {"job_id": "j-1", "command": ["python", "-c", "pass"]},
            payload=b"not a zip",
        )
        r = runner_client.post(JOBS, **kw)
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "plugin_unpack_failed"
