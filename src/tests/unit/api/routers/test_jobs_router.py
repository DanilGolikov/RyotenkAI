"""Phase 7.2 — control-plane proxy router contract.

The Web UI MVP cannot open SSH tunnels itself, so the FastAPI server
hosts a proxy router (:mod:`src.api.routers.jobs`) that opens a
short-lived tunnel + JobClient and forwards the call to the in-pod
runner. These tests pin the response shapes and error mapping so the
frontend keeps working as the proxy evolves.

We mount a minimal FastAPI app with just the jobs router (so we don't
pull in the heavyweight datasets dependency from the full app), and
patch ``_with_runner`` to bypass real SSH/HTTP — same trick the CLI
tests use. The submission file is a real on-disk JobSubmission so
the resolution path is exercised end-to-end.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.config import ApiSettings
from src.api.dependencies import get_settings
from src.api.routers import jobs as jobs_router
from src.pipeline.state.job_submission import JobSubmission, save_job_submission


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_submission_file(run_dir: Path, *, attempt_no: int = 1) -> Path:
    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_no}"
    submission = JobSubmission(
        schema_version=JobSubmission.CURRENT_VERSION,
        job_id=f"j-{attempt_no}",
        provider_name="runpod",
        pod_id="pod-xyz",
        ssh_host="1.2.3.4",
        ssh_port=22022,
        ssh_username="root",
        ssh_key_path="/k/id",
        created_at_iso="2026-04-26T00:00:00+00:00",
    )
    save_job_submission(attempt_dir, submission)
    return attempt_dir


@pytest.fixture()
def settings(tmp_path: Path) -> ApiSettings:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    return ApiSettings(
        runs_dir=runs_dir,
        projects_root=projects_root,
        serve_spa=False,
        cors_origins=["http://localhost:5173"],
    )


@pytest.fixture()
def client(settings: ApiSettings) -> Iterator[TestClient]:
    """Minimal FastAPI app with only the jobs router mounted.

    Avoids importing ``src.api.main`` (which pulls in the full router
    set including datasets, requiring the heavy ``datasets`` package)."""
    app = FastAPI()
    app.include_router(jobs_router.router, prefix="/api/v1")
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


def _patch_with_runner(client_mock: MagicMock):
    """Bypass real SSH/HTTP: ``fn(client, job_id)`` runs against the mock."""

    async def _stub(submission, fn):  # type: ignore[no-untyped-def]
        return await fn(client_mock, submission.job_id)

    return patch("src.api.routers.jobs._with_runner", side_effect=_stub)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_returns_submission_and_snapshot(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        runner = MagicMock()
        runner.get_status = AsyncMock(
            return_value={"job_id": "j-1", "state": "running", "sequence": 4},
        )

        with _patch_with_runner(runner):
            response = client.get(f"/api/v1/runs/{run_id}/job/status")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["submission"]["job_id"] == "j-1"
        assert body["snapshot"]["state"] == "running"
        assert body["snapshot"]["sequence"] == 4

    def test_picks_latest_attempt_by_default(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir, attempt_no=1)
        _make_submission_file(run_dir, attempt_no=3)

        runner = MagicMock()
        runner.get_status = AsyncMock(return_value={"state": "running"})

        with _patch_with_runner(runner):
            response = client.get(f"/api/v1/runs/{run_id}/job/status")

        assert response.status_code == 200
        # Latest attempt → submission for attempt_3.
        assert response.json()["submission"]["job_id"] == "j-3"

    def test_attempt_query_overrides_default(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir, attempt_no=1)
        _make_submission_file(run_dir, attempt_no=3)

        runner = MagicMock()
        runner.get_status = AsyncMock(return_value={"state": "running"})

        with _patch_with_runner(runner):
            response = client.get(f"/api/v1/runs/{run_id}/job/status?attempt=1")

        assert response.status_code == 200
        assert response.json()["submission"]["job_id"] == "j-1"

    def test_no_attempts_returns_404(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-empty"
        (settings.runs_dir / run_id).mkdir()

        response = client.get(f"/api/v1/runs/{run_id}/job/status")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "no_attempts"

    def test_unknown_attempt_returns_404(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir, attempt_no=1)

        response = client.get(f"/api/v1/runs/{run_id}/job/status?attempt=99")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "attempt_not_found"

    def test_missing_submission_returns_404(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        attempt_dir = settings.runs_dir / run_id / "attempts" / "attempt_1"
        attempt_dir.mkdir(parents=True)
        # No job_submission.json written.

        response = client.get(f"/api/v1/runs/{run_id}/job/status")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "job_submission_missing"

    def test_runner_failure_maps_to_502(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        runner = MagicMock()
        runner.get_status = AsyncMock(
            side_effect=ConnectionError("tunnel collapsed"),
        )

        with _patch_with_runner(runner):
            response = client.get(f"/api/v1/runs/{run_id}/job/status")

        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "runner_unreachable"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_returns_buffered_events_and_next_since(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        events: list[dict[str, Any]] = [
            {"offset": 5, "kind": "step", "payload": {"loss": 0.5}},
            {"offset": 6, "kind": "step", "payload": {"loss": 0.4}},
        ]

        async def _stream(_job_id: str, **_kwargs: Any):
            for event in events:
                yield event

        runner = MagicMock()
        runner.subscribe_events = _stream

        with _patch_with_runner(runner):
            response = client.get(f"/api/v1/runs/{run_id}/job/events?since=5")

        assert response.status_code == 200, response.text
        body = response.json()
        assert [e["offset"] for e in body["events"]] == [5, 6]
        # next_since is max(offset)+1 so the next poll skips them.
        assert body["next_since"] == 7

    def test_empty_stream_keeps_cursor(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        async def _empty(_job_id: str, **_kwargs: Any):
            return
            yield  # pragma: no cover - keep generator semantics

        runner = MagicMock()
        runner.subscribe_events = _empty

        with _patch_with_runner(runner):
            response = client.get(f"/api/v1/runs/{run_id}/job/events?since=10")

        assert response.status_code == 200
        body = response.json()
        assert body["events"] == []
        assert body["next_since"] == 10

    def test_runner_failure_returns_partial_with_error(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        """Best-effort: surface a structured error instead of failing the poll.

        The UI polls every 2 s; a transient blip should not break the
        cursor — the next poll picks up where this one left off."""
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        runner = MagicMock()
        runner.subscribe_events = MagicMock(side_effect=RuntimeError("ws collapsed"))

        with _patch_with_runner(runner):
            response = client.get(f"/api/v1/runs/{run_id}/job/events?since=4")

        assert response.status_code == 200
        body = response.json()
        assert body["events"] == []
        assert body["next_since"] == 4
        assert body["error"]["code"] == "stream_failed"


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_forwards_request_and_returns_202(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        runner = MagicMock()
        runner.request_stop = AsyncMock(
            return_value={"job_id": "j-1", "state": "stopping", "sequence": 7},
        )

        with _patch_with_runner(runner):
            response = client.post(f"/api/v1/runs/{run_id}/job/stop?grace=15")

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["state"] == "stopping"
        # ``grace`` is forwarded as ``grace_seconds`` to the JobClient.
        assert runner.request_stop.await_args.kwargs.get("grace_seconds") == 15.0

    def test_stop_without_grace_passes_none(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        runner = MagicMock()
        runner.request_stop = AsyncMock(
            return_value={"state": "stopping", "sequence": 1},
        )

        with _patch_with_runner(runner):
            response = client.post(f"/api/v1/runs/{run_id}/job/stop")

        assert response.status_code == 202
        assert runner.request_stop.await_args.kwargs.get("grace_seconds") is None

    def test_runner_failure_maps_to_502(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_id = "run-x"
        run_dir = settings.runs_dir / run_id
        _make_submission_file(run_dir)

        runner = MagicMock()
        runner.request_stop = AsyncMock(side_effect=ConnectionError("nope"))

        with _patch_with_runner(runner):
            response = client.post(f"/api/v1/runs/{run_id}/job/stop")

        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "runner_unreachable"
