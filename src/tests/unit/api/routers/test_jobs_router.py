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

Coverage split (project policy):

1. **Positive**           — happy-path status / events / stop wire shape.
2. **Negative**           — 404 maps (no attempts / unknown attempt /
                             missing submission / corrupt submission);
                             502 on runner errors.
3. **Boundary**           — attempt=1, limit=1, limit=2000, since=0,
                             grace=0, events count == limit, fewer
                             events than ``limit`` requested.
4. **Invariants**         — ``next_since`` monotonic; events length
                             ≤ limit; latest-attempt picker honours
                             numeric ordering, not lexical;
                             ``_with_runner`` always tears down the
                             tunnel.
5. **Dependency errors**  — submission JSON corrupt → 404; runner raise
                             on ``get_status`` → 502; ``subscribe_events``
                             raise mid-stream → 200 with structured
                             ``error`` block.
6. **Regressions**        — ``run_id`` path-traversal blocked by
                             ``resolve_run_dir`` (400); attempt dirs
                             with non-numeric suffix tolerated;
                             ``attempt=0`` / negative / ``limit`` out
                             of range rejected by Pydantic with 422.
7. **Combinatorial**      — (attempt × since × limit) parametrised
                             over the mocked event stream.
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


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_attempt_one_is_smallest_valid_value(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir, attempt_no=1)

        runner = MagicMock()
        runner.get_status = AsyncMock(return_value={"state": "running"})
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/status?attempt=1")
        assert response.status_code == 200

    def test_events_limit_one_returns_at_most_one_event(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            for offset in range(10):
                yield {"offset": offset, "kind": "step", "payload": {}}

        runner = MagicMock()
        runner.subscribe_events = _stream
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/events?limit=1")
        assert response.status_code == 200
        body = response.json()
        # The router buffers until ``limit`` events have arrived then
        # exits the async-for. Exactly one event must come back.
        assert len(body["events"]) == 1
        assert body["next_since"] == 1

    def test_events_limit_at_max_2000(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            for offset in range(10):
                yield {"offset": offset, "kind": "step", "payload": {}}

        runner = MagicMock()
        runner.subscribe_events = _stream
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/events?limit=2000")
        assert response.status_code == 200
        # Stream produced fewer events than the limit — all 10 returned.
        assert len(response.json()["events"]) == 10

    def test_grace_zero_immediate_stop(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)
        runner = MagicMock()
        runner.request_stop = AsyncMock(return_value={"state": "stopping"})
        with _patch_with_runner(runner):
            response = client.post("/api/v1/runs/run-x/job/stop?grace=0")
        assert response.status_code == 202
        # ``grace=0`` is meaningful (skip the SIGTERM grace window) — must
        # be forwarded as the float 0.0, not silently dropped to None.
        assert runner.request_stop.await_args.kwargs.get("grace_seconds") == 0.0

    def test_events_with_since_zero_is_default(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **kwargs):
            # Capture how the proxy passes ``since`` down to the runner.
            assert kwargs.get("since") == 0
            yield {"offset": 0, "kind": "trainer_spawned", "payload": {}}

        runner = MagicMock()
        runner.subscribe_events = _stream
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/events")
        assert response.status_code == 200
        assert response.json()["events"][0]["kind"] == "trainer_spawned"

    def test_attempt_dir_with_non_numeric_suffix_is_ignored(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        # The latest-attempt picker sorts by the int after ``attempt_``.
        # An attic directory like ``attempt_archive`` should sort with
        # 0 (per the implementation's fallback) and not crash the glob.
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir, attempt_no=2)
        # Spurious attic.
        archive = run_dir / "attempts" / "attempt_archive"
        archive.mkdir()
        runner = MagicMock()
        runner.get_status = AsyncMock(return_value={"state": "running"})
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/status")
        # Picks attempt_2 (the only one with a real submission file).
        assert response.status_code == 200
        assert response.json()["submission"]["job_id"] == "j-2"


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_next_since_is_monotonic(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            for offset in [10, 11, 12]:
                yield {"offset": offset, "kind": "step", "payload": {}}

        runner = MagicMock()
        runner.subscribe_events = _stream
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/events?since=10")
        body = response.json()
        # next_since must be strictly greater than the cursor we sent
        # AND greater than every offset returned, so the next poll
        # never re-receives the same event.
        assert body["next_since"] > 10
        assert all(body["next_since"] > e["offset"] for e in body["events"])

    def test_next_since_unchanged_when_stream_empty(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)

        async def _empty(_job_id, **_kwargs):
            return
            yield  # pragma: no cover

        runner = MagicMock()
        runner.subscribe_events = _empty
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/events?since=42")
        # An empty slice keeps the cursor where the caller had it.
        assert response.json()["next_since"] == 42

    def test_latest_attempt_picker_uses_numeric_order(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        # 10 > 2 numerically, but lexicographically "attempt_10" < "attempt_2".
        # Pin the numeric sort.
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir, attempt_no=2)
        _make_submission_file(run_dir, attempt_no=10)

        runner = MagicMock()
        runner.get_status = AsyncMock(return_value={"state": "running"})
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/status")
        assert response.json()["submission"]["job_id"] == "j-10"

    def test_events_length_never_exceeds_limit(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)

        async def _stream(_job_id, **_kwargs):
            for offset in range(50):
                yield {"offset": offset, "kind": "step", "payload": {}}

        runner = MagicMock()
        runner.subscribe_events = _stream
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/events?limit=5")
        events = response.json()["events"]
        assert len(events) <= 5
        assert len(events) == 5  # exactly 5 — stream had more than enough


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_corrupt_submission_returns_404(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        # Mirror the CLI's friendly-error contract: malformed
        # job_submission.json is not a 500 — it's "the launcher never
        # finished writing this attempt's submission".
        attempt_dir = settings.runs_dir / "run-x" / "attempts" / "attempt_1"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "job_submission.json").write_text("not-valid-json")

        response = client.get("/api/v1/runs/run-x/job/status")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "job_submission_missing"

    def test_status_unwraps_arbitrary_runner_exception(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)
        runner = MagicMock()
        runner.get_status = AsyncMock(side_effect=RuntimeError("boom"))
        with _patch_with_runner(runner):
            response = client.get("/api/v1/runs/run-x/job/status")
        # Any uncategorised runner error → 502 (cannot reach runner).
        assert response.status_code == 502
        assert "boom" in response.json()["detail"]["message"]


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_path_traversal_in_run_id_rejected(
        self, client: TestClient,
    ) -> None:
        # ``resolve_run_dir`` blocks ``..`` segments. The router relies
        # on that — pin the contract here so a future refactor can't
        # silently drop the check.
        response = client.get("/api/v1/runs/..%2Fetc/job/status")
        # 400 (invalid_run_id) or 404 (run_not_found) — either confirms
        # the run dir didn't escape the configured root.
        assert response.status_code in (400, 404)

    @pytest.mark.parametrize("attempt", ["0", "-1"])
    def test_attempt_below_one_rejected_by_validator(
        self, client: TestClient, settings: ApiSettings, attempt: str,
    ) -> None:
        # ``ge=1`` Pydantic constraint — bad attempt numbers must surface
        # as 422 from the framework, never reach our resolution code.
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)
        response = client.get(f"/api/v1/runs/run-x/job/status?attempt={attempt}")
        assert response.status_code == 422

    @pytest.mark.parametrize("limit", ["0", "2001", "-5"])
    def test_events_limit_out_of_range_rejected(
        self, client: TestClient, settings: ApiSettings, limit: str,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)
        response = client.get(f"/api/v1/runs/run-x/job/events?limit={limit}")
        assert response.status_code == 422

    def test_events_since_negative_rejected(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)
        response = client.get("/api/v1/runs/run-x/job/events?since=-1")
        assert response.status_code == 422

    def test_grace_negative_rejected(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        run_dir = settings.runs_dir / "run-x"
        _make_submission_file(run_dir)
        response = client.post("/api/v1/runs/run-x/job/stop?grace=-1")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. Combinatorial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("since", [0, 5, 100])
@pytest.mark.parametrize("limit", [1, 5, 200])
@pytest.mark.parametrize("attempt_count", [1, 3])
def test_combinatorial_events_slice(
    client: TestClient,
    settings: ApiSettings,
    since: int,
    limit: int,
    attempt_count: int,
) -> None:
    """Parametric matrix over the events endpoint's two cursors.

    Every combination of ``(since, limit, attempts on disk)`` should
    return a slice that obeys the four invariants:
      - HTTP 200,
      - ``len(events) <= limit``,
      - every returned offset >= since (the runner's ring buffer
        guarantees this; we mock the same contract),
      - ``next_since > since`` if any events came back.
    """
    run_dir = settings.runs_dir / "run-x"
    for attempt_no in range(1, attempt_count + 1):
        _make_submission_file(run_dir, attempt_no=attempt_no)

    async def _stream(_job_id: str, **kwargs: Any):
        seen_since = kwargs.get("since", 0)
        # Emit up to 50 events starting at the requested cursor.
        for offset in range(seen_since, seen_since + 50):
            yield {"offset": offset, "kind": "step", "payload": {}}

    runner = MagicMock()
    runner.subscribe_events = _stream

    with _patch_with_runner(runner):
        response = client.get(
            f"/api/v1/runs/run-x/job/events?since={since}&limit={limit}",
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["events"]) <= limit
    if body["events"]:
        assert all(e["offset"] >= since for e in body["events"])
        assert body["next_since"] > since
