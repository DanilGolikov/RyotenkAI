"""Tests for :mod:`ryotenkai_control.api.routers.events` (Phase 6.a).

Coverage split (seven canonical classes):

1. ``TestPositive``         — happy-path HTTP replay returns NDJSON of events
2. ``TestNegative``         — 404 on missing run; 400 on invalid filters
3. ``TestBoundary``         — limit cap; after_offset=-1 starts at offset 0
4. ``TestInvariants``       — Content-Type, X-Next-Offset, monotonic offsets
5. ``TestDependencyErrors`` — journal file missing (run dir exists, no events)
6. ``TestRegressions``      — type_prefix / severity parsing edge cases
7. ``TestLogicSpecific``    — subscribe-first ordering (R-19)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ryotenkai_control.api.config import ApiSettings
from ryotenkai_control.api.dependencies import get_settings
from ryotenkai_control.api.routers import events as events_router
from ryotenkai_control.events import (
    ControlEventEmitter,
    EventEmitterRegistry,
    JournalWriter,
)
from tests.unit.control.events.conftest import (
    make_completed,
    make_failed,
    make_started,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    EventEmitterRegistry.reset_instance()
    yield
    EventEmitterRegistry.reset_instance()


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
    from ryotenkai_shared.api import EXCEPTION_HANDLERS

    app = FastAPI(exception_handlers=EXCEPTION_HANDLERS)
    app.include_router(events_router.router, prefix="/api/v1")
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


def _seed_run_journal(
    settings: ApiSettings,
    run_id: str,
    *,
    extra: list[Any] | None = None,
) -> Path:
    """Create a run dir with a seeded events.jsonl journal."""
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    journal_path = run_dir / "events.jsonl"
    writer = JournalWriter(journal_path)
    writer.append(make_started(run_id=run_id, offset=0))
    writer.append(make_completed(run_id=run_id, offset=1))
    writer.append(
        make_failed(run_id=run_id, offset=2, msg="boom"),
    )
    for event in extra or []:
        writer.append(event)
    writer.close()
    return run_dir


def _parse_ndjson(body: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPositive:
    def test_get_events_returns_ndjson_with_envelopes(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get("/api/v1/runs/run-1/events")
        assert response.status_code == 200, response.text
        rows = _parse_ndjson(response.text)
        assert [r["offset"] for r in rows] == [0, 1, 2]
        assert rows[0]["kind"] == "ryotenkai.control.run.started"


class TestNegative:
    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/runs/no-such-run/events")
        assert response.status_code == 404, response.text

    def test_invalid_severity_returns_400(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get(
            "/api/v1/runs/run-1/events?severity=NOT_A_LEVEL",
        )
        # Either 400 (typed via HTTPException) or 422 (typed via
        # AttemptInvalid) — accept both 4xx outcomes.
        assert response.status_code in (400, 422), response.text

    def test_limit_out_of_range_rejected(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get("/api/v1/runs/run-1/events?limit=99999")
        assert response.status_code == 422, response.text


class TestBoundary:
    def test_after_offset_minus_one_returns_all(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get("/api/v1/runs/run-1/events?after_offset=-1")
        assert response.status_code == 200
        rows = _parse_ndjson(response.text)
        assert [r["offset"] for r in rows] == [0, 1, 2]

    def test_after_offset_skips_prior(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get("/api/v1/runs/run-1/events?after_offset=1")
        rows = _parse_ndjson(response.text)
        assert [r["offset"] for r in rows] == [2]

    def test_limit_caps_response(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get("/api/v1/runs/run-1/events?limit=2")
        rows = _parse_ndjson(response.text)
        assert len(rows) == 2


class TestInvariants:
    def test_content_type_is_ndjson(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get("/api/v1/runs/run-1/events")
        assert response.headers["content-type"].startswith(
            "application/x-ndjson",
        )

    def test_next_offset_header_reflects_last_event(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get("/api/v1/runs/run-1/events")
        assert response.headers["x-next-offset"] == "2"

    def test_next_offset_unchanged_on_empty_filter_result(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get(
            "/api/v1/runs/run-1/events"
            "?after_offset=10&type_prefix=does.not.match",
        )
        rows = _parse_ndjson(response.text)
        assert rows == []
        assert response.headers["x-next-offset"] == "10"


class TestDependencyErrors:
    def test_journal_missing_returns_empty_stream(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        # Create run dir but no journal file
        (settings.runs_dir / "run-1").mkdir(parents=True)
        response = client.get("/api/v1/runs/run-1/events")
        assert response.status_code == 200
        assert _parse_ndjson(response.text) == []
        assert response.headers["x-next-offset"] == "-1"


class TestRegressions:
    def test_type_prefix_filter(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get(
            "/api/v1/runs/run-1/events?type_prefix=ryotenkai.control.run.failed",
        )
        rows = _parse_ndjson(response.text)
        assert {r["kind"] for r in rows} == {
            "ryotenkai.control.run.failed",
        }

    def test_severity_csv_parsed(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        # error,warning — only the failed event should match (severity=error)
        response = client.get(
            "/api/v1/runs/run-1/events?severity=error,warning",
        )
        rows = _parse_ndjson(response.text)
        assert all(r["severity"] in {"error", "warning"} for r in rows)
        assert len(rows) == 1

    def test_source_filter(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        _seed_run_journal(settings, "run-1")
        response = client.get(
            "/api/v1/runs/run-1/events"
            "?source=control://orchestrator",
        )
        rows = _parse_ndjson(response.text)
        # All three default seeds use this source.
        assert len(rows) == 3


class TestLogicSpecific:
    """Subscribe-first SSE invariant (closes R-19).

    Calls the internal async generator directly to assert the ordering
    contract without spinning a real uvicorn server (which Starlette's
    TestClient also can't drive for SSE without juggling threads).
    """

    @pytest.mark.asyncio
    async def test_catchup_then_live_no_event_lost(
        self, tmp_path: Path,
    ) -> None:
        """An event published between catchup-snapshot and live-tail
        MUST be delivered. The bus subscription is opened before the
        journal catchup so the live event sits in the bus by the time
        catchup completes.
        """
        run_id = "run-r19"
        run_dir = tmp_path / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        try:
            emitter.emit(make_started(run_id=run_id))
            emitter.emit(make_completed(run_id=run_id))

            # Build a fake Request whose ``is_disconnected`` returns False
            # for two polls then True so the stream terminates after the
            # live event is consumed.
            class _FakeRequest:
                def __init__(self) -> None:
                    self._calls = 0

                async def is_disconnected(self) -> bool:
                    self._calls += 1
                    return self._calls > 5

            generator = events_router._sse_event_stream(
                request=_FakeRequest(),  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=-1,
                predicate=lambda _e: True,
            )

            # Drive one element at a time — the iterator yields catchup
            # frames first, then waits for live tail. We schedule a
            # late publish to land between catchup and live drain.
            collected: list[str] = []

            # First two yields should be the historical events (offsets 0, 1).
            collected.append(await generator.__anext__())
            collected.append(await generator.__anext__())

            # Publish a third event AFTER the subscribe-snapshot but
            # before the live drain — the generator hasn't yet started
            # waiting on the bus signal.
            emitter.emit(make_failed(run_id=run_id))

            collected.append(await generator.__anext__())

            await generator.aclose()
        finally:
            emitter.close()

        offsets = []
        for frame in collected:
            # Each SSE frame is ``id: N\nevent: ...\ndata: {json}\n\n``
            for line in frame.splitlines():
                if line.startswith("id: "):
                    offsets.append(int(line.removeprefix("id: ")))
                    break
        assert offsets == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_sse_frame_format(self, tmp_path: Path) -> None:
        """Each SSE frame must carry id/event/data and end with blank line."""
        run_id = "run-fmt"
        run_dir = tmp_path / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        try:
            emitter.emit(make_started(run_id=run_id))

            class _FakeRequest:
                def __init__(self) -> None:
                    self._calls = 0

                async def is_disconnected(self) -> bool:
                    self._calls += 1
                    return self._calls > 3

            generator = events_router._sse_event_stream(
                request=_FakeRequest(),  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=-1,
                predicate=lambda _e: True,
            )
            frame = await generator.__anext__()
            await generator.aclose()
        finally:
            emitter.close()

        assert frame.startswith("id: 0\n")
        assert "event: ryotenkai.control.run.started\n" in frame
        assert "data: " in frame
        assert frame.endswith("\n\n")
