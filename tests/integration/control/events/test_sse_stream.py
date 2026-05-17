"""SSE stream integration — closes R-19 (Phase 9).

Integration flow:

1. Start the FastAPI events router in a :class:`TestClient` (HTTP)
   AND directly invoke the SSE async generator (for stream
   semantics that TestClient can't drive).
2. Emit 50 events into the run's journal + bus.
3. Read historical events via SSE with ``after_offset=10`` — expect
   exactly 40 events monotonic.
4. Concurrent producer emits 10 more live events — the SSE
   subscriber receives them too (closes R-19 subscribe-first race).
5. Disconnect mid-stream then reconnect with
   ``Last-Event-ID: <last>`` — assert no duplicates, no missing
   events.

The "concurrent producer" and "disconnect/reconnect" scenarios use the
internal async generator directly because Starlette's
:class:`TestClient` cannot drive SSE in-process without thread juggling.
The HTTP replay endpoint IS exercised through the TestClient because
that's a regular request/response flow.
"""

from __future__ import annotations

import asyncio
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
)
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.control_run import (
    RunStartedEvent,
    RunStartedPayload,
)
from ryotenkai_shared.events.types.control_stage import (
    StageCompletedEvent,
    StageCompletedPayload,
    StageStartedEvent,
    StageStartedPayload,
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


def _make_run_started(run_id: str, source: str, offset: int) -> RunStartedEvent:
    return RunStartedEvent(
        source=source,
        run_id=run_id,
        offset=offset,
        payload=RunStartedPayload(
            run_name=f"run-{offset}",
            algorithm="sft",
            model_id="acme/test",
            dataset_id="default",
            config_hash="abc",
        ),
    )


def _make_stage_started(run_id: str, source: str, offset: int) -> StageStartedEvent:
    return StageStartedEvent(
        source=source,
        run_id=run_id,
        offset=offset,
        payload=StageStartedPayload(
            stage_name=f"stage-{offset}",
            stage_index=offset,
            total_stages=100,
            inputs_summary={},
        ),
    )


def _make_stage_completed(run_id: str, source: str, offset: int) -> StageCompletedEvent:
    return StageCompletedEvent(
        source=source,
        run_id=run_id,
        offset=offset,
        payload=StageCompletedPayload(
            stage_name=f"stage-{offset}",
            duration_s=0.5,
            outputs_summary={},
        ),
    )


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _extract_sse_offsets(frame: str) -> int | None:
    """Parse an SSE frame and return the id (offset)."""
    for line in frame.splitlines():
        if line.startswith("id: "):
            return int(line.removeprefix("id: "))
    return None


class _FakeRequest:
    """Mimics the FastAPI Request surface used by the SSE generator."""

    def __init__(self, max_calls: int = 1000) -> None:
        self._calls = 0
        self._max_calls = max_calls

    async def is_disconnected(self) -> bool:
        self._calls += 1
        return self._calls > self._max_calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSSEHistoricalReplay:
    """SSE catchup phase yields the historical window from the journal."""

    def test_http_replay_returns_filtered_window(
        self, client: TestClient, settings: ApiSettings,
    ) -> None:
        """50 events (offsets 0..49), ``after_offset=10`` → offsets 11..49 (39 events)."""
        run_id = "run-replay"
        run_dir = settings.runs_dir / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        EventEmitterRegistry.instance().register(run_id, emitter)
        try:
            for i in range(50):
                emitter.emit(_make_stage_started(run_id, emitter.source, UNKNOWN_OFFSET))
        finally:
            emitter.close()

        # HTTP replay (NDJSON, not SSE — the SSE path requires async).
        response = client.get(
            f"/api/v1/runs/{run_id}/events?after_offset=10",
        )
        assert response.status_code == 200
        rows = _parse_ndjson(response.text)
        # 50 events with offsets 0..49; after_offset=10 (exclusive) yields 11..49.
        assert len(rows) == 39
        offsets = [r["offset"] for r in rows]
        assert offsets == list(range(11, 50))
        # Monotonic guarantee.
        assert offsets == sorted(set(offsets))


class TestSSESubscribeFirst:
    """Live events published between catchup snapshot and live drain
    are still delivered (R-19)."""

    @pytest.mark.asyncio
    async def test_concurrent_producer_event_delivered(
        self, tmp_path: Path,
    ) -> None:
        run_id = "run-subscribe-first"
        run_dir = tmp_path / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        try:
            # Historical batch.
            for i in range(5):
                emitter.emit(_make_stage_started(
                    run_id, emitter.source, UNKNOWN_OFFSET,
                ))

            request = _FakeRequest(max_calls=20)
            generator = events_router._sse_event_stream(
                request=request,  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=-1,
                predicate=lambda _e: True,
            )

            # Drain the 5 historical events.
            collected_offsets: list[int] = []
            for _ in range(5):
                frame = await generator.__anext__()
                off = _extract_sse_offsets(frame)
                assert off is not None
                collected_offsets.append(off)

            # Inject one live event AFTER the catchup snapshot.
            emitter.emit(_make_stage_completed(
                run_id, emitter.source, UNKNOWN_OFFSET,
            ))
            frame = await generator.__anext__()
            live_off = _extract_sse_offsets(frame)
            assert live_off is not None
            collected_offsets.append(live_off)

            await generator.aclose()
        finally:
            emitter.close()

        # 5 historical + 1 live, strictly monotonic.
        assert collected_offsets == sorted(set(collected_offsets))
        assert len(collected_offsets) == 6

    @pytest.mark.asyncio
    async def test_multiple_live_events_after_subscribe(
        self, tmp_path: Path,
    ) -> None:
        """A burst of live events after the snapshot all arrive in order."""
        run_id = "run-burst"
        run_dir = tmp_path / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        try:
            for i in range(10):
                emitter.emit(_make_stage_started(
                    run_id, emitter.source, UNKNOWN_OFFSET,
                ))

            request = _FakeRequest(max_calls=30)
            generator = events_router._sse_event_stream(
                request=request,  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=-1,
                predicate=lambda _e: True,
            )

            # Drain historical first.
            for _ in range(10):
                await generator.__anext__()

            # Emit 5 live events.
            for _ in range(5):
                emitter.emit(_make_stage_completed(
                    run_id, emitter.source, UNKNOWN_OFFSET,
                ))

            live_offsets: list[int] = []
            for _ in range(5):
                frame = await generator.__anext__()
                off = _extract_sse_offsets(frame)
                assert off is not None
                live_offsets.append(off)
            await generator.aclose()
        finally:
            emitter.close()

        assert live_offsets == sorted(set(live_offsets))
        assert len(live_offsets) == 5


class TestSSEReconnect:
    """Reconnect with ``Last-Event-ID`` resumes from the right offset."""

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect_no_duplicates(
        self, tmp_path: Path,
    ) -> None:
        """Disconnect after offset N, reconnect with Last-Event-ID=N
        — no duplicates and the historical window picks up at N+1."""
        run_id = "run-reconnect"
        run_dir = tmp_path / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        try:
            for _ in range(20):
                emitter.emit(_make_stage_started(
                    run_id, emitter.source, UNKNOWN_OFFSET,
                ))

            # Phase A — subscribe from start, drain first 10 events.
            request_a = _FakeRequest(max_calls=15)
            gen_a = events_router._sse_event_stream(
                request=request_a,  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=-1,
                predicate=lambda _e: True,
            )
            phase_a_offsets: list[int] = []
            for _ in range(10):
                frame = await gen_a.__anext__()
                off = _extract_sse_offsets(frame)
                assert off is not None
                phase_a_offsets.append(off)
            await gen_a.aclose()

            assert phase_a_offsets == list(range(10))
            last_received = phase_a_offsets[-1]

            # Phase B — reconnect with Last-Event-ID = 9 (offset of last
            # received). The generator should resume strictly after.
            request_b = _FakeRequest(max_calls=20)
            gen_b = events_router._sse_event_stream(
                request=request_b,  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=last_received,
                predicate=lambda _e: True,
            )
            phase_b_offsets: list[int] = []
            for _ in range(10):
                frame = await gen_b.__anext__()
                off = _extract_sse_offsets(frame)
                assert off is not None
                phase_b_offsets.append(off)
            await gen_b.aclose()

            # Phase B starts at offset 10, ends at 19.
            assert phase_b_offsets == list(range(10, 20))
            # No overlap with phase A.
            assert set(phase_a_offsets).isdisjoint(set(phase_b_offsets))
        finally:
            emitter.close()


class TestSSEFiltering:
    """Filters apply uniformly across catchup and live phases."""

    @pytest.mark.asyncio
    async def test_severity_filter_applies_to_catchup(
        self, tmp_path: Path,
    ) -> None:
        """``severity=error`` skips info events during catchup."""
        run_id = "run-filter"
        run_dir = tmp_path / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        try:
            # 5 info events (StageStarted has severity=info).
            for _ in range(5):
                emitter.emit(_make_stage_started(
                    run_id, emitter.source, UNKNOWN_OFFSET,
                ))
            # Wait for the catchup to capture the snapshot before adding
            # more — we test filtering, not subscribe-first.

            request = _FakeRequest(max_calls=15)
            # Custom predicate — drop everything (severity=error would
            # match zero of our 5 info events).
            generator = events_router._sse_event_stream(
                request=request,  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=-1,
                predicate=lambda e: e.severity == "error",
            )

            # The generator should hit live-tail without yielding any
            # historical frame. Pull once with a short timeout — we
            # expect to time out via the keepalive path.
            try:
                frame = await asyncio.wait_for(generator.__anext__(), timeout=0.5)
                # If we got a frame it must be a keepalive comment.
                assert frame.startswith(":"), f"unexpected frame: {frame!r}"
            except asyncio.TimeoutError:
                # Expected — no events match the filter, no live event yet.
                pass
            finally:
                await generator.aclose()
        finally:
            emitter.close()


class TestSSEKeepalive:
    """The stream emits ``: keepalive`` when the bus is quiet."""

    @pytest.mark.asyncio
    async def test_keepalive_emitted_when_idle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force a short keepalive interval and observe the comment frame."""
        # Shorten the keepalive so we don't wait 15s.
        monkeypatch.setattr(events_router, "KEEPALIVE_INTERVAL_S", 0.1)

        run_id = "run-keepalive"
        run_dir = tmp_path / run_id
        emitter = ControlEventEmitter.for_run(
            run_id=run_id, run_directory=run_dir,
        )
        try:
            # No historical events — stream goes straight to live-tail
            # and the keepalive should fire on the first iteration.
            request = _FakeRequest(max_calls=5)
            generator = events_router._sse_event_stream(
                request=request,  # type: ignore[arg-type]
                journal_path=run_dir / "events.jsonl",
                bus=emitter.bus,
                start_offset=-1,
                predicate=lambda _e: True,
            )
            try:
                frame = await asyncio.wait_for(generator.__anext__(), timeout=1.0)
            finally:
                await generator.aclose()
        finally:
            emitter.close()

        # Keepalive frames start with ``:`` (SSE comment).
        assert frame.startswith(":"), f"expected keepalive frame, got: {frame!r}"
