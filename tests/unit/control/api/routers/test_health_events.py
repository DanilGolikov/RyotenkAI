"""Tests for ``GET /api/v1/health/events`` (Phase 8).

Seven categories per ``docs/testing/mutation_testing.md``:

1. TestPositive             — healthy state returns status=healthy.
2. TestNegative             — non-zero failure counter → status=degraded.
3. TestBoundary             — empty registry → status=no_active_runs.
4. TestInvariants           — per_run keyed by run_id.
5. TestDependencyErrors     — emitter missing for queried run_id →
                              empty per_run + no_active_runs.
6. TestRegressions          — run_id query filter scopes the response.
7. TestLogicSpecific        — aggregation across multiple runs OR's
                              the health indicators.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ryotenkai_control.api.routers.health import router as health_router
from ryotenkai_control.events import (
    ControlEventEmitter,
    EventDedup,
    EventEmitterRegistry,
    InMemoryBus,
    JournalWriter,
)
from ryotenkai_shared.api import EXCEPTION_HANDLERS

from tests.unit.control.events.conftest import make_completed, make_started


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """FastAPI test client with only the health router mounted.

    Tests don't need the runs_dir dependency — only the events
    endpoint touches the EventEmitterRegistry.
    """
    app = FastAPI(exception_handlers=EXCEPTION_HANDLERS)
    app.include_router(health_router, prefix="/api/v1")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Each test gets a clean :class:`EventEmitterRegistry` singleton."""
    EventEmitterRegistry.reset_instance()
    yield
    EventEmitterRegistry.reset_instance()


def _make_emitter(tmp_path: Path, run_id: str) -> ControlEventEmitter:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = JournalWriter(run_dir / "events.jsonl")
    bus = InMemoryBus(capacity=8)
    dedup = EventDedup()
    return ControlEventEmitter(
        run_id=run_id,
        source="control://orchestrator",
        journal=journal,
        bus=bus,
        dedup=dedup,
    )


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_healthy_when_registered_emitter_clean(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        emitter = _make_emitter(tmp_path, "run-a")
        try:
            registry = EventEmitterRegistry.instance()
            registry.register("run-a", emitter)
            emitter.emit(make_started())

            response = client.get("/api/v1/health/events")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "healthy"
            assert body["active_runs"] == ["run-a"]
            assert "run-a" in body["per_run"]
            assert body["per_run"]["run-a"]["emitter_events_emitted_total"] == 1
            assert body["health_indicators"]["any_emit_failures"] is False
            assert body["health_indicators"]["any_drops"] is False
        finally:
            registry = EventEmitterRegistry.instance()
            registry.deregister("run-a")
            emitter.close()


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_degraded_when_emit_failures_present(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        emitter = _make_emitter(tmp_path, "run-bad")
        try:
            EventEmitterRegistry.instance().register("run-bad", emitter)
            emitter._inc_emit_failed("journal_write")
            response = client.get("/api/v1/health/events")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "degraded"
            assert body["health_indicators"]["any_emit_failures"] is True
        finally:
            EventEmitterRegistry.instance().deregister("run-bad")
            emitter.close()

    def test_degraded_when_bus_overflows(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run-drop"
        run_dir.mkdir()
        journal = JournalWriter(run_dir / "events.jsonl")
        bus = InMemoryBus(capacity=1)
        dedup = EventDedup()
        emitter = ControlEventEmitter(
            run_id="run-drop",
            source="control://orchestrator",
            journal=journal,
            bus=bus,
            dedup=dedup,
        )
        try:
            EventEmitterRegistry.instance().register("run-drop", emitter)
            # Force the bus to drop.
            bus.publish(make_started(offset=0))
            bus.publish(make_completed(offset=1))

            response = client.get("/api/v1/health/events")
            body = response.json()
            assert body["status"] == "degraded"
            assert body["health_indicators"]["any_drops"] is True
        finally:
            EventEmitterRegistry.instance().deregister("run-drop")
            emitter.close()


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_empty_registry_returns_no_active_runs(
        self, client: TestClient,
    ) -> None:
        response = client.get("/api/v1/health/events")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "no_active_runs"
        assert body["active_runs"] == []
        assert body["per_run"] == {}
        # All indicators False on an empty registry.
        assert all(v is False for v in body["health_indicators"].values())

    def test_zero_counter_run_is_healthy(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        emitter = _make_emitter(tmp_path, "run-quiet")
        try:
            EventEmitterRegistry.instance().register("run-quiet", emitter)
            response = client.get("/api/v1/health/events")
            body = response.json()
            assert body["status"] == "healthy"
            assert body["per_run"]["run-quiet"]["emitter_events_emitted_total"] == 0
        finally:
            EventEmitterRegistry.instance().deregister("run-quiet")
            emitter.close()


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_per_run_keyed_by_run_id(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        emitter_a = _make_emitter(tmp_path, "run-a")
        emitter_b = _make_emitter(tmp_path, "run-b")
        try:
            registry = EventEmitterRegistry.instance()
            registry.register("run-a", emitter_a)
            registry.register("run-b", emitter_b)

            response = client.get("/api/v1/health/events")
            body = response.json()
            assert set(body["active_runs"]) == {"run-a", "run-b"}
            assert set(body["per_run"].keys()) == {"run-a", "run-b"}
        finally:
            registry = EventEmitterRegistry.instance()
            registry.deregister("run-a")
            registry.deregister("run-b")
            emitter_a.close()
            emitter_b.close()

    def test_response_shape_is_stable(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        emitter = _make_emitter(tmp_path, "run-shape")
        try:
            EventEmitterRegistry.instance().register("run-shape", emitter)
            body = client.get("/api/v1/health/events").json()
            # Top-level keys are required for clients (CLI / Web UI).
            assert {"status", "active_runs", "per_run", "health_indicators"} <= set(body)
            indicators = body["health_indicators"]
            assert {
                "any_emit_failures",
                "any_drops",
                "any_fsync_failures",
                "any_write_failures",
                "any_offset_collisions",
            } <= set(indicators)
        finally:
            EventEmitterRegistry.instance().deregister("run-shape")
            emitter.close()


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_missing_run_id_filter_returns_no_active_runs(
        self, client: TestClient,
    ) -> None:
        # Empty registry, explicit run_id filter — endpoint must not 500.
        response = client.get("/api/v1/health/events?run_id=does-not-exist")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "no_active_runs"
        assert body["active_runs"] == []
        assert body["per_run"] == {}

    def test_emitter_without_counters_still_renders(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        # An emitter whose collaborators were torn down before the
        # request hits should still snapshot (defensive getattr in
        # collect_metrics).
        emitter = _make_emitter(tmp_path, "run-torn")
        try:
            EventEmitterRegistry.instance().register("run-torn", emitter)
            emitter.close()  # close BEFORE the request.
            response = client.get("/api/v1/health/events")
            assert response.status_code == 200
            body = response.json()
            # Still keyed; counters are at zero.
            assert "run-torn" in body["per_run"]
            assert body["per_run"]["run-torn"]["emitter_events_emitted_total"] == 0
        finally:
            EventEmitterRegistry.instance().deregister("run-torn")


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_run_id_filter_returns_only_that_run(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        emitter_a = _make_emitter(tmp_path, "run-a")
        emitter_b = _make_emitter(tmp_path, "run-b")
        try:
            registry = EventEmitterRegistry.instance()
            registry.register("run-a", emitter_a)
            registry.register("run-b", emitter_b)

            response = client.get("/api/v1/health/events?run_id=run-a")
            body = response.json()
            assert body["active_runs"] == ["run-a"]
            assert list(body["per_run"].keys()) == ["run-a"]
        finally:
            registry = EventEmitterRegistry.instance()
            registry.deregister("run-a")
            registry.deregister("run-b")
            emitter_a.close()
            emitter_b.close()

    def test_status_field_lowercase_alpha(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        # Frontend grep'd on lowercase status strings — regression guard.
        emitter = _make_emitter(tmp_path, "run-case")
        try:
            EventEmitterRegistry.instance().register("run-case", emitter)
            body = client.get("/api/v1/health/events").json()
            assert body["status"] in {"healthy", "degraded", "no_active_runs"}
            assert body["status"].islower()
        finally:
            EventEmitterRegistry.instance().deregister("run-case")
            emitter.close()


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_aggregate_across_multiple_runs(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        # run-clean is healthy, run-bad has an emit failure → aggregate
        # status flips to degraded.
        emitter_clean = _make_emitter(tmp_path, "run-clean")
        emitter_bad = _make_emitter(tmp_path, "run-bad")
        try:
            registry = EventEmitterRegistry.instance()
            registry.register("run-clean", emitter_clean)
            registry.register("run-bad", emitter_bad)
            emitter_bad._inc_emit_failed("journal_write")

            response = client.get("/api/v1/health/events")
            body = response.json()
            assert body["status"] == "degraded"
            # Both runs surface in per_run.
            assert set(body["per_run"].keys()) == {"run-clean", "run-bad"}
            # Indicator turned on by run-bad alone.
            assert body["health_indicators"]["any_emit_failures"] is True
        finally:
            registry = EventEmitterRegistry.instance()
            registry.deregister("run-clean")
            registry.deregister("run-bad")
            emitter_clean.close()
            emitter_bad.close()

    def test_run_id_filter_isolates_aggregate(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        # When filtering to the clean run, the degraded run does not
        # contaminate the snapshot.
        emitter_clean = _make_emitter(tmp_path, "run-clean")
        emitter_bad = _make_emitter(tmp_path, "run-bad")
        try:
            registry = EventEmitterRegistry.instance()
            registry.register("run-clean", emitter_clean)
            registry.register("run-bad", emitter_bad)
            emitter_bad._inc_emit_failed("journal_write")

            body = client.get("/api/v1/health/events?run_id=run-clean").json()
            assert body["status"] == "healthy"
            assert body["health_indicators"]["any_emit_failures"] is False
        finally:
            registry = EventEmitterRegistry.instance()
            registry.deregister("run-clean")
            registry.deregister("run-bad")
            emitter_clean.close()
            emitter_bad.close()
