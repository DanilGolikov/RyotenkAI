"""Tests for ``GET /api/v1/resources`` (Phase 2 PR-2.2)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RYOTENKAI_RUNTIME_PROVIDER", "single_node")

from ryotenkai_pod.runner.main import create_app  # noqa: E402


class _StubSupervisor:
    is_running = False

    async def shutdown(self) -> None:
        pass


def _factory(fsm, bus, *, terminal_hook=None, stdio_log_path=None):  # type: ignore[no-untyped-def]
    return _StubSupervisor()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(supervisor_factory=_factory))


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_returns_full_snapshot(self, client: TestClient) -> None:
        snap = {
            "gpu_util_percent": 78.0,
            "gpu_memory_percent": 50.0,
            "cpu_percent": 12.5,
            "ram_used_gb": 8.0,
            "ram_total_gb": 32.0,
        }
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(return_value=snap),
        ):
            r = client.get("/api/v1/resources")
        assert r.status_code == 200
        assert r.json() == snap

    def test_all_none_snapshot_returns_200(self, client: TestClient) -> None:
        # Field semantics: None ⇒ tool unavailable. Status code stays 200.
        snap = {
            "gpu_util_percent": None,
            "gpu_memory_percent": None,
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
        }
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(return_value=snap),
        ):
            r = client.get("/api/v1/resources")
        assert r.status_code == 200
        assert r.json() == snap


# ---------------------------------------------------------------------------
# Negative — provider raises
# ---------------------------------------------------------------------------


class TestNegative:
    def test_provider_exception_returns_502_problem_json(
        self, client: TestClient,
    ) -> None:
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(side_effect=RuntimeError("nvidia-smi crashed")),
        ):
            r = client.get("/api/v1/resources")
        assert r.status_code == 502
        assert r.headers["content-type"].startswith("application/problem+json")
        body = r.json()
        assert body["code"] == "RESOURCES_UNAVAILABLE"
        assert "nvidia-smi crashed" in (body.get("detail") or "")


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_zero_values_accepted(self, client: TestClient) -> None:
        snap = {
            "gpu_util_percent": 0.0,
            "gpu_memory_percent": 0.0,
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
        }
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(return_value=snap),
        ):
            r = client.get("/api/v1/resources")
        assert r.status_code == 200

    def test_full_utilization_accepted(self, client: TestClient) -> None:
        snap = {
            "gpu_util_percent": 100.0,
            "gpu_memory_percent": 100.0,
            "cpu_percent": 100.0,
            "ram_used_gb": 32.0,
            "ram_total_gb": 32.0,
        }
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(return_value=snap),
        ):
            r = client.get("/api/v1/resources")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Invariant — content-type
# ---------------------------------------------------------------------------


class TestInvariant:
    def test_success_returns_application_json(self, client: TestClient) -> None:
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(return_value={
                "gpu_util_percent": None, "gpu_memory_percent": None,
                "cpu_percent": None, "ram_used_gb": None,
                "ram_total_gb": None,
            }),
        ):
            r = client.get("/api/v1/resources")
        assert r.headers["content-type"].startswith("application/json")

    def test_failure_returns_problem_json(self, client: TestClient) -> None:
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(side_effect=OSError("x")),
        ):
            r = client.get("/api/v1/resources")
        assert r.headers["content-type"].startswith("application/problem+json")


# ---------------------------------------------------------------------------
# Combinatorial: tool availability matrix
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("gpu_present", [True, False])
    @pytest.mark.parametrize("psutil_present", [True, False])
    def test_partial_availability(
        self,
        client: TestClient,
        gpu_present: bool,
        psutil_present: bool,
    ) -> None:
        snap = {
            "gpu_util_percent": 50.0 if gpu_present else None,
            "gpu_memory_percent": 30.0 if gpu_present else None,
            "cpu_percent": 10.0 if psutil_present else None,
            "ram_used_gb": 8.0 if psutil_present else None,
            "ram_total_gb": 16.0 if psutil_present else None,
        }
        with patch(
            "ryotenkai_pod.runner.api.resources.default_health_snapshot",
            new=AsyncMock(return_value=snap),
        ):
            r = client.get("/api/v1/resources")
        assert r.status_code == 200
        body = r.json()
        if gpu_present:
            assert body["gpu_util_percent"] == 50.0
        else:
            assert body["gpu_util_percent"] is None
        if psutil_present:
            assert body["cpu_percent"] == 10.0
        else:
            assert body["cpu_percent"] is None
