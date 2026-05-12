"""End-to-end tests for ``GET /api/v1/diagnostics`` (Phase 2 PR-2.1)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Required by main.py lifespan — set BEFORE create_app import.
os.environ.setdefault("RYOTENKAI_RUNTIME_PROVIDER", "single_node")

from ryotenkai_pod.runner.main import create_app
from ryotenkai_shared.contracts.runner_api.diagnostics import (
    DmesgReport,
    GpuReport,
    GpuRow,
    KernelSignalsReport,
)


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
    def test_default_returns_all_blocks(self, client: TestClient) -> None:
        with patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_dmesg",
            return_value=DmesgReport(lines=["a", "b"]),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_nvidia_smi",
            return_value=GpuReport(rows=[GpuRow(
                name="RTX", utilization_gpu_percent=10,
                memory_used_mib=100, memory_total_mib=1000,
            )]),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_kernel_signals",
            return_value=KernelSignalsReport(matches=["NVRM: ..."]),
        ):
            r = client.get("/api/v1/diagnostics")
        assert r.status_code == 200
        body = r.json()
        assert body["dmesg"]["lines"] == ["a", "b"]
        assert body["gpu"]["rows"][0]["name"] == "RTX"
        assert body["kernel_signals"]["matches"] == ["NVRM: ..."]

    def test_include_filter(self, client: TestClient) -> None:
        with patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_dmesg",
        ) as dmesg_mock, patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_nvidia_smi",
            return_value=GpuReport(rows=[]),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_kernel_signals",
        ) as kern_mock:
            r = client.get("/api/v1/diagnostics?include=gpu")
        assert r.status_code == 200
        body = r.json()
        # Other collectors NOT invoked
        dmesg_mock.assert_not_called()
        kern_mock.assert_not_called()
        # Only gpu in response
        assert "gpu" in body
        assert "dmesg" not in body
        assert "kernel_signals" not in body


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_invalid_include_returns_422_problem_json(self, client: TestClient) -> None:
        r = client.get("/api/v1/diagnostics?include=evilcommand")
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")
        body = r.json()
        assert body["code"] == "DIAGNOSTIC_INVALID_INCLUDE"
        assert "evilcommand" in body["detail"]


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_include_list_means_all_blocks(self, client: TestClient) -> None:
        # ``?include=`` with empty value → all blocks (operator-friendly).
        with patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_dmesg",
            return_value=DmesgReport(),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_nvidia_smi",
            return_value=GpuReport(),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_kernel_signals",
            return_value=KernelSignalsReport(),
        ):
            r = client.get("/api/v1/diagnostics?include=")
        assert r.status_code == 200
        assert sorted(r.json().keys()) == ["dmesg", "gpu", "kernel_signals"]

    def test_comma_separated_repeats_supported(self, client: TestClient) -> None:
        with patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_dmesg",
            return_value=DmesgReport(),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_nvidia_smi",
            return_value=GpuReport(),
        ):
            r = client.get("/api/v1/diagnostics?include=dmesg,gpu")
        assert r.status_code == 200
        assert sorted(r.json().keys()) == ["dmesg", "gpu"]


# ---------------------------------------------------------------------------
# Invariant
# ---------------------------------------------------------------------------


class TestInvariant:
    def test_per_block_failure_keeps_http_200(self, client: TestClient) -> None:
        """Whole point of RP2: dmesg PERMISSION_DENIED while GPU is
        fine should still ship the GPU block + 200."""
        from ryotenkai_shared.contracts.runner_api.diagnostics import (
            DiagnosticsBlockError,
        )

        with patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_dmesg",
            return_value=DmesgReport(error=DiagnosticsBlockError.PERMISSION_DENIED),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_nvidia_smi",
            return_value=GpuReport(rows=[GpuRow(
                name="RTX", utilization_gpu_percent=0,
                memory_used_mib=0, memory_total_mib=1,
            )]),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_kernel_signals",
            return_value=KernelSignalsReport(error=DiagnosticsBlockError.PERMISSION_DENIED),
        ):
            r = client.get("/api/v1/diagnostics")
        assert r.status_code == 200
        body = r.json()
        assert body["dmesg"]["error"] == "permission_denied"
        # ``error`` null-stripped on healthy block (response_model_exclude_none).
        assert "error" not in body["gpu"]
        assert body["kernel_signals"]["error"] == "permission_denied"


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("query", [
        "include=dmesg",
        "include=gpu",
        "include=kernel_signals",
        "include=dmesg&include=gpu",
        "include=dmesg,gpu,kernel_signals",
    ])
    def test_query_variants_return_200(
        self, client: TestClient, query: str,
    ) -> None:
        with patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_dmesg",
            return_value=DmesgReport(),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_nvidia_smi",
            return_value=GpuReport(),
        ), patch(
            "ryotenkai_pod.runner.api.diagnostics.collect_kernel_signals",
            return_value=KernelSignalsReport(),
        ):
            r = client.get(f"/api/v1/diagnostics?{query}")
        assert r.status_code == 200, query
