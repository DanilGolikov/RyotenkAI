"""Tests for ``POST /api/v1/runtime/import-check`` (Phase 2 PR-2.5)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RYOTENKAI_RUNTIME_PROVIDER", "single_node")

from ryotenkai_pod.runner.main import create_app
from ryotenkai_shared.contracts.runner_api.runtime import (
    MAX_MODULES_PER_REQUEST,
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
    def test_real_modules_succeed(self, client: TestClient) -> None:
        # ``json`` and ``os`` are stdlib — guaranteed importable.
        r = client.post(
            "/api/v1/runtime/import-check",
            json={"modules": ["json", "os"]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["results"]) == 2
        assert all(r["importable"] for r in body["results"])

    def test_missing_module_returns_200_with_error(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/runtime/import-check",
            json={"modules": ["definitely_not_a_module_xyz"]},
        )
        assert r.status_code == 200
        result = r.json()["results"][0]
        assert result["importable"] is False
        # Last stderr line is the canonical Python exception:
        # ``ModuleNotFoundError: No module named '...'``.
        assert "ModuleNotFoundError" in (result["error"] or "")


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_empty_list_rejected(self, client: TestClient) -> None:
        # min_length=1 on the field
        r = client.post(
            "/api/v1/runtime/import-check", json={"modules": []},
        )
        assert r.status_code == 422

    def test_invalid_module_name_returns_422(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/runtime/import-check",
            json={"modules": ["os.system('rm -rf /')"]},
        )
        assert r.status_code == 422
        assert r.json()["code"] == "IMPORT_CHECK_INVALID_MODULE_NAME"

    def test_too_many_modules_returns_422(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/runtime/import-check",
            json={"modules": [f"mod{i}" for i in range(MAX_MODULES_PER_REQUEST + 1)]},
        )
        assert r.status_code == 422
        assert r.json()["code"] == "IMPORT_CHECK_TOO_MANY_MODULES"


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_exactly_max_modules_succeeds(self, client: TestClient) -> None:
        # Use ``json`` repeated — small subprocess overhead × MAX is
        # within reason, kept off slow-CI by capping at 50 in DTO.
        modules = ["json"] * MAX_MODULES_PER_REQUEST
        r = client.post(
            "/api/v1/runtime/import-check",
            json={"modules": modules},
        )
        assert r.status_code == 200
        assert len(r.json()["results"]) == MAX_MODULES_PER_REQUEST


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    @pytest.mark.parametrize("name", [
        "json",                       # stdlib top-level
        "json.decoder",              # stdlib dotted
        "ryotenkai_shared",           # workspace top-level
        "ryotenkai_shared.constants", # workspace dotted
    ])
    def test_dotted_modules_accepted(self, client: TestClient, name: str) -> None:
        r = client.post(
            "/api/v1/runtime/import-check", json={"modules": [name]},
        )
        assert r.status_code == 200
        result = r.json()["results"][0]
        assert result["module"] == name
        assert result["importable"] is True


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("modules,expected_failed", [
        (["json"], []),
        (["json", "os"], []),
        (["json", "doesnt_exist"], ["doesnt_exist"]),
        (["doesnt_exist_a", "doesnt_exist_b"], ["doesnt_exist_a", "doesnt_exist_b"]),
    ])
    def test_matrix(
        self,
        client: TestClient,
        modules: list[str],
        expected_failed: list[str],
    ) -> None:
        r = client.post(
            "/api/v1/runtime/import-check", json={"modules": modules},
        )
        assert r.status_code == 200
        body = r.json()
        failed = [r["module"] for r in body["results"] if not r["importable"]]
        assert failed == expected_failed
