"""Integration tests for ``GET /projects/{id}/runs`` (Step 6).

The endpoint surfaces the per-project ``runs/index.json`` ledger to
the frontend's Runs tab. These tests exercise the FastAPI surface end-
to-end through the test client, with the project registry and stores
rooted at ``projects_root`` (per the conftest fixtures).

Categories: positive, negative, boundary, regression, logic-specific.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.workspace.projects.store import ProjectStore


def _create_project(client: TestClient, project_id: str = "p1") -> None:
    """Create a project via the API so the registry stays consistent."""
    resp = client.post(
        "/api/v1/projects",
        json={"name": project_id.replace("-", " ").title(), "id": project_id},
    )
    assert resp.status_code == 201, resp.text


def _project_path(projects_root: Path, project_id: str) -> Path:
    return projects_root / "projects" / project_id


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_runs_endpoint_empty_for_fresh_project(
    client: TestClient,
) -> None:
    """No launches yet → empty list, 200 OK (not 404)."""
    _create_project(client, "p1")

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_runs_endpoint_lists_registered_runs(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    store = ProjectStore(_project_path(projects_root, "p1"))
    store.register_run(
        run_id="run-A",
        started_at="2026-04-10T10:00:00Z",
        mlflow_run_id="ml-A",
        actor="alice",
    )
    store.register_run(
        run_id="run-B",
        started_at="2026-04-11T10:00:00Z",
        actor="bob",
    )

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["runs"]) == 2
    # Newest first.
    assert [r["run_id"] for r in body["runs"]] == ["run-B", "run-A"]
    # Schema fields surface.
    assert body["runs"][1]["mlflow_run_id"] == "ml-A"


def test_runs_endpoint_filters_by_status(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    store = ProjectStore(_project_path(projects_root, "p1"))
    store.register_run(run_id="r1", status="running")
    store.register_run(run_id="r2", status="completed")
    store.register_run(run_id="r3", status="failed")

    resp = client.get("/api/v1/projects/p1/runs?status=completed")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r2"


def test_runs_endpoint_caps_with_limit(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    store = ProjectStore(_project_path(projects_root, "p1"))
    for i in range(5):
        store.register_run(
            run_id=f"r{i}",
            started_at=f"2026-04-{i + 1:02d}T00:00:00Z",
        )

    resp = client.get("/api/v1/projects/p1/runs?limit=2")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 2
    # Newest first → r4, r3.
    assert [r["run_id"] for r in runs] == ["r4", "r3"]


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_runs_endpoint_unknown_project_404(client: TestClient) -> None:
    resp = client.get("/api/v1/projects/does-not-exist/runs")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Boundary / Regression — corrupt index doesn't 500 the endpoint
# ---------------------------------------------------------------------------


def test_runs_endpoint_handles_malformed_index(
    client: TestClient, projects_root: Path,
) -> None:
    """``index.json`` corrupted on disk → endpoint returns empty list,
    not 500. The store's defensive read paves the way; this pins that
    the router doesn't accidentally re-introduce the failure mode."""
    _create_project(client, "p1")
    store = ProjectStore(_project_path(projects_root, "p1"))
    store.runs_dir.mkdir(parents=True, exist_ok=True)
    store.runs_index_path.write_text("garbage", encoding="utf-8")

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_runs_endpoint_skips_malformed_entries(
    client: TestClient, projects_root: Path,
) -> None:
    """A row missing required keys should be dropped silently, not
    5XX'd. The valid entries beside it still surface."""
    _create_project(client, "p1")
    store = ProjectStore(_project_path(projects_root, "p1"))
    store.runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "runs": [
            {
                "run_id": "good",
                "started_at": "2026-04-10T10:00:00Z",
                "status": "running",
            },
            # Missing started_at / status — invalid.
            {"run_id": "bad"},
        ],
    }
    store.runs_index_path.write_text(json.dumps(payload), encoding="utf-8")

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert [r["run_id"] for r in runs] == ["good"]


# ---------------------------------------------------------------------------
# Logic-specific — filter+limit composition
# ---------------------------------------------------------------------------


def test_runs_endpoint_status_and_limit_compose(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    store = ProjectStore(_project_path(projects_root, "p1"))
    for i in range(6):
        store.register_run(
            run_id=f"r{i}",
            status="completed" if i % 2 == 0 else "running",
            started_at=f"2026-04-{i + 1:02d}T00:00:00Z",
        )

    resp = client.get("/api/v1/projects/p1/runs?status=completed&limit=2")
    runs = resp.json()["runs"]
    assert len(runs) == 2
    # All returned must be completed.
    assert all(r["status"] == "completed" for r in runs)
    # Newest first within the matching status.
    assert [r["run_id"] for r in runs] == ["r4", "r2"]
