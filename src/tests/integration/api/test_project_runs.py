"""Integration tests for ``GET /projects/{id}/runs``.

The endpoint walks ``<project>/runs/`` directly — every sub-directory
containing ``pipeline_state.json`` is treated as a run. Tests seed
actual run dirs (the same shape ``LaunchPreparator`` writes) and
exercise the FastAPI surface end-to-end.

Categories: positive, negative, boundary, regression, logic-specific.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.pipeline.state import (
    PipelineAttemptState,
    PipelineState,
    PipelineStateStore,
    StageRunState,
)
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


def _seed_run(
    project_runs_dir: Path,
    *,
    run_id: str,
    status: str = StageRunState.STATUS_COMPLETED,
    actor: str | None = None,
    config_version_hash: str | None = None,
    started_at: str = "2026-04-10T10:00:00+00:00",
    completed_at: str | None = "2026-04-10T10:30:00+00:00",
    mlflow_run_id: str | None = None,
) -> Path:
    """Materialise a fully-formed run dir inside ``<project>/runs/``.

    Mirrors what :class:`LaunchPreparator` writes for a fresh launch —
    ``pipeline_state.json`` + ``attempts/attempt_1/`` so
    :func:`scan_runs_dir` picks it up.
    """
    run_dir = project_runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    store = PipelineStateStore(run_dir)
    attempt = PipelineAttemptState(
        attempt_id=f"{run_id}:attempt:1",
        attempt_no=1,
        runtime_name="single_node",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        training_critical_config_hash="train-hash",
        late_stage_config_hash="late-hash",
        model_dataset_config_hash="md-hash",
        root_mlflow_run_id=mlflow_run_id,
        enabled_stage_names=["Dataset Validator"],
        stage_runs={},
    )
    metadata: dict[str, object] = {}
    if actor is not None:
        metadata["actor"] = actor
    if config_version_hash is not None:
        metadata["config_version_hash"] = config_version_hash

    state = PipelineState(
        schema_version=1,
        logical_run_id=run_id,
        run_directory=str(run_dir),
        config_path="configs/current.yaml",
        active_attempt_id=attempt.attempt_id,
        pipeline_status=status,
        training_critical_config_hash="train-hash",
        late_stage_config_hash="late-hash",
        model_dataset_config_hash="md-hash",
        root_mlflow_run_id=mlflow_run_id,
        attempts=[attempt],
        current_output_lineage={},
        metadata=metadata,
    )
    store.save(state)
    return run_dir


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_runs_endpoint_empty_for_fresh_project(
    client: TestClient,
) -> None:
    """Fresh project → no runs subdirs → empty list, 200 OK (not 404)."""
    _create_project(client, "p1")

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_runs_endpoint_lists_seeded_runs(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    runs_dir = ProjectStore(_project_path(projects_root, "p1")).runs_dir
    _seed_run(
        runs_dir,
        run_id="run-A",
        started_at="2026-04-10T10:00:00+00:00",
        actor="alice",
        mlflow_run_id="ml-A",
    )
    _seed_run(
        runs_dir,
        run_id="run-B",
        started_at="2026-04-11T10:00:00+00:00",
        actor="bob",
    )

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["runs"]) == 2
    # Newest first by directory name (run-B sorts after run-A).
    ids = [r["run_id"] for r in body["runs"]]
    assert "run-A" in ids and "run-B" in ids
    # Schema fields surface from pipeline_state.json.
    a_row = next(r for r in body["runs"] if r["run_id"] == "run-A")
    assert a_row["mlflow_run_id"] == "ml-A"
    assert a_row["actor"] == "alice"


def test_runs_endpoint_filters_by_status(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    runs_dir = ProjectStore(_project_path(projects_root, "p1")).runs_dir
    _seed_run(runs_dir, run_id="r1", status=StageRunState.STATUS_RUNNING)
    _seed_run(runs_dir, run_id="r2", status=StageRunState.STATUS_COMPLETED)
    _seed_run(runs_dir, run_id="r3", status=StageRunState.STATUS_FAILED)

    resp = client.get(
        f"/api/v1/projects/p1/runs?status={StageRunState.STATUS_COMPLETED}"
    )
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r2"


def test_runs_endpoint_caps_with_limit(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    runs_dir = ProjectStore(_project_path(projects_root, "p1")).runs_dir
    for i in range(5):
        _seed_run(runs_dir, run_id=f"r{i}")

    resp = client.get("/api/v1/projects/p1/runs?limit=2")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 2


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_runs_endpoint_unknown_project_404(client: TestClient) -> None:
    resp = client.get("/api/v1/projects/does-not-exist/runs")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Boundary / Regression
# ---------------------------------------------------------------------------


def test_runs_endpoint_skips_dirs_without_pipeline_state(
    client: TestClient, projects_root: Path,
) -> None:
    """A subdir with no ``pipeline_state.json`` (interrupted creation,
    leftover artefact) is silently skipped — endpoint must not 5XX."""
    _create_project(client, "p1")
    runs_dir = ProjectStore(_project_path(projects_root, "p1")).runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Empty dir — no state file.
    (runs_dir / "leftover").mkdir()
    # Real run.
    _seed_run(runs_dir, run_id="real")

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert [r["run_id"] for r in runs] == ["real"]


def test_runs_endpoint_handles_corrupt_state_file(
    client: TestClient, projects_root: Path,
) -> None:
    """A run dir with a malformed ``pipeline_state.json`` still
    appears in the listing (with ``status: unknown``) — surface stays
    usable while diagnostics flow through."""
    _create_project(client, "p1")
    runs_dir = ProjectStore(_project_path(projects_root, "p1")).runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    bad_run = runs_dir / "corrupt"
    bad_run.mkdir()
    (bad_run / "pipeline_state.json").write_text(
        "garbage not json", encoding="utf-8"
    )

    resp = client.get("/api/v1/projects/p1/runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert any(r["run_id"] == "corrupt" for r in runs)


# ---------------------------------------------------------------------------
# Logic-specific — filter+limit composition
# ---------------------------------------------------------------------------


def test_runs_endpoint_status_and_limit_compose(
    client: TestClient, projects_root: Path,
) -> None:
    _create_project(client, "p1")
    runs_dir = ProjectStore(_project_path(projects_root, "p1")).runs_dir
    for i in range(6):
        _seed_run(
            runs_dir,
            run_id=f"r{i}",
            status=(
                StageRunState.STATUS_COMPLETED if i % 2 == 0
                else StageRunState.STATUS_RUNNING
            ),
        )

    resp = client.get(
        f"/api/v1/projects/p1/runs"
        f"?status={StageRunState.STATUS_COMPLETED}&limit=2"
    )
    runs = resp.json()["runs"]
    assert len(runs) == 2
    assert all(r["status"] == StageRunState.STATUS_COMPLETED for r in runs)
