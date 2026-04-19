from __future__ import annotations

from pathlib import Path


def test_list_runs_empty(client) -> None:
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    body = response.json()
    assert body["groups"] == {}


def test_list_runs_returns_completed_run(client, seed_completed_run) -> None:
    seed_completed_run("run_alpha")
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    body = response.json()
    assert "(root)" in body["groups"]
    rows = body["groups"]["(root)"]
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run_alpha"
    assert row["status"] == "completed"
    assert row["attempts"] == 1
    assert row["status_icon"]  # enrichment works


def test_get_run_detail_returns_enriched_state(client, seed_completed_run) -> None:
    seed_completed_run("run_beta")
    response = client.get("/api/v1/runs/run_beta")
    assert response.status_code == 200
    body = response.json()
    assert body["logical_run_id"] == "run_beta"
    assert body["status"] == "completed"
    assert body["next_attempt_no"] == 2
    assert body["running_attempt_no"] is None
    assert body["is_locked"] is False


def test_get_run_detail_404_for_missing_run(client) -> None:
    response = client.get("/api/v1/runs/does_not_exist")
    assert response.status_code == 404


def test_get_run_rejects_path_traversal(client) -> None:
    response = client.get("/api/v1/runs/..%2F..%2Fetc")
    assert response.status_code in (400, 404)


def test_create_run_auto_id(client, runs_dir: Path) -> None:
    response = client.post("/api/v1/runs", json={})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["attempts"] == 0
    created_path = Path(body["run_dir"])
    assert created_path.is_dir()
    assert created_path.parent == runs_dir.resolve()


def test_create_run_rejects_duplicate(client, seed_completed_run) -> None:
    seed_completed_run("dup_run")
    response = client.post("/api/v1/runs", json={"run_id": "dup_run"})
    assert response.status_code == 422


def test_list_runs_grouped_by_subgroup(client, seed_completed_run) -> None:
    seed_completed_run("sub_run")
    # move into a subgroup
    (client.app.state.settings.runs_dir_resolved / "experiments").mkdir(parents=True, exist_ok=True)
    response = client.post("/api/v1/runs", json={"run_id": "nested", "subgroup": "experiments"})
    assert response.status_code == 201
