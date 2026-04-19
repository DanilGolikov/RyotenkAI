from __future__ import annotations


def test_get_attempt_detail(client, seed_completed_run) -> None:
    seed_completed_run("run_gamma")
    response = client.get("/api/v1/runs/run_gamma/attempts/1")
    assert response.status_code == 200
    body = response.json()
    assert body["attempt_no"] == 1
    assert body["status"] == "completed"
    assert "Dataset Validator" in body["stage_runs"]
    assert body["stage_runs"]["Dataset Validator"]["status"] == "completed"
    assert body["stage_runs"]["Dataset Validator"]["mode_label"]


def test_get_stages_ordered_by_enabled_stage_names(client, seed_completed_run) -> None:
    seed_completed_run("run_delta")
    response = client.get("/api/v1/runs/run_delta/attempts/1/stages")
    assert response.status_code == 200
    stages = response.json()["stages"]
    assert len(stages) == 1
    assert stages[0]["stage_name"] == "Dataset Validator"


def test_get_attempt_missing(client, seed_completed_run) -> None:
    seed_completed_run("run_epsilon")
    response = client.get("/api/v1/runs/run_epsilon/attempts/99")
    assert response.status_code == 404


def test_read_log_chunk(client, seed_completed_run) -> None:
    seed_completed_run("run_zeta")
    response = client.get("/api/v1/runs/run_zeta/attempts/1/logs?file=pipeline.log")
    assert response.status_code == 200
    body = response.json()
    assert body["file"] == "pipeline.log"
    assert "pipeline start" in body["content"]
    assert body["eof"] is True
    assert body["next_offset"] > 0


def test_read_log_rejects_unknown_file(client, seed_completed_run) -> None:
    seed_completed_run("run_eta")
    response = client.get("/api/v1/runs/run_eta/attempts/1/logs?file=../secrets.txt")
    assert response.status_code == 400


def test_list_log_files(client, seed_completed_run) -> None:
    seed_completed_run("run_theta")
    response = client.get("/api/v1/runs/run_theta/attempts/1/logs/files")
    assert response.status_code == 200
    files = response.json()
    names = {f["name"]: f for f in files}
    assert names["pipeline.log"]["exists"] is True
    assert names["training.log"]["exists"] is False
