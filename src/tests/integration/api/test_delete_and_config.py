from __future__ import annotations

from pathlib import Path

from src.api.services import delete_service
from src.pipeline.deletion import DeleteIssue, DeleteResult, RunDeleter


def test_delete_run_happy_path(client, seed_completed_run, monkeypatch) -> None:
    run_dir = seed_completed_run("run_delete_1")

    def _fake_delete(self, target, *, mode):  # type: ignore[no-untyped-def]
        return DeleteResult(
            target=target,
            run_dirs=(target,),
            deleted_mlflow_run_ids=("mlflow-1",),
            local_deleted=True,
            issues=(),
        )

    monkeypatch.setattr(RunDeleter, "delete_target", _fake_delete)

    response = client.delete("/api/v1/runs/run_delete_1")
    assert response.status_code == 200
    body = response.json()
    assert body["local_deleted"] is True
    assert body["deleted_mlflow_run_ids"] == ["mlflow-1"]
    assert body["is_success"] is True


def test_delete_run_reports_issues(client, seed_completed_run, monkeypatch) -> None:
    seed_completed_run("run_delete_2")

    def _fake_delete(self, target, *, mode):  # type: ignore[no-untyped-def]
        return DeleteResult(
            target=target,
            run_dirs=(target,),
            deleted_mlflow_run_ids=(),
            local_deleted=True,
            issues=(DeleteIssue(run_dir=target, phase="mlflow_runtime_contract", message="no tracking uri"),),
        )

    monkeypatch.setattr(RunDeleter, "delete_target", _fake_delete)

    response = client.delete("/api/v1/runs/run_delete_2")
    assert response.status_code == 200
    body = response.json()
    assert body["local_deleted"] is True
    assert body["is_success"] is False
    assert body["issues"][0]["phase"] == "mlflow_runtime_contract"


def test_delete_run_rejects_when_active(client, seed_running_run, monkeypatch) -> None:
    seed_running_run("run_delete_3", pid=111)
    monkeypatch.setattr(delete_service, "is_process_alive", lambda pid: True)

    response = client.delete("/api/v1/runs/run_delete_3")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "run_active"


def test_config_validate_reads_yaml(client, tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("stub", encoding="utf-8")

    from src.api.schemas.config_validate import ConfigCheck, ConfigValidationResult
    from src.api.services import config_service

    def _fake_validate(path: Path) -> ConfigValidationResult:
        return ConfigValidationResult(
            ok=True,
            config_path=str(path),
            checks=[ConfigCheck(label="YAML schema valid (Pydantic)", status="ok")],
        )

    monkeypatch.setattr(config_service, "validate_config", _fake_validate)
    response = client.post("/api/v1/config/validate", json={"config_path": str(config_file)})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["checks"][0]["status"] == "ok"


def test_config_validate_404_when_missing(client) -> None:
    response = client.post("/api/v1/config/validate", json={"config_path": "/tmp/does-not-exist.yaml"})
    assert response.status_code == 404


def test_config_default_lists_templates(client) -> None:
    response = client.get("/api/v1/config/default")
    assert response.status_code == 200
    body = response.json()
    assert body["runs_dir"]
    assert isinstance(body["config_templates"], list)
