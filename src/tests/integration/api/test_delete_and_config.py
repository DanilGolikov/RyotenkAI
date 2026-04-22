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


def test_config_presets_list_exposes_scope_and_requirements(client) -> None:
    """GET /config/presets surfaces v2 manifest fields for the three shipped presets."""
    response = client.get("/api/v1/config/presets")
    assert response.status_code == 200
    body = response.json()
    presets_by_id = {p["name"]: p for p in body["presets"]}

    # All three shipped presets declare scope/requirements after the v2 upgrade.
    for pid in ("01-small", "02-medium", "03-large"):
        assert pid in presets_by_id, f"preset {pid!r} missing"
        p = presets_by_id[pid]
        assert p["scope"] is not None
        assert p["scope"]["replaces"] == ["model", "training"]
        assert "datasets" in p["scope"]["preserves"]
        assert p["requirements"] is not None
        # Placeholders surface the dataset-path hint
        assert any("datasets.default" in path for path in p["placeholders"])


def test_config_preview_preserves_user_datasets_and_providers(client) -> None:
    """POST /config/presets/{id}/preview keeps datasets+providers when the
    manifest scope lists them under ``preserves``."""
    user_config = {
        "model": {"name": "old-model"},
        "training": {"type": "sft"},
        "datasets": {"mine": {"source_type": "local"}},
        "providers": {"my_provider": {"kind": "single_node"}},
    }
    response = client.post(
        "/api/v1/config/presets/01-small/preview",
        json={"current_config": user_config},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # model and training are replaced; datasets/providers preserved verbatim
    assert body["resulting_config"]["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert body["resulting_config"]["training"]["type"] == "qlora"
    assert body["resulting_config"]["datasets"] == user_config["datasets"]
    assert body["resulting_config"]["providers"] == user_config["providers"]

    # Diff carries reasons per key
    diff_by_key = {d["key"]: d for d in body["diff"]}
    assert diff_by_key["model"]["reason"] == "preset_replaced"
    assert diff_by_key["datasets"]["reason"] == "preset_preserved"
    assert diff_by_key["providers"]["reason"] == "preset_preserved"

    # Requirements surfaced
    labels = {r["label"] for r in body["requirements"]}
    assert "Provider kind" in labels

    # Placeholders surfaced
    assert any("datasets.default" in ph["path"] for ph in body["placeholders"])


def test_config_preview_404_on_unknown_preset(client) -> None:
    response = client.post(
        "/api/v1/config/presets/does-not-exist/preview",
        json={"current_config": {}},
    )
    assert response.status_code == 404
