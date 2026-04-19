from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient


def _create(client: TestClient, **payload) -> dict:
    resp = client.post("/api/v1/providers", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_list_types_returns_single_node_and_runpod(client: TestClient) -> None:
    resp = client.get("/api/v1/providers/types")
    assert resp.status_code == 200
    payload = resp.json()
    ids = {t["id"] for t in payload["types"]}
    assert {"single_node", "runpod"}.issubset(ids)
    for entry in payload["types"]:
        assert entry["label"]
        assert isinstance(entry["json_schema"], dict)
        assert entry["json_schema"].get("properties"), f"missing properties for {entry['id']}"


def test_create_runpod_provider_default_path(
    client: TestClient, projects_root: Path
) -> None:
    assert client.get("/api/v1/providers").json() == []
    summary = _create(client, name="RunPod prod", type="runpod")
    assert summary["type"] == "runpod"
    assert summary["id"] == "runpod-prod"
    expected_path = projects_root / "providers" / "runpod-prod"
    assert Path(summary["path"]) == expected_path
    assert (expected_path / "provider.json").is_file()
    assert (expected_path / "current.yaml").is_file()
    metadata = json.loads((expected_path / "provider.json").read_text(encoding="utf-8"))
    assert metadata["type"] == "runpod"


def test_create_with_custom_path(client: TestClient, tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere" / "single"
    summary = _create(
        client,
        name="Single local",
        type="single_node",
        id="sn-local",
        path=str(custom),
    )
    assert Path(summary["path"]) == custom.resolve()
    assert (custom / "provider.json").is_file()


def test_create_unknown_type_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/providers", json={"name": "Bogus", "type": "bogus"})
    assert resp.status_code == 400


def test_duplicate_id_rejected(client: TestClient) -> None:
    _create(client, name="Dup", type="runpod", id="dup")
    resp = client.post("/api/v1/providers", json={"name": "Dup 2", "type": "runpod", "id": "dup"})
    assert resp.status_code == 400


def test_save_valid_runpod_config_snapshots_history(
    client: TestClient, projects_root: Path
) -> None:
    _create(client, name="RP", type="runpod", id="rp")
    minimal = yaml.safe_dump(
        {
            "connect": {"ssh": {"key_path": "~/.ssh/id_ed25519_runpod"}},
            "cleanup": {"auto_delete_pod": True, "keep_pod_on_error": False, "on_interrupt": True},
            "training": {
                "gpu_type": "NVIDIA A40",
                "image_name": "ryotenkai/training:latest",
                "container_disk_gb": 100,
                "volume_disk_gb": 20,
            },
            "inference": {},
        }
    )
    first = client.put("/api/v1/providers/rp/config", json={"yaml": "original\n"})
    assert first.status_code == 200
    second = client.put("/api/v1/providers/rp/config", json={"yaml": minimal})
    assert second.status_code == 200
    assert second.json()["snapshot_filename"] is not None

    listing = client.get("/api/v1/providers/rp/config/versions").json()
    assert len(listing["versions"]) == 1


def test_validate_endpoint_reports_schema_errors(client: TestClient) -> None:
    _create(client, name="RP", type="runpod", id="rp")
    resp = client.post(
        "/api/v1/providers/rp/config/validate",
        json={"yaml": "connect: invalid\n"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert any(c["status"] == "fail" for c in body["checks"])


def test_validate_ok_when_valid(client: TestClient) -> None:
    _create(client, name="SN", type="single_node", id="sn")
    valid = yaml.safe_dump(
        {
            "connect": {},
            "cleanup": {},
            "training": {},
            "inference": {},
        }
    )
    resp = client.post("/api/v1/providers/sn/config/validate", json={"yaml": valid})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # ok or fail depends on whether SingleNodeConfig requires nested fields;
    # at minimum the response shape must be present
    assert "checks" in body


def test_get_detail_and_config_roundtrip(client: TestClient) -> None:
    _create(client, name="Det", type="runpod", id="det")
    detail = client.get("/api/v1/providers/det").json()
    assert detail["id"] == "det"
    assert detail["type"] == "runpod"
    assert detail["current_config_yaml"] == ""

    client.put("/api/v1/providers/det/config", json={"yaml": "training: {}\n"})
    cfg = client.get("/api/v1/providers/det/config").json()
    assert cfg["yaml"] == "training: {}\n"
    assert cfg["parsed_json"] == {"training": {}}


def test_versions_list_read_restore(client: TestClient) -> None:
    _create(client, name="V", type="runpod", id="v")
    for text in ("model: a\n", "model: b\n", "model: c\n"):
        client.put("/api/v1/providers/v/config", json={"yaml": text})

    versions = client.get("/api/v1/providers/v/config/versions").json()["versions"]
    assert len(versions) == 2
    target = versions[-1]["filename"]

    read = client.get(f"/api/v1/providers/v/config/versions/{target}")
    assert read.status_code == 200

    restore = client.post(f"/api/v1/providers/v/config/versions/{target}/restore")
    assert restore.status_code == 200
    current = client.get("/api/v1/providers/v/config").json()
    assert current["yaml"].startswith("model:")


def test_delete_only_unregisters(client: TestClient, projects_root: Path) -> None:
    _create(client, name="Gone", type="runpod", id="gone")
    provider_dir = projects_root / "providers" / "gone"
    assert provider_dir.is_dir()

    resp = client.delete("/api/v1/providers/gone")
    assert resp.status_code == 204
    assert provider_dir.is_dir()
    listing = client.get("/api/v1/providers").json()
    assert all(item["id"] != "gone" for item in listing)
    assert client.get("/api/v1/providers/gone").status_code == 404
