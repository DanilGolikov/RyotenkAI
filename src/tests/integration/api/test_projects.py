from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def _create(client: TestClient, **payload) -> dict:
    resp = client.post("/api/v1/projects", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_create_and_list_project_in_default_root(
    client: TestClient, projects_root: Path
) -> None:
    # Empty initial list
    resp = client.get("/api/v1/projects")
    assert resp.status_code == 200
    assert resp.json() == []

    summary = _create(client, name="Demo project", description="hello")
    assert summary["id"] == "demo-project"
    assert summary["name"] == "Demo project"
    expected_path = projects_root / "projects" / "demo-project"
    assert Path(summary["path"]) == expected_path
    assert expected_path.is_dir()
    assert (expected_path / "project.json").is_file()
    assert (expected_path / "configs" / "current.yaml").is_file()

    resp = client.get("/api/v1/projects")
    assert resp.status_code == 200
    items = resp.json()
    assert [it["id"] for it in items] == ["demo-project"]


def test_create_project_with_custom_path(client: TestClient, tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere" / "my_exp"
    summary = _create(client, name="Custom", id="custom_one", path=str(custom))
    assert summary["id"] == "custom_one"
    assert Path(summary["path"]) == custom.resolve()
    assert (custom / "project.json").is_file()


def test_cannot_create_with_duplicate_id(client: TestClient) -> None:
    _create(client, name="Alpha", id="alpha")
    resp = client.post("/api/v1/projects", json={"name": "Alpha again", "id": "alpha"})
    assert resp.status_code == 400


def test_invalid_id_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/projects", json={"name": "Bad", "id": "Has Spaces"})
    assert resp.status_code == 400


def test_get_detail_returns_current_config(client: TestClient) -> None:
    _create(client, name="Inspect", id="inspect")
    resp = client.get("/api/v1/projects/inspect")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == "inspect"
    assert detail["current_config_yaml"] == ""


def test_save_config_creates_history_snapshot(
    client: TestClient, projects_root: Path
) -> None:
    _create(client, name="Hist", id="hist")

    first = client.put(
        "/api/v1/projects/hist/config", json={"yaml": "model: foo\n"}
    )
    assert first.status_code == 200
    # No prior config → no snapshot.
    assert first.json() == {"ok": True, "snapshot_filename": None}

    second = client.put(
        "/api/v1/projects/hist/config", json={"yaml": "model: bar\n"}
    )
    assert second.status_code == 200
    snapshot_name = second.json()["snapshot_filename"]
    assert snapshot_name is not None
    snapshot_path = (
        projects_root / "projects" / "hist" / "configs" / "history" / snapshot_name
    )
    assert snapshot_path.is_file()
    assert snapshot_path.read_text(encoding="utf-8") == "model: foo\n"

    current = client.get("/api/v1/projects/hist/config")
    assert current.status_code == 200
    assert current.json()["yaml"] == "model: bar\n"


def test_versions_endpoint_lists_snapshots(client: TestClient) -> None:
    _create(client, name="Versions", id="versions")
    for text in ("model: a\n", "model: b\n", "model: c\n"):
        resp = client.put("/api/v1/projects/versions/config", json={"yaml": text})
        assert resp.status_code == 200

    listing = client.get("/api/v1/projects/versions/config/versions")
    assert listing.status_code == 200
    versions = listing.json()["versions"]
    assert len(versions) == 2

    first_snap = versions[-1]["filename"]
    detail = client.get(f"/api/v1/projects/versions/config/versions/{first_snap}")
    assert detail.status_code == 200
    assert detail.json()["yaml"].startswith("model:")


def test_restore_version_replaces_current(client: TestClient) -> None:
    _create(client, name="Restore", id="restore")
    client.put("/api/v1/projects/restore/config", json={"yaml": "model: v1\n"})
    client.put("/api/v1/projects/restore/config", json={"yaml": "model: v2\n"})
    versions = client.get(
        "/api/v1/projects/restore/config/versions"
    ).json()["versions"]
    assert versions, "expected at least one snapshot"
    target = versions[-1]["filename"]  # oldest — the "v1" snapshot

    restore = client.post(
        f"/api/v1/projects/restore/config/versions/{target}/restore"
    )
    assert restore.status_code == 200
    current = client.get("/api/v1/projects/restore/config").json()
    assert current["yaml"] == "model: v1\n"


def test_validate_endpoint_surfaces_errors(client: TestClient) -> None:
    _create(client, name="Val", id="val")
    resp = client.post(
        "/api/v1/projects/val/config/validate",
        json={"yaml": "not: a valid pipeline config\n"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert any(c["status"] == "fail" for c in body["checks"])


def test_delete_only_unregisters(
    client: TestClient, projects_root: Path
) -> None:
    _create(client, name="Gone", id="gone")
    project_dir = projects_root / "projects" / "gone"
    assert project_dir.is_dir()

    resp = client.delete("/api/v1/projects/gone")
    assert resp.status_code == 204

    # Directory still exists — delete is only an unregister.
    assert project_dir.is_dir()

    listing = client.get("/api/v1/projects").json()
    assert all(item["id"] != "gone" for item in listing)

    # And a follow-up detail/GET returns 404.
    assert client.get("/api/v1/projects/gone").status_code == 404


def test_config_schema_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/config/schema")
    assert resp.status_code == 200
    schema = resp.json()
    props = schema.get("properties", {})
    # All 7 top-level groups must be present for the UI builder.
    for key in (
        "model",
        "datasets",
        "training",
        "providers",
        "inference",
        "evaluation",
        "experiment_tracking",
    ):
        assert key in props, f"missing {key} from config schema"


def test_favorite_versions_toggle(client: TestClient) -> None:
    _create(client, name="Fav", id="fav")
    # two saves → one snapshot in history
    client.put("/api/v1/projects/fav/config", json={"yaml": "model: v1\n"})
    client.put("/api/v1/projects/fav/config", json={"yaml": "model: v2\n"})
    versions = client.get("/api/v1/projects/fav/config/versions").json()["versions"]
    assert versions and not versions[0]["is_favorite"]
    target = versions[0]["filename"]

    pinned = client.put(
        f"/api/v1/projects/fav/config/versions/{target}/favorite",
        json={"favorite": True},
    )
    assert pinned.status_code == 200
    assert pinned.json()["favorite_versions"] == [target]

    after = client.get("/api/v1/projects/fav/config/versions").json()["versions"]
    assert after[0]["filename"] == target
    assert after[0]["is_favorite"] is True

    unpinned = client.put(
        f"/api/v1/projects/fav/config/versions/{target}/favorite",
        json={"favorite": False},
    )
    assert unpinned.json()["favorite_versions"] == []


def test_favorite_rejects_unknown_filename(client: TestClient) -> None:
    _create(client, name="FavX", id="favx")
    resp = client.put(
        "/api/v1/projects/favx/config/versions/bogus.yaml/favorite",
        json={"favorite": True},
    )
    assert resp.status_code == 404


def test_registry_file_is_written(client: TestClient, projects_root: Path) -> None:
    _create(client, name="Persisted", id="persisted")
    registry_file = projects_root / "projects.json"
    assert registry_file.is_file()
    payload = json.loads(registry_file.read_text(encoding="utf-8"))
    ids = [p["id"] for p in payload.get("projects", [])]
    assert "persisted" in ids
