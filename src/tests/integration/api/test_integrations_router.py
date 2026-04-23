from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _ephemeral_master_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin an isolated master key per test so the module-level cache in
    ``get_token_crypto`` does not leak state, and so crypto never writes
    into the real ``~/.ryotenkai`` during CI."""
    monkeypatch.setenv(
        "RYOTENKAI_SECRET_KEY", base64.b64encode(os.urandom(32)).decode("ascii")
    )
    from src.api import dependencies

    dependencies.get_token_crypto.cache_clear()


def _create_integration(client: TestClient, **payload) -> dict:
    resp = client.post("/api/v1/integrations", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_list_types(client: TestClient) -> None:
    resp = client.get("/api/v1/integrations/types")
    assert resp.status_code == 200
    ids = {t["id"] for t in resp.json()["types"]}
    assert {"mlflow", "huggingface"}.issubset(ids)
    for t in resp.json()["types"]:
        # Integration schemas MUST NOT declare a token field — that
        # would leak via GET /integrations/{id}.
        props = (t["json_schema"].get("properties") or {}).keys()
        assert "token" not in props


def test_crud_roundtrip(client: TestClient, projects_root: Path) -> None:
    assert client.get("/api/v1/integrations").json() == []

    summary = _create_integration(client, name="MLflow prod", type="mlflow")
    assert summary["type"] == "mlflow"
    assert summary["has_token"] is False
    workspace = projects_root / "integrations" / "mlflow-prod"
    assert workspace.is_dir()
    assert (workspace / "integration.json").is_file()

    detail = client.get(f"/api/v1/integrations/{summary['id']}").json()
    assert detail["id"] == summary["id"]
    assert detail["has_token"] is False
    assert detail["current_config_yaml"] == ""

    del_resp = client.delete(f"/api/v1/integrations/{summary['id']}")
    assert del_resp.status_code == 204
    assert client.get(f"/api/v1/integrations/{summary['id']}").status_code == 404


def test_save_and_validate_mlflow_yaml(client: TestClient) -> None:
    summary = _create_integration(client, name="M", type="mlflow", id="mm")
    ok_yaml = "tracking_uri: http://localhost:5002\n"
    save = client.put(f"/api/v1/integrations/{summary['id']}/config", json={"yaml": ok_yaml})
    assert save.status_code == 200

    valid = client.post(
        f"/api/v1/integrations/{summary['id']}/config/validate", json={"yaml": ok_yaml}
    )
    assert valid.status_code == 200
    assert valid.json()["ok"] is True

    # Unknown field is rejected by StrictBaseModel.
    bad = client.post(
        f"/api/v1/integrations/{summary['id']}/config/validate",
        json={"yaml": "bogus_field: hi\n"},
    )
    assert bad.status_code == 200
    assert bad.json()["ok"] is False


def test_token_put_never_echoes_and_has_token_flips(
    client: TestClient,
) -> None:
    summary = _create_integration(client, name="HF", type="huggingface", id="hf")
    assert summary["has_token"] is False

    put = client.put(
        f"/api/v1/integrations/{summary['id']}/token", json={"token": "hf_secret_xyz"}
    )
    assert put.status_code == 204
    assert put.content in (b"", b"null")
    # Response body MUST NOT contain the token string.
    assert b"hf_secret_xyz" not in put.content

    detail = client.get(f"/api/v1/integrations/{summary['id']}").json()
    assert detail["has_token"] is True
    # Detail payload must not leak the plaintext.
    assert "hf_secret_xyz" not in str(detail)

    delete = client.delete(f"/api/v1/integrations/{summary['id']}/token")
    assert delete.status_code == 204
    assert client.get(f"/api/v1/integrations/{summary['id']}").json()["has_token"] is False


def test_token_stored_encrypted(client: TestClient, projects_root: Path) -> None:
    summary = _create_integration(client, name="E", type="huggingface", id="enc")
    client.put(f"/api/v1/integrations/{summary['id']}/token", json={"token": "plaintext-probe"})
    blob = (projects_root / "integrations" / "enc" / "token.enc").read_bytes()
    assert b"plaintext-probe" not in blob, "token.enc must not contain plaintext"


def test_versions_listed(client: TestClient) -> None:
    summary = _create_integration(client, name="V", type="mlflow", id="vv")
    for text in ("tracking_uri: http://a\n", "tracking_uri: http://b\n", "tracking_uri: http://c\n"):
        client.put(f"/api/v1/integrations/{summary['id']}/config", json={"yaml": text})
    versions = client.get(f"/api/v1/integrations/{summary['id']}/config/versions").json()["versions"]
    assert len(versions) == 2


def test_create_unknown_type_rejected(client: TestClient) -> None:
    resp = client.post("/api/v1/integrations", json={"name": "x", "type": "bogus"})
    assert resp.status_code == 400
