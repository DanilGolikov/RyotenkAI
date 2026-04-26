from __future__ import annotations

from pathlib import Path

import pytest

from src.workspace.integrations import IntegrationStore, IntegrationStoreError


def test_create_and_load(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "hf-prod")
    assert not store.exists()

    metadata = store.create(id="hf-prod", name="HF prod", type="huggingface")
    assert store.exists()
    assert metadata.id == "hf-prod"

    reloaded = store.load()
    assert reloaded.name == "HF prod"
    assert reloaded.type == "huggingface"


def test_save_config_snapshots_previous_non_empty(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "mlflow-stg")
    store.create(id="mlflow-stg", name="MLflow stg", type="mlflow")

    # First save from empty — no snapshot of empty text.
    first = store.save_config("tracking_uri: http://a\n")
    assert first is None

    # Second save snapshots the previous.
    second = store.save_config("tracking_uri: http://b\n")
    assert second is not None
    assert store.current_yaml_text().startswith("tracking_uri: http://b")

    versions = store.list_versions()
    assert len(versions) == 1
    assert versions[0].filename == second


def test_list_versions_sorted_desc(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "m")
    store.create(id="m", name="M", type="mlflow")
    for text in ("a", "b", "c"):
        store.save_config(text + "\n")
    versions = store.list_versions()
    filenames = [v.filename for v in versions]
    assert filenames == sorted(filenames, reverse=True)


def test_read_version_rejects_traversal(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "t")
    store.create(id="t", name="T", type="mlflow")
    with pytest.raises(IntegrationStoreError):
        store.read_version("../etc/passwd")


def test_restore_version_roundtrip(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "rt")
    store.create(id="rt", name="RT", type="mlflow")
    store.save_config("v1\n")
    store.save_config("v2\n")

    versions = store.list_versions()
    target = versions[-1].filename  # oldest — contains "v1"
    store.restore_version(target)
    assert store.current_yaml_text() == "v1\n"


def test_has_token_reflects_token_file(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "tk")
    store.create(id="tk", name="T", type="huggingface")
    assert store.has_token() is False
    store.token_path.write_bytes(b"encrypted-bytes")
    assert store.has_token() is True
