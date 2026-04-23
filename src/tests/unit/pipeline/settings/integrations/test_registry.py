from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.settings.integrations import (
    IntegrationRegistry,
    IntegrationRegistryError,
)


def test_list_empty_when_fresh(tmp_path: Path) -> None:
    registry = IntegrationRegistry(tmp_path)
    assert registry.list() == []


def test_register_and_resolve(tmp_path: Path) -> None:
    registry = IntegrationRegistry(tmp_path)
    registry.register(
        integration_id="mlflow-prod",
        name="MLflow prod",
        type="mlflow",
        path=tmp_path / "integrations" / "mlflow-prod",
    )
    entry = registry.resolve("mlflow-prod")
    assert entry.id == "mlflow-prod"
    assert entry.type == "mlflow"


def test_duplicate_id_rejected(tmp_path: Path) -> None:
    registry = IntegrationRegistry(tmp_path)
    registry.register(
        integration_id="m",
        name="M",
        type="mlflow",
        path=tmp_path / "integrations" / "m",
    )
    with pytest.raises(IntegrationRegistryError):
        registry.register(
            integration_id="m",
            name="M2",
            type="mlflow",
            path=tmp_path / "integrations" / "m2",
        )


def test_invalid_id_rejected(tmp_path: Path) -> None:
    registry = IntegrationRegistry(tmp_path)
    with pytest.raises(IntegrationRegistryError):
        registry.register(
            integration_id="bad id with spaces",
            name="X",
            type="mlflow",
            path=tmp_path / "x",
        )


def test_unregister(tmp_path: Path) -> None:
    registry = IntegrationRegistry(tmp_path)
    registry.register(
        integration_id="u",
        name="U",
        type="huggingface",
        path=tmp_path / "integrations" / "u",
    )
    assert registry.unregister("u") is True
    assert registry.unregister("u") is False
    with pytest.raises(IntegrationRegistryError):
        registry.resolve("u")


def test_resolve_missing(tmp_path: Path) -> None:
    registry = IntegrationRegistry(tmp_path)
    with pytest.raises(IntegrationRegistryError):
        registry.resolve("ghost")
