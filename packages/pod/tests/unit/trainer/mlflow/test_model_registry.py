"""
Isolated unit tests for MLflowModelRegistry.
No real MLflow SDK calls — gateway and mlflow module are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.training.mlflow.model_registry import MLflowModelRegistry


def _make_registry(log_model: bool = True) -> tuple[MLflowModelRegistry, MagicMock, MagicMock]:
    """Helper: create registry with mocked gateway and mlflow module."""
    gateway = MagicMock()
    mlflow = MagicMock()
    registry = MLflowModelRegistry(gateway, mlflow, log_model_enabled=log_model)
    return registry, gateway, mlflow


class TestMLflowModelRegistryRegisterModel:
    def test_register_model_success(self):
        registry, gateway, mlflow = _make_registry()
        mlflow.register_model.return_value = MagicMock(version="1")
        version = registry.register_model("ryotenkai", run_id="run123")
        assert version == "1"
        mlflow.register_model.assert_called_once_with("runs:/run123/model", "ryotenkai")

    def test_register_model_custom_uri(self):
        registry, gateway, mlflow = _make_registry()
        mlflow.register_model.return_value = MagicMock(version="2")
        version = registry.register_model("ryotenkai", run_id="run123", model_uri="runs:/run123/adapter")
        mlflow.register_model.assert_called_once_with("runs:/run123/adapter", "ryotenkai")

    def test_register_model_with_alias(self):
        registry, gateway, mlflow = _make_registry()
        mlflow.register_model.return_value = MagicMock(version="1")
        mock_client = MagicMock()
        gateway.get_client.return_value = mock_client

        registry.register_model("ryotenkai", run_id="run123", alias="champion")
        mock_client.set_registered_model_alias.assert_called_once_with("ryotenkai", "champion", "1")

    def test_register_model_with_tags(self):
        registry, gateway, mlflow = _make_registry()
        mlflow.register_model.return_value = MagicMock(version="3")
        mock_client = MagicMock()
        gateway.get_client.return_value = mock_client

        registry.register_model("ryotenkai", run_id="run123", tags={"phase": "sft"})
        mock_client.set_model_version_tag.assert_called_once_with("ryotenkai", "3", "phase", "sft")

    def test_register_model_disabled(self):
        registry, _, _ = _make_registry(log_model=False)
        version = registry.register_model("ryotenkai", run_id="run123")
        assert version is None

    def test_register_model_no_run_id(self):
        registry, _, _ = _make_registry()
        version = registry.register_model("ryotenkai", run_id="")
        assert version is None

    def test_register_model_no_mlflow(self):
        gateway = MagicMock()
        registry = MLflowModelRegistry(gateway, None, log_model_enabled=True)
        assert registry.register_model("ryotenkai", run_id="run123") is None

    def test_register_model_exception_returns_none(self):
        registry, _, mlflow = _make_registry()
        mlflow.register_model.side_effect = Exception("MLflow error")
        version = registry.register_model("ryotenkai", run_id="run123")
        assert version is None


class TestMLflowModelRegistryAliases:
    def test_set_model_alias_success(self):
        registry, gateway, mlflow = _make_registry()
        mock_client = MagicMock()
        gateway.get_client.return_value = mock_client

        result = registry.set_model_alias("ryotenkai", "staging", 3)
        assert result is True
        mock_client.set_registered_model_alias.assert_called_once_with("ryotenkai", "staging", "3")

    def test_set_model_alias_no_mlflow(self):
        gateway = MagicMock()
        registry = MLflowModelRegistry(gateway, None)
        assert registry.set_model_alias("m", "a", 1) is False

    def test_set_model_alias_exception(self):
        registry, gateway, _ = _make_registry()
        gateway.get_client.side_effect = Exception("gateway down")
        result = registry.set_model_alias("ryotenkai", "staging", 1)
        assert result is False

    def test_get_model_by_alias_success(self):
        registry, gateway, mlflow = _make_registry()
        mock_client = MagicMock()
        gateway.get_client.return_value = mock_client
        mv = MagicMock()
        mv.version = "2"
        mv.run_id = "run-abc"
        mv.source = "runs:/abc/model"
        mv.status = "READY"
        mv.creation_timestamp = 123
        mv.last_updated_timestamp = 456
        mv.description = "desc"
        mv.tags = {}
        mock_client.get_model_version_by_alias.return_value = mv

        result = registry.get_model_by_alias("ryotenkai", "champion")
        assert result is not None
        assert result["version"] == "2"
        assert result["run_id"] == "run-abc"

    def test_get_model_by_alias_not_found(self):
        registry, gateway, _ = _make_registry()
        gateway.get_client.side_effect = Exception("not found")
        result = registry.get_model_by_alias("ryotenkai", "champion")
        assert result is None

    def test_get_model_by_alias_no_mlflow(self):
        gateway = MagicMock()
        registry = MLflowModelRegistry(gateway, None)
        assert registry.get_model_by_alias("m", "a") is None

    def test_delete_model_alias_success(self):
        registry, gateway, mlflow = _make_registry()
        mock_client = MagicMock()
        gateway.get_client.return_value = mock_client
        result = registry.delete_model_alias("ryotenkai", "staging")
        assert result is True
        mock_client.delete_registered_model_alias.assert_called_once_with("ryotenkai", "staging")

    def test_delete_model_alias_exception(self):
        registry, gateway, _ = _make_registry()
        gateway.get_client.side_effect = Exception("error")
        result = registry.delete_model_alias("ryotenkai", "staging")
        assert result is False


class TestMLflowModelRegistryPromote:
    def test_promote_model_success(self):
        registry, gateway, mlflow = _make_registry()
        mock_client = MagicMock()
        gateway.get_client.return_value = mock_client
        mv = MagicMock()
        mv.version = "3"
        mv.run_id = "r"
        mv.source = "s"
        mv.status = "READY"
        mv.creation_timestamp = 1
        mv.last_updated_timestamp = 2
        mv.description = ""
        mv.tags = {}
        mock_client.get_model_version_by_alias.return_value = mv

        result = registry.promote_model("ryotenkai", "staging", "champion")
        assert result is True

    def test_promote_model_source_alias_not_found(self):
        registry, gateway, mlflow = _make_registry()
        gateway.get_client.side_effect = Exception("not found")
        result = registry.promote_model("ryotenkai", "staging", "champion")
        assert result is False

    def test_promote_model_no_mlflow(self):
        gateway = MagicMock()
        registry = MLflowModelRegistry(gateway, None)
        result = registry.promote_model("m")
        assert result is False


class TestMLflowModelRegistryGetAliases:
    def test_get_model_aliases_success(self):
        registry, gateway, mlflow = _make_registry()
        mock_client = MagicMock()
        gateway.get_client.return_value = mock_client
        model = MagicMock()
        model.aliases = {"champion": "3", "staging": "4"}
        mock_client.get_registered_model.return_value = model

        aliases = registry.get_model_aliases("ryotenkai")
        assert aliases == {"champion": 3, "staging": 4}

    def test_get_model_aliases_empty_on_exception(self):
        registry, gateway, _ = _make_registry()
        gateway.get_client.side_effect = Exception("error")
        assert registry.get_model_aliases("ryotenkai") == {}

    def test_get_model_aliases_no_mlflow(self):
        gateway = MagicMock()
        registry = MLflowModelRegistry(gateway, None)
        assert registry.get_model_aliases("m") == {}


class TestMLflowModelRegistryLoadModel:
    def test_load_model_by_alias_success(self):
        registry, _, mlflow = _make_registry()
        mock_model = MagicMock()
        mlflow.pyfunc.load_model.return_value = mock_model

        result = registry.load_model_by_alias("ryotenkai", "champion")
        assert result is mock_model
        mlflow.pyfunc.load_model.assert_called_once_with("models:/ryotenkai@champion")

    def test_load_model_by_alias_default_is_champion(self):
        registry, _, mlflow = _make_registry()
        mlflow.pyfunc.load_model.return_value = MagicMock()
        registry.load_model_by_alias("ryotenkai")
        mlflow.pyfunc.load_model.assert_called_once_with("models:/ryotenkai@champion")

    def test_load_model_by_alias_exception(self):
        registry, _, mlflow = _make_registry()
        mlflow.pyfunc.load_model.side_effect = Exception("not found")
        result = registry.load_model_by_alias("ryotenkai", "champion")
        assert result is None

    def test_load_model_by_alias_no_mlflow(self):
        gateway = MagicMock()
        registry = MLflowModelRegistry(gateway, None)
        assert registry.load_model_by_alias("m") is None
