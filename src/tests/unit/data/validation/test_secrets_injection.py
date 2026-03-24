"""
Tests for DTST_* secret injection in DatasetValidator.

Verifies that:
- Plugins decorated with @requires_secrets("DTST_*") get secrets injected
- Plugins without @requires_secrets work as before
- Missing secrets raises RuntimeError
- Missing Secrets model when plugin needs secrets raises RuntimeError
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry
from src.data.validation.secrets import requires_secrets
from src.pipeline.stages.dataset_validator import DatasetValidator
from src.utils.config import DatasetConfig, PipelineConfig


def _mk_config(ds: DatasetConfig) -> MagicMock:
    cfg = MagicMock(spec=PipelineConfig)
    cfg.get_primary_dataset.return_value = ds
    cfg.training = MagicMock()
    cfg.training.strategies = []
    return cfg


def _local_ds(path: str, *, plugins: list[dict], critical_failures: int = 0) -> DatasetConfig:
    return DatasetConfig(
        source_type="local",
        source_local={"local_paths": {"train": path, "eval": None}},
        validations={"plugins": plugins, "mode": "fast", "critical_failures": critical_failures},
    )


def _make_secrets_mock(extra: dict[str, str]) -> MagicMock:
    s = MagicMock()
    s.model_extra = extra
    return s


class _PlainPlugin(ValidationPlugin):
    """Plugin that does NOT require secrets."""

    name = "test_plain"
    priority = 50

    @classmethod
    def get_description(cls) -> str:
        return "Plain plugin for testing"

    def validate(self, dataset) -> ValidationResult:
        return ValidationResult(
            plugin_name=self.name, passed=True, params={}, thresholds={},
            metrics={"ok": 1.0}, warnings=[], errors=[], execution_time_ms=0.0,
        )

    def get_recommendations(self, result):
        return []


@requires_secrets("DTST_TEST_API_KEY")
class _SecretPlugin(ValidationPlugin):
    """Plugin that REQUIRES a DTST_* secret."""

    name = "test_secret"
    priority = 50
    _secrets: dict[str, str]

    @classmethod
    def get_description(cls) -> str:
        return "Secret-requiring plugin for testing"

    def validate(self, dataset) -> ValidationResult:
        _key = self._secrets["DTST_TEST_API_KEY"]
        return ValidationResult(
            plugin_name=self.name, passed=True, params={}, thresholds={},
            metrics={"ok": 1.0}, warnings=[], errors=[], execution_time_ms=0.0,
        )

    def get_recommendations(self, result):
        return []


@requires_secrets("DTST_KEY_A", "DTST_KEY_B")
class _MultiSecretPlugin(ValidationPlugin):
    """Plugin that requires multiple DTST_* secrets."""

    name = "test_multi_secret"
    priority = 50
    _secrets: dict[str, str]

    @classmethod
    def get_description(cls) -> str:
        return "Multi-secret plugin for testing"

    def validate(self, dataset) -> ValidationResult:
        return ValidationResult(
            plugin_name=self.name, passed=True, params={}, thresholds={},
            metrics={"ok": 1.0}, warnings=[], errors=[], execution_time_ms=0.0,
        )

    def get_recommendations(self, result):
        return []


@pytest.fixture(autouse=True)
def _register_test_plugins():
    """Register test plugins before each test and clean up after."""
    ValidationPluginRegistry.register(_PlainPlugin)
    ValidationPluginRegistry.register(_SecretPlugin)
    ValidationPluginRegistry.register(_MultiSecretPlugin)
    yield
    ValidationPluginRegistry.clear()


class TestSecretsInjection:
    @patch("src.pipeline.stages.dataset_validator.DatasetLoaderFactory")
    def test_plugin_without_secrets_works_normally(self, _loader_factory):
        """Plain plugins instantiate fine without any secrets."""
        ds = _local_ds("data/train.jsonl", plugins=[
            {"id": "p1", "plugin": "test_plain", "apply_to": ["train"]},
        ])
        cfg = _mk_config(ds)
        validator = DatasetValidator(cfg)

        assert len(validator._plugins) == 1
        _pid, _pname, plugin, _apply_to = validator._plugins[0]
        assert not hasattr(plugin, "_secrets")

    @patch("src.pipeline.stages.dataset_validator.DatasetLoaderFactory")
    def test_plugin_with_secrets_gets_injected(self, _loader_factory):
        """Plugins decorated with @requires_secrets get _secrets injected."""
        secrets = _make_secrets_mock({"dtst_test_api_key": "test-token-123"})
        ds = _local_ds("data/train.jsonl", plugins=[
            {"id": "s1", "plugin": "test_secret", "apply_to": ["train"]},
        ])
        cfg = _mk_config(ds)
        validator = DatasetValidator(cfg, secrets=secrets)

        assert len(validator._plugins) == 1
        _pid, _pname, plugin, _apply_to = validator._plugins[0]
        assert hasattr(plugin, "_secrets")
        assert plugin._secrets == {"DTST_TEST_API_KEY": "test-token-123"}

    @patch("src.pipeline.stages.dataset_validator.DatasetLoaderFactory")
    def test_plugin_with_multiple_secrets_all_injected(self, _loader_factory):
        """Plugin requiring multiple secrets gets all of them injected."""
        secrets = _make_secrets_mock({
            "dtst_key_a": "value-a",
            "dtst_key_b": "value-b",
        })
        ds = _local_ds("data/train.jsonl", plugins=[
            {"id": "m1", "plugin": "test_multi_secret", "apply_to": ["train"]},
        ])
        cfg = _mk_config(ds)
        validator = DatasetValidator(cfg, secrets=secrets)

        assert len(validator._plugins) == 1
        _pid, _pname, plugin, _apply_to = validator._plugins[0]
        assert plugin._secrets == {"DTST_KEY_A": "value-a", "DTST_KEY_B": "value-b"}

    @patch("src.pipeline.stages.dataset_validator.DatasetLoaderFactory")
    def test_plugin_needs_secrets_but_no_secrets_model_raises(self, _loader_factory):
        """Plugin needs secrets, but DatasetValidator has no Secrets model → RuntimeError."""
        ds = _local_ds("data/train.jsonl", plugins=[
            {"id": "s1", "plugin": "test_secret", "apply_to": ["train"]},
        ])
        cfg = _mk_config(ds)

        with pytest.raises(RuntimeError, match="requires secrets"):
            DatasetValidator(cfg)  # no secrets=...

    @patch("src.pipeline.stages.dataset_validator.DatasetLoaderFactory")
    def test_plugin_needs_secrets_but_key_missing_in_env_raises(self, _loader_factory):
        """Plugin needs DTST_TEST_API_KEY but it's not in secrets.env → RuntimeError."""
        secrets = _make_secrets_mock({})  # empty
        ds = _local_ds("data/train.jsonl", plugins=[
            {"id": "s1", "plugin": "test_secret", "apply_to": ["train"]},
        ])
        cfg = _mk_config(ds)

        with pytest.raises(RuntimeError, match="DTST_TEST_API_KEY"):
            DatasetValidator(cfg, secrets=secrets)

    @patch("src.pipeline.stages.dataset_validator.DatasetLoaderFactory")
    def test_mixed_plugins_plain_and_secret(self, _loader_factory):
        """Both plain and secret-requiring plugins coexist correctly."""
        secrets = _make_secrets_mock({"dtst_test_api_key": "tok"})
        ds = _local_ds("data/train.jsonl", plugins=[
            {"id": "p1", "plugin": "test_plain", "apply_to": ["train"]},
            {"id": "s1", "plugin": "test_secret", "apply_to": ["train"]},
        ])
        cfg = _mk_config(ds)
        validator = DatasetValidator(cfg, secrets=secrets)

        assert len(validator._plugins) == 2
        plain = next(p for _, _, p, _ in validator._plugins if p.name == "test_plain")
        secret = next(p for _, _, p, _ in validator._plugins if p.name == "test_secret")

        assert not hasattr(plain, "_secrets")
        assert secret._secrets == {"DTST_TEST_API_KEY": "tok"}
