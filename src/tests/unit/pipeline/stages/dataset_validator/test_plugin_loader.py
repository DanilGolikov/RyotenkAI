"""Unit tests for src.pipeline.stages.dataset_validator.plugin_loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.dataset_validator.plugin_loader import PluginLoader
from src.utils.config import DatasetConfig, PipelineConfig

pytestmark = pytest.mark.unit


def _local_ds(*, plugins: list[dict] | None = None) -> DatasetConfig:
    return DatasetConfig(
        source_type="local",
        source_local={"local_paths": {"train": "data/train.jsonl", "eval": None}},
        validations={"plugins": plugins or [], "mode": "fast", "critical_failures": 1},
    )


def _mk_primary_only_config(ds: DatasetConfig) -> MagicMock:
    cfg = MagicMock(spec=PipelineConfig)
    cfg.get_primary_dataset.return_value = ds
    cfg.training = MagicMock()
    cfg.training.strategies = []
    return cfg


@patch("src.pipeline.stages.dataset_validator.plugin_loader.validation_registry")
def test_load_for_dataset_empty_plugins_returns_empty_list(mock_registry):
    """Empty plugins config → no plugins run (no hidden defaults)."""
    ds = _local_ds(plugins=[])
    cfg = _mk_primary_only_config(ds)

    loader = PluginLoader(config=cfg, secrets=None)
    plugins = loader.load_for_dataset(ds)

    assert plugins == []
    mock_registry.instantiate.assert_not_called()


@patch("src.pipeline.stages.dataset_validator.plugin_loader.validation_registry")
def test_load_for_dataset_no_validations_attr_returns_empty_list(mock_registry):
    """Dataset with no validations attr → no plugins run (no hidden defaults)."""
    cfg = _mk_primary_only_config(_local_ds(plugins=[]))
    bare_ds = MagicMock(spec=[])  # no validations attr

    loader = PluginLoader(config=cfg, secrets=None)
    plugins = loader.load_for_dataset(bare_ds)

    assert plugins == []
    mock_registry.instantiate.assert_not_called()


@patch("src.pipeline.stages.dataset_validator.plugin_loader.validation_registry")
def test_load_for_dataset_uses_configured_plugins(mock_registry):
    """Explicit plugins config → exactly those plugins instantiated."""
    ds = _local_ds(
        plugins=[
            {"id": "custom_main", "plugin": "custom_plugin", "params": {"threshold": 100}, "apply_to": ["train"]},
            {"id": "another_main", "plugin": "another_plugin", "params": {}, "apply_to": ["train"]},
        ]
    )
    cfg = _mk_primary_only_config(ds)
    mock_registry.instantiate.return_value = MagicMock(name="x")

    loader = PluginLoader(config=cfg, secrets=None)
    plugins = loader.load_for_dataset(ds)

    assert len(plugins) == 2
    assert mock_registry.instantiate.call_count == 2


@patch("src.pipeline.stages.dataset_validator.plugin_loader.validation_registry")
def test_load_for_dataset_per_dataset_resolution(mock_registry):
    """load_for_dataset uses the dataset's own validations.plugins, not config defaults."""
    cfg = _mk_primary_only_config(_local_ds(plugins=[]))
    mock_registry.instantiate.return_value = MagicMock(name="x")

    other_ds = _local_ds(plugins=[{"id": "p", "plugin": "p", "params": {}, "apply_to": ["train"]}])

    loader = PluginLoader(config=cfg, secrets=None)
    plugins = loader.load_for_dataset(other_ds)

    assert len(plugins) == 1
    plugin_id, plugin_name, _plugin_instance, apply_to = plugins[0]
    assert plugin_id == "p"
    assert plugin_name == "p"
    assert apply_to == {"train"}


@patch("src.pipeline.stages.dataset_validator.plugin_loader.validation_registry")
def test_load_raises_keyerror_for_unknown_plugin(mock_registry):
    """Registry KeyError propagates with available-plugins log."""
    ds = _local_ds(plugins=[{"id": "x", "plugin": "missing_plugin", "params": {}, "apply_to": ["train"]}])
    cfg = _mk_primary_only_config(ds)
    mock_registry.instantiate.side_effect = KeyError("missing_plugin")
    mock_registry.list_ids.return_value = ["a", "b", "c"]

    loader = PluginLoader(config=cfg, secrets=None)
    with pytest.raises(KeyError):
        loader.load_for_dataset(ds)


@patch("src.pipeline.stages.dataset_validator.plugin_loader.validation_registry")
def test_secrets_resolver_built_when_secrets_provided(mock_registry):
    """When secrets is non-None, resolver is constructed and passed to instantiate()."""
    ds = _local_ds(plugins=[{"id": "p", "plugin": "p", "params": {}, "apply_to": ["train"]}])
    cfg = _mk_primary_only_config(ds)
    mock_registry.instantiate.return_value = MagicMock(name="x")

    secrets = MagicMock()
    loader = PluginLoader(config=cfg, secrets=secrets)
    loader.load_for_dataset(ds)

    _args, kwargs = mock_registry.instantiate.call_args
    assert kwargs["resolver"] is not None


@patch("src.pipeline.stages.dataset_validator.plugin_loader.validation_registry")
def test_secrets_resolver_is_none_when_no_secrets(mock_registry):
    """When secrets is None, resolver passed to instantiate() is None too."""
    ds = _local_ds(plugins=[{"id": "p", "plugin": "p", "params": {}, "apply_to": ["train"]}])
    cfg = _mk_primary_only_config(ds)
    mock_registry.instantiate.return_value = MagicMock(name="x")

    loader = PluginLoader(config=cfg, secrets=None)
    loader.load_for_dataset(ds)

    _args, kwargs = mock_registry.instantiate.call_args
    assert kwargs["resolver"] is None
