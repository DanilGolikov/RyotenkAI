"""Unit tests for community manifest pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.community.manifest import PluginManifest, PresetManifest


def _base_plugin_body(**overrides):
    body = {
        "plugin": {
            "id": "my_plugin",
            "kind": "validation",
            "version": "1.0.0",
            "priority": 50,
            "entry_point": {"module": "plugin", "class": "MyPlugin"},
        }
    }
    body["plugin"].update(overrides)
    return body


def test_plugin_manifest_minimum_fields() -> None:
    manifest = PluginManifest.model_validate(_base_plugin_body())
    assert manifest.plugin.id == "my_plugin"
    assert manifest.plugin.name == "my_plugin"  # defaulted to id
    assert manifest.plugin.entry_point.class_name == "MyPlugin"


def test_plugin_manifest_rejects_suggested_keys_outside_schema() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"known": {"type": "integer"}}
    body["suggested_params"] = {"unknown": 1}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_plugin_manifest_secrets_default_empty() -> None:
    manifest = PluginManifest.model_validate(_base_plugin_body())
    assert manifest.secrets.required == []


def test_plugin_manifest_ui_manifest_shape() -> None:
    body = _base_plugin_body(description="desc", category="basic")
    body["thresholds_schema"] = {"threshold": {"type": "integer", "default": 10}}
    body["suggested_thresholds"] = {"threshold": 10}
    manifest = PluginManifest.model_validate(body)
    ui = manifest.ui_manifest()
    assert ui["id"] == "my_plugin"
    assert ui["description"] == "desc"
    assert ui["category"] == "basic"
    assert ui["suggested_thresholds"] == {"threshold": 10}


def test_preset_manifest_fills_name_from_id() -> None:
    body = {
        "preset": {
            "id": "starter",
            "entry_point": {"file": "preset.yaml"},
        }
    }
    manifest = PresetManifest.model_validate(body)
    assert manifest.preset.name == "starter"
    assert manifest.preset.size_tier == ""


def test_plugin_manifest_rejects_unknown_top_level_key() -> None:
    body = _base_plugin_body()
    body["unknown"] = 1
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)
