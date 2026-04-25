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


def test_plugin_manifest_required_env_default_empty() -> None:
    """No ``[[required_env]]`` declared → empty list, no derived secrets.

    The legacy ``[secrets]`` block was removed in schema v4; envs (secret
    or not) are now declared via ``[[required_env]]`` and the runtime
    ``_required_secrets`` ClassVar comes from the
    ``required_secret_names()`` helper.
    """
    manifest = PluginManifest.model_validate(_base_plugin_body())
    assert manifest.required_env == []
    assert manifest.required_secret_names() == ()


def test_plugin_manifest_required_secret_names_filters() -> None:
    """Only ``secret=true, optional=false`` envs end up in the runtime tuple."""
    body = _base_plugin_body()
    body["required_env"] = [
        {"name": "EVAL_REAL_SECRET", "secret": True, "optional": False},
        {"name": "EVAL_OPTIONAL_SECRET", "secret": True, "optional": True},
        {"name": "EVAL_PUBLIC_URL", "secret": False, "optional": False},
    ]
    manifest = PluginManifest.model_validate(body)
    assert manifest.required_secret_names() == ("EVAL_REAL_SECRET",)


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


def test_reports_kind_does_not_require_manifest_block() -> None:
    """Report-plugin section order lives in pipeline config, not the manifest —
    no `[reports]` block is expected on the manifest side any more."""
    body = _base_plugin_body(kind="reports")
    manifest = PluginManifest.model_validate(body)
    assert manifest.plugin.kind == "reports"
    # ui_manifest never emits `order` — it's a runtime detail assigned by the registry.
    assert "order" not in manifest.ui_manifest()


def test_reports_block_is_not_accepted_on_any_kind() -> None:
    """The old `[reports]` block is removed; passing it now fails as unknown key."""
    body = _base_plugin_body(kind="reports")
    body["reports"] = {"order": 10}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


# ---------------------------------------------------------------------------
# Schema v3 — ParamFieldSchema & JSON Schema transform.
# ---------------------------------------------------------------------------


def test_param_field_schema_required_and_default_are_mutually_exclusive() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {
        "foo": {"type": "integer", "required": True, "default": 10},
    }
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_param_field_schema_enum_requires_options() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"foo": {"type": "enum"}}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_param_field_schema_options_invalid_for_non_enum() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"foo": {"type": "integer", "options": [1, 2, 3]}}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_param_field_schema_min_greater_than_max() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"foo": {"type": "integer", "min": 10, "max": 5}}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_param_field_schema_default_out_of_range() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"foo": {"type": "integer", "min": 0, "max": 10, "default": 100}}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_param_field_schema_default_not_in_enum() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {
        "mode": {"type": "enum", "options": ["a", "b"], "default": "c"},
    }
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_param_field_schema_min_max_only_for_numeric() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"foo": {"type": "string", "min": 0}}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_param_field_schema_secret_only_for_string() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"foo": {"type": "integer", "secret": True}}
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_ui_manifest_emits_json_schema_for_params() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {
        "sample_size": {
            "type": "integer",
            "min": 1,
            "max": 1000,
            "default": 10,
            "title": "Sample size",
            "description": "Rows drawn for the check.",
        },
        "mode": {
            "type": "enum",
            "options": ["compile", "semantic_only"],
            "default": "compile",
        },
        "must_set": {"type": "string", "required": True},
    }
    ui = PluginManifest.model_validate(body).ui_manifest()
    js = ui["params_schema"]
    assert js["type"] == "object"
    assert js["additionalProperties"] is False
    assert js["required"] == ["must_set"]  # only required=True fields
    assert js["properties"]["sample_size"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 1000,
        "default": 10,
        "title": "Sample size",
        "description": "Rows drawn for the check.",
    }
    assert js["properties"]["mode"] == {
        "type": "string",
        "enum": ["compile", "semantic_only"],
        "default": "compile",
    }
    assert js["properties"]["must_set"] == {"type": "string"}


def test_ui_manifest_marks_secret_fields() -> None:
    body = _base_plugin_body()
    body["params_schema"] = {"api_key": {"type": "string", "secret": True, "required": True}}
    ui = PluginManifest.model_validate(body).ui_manifest()
    assert ui["params_schema"]["properties"]["api_key"]["x-secret"] is True


# ---------------------------------------------------------------------------
# supported_strategies — reward-only, required.
# ---------------------------------------------------------------------------


def test_reward_plugin_requires_supported_strategies() -> None:
    body = _base_plugin_body(kind="reward")
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)


def test_reward_plugin_accepts_supported_strategies() -> None:
    body = _base_plugin_body(kind="reward", supported_strategies=["grpo", "sapo"])
    manifest = PluginManifest.model_validate(body)
    assert manifest.plugin.supported_strategies == ["grpo", "sapo"]
    assert manifest.ui_manifest()["supported_strategies"] == ["grpo", "sapo"]


def test_validation_plugin_rejects_supported_strategies() -> None:
    body = _base_plugin_body(kind="validation", supported_strategies=["grpo"])
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(body)
