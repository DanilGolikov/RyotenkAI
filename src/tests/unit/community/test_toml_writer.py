"""Unit tests for :mod:`src.community.toml_writer`."""

from __future__ import annotations

import tomllib

from src.community.toml_writer import dump_manifest_toml


def test_plugin_manifest_round_trip() -> None:
    manifest = {
        "plugin": {
            "id": "demo",
            "kind": "validation",
            "name": "Demo",
            "version": "0.1.0",
            "description": "hi",
            "entry_point": {"module": "plugin", "class": "DemoPlugin"},
        },
        "params_schema": {
            "threshold": {"type": "integer", "default": 10, "min": 1},
        },
        "suggested_params": {"threshold": 10},
    }
    text = dump_manifest_toml(manifest)
    parsed = tomllib.loads(text)
    assert parsed == manifest


def test_plugin_section_order_is_stable() -> None:
    """``[plugin]`` scalars follow a fixed order (id → kind → name → version …)."""
    manifest = {
        "plugin": {
            "description": "x",
            "version": "0.1.0",
            "category": "basic",
            "name": "Name",
            "kind": "evaluation",
            "id": "x",
            "entry_point": {"module": "plugin", "class": "X"},
        }
    }
    text = dump_manifest_toml(manifest)
    lines = [line for line in text.splitlines() if "=" in line and not line.startswith("[")]
    # `id`, `kind`, `name`, `version`, `category` must appear in that fixed order.
    idx = {key: i for i, line in enumerate(lines) for key in [line.split("=", 1)[0].strip()]}
    assert idx["id"] < idx["kind"] < idx["name"] < idx["version"] < idx["category"]


def test_reports_block_renders_after_plugin() -> None:
    """Report manifests get a dedicated [reports] section with `order`."""
    manifest = {
        "plugin": {
            "id": "header",
            "kind": "reports",
            "entry_point": {"module": "plugin", "class": "HeaderPlugin"},
        },
        "reports": {"order": 10},
    }
    text = dump_manifest_toml(manifest)
    assert "[reports]" in text
    assert "order = 10" in text
    # reports section should land after the plugin section.
    assert text.index("[plugin]") < text.index("[reports]")


def test_nested_tables_never_inline() -> None:
    """A nested dict is always emitted as ``[section.sub]``, never inline."""
    manifest = {
        "plugin": {
            "id": "x",
            "kind": "validation",
            "entry_point": {"module": "plugin", "class": "X"},
        },
        "params_schema": {
            "a": {"type": "integer", "default": 1},
            "b": {"type": "string", "default": "hi"},
        },
    }
    text = dump_manifest_toml(manifest)
    # params_schema itself has no scalars → no bare "[params_schema]" header.
    assert "\n[params_schema]\n" not in text
    assert "[params_schema.a]" in text
    assert "[params_schema.b]" in text


def test_todo_marker_emitted() -> None:
    manifest = {
        "plugin": {
            "id": "x",
            "kind": "validation",
            "category": "",
            "entry_point": {"module": "plugin", "class": "X"},
        }
    }
    text = dump_manifest_toml(manifest, todo_fields={"plugin.category"})
    assert 'category = ""  # TODO: fill in' in text


def test_preset_default_section_order() -> None:
    manifest = {
        "preset": {
            "id": "x",
            "name": "x",
            "version": "0.1.0",
            "description": "",
            "size_tier": "",
            "entry_point": {"file": "preset.yaml"},
        }
    }
    text = dump_manifest_toml(manifest)
    assert text.index("[preset]") < text.index("[preset.entry_point]")
    parsed = tomllib.loads(text)
    assert parsed == manifest


def test_list_of_scalars() -> None:
    manifest = {
        "plugin": {
            "id": "x",
            "kind": "validation",
            "entry_point": {"module": "plugin", "class": "X"},
        },
        "secrets": {"required": ["EVAL_A", "EVAL_B"]},
    }
    text = dump_manifest_toml(manifest)
    assert 'required = ["EVAL_A", "EVAL_B"]' in text
    assert tomllib.loads(text)["secrets"]["required"] == ["EVAL_A", "EVAL_B"]
