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
    """Plain list-of-strings inside a section round-trips."""
    manifest = {
        "plugin": {
            "id": "x",
            "kind": "reward",
            "supported_strategies": ["grpo", "sapo"],
            "entry_point": {"module": "plugin", "class": "X"},
        },
    }
    text = dump_manifest_toml(manifest)
    assert 'supported_strategies = ["grpo", "sapo"]' in text


def test_array_of_tables_for_required_env() -> None:
    """``required_env`` renders as ``[[required_env]]`` blocks in a fixed
    field order so hand-edited diffs stay stable."""
    manifest = {
        "plugin": {
            "id": "x",
            "kind": "evaluation",
            "entry_point": {"module": "plugin", "class": "X"},
        },
        "required_env": [
            {
                "name": "EVAL_KEY_A",
                "description": "first key",
                "optional": False,
                "secret": True,
                "managed_by": "",
            },
            {
                "name": "EVAL_KEY_B",
                "description": "second key",
                "optional": True,
                "secret": False,
                "managed_by": "integrations",
            },
        ],
    }
    text = dump_manifest_toml(manifest)
    # Two AOT blocks, in the same order they appeared in the input.
    assert text.count("[[required_env]]") == 2
    a_idx = text.index('name = "EVAL_KEY_A"')
    b_idx = text.index('name = "EVAL_KEY_B"')
    assert a_idx < b_idx
    # name comes before description per _REQUIRED_ENV_FIELD_ORDER.
    assert text.index("name", a_idx) < text.index("description", a_idx)
    # Round-trips through tomllib unchanged.
    parsed = tomllib.loads(text)
    assert parsed["required_env"] == manifest["required_env"]
