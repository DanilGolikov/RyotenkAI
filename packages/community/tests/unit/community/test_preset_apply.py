"""Unit tests for :mod:`src.community.preset_apply`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.community.loader import LoadedPreset, load_presets
from src.community.manifest import PresetManifest
from src.community.preset_apply import apply_preset


def _build_preset(
    root: Path,
    *,
    manifest_toml: str,
    preset_yaml: str,
    preset_id: str = "demo",
) -> LoadedPreset:
    """Write a preset into ``root/presets/<id>/`` and load it via the real loader."""
    preset_dir = root / "presets" / preset_id
    preset_dir.mkdir(parents=True)
    (preset_dir / "manifest.toml").write_text(manifest_toml)
    (preset_dir / "preset.yaml").write_text(preset_yaml)
    (loaded,) = load_presets(root=root)
    assert loaded.manifest.preset.id == preset_id
    return loaded


_SCOPED_MANIFEST = textwrap.dedent('''
    [preset]
    id = "demo"
    name = "Demo"
    description = "."
    size_tier = "small"
    version = "0.1.0"

    [preset.entry_point]
    file = "preset.yaml"

    [preset.scope]
    replaces  = ["model", "training"]
    preserves = ["datasets", "providers"]
''')


# ---------------------------------------------------------------------------
# Scope-aware merge (v2)
# ---------------------------------------------------------------------------


def test_scope_replaces_only_declared_keys(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_SCOPED_MANIFEST,
        preset_yaml="model:\n  name: new-model\ntraining:\n  type: qlora\n",
    )
    user = {
        "model": {"name": "old-model"},
        "training": {"type": "sft"},
        "datasets": {"default": {"source_type": "local", "source_local": {"local_paths": {"train": "./my.jsonl"}}}},
        "providers": {"my_provider": {"kind": "runpod"}},
    }
    preview = apply_preset(user, loaded)

    # model and training got overwritten by preset
    assert preview.resulting_config["model"] == {"name": "new-model"}
    assert preview.resulting_config["training"] == {"type": "qlora"}
    # datasets and providers are preserved unchanged
    assert preview.resulting_config["datasets"] == user["datasets"]
    assert preview.resulting_config["providers"] == user["providers"]
    # No v1 warning when scope is declared
    assert preview.warnings == []


def test_preserved_key_from_preset_yaml_is_ignored(tmp_path: Path) -> None:
    """If the preset YAML includes a ``preserves`` key (e.g. for YAML
    completeness), that content is ignored at apply time."""
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_SCOPED_MANIFEST,
        preset_yaml=textwrap.dedent("""
            model: {name: new-model}
            training: {type: qlora}
            datasets: {placeholder: true}        # in preserves → ignored
            providers: {}                         # in preserves → ignored
        """),
    )
    user = {"datasets": {"default": {"source_type": "local"}}, "providers": {"mine": {"kind": "x"}}}
    preview = apply_preset(user, loaded)

    assert preview.resulting_config["datasets"] == user["datasets"]
    assert preview.resulting_config["providers"] == user["providers"]


def test_diff_reports_reason_per_key(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_SCOPED_MANIFEST,
        preset_yaml="model: {name: new}\ntraining: {type: qlora}\n",
    )
    user = {
        "model": {"name": "old"},
        "datasets": {"default": {"x": 1}},
    }
    preview = apply_preset(user, loaded)

    diff_by_key = {d.key: d for d in preview.diff}
    assert diff_by_key["model"].kind == "changed"
    assert diff_by_key["model"].reason == "preset_replaced"
    assert diff_by_key["training"].kind == "added"
    assert diff_by_key["training"].reason == "preset_added"
    assert diff_by_key["datasets"].kind == "unchanged"
    assert diff_by_key["datasets"].reason == "preset_preserved"


def test_replaces_key_absent_from_yaml_removes_user_value(tmp_path: Path) -> None:
    """If ``scope.replaces`` lists a key but the preset YAML omits it,
    the user's value for that key is dropped — the preset is still
    authoritative (it just chose emptiness)."""
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_SCOPED_MANIFEST,
        preset_yaml="model: {name: new}\n",   # no `training:` block
    )
    user = {"model": {"name": "old"}, "training": {"type": "sft"}}
    preview = apply_preset(user, loaded)
    assert "training" not in preview.resulting_config
    diff_by_key = {d.key: d for d in preview.diff}
    assert diff_by_key["training"].kind == "removed"


# ---------------------------------------------------------------------------
# Backward compatibility — no [preset.scope] block at all
# ---------------------------------------------------------------------------


def test_no_scope_falls_back_to_full_replace(tmp_path: Path) -> None:
    manifest = textwrap.dedent('''
        [preset]
        id = "legacy"
        name = "Legacy"
        description = "."
        size_tier = "small"
        version = "0.1.0"

        [preset.entry_point]
        file = "preset.yaml"
    ''')
    loaded = _build_preset(
        tmp_path,
        manifest_toml=manifest,
        preset_yaml="model: {name: only-this}\n",
        preset_id="legacy",
    )
    user = {"model": {"name": "old"}, "datasets": {"d": 1}, "providers": {"p": 1}}
    preview = apply_preset(user, loaded)

    # v1 semantics: preset YAML becomes the whole config; datasets/providers lost
    assert preview.resulting_config == {"model": {"name": "only-this"}}
    assert any("full overwrite" in w for w in preview.warnings)


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


_REQS_MANIFEST = textwrap.dedent('''
    [preset]
    id = "reqs"
    name = "Reqs"
    description = "."
    size_tier = "large"
    version = "0.1.0"

    [preset.entry_point]
    file = "preset.yaml"

    [preset.scope]
    replaces = ["model"]

    [preset.requirements]
    hub_models       = ["gated/model"]
    provider_kind    = ["runpod"]
    required_plugins = ["validation:min_samples"]
    min_vram_gb      = 80
''')


def test_requirements_hf_token_missing(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_REQS_MANIFEST,
        preset_yaml="model: {name: x}\n",
        preset_id="reqs",
    )
    preview = apply_preset({}, loaded, secrets_model_extra={})
    hf_check = next(r for r in preview.requirements if r.label == "HF Hub access")
    assert hf_check.status == "warning"
    assert "HF_TOKEN" in hf_check.detail


def test_requirements_hf_token_present(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_REQS_MANIFEST,
        preset_yaml="model: {name: x}\n",
        preset_id="reqs",
    )
    preview = apply_preset({}, loaded, secrets_model_extra={"hf_token": "hf_xxx"})
    hf_check = next(r for r in preview.requirements if r.label == "HF Hub access")
    assert hf_check.status == "ok"


def test_requirements_provider_kind_matches(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_REQS_MANIFEST,
        preset_yaml="model: {name: x}\n",
        preset_id="reqs",
    )
    user = {"providers": {"cloud_A": {"kind": "runpod"}}}
    preview = apply_preset(user, loaded)
    provider_check = next(r for r in preview.requirements if r.label == "Provider kind")
    assert provider_check.status == "ok"
    assert "runpod" in provider_check.detail


def test_requirements_provider_kind_mismatch(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_REQS_MANIFEST,
        preset_yaml="model: {name: x}\n",
        preset_id="reqs",
    )
    user = {"providers": {"local": {"kind": "single_node"}}}
    preview = apply_preset(user, loaded)
    provider_check = next(r for r in preview.requirements if r.label == "Provider kind")
    assert provider_check.status == "warning"
    assert "single_node" in provider_check.detail


def test_requirements_required_plugins_missing(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_REQS_MANIFEST,
        preset_yaml="model: {name: x}\n",
        preset_id="reqs",
    )
    preview = apply_preset({}, loaded, available_plugin_ids_by_kind={"validation": set()})
    plugin_check = next(r for r in preview.requirements if r.label == "Required plugins")
    assert plugin_check.status == "missing"
    assert "validation:min_samples" in plugin_check.detail


def test_requirements_required_plugins_present(tmp_path: Path) -> None:
    loaded = _build_preset(
        tmp_path,
        manifest_toml=_REQS_MANIFEST,
        preset_yaml="model: {name: x}\n",
        preset_id="reqs",
    )
    preview = apply_preset(
        {}, loaded,
        available_plugin_ids_by_kind={"validation": {"min_samples"}},
    )
    plugin_check = next(r for r in preview.requirements if r.label == "Required plugins")
    assert plugin_check.status == "ok"


def test_placeholders_surfaced(tmp_path: Path) -> None:
    manifest = textwrap.dedent('''
        [preset]
        id = "ph"
        name = "PH"
        description = "."
        size_tier = "small"
        version = "0.1.0"

        [preset.entry_point]
        file = "preset.yaml"

        [preset.scope]
        replaces = ["model"]

        [preset.placeholders]
        "datasets.default.source_local.local_paths.train" = "Your JSONL path"
    ''')
    loaded = _build_preset(
        tmp_path, manifest_toml=manifest, preset_yaml="model: {}\n", preset_id="ph",
    )
    preview = apply_preset({}, loaded)
    assert len(preview.placeholders) == 1
    assert preview.placeholders[0].path == "datasets.default.source_local.local_paths.train"
    assert "JSONL" in preview.placeholders[0].hint


# ---------------------------------------------------------------------------
# Manifest-level validation
# ---------------------------------------------------------------------------


def test_manifest_rejects_overlapping_scope_sets() -> None:
    with pytest.raises(ValidationError):
        PresetManifest.model_validate({
            "preset": {
                "id": "bad",
                "entry_point": {"file": "preset.yaml"},
                "scope": {"replaces": ["model"], "preserves": ["model"]},
            },
        })
