"""Generate fresh ``manifest.toml`` files for plugins and presets.

Called from ``community scaffold <path>`` CLI. Uses AST-only inference
(:mod:`src.community.inference`) — never executes the plugin's own code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.community.inference import InferredField, InferredPlugin, infer_plugin
from src.community.manifest import PluginManifest, PresetManifest
from src.community.toml_writer import dump_manifest_toml

_PLUGIN_TODO_FIELDS = frozenset(
    {
        "plugin.category",
        "plugin.stability",
        "compat.min_core_version",
        "reports.order",
    }
)

_PRESET_TODO_FIELDS = frozenset(
    {
        "preset.description",
        "preset.size_tier",
    }
)


def _field_to_dict(field: InferredField) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": field.type}
    if field.default is not None:
        entry["default"] = field.default
    return entry


def _schema_from_inferred(inferred: dict[str, InferredField]) -> dict[str, dict[str, Any]]:
    return {key: _field_to_dict(value) for key, value in inferred.items()}


def _defaults_from_inferred(inferred: dict[str, InferredField]) -> dict[str, Any]:
    return {key: value.default for key, value in inferred.items() if value.default is not None}


def build_plugin_manifest_dict(
    plugin_id: str, inferred: InferredPlugin
) -> dict[str, Any]:
    """Assemble the dict structure that will be rendered to TOML."""
    manifest: dict[str, Any] = {
        "plugin": {
            "id": plugin_id,
            "kind": inferred.kind,
            "name": plugin_id,
            "version": "0.1.0",
            "category": "",
            "stability": "experimental",
            "description": inferred.description,
            "entry_point": {
                "module": inferred.entry_module,
                "class": inferred.entry_class,
            },
        },
    }
    params_schema = _schema_from_inferred(inferred.params)
    thresholds_schema = _schema_from_inferred(inferred.thresholds)
    if params_schema:
        manifest["params_schema"] = params_schema
    if thresholds_schema:
        manifest["thresholds_schema"] = thresholds_schema
    suggested_params = _defaults_from_inferred(inferred.params)
    suggested_thresholds = _defaults_from_inferred(inferred.thresholds)
    if suggested_params:
        manifest["suggested_params"] = suggested_params
    if suggested_thresholds:
        manifest["suggested_thresholds"] = suggested_thresholds
    if inferred.required_secrets:
        manifest["secrets"] = {"required": list(inferred.required_secrets)}
    # Report plugins get a [reports] block with a placeholder order. Authors
    # must fill in the actual section order manually — it's unique per
    # report plugin and drives Markdown section ordering.
    if inferred.kind == "reports":
        manifest["reports"] = {"order": 50}
    # ``[compat]`` is intentionally omitted: adding an empty block just makes
    # noise in the rendered TOML. Authors can add it by hand when they have a
    # real ``min_core_version`` to pin.
    return manifest


def scaffold_plugin_manifest(plugin_dir: Path) -> str:
    """Return manifest.toml text for a plugin folder (validated)."""
    inferred = infer_plugin(plugin_dir)
    manifest = build_plugin_manifest_dict(plugin_dir.name, inferred)
    # Validate before writing — fail fast if inference produced garbage.
    PluginManifest.model_validate(manifest)
    return dump_manifest_toml(manifest, todo_fields=_PLUGIN_TODO_FIELDS)


def scaffold_preset_manifest(preset_dir: Path) -> str:
    """Return manifest.toml text for a preset folder (validated)."""
    yaml_path = _pick_preset_yaml(preset_dir)
    manifest: dict[str, Any] = {
        "preset": {
            "id": preset_dir.name,
            "name": preset_dir.name,
            "version": "0.1.0",
            "size_tier": "",
            "description": "",
            "entry_point": {"file": yaml_path.name},
        }
    }
    PresetManifest.model_validate(manifest)
    return dump_manifest_toml(manifest, todo_fields=_PRESET_TODO_FIELDS)


def _pick_preset_yaml(preset_dir: Path) -> Path:
    preferred = preset_dir / "preset.yaml"
    if preferred.is_file():
        return preferred
    yamls = sorted(p for p in preset_dir.glob("*.yaml") if p.is_file())
    if not yamls:
        raise FileNotFoundError(f"no *.yaml found in {preset_dir}")
    return yamls[0]


__all__ = [
    "build_plugin_manifest_dict",
    "scaffold_plugin_manifest",
    "scaffold_preset_manifest",
]
