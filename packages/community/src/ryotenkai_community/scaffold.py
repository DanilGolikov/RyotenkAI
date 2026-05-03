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
        "plugin.supported_strategies",
        "compat.min_core_version",
    }
)

_PRESET_TODO_FIELDS = frozenset(
    {
        "preset.description",
        "preset.size_tier",
    }
)


def _field_to_dict(field: InferredField) -> dict[str, Any]:
    """Render one inferred field into a ParamFieldSchema-shaped dict.

    Inference only knows type + default; authors are expected to fill in
    ``title`` / ``description`` themselves — we seed them as empty strings
    so they're visible in the rendered TOML (and TODO-marked via
    ``todo_fields``)."""
    entry: dict[str, Any] = {"type": field.type}
    if field.default is not None:
        entry["default"] = field.default
    entry["title"] = ""
    entry["description"] = ""
    return entry


def _schema_from_inferred(inferred: dict[str, InferredField]) -> dict[str, dict[str, Any]]:
    return {key: _field_to_dict(value) for key, value in inferred.items()}


def _defaults_from_inferred(inferred: dict[str, InferredField]) -> dict[str, Any]:
    return {key: value.default for key, value in inferred.items() if value.default is not None}


def _field_todo_paths(
    inferred: dict[str, InferredField], section: str
) -> set[str]:
    """TODO-mark the new ``title`` / ``description`` scalars in every field."""
    return {f"{section}.{key}.{attr}" for key in inferred for attr in ("title", "description")}


def build_plugin_manifest_dict(
    plugin_id: str, inferred: InferredPlugin
) -> dict[str, Any]:
    """Assemble the dict structure that will be rendered to TOML."""
    plugin_body: dict[str, Any] = {
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
    }
    # Reward plugins MUST declare supported_strategies (see PluginSpec).
    # Emit an empty list as a TODO so the author fills it in before the
    # loader accepts the manifest.
    if inferred.kind == "reward":
        plugin_body["supported_strategies"] = []
    manifest: dict[str, Any] = {"plugin": plugin_body}
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
        # Each inferred ``self._secrets["KEY"]`` access becomes a
        # ``[[required_env]]`` block with ``secret=true, optional=false``
        # — the safe default for any key that the plugin code dereferences
        # unconditionally. Authors flip ``secret=false`` for non-credential
        # envs (e.g. CLI paths) post-scaffold.
        manifest["required_env"] = [
            {
                "name": name,
                "description": "",
                "optional": False,
                "secret": True,
                "managed_by": "",
            }
            for name in inferred.required_secrets
        ]
    # ``[compat]`` is intentionally omitted: adding an empty block just makes
    # noise in the rendered TOML. Authors can add it by hand when they have a
    # real ``min_core_version`` to pin.
    # Report plugins don't carry section order in the manifest — order lives
    # in the pipeline config's ``reports.sections`` list (or the built-in
    # default). Authors opt-in to a section by adding the plugin id there.
    return manifest


def scaffold_plugin_manifest(plugin_dir: Path) -> str:
    """Return manifest.toml text for a plugin folder.

    Note: we intentionally do NOT validate against :class:`PluginManifest`
    here — the scaffolded manifest is expected to contain TODO-marked
    empty placeholders (e.g. ``supported_strategies = []`` for reward
    plugins) that the author has to fill in before the loader accepts
    the plugin. Validation happens on load, where it belongs.
    """
    inferred = infer_plugin(plugin_dir)
    manifest = build_plugin_manifest_dict(plugin_dir.name, inferred)
    todo_fields = set(_PLUGIN_TODO_FIELDS)
    todo_fields |= _field_todo_paths(inferred.params, "params_schema")
    todo_fields |= _field_todo_paths(inferred.thresholds, "thresholds_schema")
    return dump_manifest_toml(manifest, todo_fields=todo_fields)


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
