"""Re-infer a plugin/preset and 3-way-merge the result with an existing
``manifest.toml``.

Merge rules are documented in ``docs/plans/tidy-leaping-lighthouse.md``.
Short version: MANAGED fields are overwritten from inference, USER-OWNED
are preserved, ``version`` is bumped by ``--bump patch|minor|major``.
"""

from __future__ import annotations

import difflib
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.community.inference import InferredPlugin, bump_version, infer_plugin
from src.community.libs import libs_root_for, preload_community_libs
from src.community.loader import _import_plugin_class
from src.community.manifest import PluginManifest, PresetManifest
from src.community.scaffold import build_plugin_manifest_dict
from src.community.toml_writer import dump_manifest_toml


def _preload_libs_for(plugin_dir: Path) -> None:
    """Preload ``community_libs.*`` for the catalog that owns ``plugin_dir``.

    Plugins under ``community/<kind>/<id>/`` may do
    ``from community_libs.<lib> import …`` at module load time. Sync
    commands import the plugin module directly via
    :func:`_import_plugin_class`, which short-circuits the catalog —
    so the namespace would otherwise be missing and the import would
    raise ``ModuleNotFoundError``. We do the preload ourselves so
    sync-envs / sync-libs can be invoked against an arbitrary plugin
    folder without first warming up the catalog.

    Idempotent (the catalog or another sync call may already have
    preloaded the same root).
    """
    community_root = plugin_dir.parents[1]
    preload_community_libs(libs_root_for(community_root))

Bump = Literal["patch", "minor", "major"]


@dataclass(frozen=True, slots=True)
class SyncResult:
    new_text: str         # full content to be written to manifest.toml
    old_text: str         # previous content (for diffing in CLI)
    changed: bool
    diff: str             # unified diff (one hunk), or "" when unchanged


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------


_USER_OWNED_PLUGIN_FIELDS = (
    "id",
    "name",
    "description",
    "category",
    "stability",
)
_USER_OWNED_SCHEMA_KEYS = ("min", "max", "options", "description")


def _merge_schema(
    existing: dict[str, dict[str, Any]],
    inferred: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-key schema merge.

    - Keys only in ``inferred`` → added verbatim (NEW-DISCOVERED).
    - Keys only in ``existing`` → dropped (code no longer references them).
    - Keys in both → start from existing, overwrite MANAGED bits
      (``type``, ``default`` when existing value has none), preserve
      USER-OWNED bits (``min``, ``max``, ``options``, ``description``).
    """
    out: dict[str, dict[str, Any]] = {}
    for key, inferred_entry in inferred.items():
        if key not in existing:
            out[key] = dict(inferred_entry)
            continue
        existing_entry = dict(existing[key])
        merged: dict[str, Any] = {}
        # MANAGED: type is rewritten from inference unless the author set
        # an explicit non-empty value in the manifest.
        merged["type"] = existing_entry.get("type") or inferred_entry["type"]
        # default: prefer existing author-provided value; fall back to inference.
        if "default" in existing_entry:
            merged["default"] = existing_entry["default"]
        elif "default" in inferred_entry:
            merged["default"] = inferred_entry["default"]
        # USER-OWNED bits: keep whatever was there.
        for owned_key in _USER_OWNED_SCHEMA_KEYS:
            if owned_key in existing_entry:
                merged[owned_key] = existing_entry[owned_key]
        # Carry over any other manually-added keys (forward-compat).
        for extra_key, extra_val in existing_entry.items():
            merged.setdefault(extra_key, extra_val)
        out[key] = merged
    return out


def _merge_suggestions(
    existing: dict[str, Any], schema: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Keep user suggestions, drop keys that are no longer in schema."""
    return {k: v for k, v in existing.items() if k in schema}


def _merge_required_env(
    existing: list[dict[str, Any]] | None,
    inferred: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Merge ``[[required_env]]`` entries.

    - Existing entries (with hand-edited descriptions / managed_by hints
      / non-default secret/optional flags) win — sync never destroys
      author-provided metadata.
    - Inferred env names that don't appear in ``existing`` are appended
      with the safe defaults (``secret=true, optional=false, managed_by=""``).
    - Existing entries whose ``name`` is no longer inferred *stay*: the
      runtime cross-check (PR8 / A7) will catch genuine drift between
      Python and TOML; here we err on the side of preservation.

    The output preserves the order ``inferred → existing-only`` so a
    fresh scaffold places newly-discovered envs at the top of the list.
    """
    existing_by_name: dict[str, dict[str, Any]] = {
        entry["name"]: dict(entry)
        for entry in (existing or [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _default_entry(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": "",
            "optional": False,
            "secret": True,
            "managed_by": "",
        }

    for name in inferred:
        if name in seen:
            continue
        out.append(existing_by_name.get(name) or _default_entry(name))
        seen.add(name)

    for name, entry in existing_by_name.items():
        if name in seen:
            continue
        out.append(entry)
        seen.add(name)

    return out


def _merge_plugin_manifest(
    existing: dict[str, Any],
    inferred: InferredPlugin,
    *,
    plugin_id: str,
    bump: Bump,
) -> dict[str, Any]:
    # Full scaffolded baseline; we'll overlay existing user-owned bits.
    baseline = build_plugin_manifest_dict(plugin_id, inferred)
    merged = deepcopy(baseline)

    existing_plugin = existing.get("plugin", {}) or {}

    # USER-OWNED scalars under [plugin]
    for field in _USER_OWNED_PLUGIN_FIELDS:
        if field in existing_plugin:
            merged["plugin"][field] = existing_plugin[field]

    # version: bump from existing; if existing lacks one, start at 0.1.0.
    existing_version = existing_plugin.get("version") or "0.1.0"
    merged["plugin"]["version"] = bump_version(existing_version, bump)

    # params_schema / thresholds_schema — per-key merge
    existing_params_schema = existing.get("params_schema", {}) or {}
    existing_thresholds_schema = existing.get("thresholds_schema", {}) or {}
    inferred_params_schema = merged.get("params_schema", {})
    inferred_thresholds_schema = merged.get("thresholds_schema", {})

    new_params_schema = _merge_schema(existing_params_schema, inferred_params_schema)
    new_thresholds_schema = _merge_schema(
        existing_thresholds_schema, inferred_thresholds_schema
    )

    if new_params_schema:
        merged["params_schema"] = new_params_schema
    else:
        merged.pop("params_schema", None)
    if new_thresholds_schema:
        merged["thresholds_schema"] = new_thresholds_schema
    else:
        merged.pop("thresholds_schema", None)

    # suggested_* — keep user suggestions, drop orphans
    existing_sp = existing.get("suggested_params", {}) or {}
    existing_st = existing.get("suggested_thresholds", {}) or {}
    merged_sp = _merge_suggestions(existing_sp, new_params_schema)
    merged_st = _merge_suggestions(existing_st, new_thresholds_schema)
    if merged_sp:
        merged["suggested_params"] = merged_sp
    else:
        merged.pop("suggested_params", None)
    if merged_st:
        merged["suggested_thresholds"] = merged_st
    else:
        merged.pop("suggested_thresholds", None)

    # required_env — merge per-name; never delete author-provided metadata.
    existing_required_env = existing.get("required_env") or []
    merged_required_env = _merge_required_env(
        existing_required_env, inferred.required_secrets
    )
    if merged_required_env:
        merged["required_env"] = merged_required_env
    else:
        merged.pop("required_env", None)

    # compat — preserve existing if set
    existing_compat = existing.get("compat", {}) or {}
    if existing_compat.get("min_core_version"):
        merged["compat"] = {"min_core_version": existing_compat["min_core_version"]}

    return merged


def sync_plugin_manifest(
    plugin_dir: Path,
    *,
    bump: Bump = "patch",
) -> SyncResult:
    manifest_path = plugin_dir / "manifest.toml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.toml in {plugin_dir}")

    old_text = manifest_path.read_text(encoding="utf-8")
    existing = tomllib.loads(old_text)
    inferred = infer_plugin(plugin_dir)

    merged = _merge_plugin_manifest(
        existing, inferred, plugin_id=plugin_dir.name, bump=bump
    )
    # Validate before writing — fail fast if merge produced something invalid.
    PluginManifest.model_validate(merged)

    new_text = dump_manifest_toml(merged)
    changed = new_text != old_text
    diff = _unified_diff(old_text, new_text, path=str(manifest_path))
    return SyncResult(new_text=new_text, old_text=old_text, changed=changed, diff=diff)


def sync_plugin_envs(plugin_dir: Path) -> SyncResult:
    """Rewrite manifest's ``[[required_env]]`` from the class's REQUIRED_ENV.

    Imports the plugin module (just like the loader does), reads the
    Python-side ``REQUIRED_ENV`` ClassVar, and writes that as the
    manifest's ``required_env`` block. Use this after editing the env
    contract in code so the TOML side never drifts and the load-time
    cross-check in :func:`_crosscheck_required_env` keeps passing.

    No-op when the class doesn't declare ``REQUIRED_ENV`` (or declares
    it empty) — the manifest stays the only source of truth.
    """
    manifest_path = plugin_dir / "manifest.toml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.toml in {plugin_dir}")

    old_text = manifest_path.read_text(encoding="utf-8")
    existing = tomllib.loads(old_text)
    plugin_block = existing.get("plugin", {}) or {}
    entry_point = plugin_block.get("entry_point", {}) or {}
    module_name = entry_point.get("module", "plugin")
    class_name = entry_point.get("class") or entry_point.get("class_name")
    if not class_name:
        raise ValueError(
            f"manifest at {manifest_path} has no [plugin.entry_point].class — "
            "cannot import to read REQUIRED_ENV"
        )

    _preload_libs_for(plugin_dir)
    plugin_cls = _import_plugin_class(plugin_dir, module_name, class_name)
    declared = getattr(plugin_cls, "REQUIRED_ENV", ())

    merged = dict(existing)
    if declared:
        merged["required_env"] = [
            spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
            for spec in declared
        ]
    else:
        merged.pop("required_env", None)

    # Re-validate the whole manifest so a bad REQUIRED_ENV declaration
    # surfaces here instead of at the next loader run.
    PluginManifest.model_validate(merged)

    new_text = dump_manifest_toml(merged)
    changed = new_text != old_text
    diff = _unified_diff(old_text, new_text, path=str(manifest_path))
    return SyncResult(new_text=new_text, old_text=old_text, changed=changed, diff=diff)


def sync_plugin_libs(plugin_dir: Path) -> SyncResult:
    """Rewrite manifest's ``[plugin].libs`` from the class's REQUIRED_LIBS.

    Imports the plugin module the same way the loader does, reads the
    Python-side ``REQUIRED_LIBS`` ClassVar, and writes that as the
    manifest's ``[plugin].libs`` field. Use this after editing the
    declared lib dependencies in code so the TOML side never drifts and
    the load-time cross-check in :func:`_crosscheck_required_libs`
    keeps passing.

    Order semantics: the field on the class is a tuple, but the TOML
    list is normalised to **sorted order** at write time. Cross-check
    is set-based, so this normalisation is invisible to the loader and
    keeps the diff stable across re-runs.

    No-op when the class doesn't declare ``REQUIRED_LIBS`` (or declares
    it empty) — the manifest's ``libs`` field stays the only source of
    truth and is removed if it was previously synced from a now-empty
    ClassVar.
    """
    manifest_path = plugin_dir / "manifest.toml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.toml in {plugin_dir}")

    old_text = manifest_path.read_text(encoding="utf-8")
    existing = tomllib.loads(old_text)
    plugin_block = existing.get("plugin", {}) or {}
    entry_point = plugin_block.get("entry_point", {}) or {}
    module_name = entry_point.get("module", "plugin")
    class_name = entry_point.get("class") or entry_point.get("class_name")
    if not class_name:
        raise ValueError(
            f"manifest at {manifest_path} has no [plugin.entry_point].class — "
            "cannot import to read REQUIRED_LIBS"
        )

    _preload_libs_for(plugin_dir)
    plugin_cls = _import_plugin_class(plugin_dir, module_name, class_name)
    declared = getattr(plugin_cls, "REQUIRED_LIBS", ())
    if declared and (
        not isinstance(declared, tuple)
        or any(not isinstance(x, str) for x in declared)
    ):
        raise TypeError(
            f"plugin {plugin_cls.__name__}.REQUIRED_LIBS must be a tuple of "
            f"strings; got {type(declared).__name__}"
        )

    merged = dict(existing)
    merged_plugin = dict(plugin_block)
    if declared:
        merged_plugin["libs"] = sorted(set(declared))
    else:
        merged_plugin.pop("libs", None)
    merged["plugin"] = merged_plugin

    # Re-validate so a bogus REQUIRED_LIBS surfaces here, not on next load.
    PluginManifest.model_validate(merged)

    new_text = dump_manifest_toml(merged)
    changed = new_text != old_text
    diff = _unified_diff(old_text, new_text, path=str(manifest_path))
    return SyncResult(new_text=new_text, old_text=old_text, changed=changed, diff=diff)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def _pick_yaml(preset_dir: Path) -> str:
    preferred = preset_dir / "preset.yaml"
    if preferred.is_file():
        return preferred.name
    for path in sorted(preset_dir.glob("*.yaml")):
        if path.is_file():
            return path.name
    raise FileNotFoundError(f"no *.yaml found in {preset_dir}")


def _merge_preset_manifest(
    existing: dict[str, Any],
    preset_dir: Path,
    *,
    bump: Bump,
) -> dict[str, Any]:
    existing_preset = existing.get("preset", {}) or {}
    existing_entry = existing_preset.get("entry_point", {}) or {}

    merged_preset: dict[str, Any] = dict(existing_preset)
    merged_preset.setdefault("id", preset_dir.name)
    merged_preset.setdefault("name", existing_preset.get("id") or preset_dir.name)
    merged_preset.setdefault("description", "")
    merged_preset.setdefault("size_tier", "")
    # MANAGED: entry_point.file is refreshed from disk.
    merged_preset["entry_point"] = {"file": _pick_yaml(preset_dir)}
    # Preserve any unrelated fields the user put under [preset.entry_point]
    for key, value in existing_entry.items():
        if key == "file":
            continue
        merged_preset["entry_point"][key] = value
    # version bump
    existing_version = existing_preset.get("version") or "0.1.0"
    merged_preset["version"] = bump_version(existing_version, bump)

    return {"preset": merged_preset}


def sync_preset_manifest(
    preset_dir: Path,
    *,
    bump: Bump = "patch",
) -> SyncResult:
    manifest_path = preset_dir / "manifest.toml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.toml in {preset_dir}")

    old_text = manifest_path.read_text(encoding="utf-8")
    existing = tomllib.loads(old_text)
    merged = _merge_preset_manifest(existing, preset_dir, bump=bump)
    PresetManifest.model_validate(merged)

    new_text = dump_manifest_toml(merged)
    changed = new_text != old_text
    diff = _unified_diff(old_text, new_text, path=str(manifest_path))
    return SyncResult(new_text=new_text, old_text=old_text, changed=changed, diff=diff)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unified_diff(a: str, b: str, *, path: str) -> str:
    if a == b:
        return ""
    lines = list(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
        )
    )
    return "".join(lines)


__all__ = [
    "SyncResult",
    "sync_plugin_envs",
    "sync_plugin_libs",
    "sync_plugin_manifest",
    "sync_preset_manifest",
]
