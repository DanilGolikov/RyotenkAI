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
from src.community.manifest import PluginManifest, PresetManifest
from src.community.scaffold import build_plugin_manifest_dict
from src.community.toml_writer import dump_manifest_toml

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


def _merge_secrets(
    existing: list[str] | None, inferred: tuple[str, ...]
) -> list[str]:
    """Union: inferred keys first (deterministic), then existing-only keys."""
    existing_list = list(existing or [])
    seen: set[str] = set()
    out: list[str] = []
    for key in list(inferred) + existing_list:
        if key not in seen:
            out.append(key)
            seen.add(key)
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

    # secrets — supplement union (never delete)
    existing_secrets = (existing.get("secrets", {}) or {}).get("required") or []
    union = _merge_secrets(existing_secrets, inferred.required_secrets)
    if union:
        merged["secrets"] = {"required": union}
    else:
        merged.pop("secrets", None)

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
    "sync_plugin_manifest",
    "sync_preset_manifest",
]
