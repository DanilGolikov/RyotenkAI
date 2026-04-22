"""Apply a preset to a user config — pure function, no I/O.

Public entry point: :func:`apply_preset`. Takes the user's current config
dict and a :class:`LoadedPreset`; returns a :class:`PresetPreview` with the
resulting config, a field-level diff, and a requirements check. The backend
endpoint ``POST /config/presets/{id}/preview`` is a thin wrapper around this.

v1 compatibility mode: if the preset manifest has **no** ``[preset.scope]``
block, we fall back to the historical behaviour — full replacement of the
user config. That keeps pre-existing presets working without edits.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from src.community.loader import LoadedPreset

DiffKind = Literal["added", "removed", "changed", "unchanged"]
Reason = Literal["preset_replaced", "preset_added", "preset_preserved", "no_scope"]
Status = Literal["ok", "missing", "warning"]


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """One top-level key in the resulting config, described for the UI."""

    key: str
    kind: DiffKind
    reason: Reason
    before: Any = None
    after: Any = None


@dataclass(frozen=True, slots=True)
class RequirementCheck:
    """One line on the ``Requirements`` panel in the UI."""

    label: str
    status: Status
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PlaceholderHint:
    path: str
    hint: str


@dataclass(slots=True)
class PresetPreview:
    """Return shape of :func:`apply_preset` — matches the API response."""

    resulting_config: dict[str, Any]
    diff: list[DiffEntry]
    requirements: list[RequirementCheck]
    placeholders: list[PlaceholderHint]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------


def _merge(
    current: dict[str, Any],
    preset_body: dict[str, Any],
    *,
    replaces: list[str],
    preserves: list[str],
) -> tuple[dict[str, Any], list[DiffEntry]]:
    """Top-level key merge.

    Precedence rules (iterating over the union of keys in both dicts):

    1. ``preserves`` — keep user value; the preset's value for this key is
       dropped even if present.
    2. ``replaces`` — overwrite with preset value. Absence in preset means
       "clear the key from user config".
    3. Keys that appear only in the preset YAML but aren't in either list
       fall into ``replaces`` (permissive default).
    4. Keys only in the user config (not mentioned in preset YAML, not in
       any scope list) are kept.
    """
    replaces_set = set(replaces)
    preserves_set = set(preserves)
    result = deepcopy(current)
    diff: list[DiffEntry] = []

    preset_touches = replaces_set | set(preset_body.keys())

    for key in sorted(preset_touches | set(current.keys())):
        in_user = key in current
        in_preset = key in preset_body

        if key in preserves_set:
            if in_user:
                diff.append(
                    DiffEntry(key=key, kind="unchanged", reason="preset_preserved",
                              before=current[key], after=current[key])
                )
            continue

        if key in replaces_set or in_preset:
            before = current.get(key)
            after = preset_body.get(key)
            result.pop(key, None)
            if in_preset:
                result[key] = deepcopy(after)
                if not in_user:
                    diff.append(DiffEntry(key=key, kind="added",
                                          reason="preset_added", after=after))
                elif before != after:
                    diff.append(DiffEntry(key=key, kind="changed",
                                          reason="preset_replaced",
                                          before=before, after=after))
                else:
                    diff.append(DiffEntry(key=key, kind="unchanged",
                                          reason="preset_replaced",
                                          before=before, after=after))
            else:
                # replaces declares the key but preset.yaml omits it → drop.
                if in_user:
                    diff.append(DiffEntry(key=key, kind="removed",
                                          reason="preset_replaced", before=before))
            continue

        # Not in any scope list and not in preset YAML → keep user value.
        if in_user:
            diff.append(DiffEntry(key=key, kind="unchanged",
                                  reason="preset_preserved", before=current[key],
                                  after=current[key]))

    return result, diff


def _v1_full_replace(
    current: dict[str, Any], preset_body: dict[str, Any]
) -> tuple[dict[str, Any], list[DiffEntry]]:
    """Legacy behaviour: the preset body is the resulting config."""
    result = deepcopy(preset_body)
    diff: list[DiffEntry] = []
    for key in sorted(set(current.keys()) | set(preset_body.keys())):
        before = current.get(key)
        after = preset_body.get(key)
        if key not in preset_body:
            diff.append(DiffEntry(key=key, kind="removed", reason="no_scope",
                                  before=before))
        elif key not in current:
            diff.append(DiffEntry(key=key, kind="added", reason="no_scope",
                                  after=after))
        elif before != after:
            diff.append(DiffEntry(key=key, kind="changed", reason="no_scope",
                                  before=before, after=after))
        else:
            diff.append(DiffEntry(key=key, kind="unchanged", reason="no_scope",
                                  before=before, after=after))
    return result, diff


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


def _check_requirements(
    preset: LoadedPreset,
    *,
    current_config: dict[str, Any],
    secrets_model_extra: dict[str, Any] | None,
    available_plugin_ids_by_kind: dict[str, set[str]],
) -> list[RequirementCheck]:
    """Build the requirements panel from the environment the caller gave us.

    This function **never** hits the network; the caller feeds it the facts
    (is HF_TOKEN set, which plugins the catalog has, which providers are
    configured). That keeps the unit under test small and the API endpoint
    responsible for gathering environment.
    """
    req = preset.manifest.preset.requirements
    if req is None:
        return []

    checks: list[RequirementCheck] = []

    if req.hub_models:
        has_token = bool(
            (secrets_model_extra or {}).get("hf_token")
            or (secrets_model_extra or {}).get("HF_TOKEN")
        )
        label = "HF Hub access"
        if has_token:
            checks.append(RequirementCheck(
                label=label, status="ok",
                detail=f"HF_TOKEN is set — {len(req.hub_models)} model(s) referenced",
            ))
        else:
            checks.append(RequirementCheck(
                label=label, status="warning",
                detail=(
                    f"HF_TOKEN is not set; gated models "
                    f"({', '.join(req.hub_models)}) will 401 at download time"
                ),
            ))

    if req.provider_kind:
        configured = _configured_provider_kinds(current_config)
        match = [pk for pk in req.provider_kind if pk in configured]
        label = "Provider kind"
        if match:
            checks.append(RequirementCheck(
                label=label, status="ok",
                detail=f"configured: {', '.join(sorted(match))}",
            ))
        elif configured:
            checks.append(RequirementCheck(
                label=label, status="warning",
                detail=(
                    f"preset recommends {req.provider_kind}; "
                    f"your config uses {sorted(configured)}"
                ),
            ))
        else:
            checks.append(RequirementCheck(
                label=label, status="missing",
                detail=f"preset recommends {req.provider_kind}; no provider configured",
            ))

    if req.required_plugins:
        missing: list[str] = []
        for token in req.required_plugins:
            if ":" not in token:
                missing.append(token)
                continue
            kind, plugin_id = token.split(":", 1)
            if plugin_id not in available_plugin_ids_by_kind.get(kind, set()):
                missing.append(token)
        if missing:
            checks.append(RequirementCheck(
                label="Required plugins", status="missing",
                detail=f"not loaded: {', '.join(missing)}",
            ))
        else:
            checks.append(RequirementCheck(
                label="Required plugins", status="ok",
                detail=f"{len(req.required_plugins)} present in catalog",
            ))

    if req.min_vram_gb is not None:
        checks.append(RequirementCheck(
            label="GPU memory",
            status="warning",     # informational — can't actually probe here
            detail=f"preset expects ≥ {req.min_vram_gb} GB VRAM",
        ))

    return checks


def _configured_provider_kinds(cfg: dict[str, Any]) -> set[str]:
    providers = cfg.get("providers") or {}
    out: set[str] = set()
    if isinstance(providers, dict):
        for value in providers.values():
            if isinstance(value, dict):
                kind = value.get("kind") or value.get("type")
                if isinstance(kind, str):
                    out.add(kind)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_preset(
    current_config: dict[str, Any],
    preset: LoadedPreset,
    *,
    secrets_model_extra: dict[str, Any] | None = None,
    available_plugin_ids_by_kind: dict[str, set[str]] | None = None,
) -> PresetPreview:
    """Compute the preview shown to the user before they click Apply."""
    preset_body = yaml.safe_load(preset.yaml_text) or {}
    if not isinstance(preset_body, dict):
        raise ValueError(f"preset YAML must be a mapping, got {type(preset_body).__name__}")

    spec = preset.manifest.preset
    scope = spec.scope
    if scope is None:
        resulting, diff = _v1_full_replace(current_config, preset_body)
    else:
        resulting, diff = _merge(
            current_config, preset_body,
            replaces=scope.replaces, preserves=scope.preserves,
        )

    reqs = _check_requirements(
        preset,
        current_config=current_config,
        secrets_model_extra=secrets_model_extra,
        available_plugin_ids_by_kind=available_plugin_ids_by_kind or {},
    )

    placeholders = [PlaceholderHint(path=p, hint=h) for p, h in spec.placeholders.items()]

    warnings: list[str] = []
    if scope is None:
        warnings.append(
            "preset has no [preset.scope] block — applying full overwrite "
            "(user datasets/providers/evaluation will be replaced by preset YAML)"
        )

    return PresetPreview(
        resulting_config=resulting,
        diff=diff,
        requirements=reqs,
        placeholders=placeholders,
        warnings=warnings,
    )


__all__ = [
    "DiffEntry",
    "PlaceholderHint",
    "PresetPreview",
    "RequirementCheck",
    "apply_preset",
]
