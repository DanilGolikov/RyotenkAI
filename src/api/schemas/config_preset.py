"""Pydantic schemas for the ``/config/presets`` endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PresetScopeOut(BaseModel):
    """Declared authority of a preset over top-level config keys."""

    replaces: list[str] = Field(default_factory=list)
    preserves: list[str] = Field(default_factory=list)


class PresetRequirementsOut(BaseModel):
    """Environmental prerequisites surfaced to the UI."""

    hub_models: list[str] = Field(default_factory=list)
    provider_kind: list[str] = Field(default_factory=list)
    required_plugins: list[str] = Field(default_factory=list)
    min_vram_gb: int | None = None


class ConfigPreset(BaseModel):
    name: str
    """Stable id — preset manifest.preset.id (used for ordering and permalinks)."""
    display_name: str = ""
    """Human-readable label for the dropdown."""
    description: str = ""
    yaml: str
    size_tier: str = ""
    """Optional coarse size classification (e.g. ``small``/``medium``/``large``)."""
    scope: PresetScopeOut | None = None
    """Declared ownership over top-level config keys. ``null`` → v1 full replace."""
    requirements: PresetRequirementsOut | None = None
    """Environment prerequisites, displayed on the preview modal."""
    placeholders: dict[str, str] = Field(default_factory=dict)
    """Dotted JSONPath → user-facing hint for fields that must be filled manually."""


class ConfigPresetsResponse(BaseModel):
    presets: list[ConfigPreset] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Preview (POST /config/presets/{id}/preview)
# ---------------------------------------------------------------------------


class PresetPreviewRequest(BaseModel):
    current_config: dict[str, Any] = Field(default_factory=dict)


DiffKind = Literal["added", "removed", "changed", "unchanged"]
DiffReason = Literal["preset_replaced", "preset_added", "preset_preserved", "no_scope"]
RequirementStatus = Literal["ok", "missing", "warning"]


class PresetDiffEntry(BaseModel):
    key: str
    kind: DiffKind
    reason: DiffReason
    before: Any = None
    after: Any = None


class PresetRequirementCheck(BaseModel):
    label: str
    status: RequirementStatus
    detail: str = ""


class PresetPlaceholderHint(BaseModel):
    path: str
    hint: str


class PresetPreviewResponse(BaseModel):
    resulting_config: dict[str, Any]
    diff: list[PresetDiffEntry]
    requirements: list[PresetRequirementCheck]
    placeholders: list[PresetPlaceholderHint]
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "ConfigPreset",
    "ConfigPresetsResponse",
    "PresetDiffEntry",
    "PresetPlaceholderHint",
    "PresetPreviewRequest",
    "PresetPreviewResponse",
    "PresetRequirementCheck",
    "PresetRequirementsOut",
    "PresetScopeOut",
]
