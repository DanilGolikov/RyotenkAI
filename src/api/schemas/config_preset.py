"""Pydantic schemas for the ``/config/presets`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigPreset(BaseModel):
    name: str
    """Stable id derived from the filename stem — used for ordering and
    permalinks. NOT shown to users directly (ugly kebab/digit-prefix)."""
    display_name: str = ""
    """Human-readable label for the dropdown. Falls back to ``name``
    when the preset YAML omits the ``# Preset: <label>`` comment."""
    description: str = ""
    yaml: str


class ConfigPresetsResponse(BaseModel):
    presets: list[ConfigPreset] = Field(default_factory=list)


__all__ = ["ConfigPreset", "ConfigPresetsResponse"]
