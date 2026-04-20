"""Pydantic schemas for the ``/config/presets`` endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigPreset(BaseModel):
    name: str
    description: str = ""
    yaml: str


class ConfigPresetsResponse(BaseModel):
    presets: list[ConfigPreset] = Field(default_factory=list)


__all__ = ["ConfigPreset", "ConfigPresetsResponse"]
