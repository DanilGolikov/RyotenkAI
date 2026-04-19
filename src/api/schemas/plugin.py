"""Pydantic schemas for the plugin catalogue endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PluginKind = Literal["reward", "validation", "evaluation"]


class PluginManifest(BaseModel):
    """Normalised manifest surfaced to the web UI (Grafana-style)."""

    id: str
    name: str
    version: str = "1.0.0"
    priority: int = 50
    description: str = ""
    category: str = ""
    stability: str = "stable"
    params_schema: dict[str, Any] = Field(default_factory=dict)
    thresholds_schema: dict[str, Any] = Field(default_factory=dict)
    suggested_params: dict[str, Any] = Field(default_factory=dict)
    suggested_thresholds: dict[str, Any] = Field(default_factory=dict)


class PluginListResponse(BaseModel):
    kind: PluginKind
    plugins: list[PluginManifest]


__all__ = ["PluginKind", "PluginManifest", "PluginListResponse"]
