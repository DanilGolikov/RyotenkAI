"""Pydantic schemas for the plugin catalogue endpoints.

Surfaces the shape of :meth:`src.community.manifest.PluginManifest.ui_manifest`
— ``params_schema`` / ``thresholds_schema`` come out as JSON Schema objects
that the UI can drop straight into ``FieldRenderer``.

``RequiredEnvSpec`` is re-exported from :mod:`src.community.manifest` so the
plugin catalog has **a single source of truth**: the canonical Pydantic model
lives next to the manifest loader, and any schema change automatically flows
to the OpenAPI surface (which `web/src/api/schema.d.ts` is generated from).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Re-export the canonical RequiredEnvSpec — see module docstring for
# rationale. The CI gate (`make check-openapi`) ensures the front-end
# generated types stay in sync.
from src.community.manifest import RequiredEnvSpec

PluginKind = Literal["reward", "validation", "evaluation", "reports"]


class PluginManifest(BaseModel):
    """Normalised manifest surfaced to the web UI."""

    schema_version: int = 1
    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    category: str = ""
    stability: str = "stable"
    kind: PluginKind
    #: For reward plugins — which ``strategy_type`` values are supported.
    #: Empty for all other kinds.
    supported_strategies: list[str] = Field(default_factory=list)
    #: JSON Schema object describing plugin parameters.
    params_schema: dict[str, Any] = Field(default_factory=dict)
    #: JSON Schema object describing plugin thresholds.
    thresholds_schema: dict[str, Any] = Field(default_factory=dict)
    suggested_params: dict[str, Any] = Field(default_factory=dict)
    suggested_thresholds: dict[str, Any] = Field(default_factory=dict)
    #: Declarative env contract — UI surfaces these as inputs in the
    #: Configure modal. Empty list when the plugin doesn't need envs.
    required_env: list[RequiredEnvSpec] = Field(default_factory=list)


class PluginListResponse(BaseModel):
    kind: PluginKind
    plugins: list[PluginManifest]


__all__ = ["PluginKind", "PluginManifest", "PluginListResponse", "RequiredEnvSpec"]
