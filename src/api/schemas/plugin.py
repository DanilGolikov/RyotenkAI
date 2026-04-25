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


class PluginLoadError(BaseModel):
    """One per-entry failure surfaced from the community loader.

    Mirrors :class:`src.community.loader.LoadFailure` for the OpenAPI
    surface — kept as a plain Pydantic model (rather than re-exporting
    the dataclass) so the API stays decoupled from internal shape
    changes. The UI reads ``error_type`` to pick an icon, ``message``
    for the headline, and ``traceback`` for the developer drilldown.
    """

    entry_name: str
    plugin_id: str | None = None
    error_type: str
    message: str
    traceback: str = ""


class PluginListResponse(BaseModel):
    kind: PluginKind
    plugins: list[PluginManifest]
    #: Per-entry load failures from the most recent catalog refresh.
    #: Empty when every plugin in the kind directory loaded cleanly.
    errors: list[PluginLoadError] = Field(default_factory=list)


class MissingEnvSchema(BaseModel):
    """One required env the preflight gate couldn't resolve.

    Mirrors :class:`src.community.preflight.MissingEnv` — the API
    re-shapes the dataclass into a Pydantic model so OpenAPI emits a
    proper schema and the front-end gets typed access.
    """

    plugin_kind: PluginKind
    plugin_name: str
    plugin_instance_id: str
    name: str
    description: str = ""
    secret: bool = True
    managed_by: str = ""


class PreflightRequest(BaseModel):
    """Run preflight against an in-memory config payload.

    The Launch modal pulls the project's saved config YAML into JSON
    and POSTs it here so the user sees env errors *before* hitting
    the actual launch endpoint. ``project_env`` is the same dict the
    launcher will merge on top of process env at fork time.
    """

    config: dict[str, Any]
    project_env: dict[str, str] = Field(default_factory=dict)


class PreflightResponse(BaseModel):
    """Result envelope for ``POST /plugins/preflight``."""

    ok: bool
    missing: list[MissingEnvSchema] = Field(default_factory=list)


__all__ = [
    "MissingEnvSchema",
    "PluginKind",
    "PluginListResponse",
    "PluginLoadError",
    "PluginManifest",
    "PreflightRequest",
    "PreflightResponse",
    "RequiredEnvSpec",
]
