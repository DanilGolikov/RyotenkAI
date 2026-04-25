"""Pydantic schemas for project workspace endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.api.schemas.config_validate import ConfigValidationResult


class ProjectSummary(BaseModel):
    id: str
    name: str
    path: str
    created_at: str
    description: str = ""


class ProjectDetail(BaseModel):
    id: str
    name: str
    path: str
    description: str
    created_at: str
    updated_at: str
    current_config_yaml: str = ""


class UpdateProjectDescriptionRequest(BaseModel):
    description: str = Field(default="", max_length=2000)


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    id: str | None = Field(
        default=None,
        description="Slug used on disk. Derived from name when omitted.",
    )
    path: str | None = Field(
        default=None,
        description="Absolute path; when omitted the project lives under "
        "~/.ryotenkai/projects/<id>/.",
    )
    description: str = ""


class SaveConfigRequest(BaseModel):
    yaml: str


class SaveConfigResponse(BaseModel):
    ok: bool
    snapshot_filename: str | None = None


class StalePluginEntry(BaseModel):
    """One plugin reference in the saved config that no longer matches
    a registered community plugin.

    Mirrors :class:`src.community.stale_plugins.StalePluginRef` —
    reshaped as a Pydantic model so it lands in OpenAPI cleanly.
    """

    plugin_kind: str  # "validation" | "evaluation" | "reward" | "reports"
    plugin_name: str
    instance_id: str
    location: str


class ConfigResponse(BaseModel):
    yaml: str
    parsed_json: dict[str, Any] | None = None
    #: Plugin references in ``yaml`` whose target id is not currently
    #: registered in the community catalog. Populated when the config
    #: parses cleanly enough to walk the plugin enumerators (otherwise
    #: empty — config-level errors surface via the validate endpoint).
    #: The UI renders a "Remove from config" button per row.
    stale_plugins: list[StalePluginEntry] = []


class ConfigVersion(BaseModel):
    filename: str
    created_at: str
    size_bytes: int
    is_favorite: bool = False


class ConfigVersionsResponse(BaseModel):
    versions: list[ConfigVersion]


class ToggleFavoriteRequest(BaseModel):
    favorite: bool


class ToggleFavoriteResponse(BaseModel):
    favorite_versions: list[str]


class ConfigVersionDetail(BaseModel):
    filename: str
    yaml: str


class ConfigValidateResponse(BaseModel):
    result: ConfigValidationResult


class ProjectEnvResponse(BaseModel):
    """Project-scoped environment overrides (``HF_TOKEN`` & friends)."""

    env: dict[str, str] = {}


class ProjectEnvRequest(BaseModel):
    env: dict[str, str]


__all__ = [
    "ConfigResponse",
    "ConfigValidateResponse",
    "ConfigVersion",
    "ConfigVersionDetail",
    "ConfigVersionsResponse",
    "CreateProjectRequest",
    "ProjectDetail",
    "ProjectEnvRequest",
    "ProjectEnvResponse",
    "ProjectSummary",
    "SaveConfigRequest",
    "SaveConfigResponse",
    "ToggleFavoriteRequest",
    "ToggleFavoriteResponse",
    "UpdateProjectDescriptionRequest",
]
