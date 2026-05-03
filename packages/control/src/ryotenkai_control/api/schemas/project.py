"""Pydantic schemas for project workspace endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ryotenkai_control.api.schemas.config_validate import ConfigValidationResult


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


# ---------------------------------------------------------------------------
# Project runs
# ---------------------------------------------------------------------------


class ProjectRunEntry(BaseModel):
    """Summary of one run that lives inside a project's ``runs/`` dir.

    Built by walking ``<project>/runs/`` and reading each subdir's
    ``pipeline_state.json``. ``run_id`` / ``started_at`` / ``status``
    are required; the rest are optional audit breadcrumbs surfaced to
    the UI when present in ``state.metadata``.
    """

    run_id: str
    started_at: str
    status: str
    finished_at: str | None = None
    mlflow_run_id: str | None = None
    config_version_hash: str | None = None
    actor: str | None = None
    run_directory: str | None = None


class ProjectRunsResponse(BaseModel):
    runs: list[ProjectRunEntry] = []


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
    "ProjectRunEntry",
    "ProjectRunsResponse",
    "ProjectSummary",
    "SaveConfigRequest",
    "SaveConfigResponse",
    "ToggleFavoriteRequest",
    "ToggleFavoriteResponse",
    "UpdateProjectDescriptionRequest",
]
