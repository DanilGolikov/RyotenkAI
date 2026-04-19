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


class ConfigResponse(BaseModel):
    yaml: str
    parsed_json: dict[str, Any] | None = None


class ConfigVersion(BaseModel):
    filename: str
    created_at: str
    size_bytes: int


class ConfigVersionsResponse(BaseModel):
    versions: list[ConfigVersion]


class ConfigVersionDetail(BaseModel):
    filename: str
    yaml: str


class ConfigValidateResponse(BaseModel):
    result: ConfigValidationResult


__all__ = [
    "ConfigResponse",
    "ConfigValidateResponse",
    "ConfigVersion",
    "ConfigVersionDetail",
    "ConfigVersionsResponse",
    "CreateProjectRequest",
    "ProjectDetail",
    "ProjectSummary",
    "SaveConfigRequest",
    "SaveConfigResponse",
]
