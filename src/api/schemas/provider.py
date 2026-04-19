"""Pydantic schemas for provider registry endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProviderTypeInfo(BaseModel):
    id: str
    label: str
    json_schema: dict[str, Any]


class ProviderTypesResponse(BaseModel):
    types: list[ProviderTypeInfo]


class ProviderSummary(BaseModel):
    id: str
    name: str
    type: str
    path: str
    created_at: str
    description: str = ""


class ProviderDetail(ProviderSummary):
    updated_at: str
    current_config_yaml: str = ""


class CreateProviderRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    type: str = Field(..., min_length=1)
    id: str | None = Field(default=None, description="Slug used on disk. Derived from name when omitted.")
    path: str | None = Field(
        default=None,
        description="Absolute path; when omitted the provider lives under "
        "~/.ryotenkai/providers/<id>/.",
    )
    description: str = ""


class ProviderConfigResponse(BaseModel):
    yaml: str
    parsed_json: dict[str, Any] | None = None


class ProviderSaveConfigRequest(BaseModel):
    yaml: str


class ProviderSaveConfigResponse(BaseModel):
    ok: bool
    snapshot_filename: str | None = None


class ProviderConfigVersion(BaseModel):
    filename: str
    created_at: str
    size_bytes: int


class ProviderConfigVersionsResponse(BaseModel):
    versions: list[ProviderConfigVersion]


class ProviderConfigVersionDetail(BaseModel):
    filename: str
    yaml: str


__all__ = [
    "CreateProviderRequest",
    "ProviderConfigResponse",
    "ProviderConfigVersion",
    "ProviderConfigVersionDetail",
    "ProviderConfigVersionsResponse",
    "ProviderDetail",
    "ProviderSaveConfigRequest",
    "ProviderSaveConfigResponse",
    "ProviderSummary",
    "ProviderTypeInfo",
    "ProviderTypesResponse",
]
