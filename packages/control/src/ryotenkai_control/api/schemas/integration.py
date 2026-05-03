"""Pydantic schemas for integration registry endpoints.

Note: no response model here exposes a ``token`` field. Tokens live in
``token.enc`` inside each integration's workspace and are never returned
through the API (see ``src/tests/integration/api/test_no_secret_leaks.py``
for the invariant check).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IntegrationTypeInfo(BaseModel):
    id: str
    label: str
    requires_token: bool
    json_schema: dict[str, Any]


class IntegrationTypesResponse(BaseModel):
    types: list[IntegrationTypeInfo]


class IntegrationSummary(BaseModel):
    id: str
    name: str
    type: str
    path: str
    created_at: str
    description: str = ""
    has_token: bool = False


class IntegrationDetail(IntegrationSummary):
    updated_at: str
    current_config_yaml: str = ""


class CreateIntegrationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    type: str = Field(..., min_length=1)
    id: str | None = Field(
        default=None,
        description="Slug used on disk. Derived from name when omitted.",
    )
    path: str | None = Field(
        default=None,
        description="Absolute path; when omitted the integration lives under "
        "~/.ryotenkai/integrations/<id>/.",
    )
    description: str = ""


class IntegrationConfigResponse(BaseModel):
    yaml: str
    parsed_json: dict[str, Any] | None = None


class IntegrationSaveConfigRequest(BaseModel):
    yaml: str


class IntegrationSaveConfigResponse(BaseModel):
    ok: bool
    snapshot_filename: str | None = None


class IntegrationConfigVersion(BaseModel):
    filename: str
    created_at: str
    size_bytes: int


class IntegrationConfigVersionsResponse(BaseModel):
    versions: list[IntegrationConfigVersion]


class IntegrationConfigVersionDetail(BaseModel):
    filename: str
    yaml: str


class IntegrationTokenRequest(BaseModel):
    """Body for ``PUT /integrations/{id}/token``.

    The ``token`` field is write-only — responses never echo it back.
    """

    token: str = Field(..., min_length=1)


class ConnectionTestResult(BaseModel):
    """Outcome of a ``POST …/test-connection`` call.

    Always returned with HTTP 200; network failures show up as ``ok=false``
    with a ``detail`` message.
    """

    ok: bool
    latency_ms: int | None = None
    detail: str = ""


__all__ = [
    "ConnectionTestResult",
    "CreateIntegrationRequest",
    "IntegrationConfigResponse",
    "IntegrationConfigVersion",
    "IntegrationConfigVersionDetail",
    "IntegrationConfigVersionsResponse",
    "IntegrationDetail",
    "IntegrationSaveConfigRequest",
    "IntegrationSaveConfigResponse",
    "IntegrationSummary",
    "IntegrationTokenRequest",
    "IntegrationTypeInfo",
    "IntegrationTypesResponse",
]
