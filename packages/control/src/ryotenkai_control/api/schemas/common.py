from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    detail: str
    code: str | None = None
    field_errors: dict[str, str] | None = Field(default=None, description="Per-field validation errors")
