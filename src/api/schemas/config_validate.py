from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CheckStatus = Literal["ok", "warn", "fail"]


class ConfigCheck(BaseModel):
    label: str
    status: CheckStatus
    detail: str = ""


class ConfigValidationResult(BaseModel):
    ok: bool
    config_path: str
    checks: list[ConfigCheck] = Field(default_factory=list)
