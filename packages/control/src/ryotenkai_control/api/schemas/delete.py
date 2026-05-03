from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DeleteMode = Literal["local_and_mlflow", "local_only"]


class DeleteIssueSchema(BaseModel):
    run_dir: str
    phase: str
    message: str


class DeleteResultSchema(BaseModel):
    target: str
    run_dirs: list[str] = Field(default_factory=list)
    deleted_mlflow_run_ids: list[str] = Field(default_factory=list)
    local_deleted: bool
    issues: list[DeleteIssueSchema] = Field(default_factory=list)
    is_success: bool
