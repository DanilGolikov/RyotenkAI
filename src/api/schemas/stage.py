from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StageArtifactResponse(BaseModel):
    stage: str
    status: str
    started_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(description="Where artifact came from: 'local' | 'mlflow'")
