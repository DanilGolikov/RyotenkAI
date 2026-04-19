from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

HealthState = Literal["ok", "degraded"]


class HealthStatus(BaseModel):
    status: HealthState
    runs_dir: str
    runs_dir_readable: bool
    version: str = "v0.1.0"
