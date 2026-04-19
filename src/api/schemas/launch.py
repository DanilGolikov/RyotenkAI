from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LaunchMode = Literal["new_run", "fresh", "resume", "restart"]
LaunchLogLevel = Literal["INFO", "DEBUG"]


class LaunchRequestSchema(BaseModel):
    mode: LaunchMode
    config_path: str | None = None
    restart_from_stage: str | None = None
    log_level: LaunchLogLevel = "INFO"


class LaunchResponse(BaseModel):
    pid: int
    launched_at: str
    command: list[str]
    launcher_log: str
    run_dir: str


class InterruptResponse(BaseModel):
    interrupted: bool
    pid: int | None = None
    reason: str | None = None


class RestartPoint(BaseModel):
    stage: str
    available: bool
    mode: str
    reason: str


class RestartPointsResponse(BaseModel):
    config_path: str
    points: list[RestartPoint] = Field(default_factory=list)
