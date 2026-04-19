from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StageRun(BaseModel):
    stage_name: str
    status: str
    status_icon: str | None = None
    status_color: str | None = None
    execution_mode: str | None = None
    mode_label: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    failure_kind: str | None = None
    reuse_from: dict[str, Any] | None = None
    skip_reason: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None


class AttemptDetail(BaseModel):
    attempt_id: str
    attempt_no: int
    runtime_name: str
    requested_action: str
    effective_action: str
    restart_from_stage: str | None = None
    status: str
    status_icon: str | None = None
    status_color: str | None = None
    started_at: str
    completed_at: str | None = None
    error: str | None = None
    training_critical_config_hash: str = ""
    late_stage_config_hash: str = ""
    model_dataset_config_hash: str = ""
    root_mlflow_run_id: str | None = None
    pipeline_attempt_mlflow_run_id: str | None = None
    training_run_id: str | None = None
    enabled_stage_names: list[str] = Field(default_factory=list)
    stage_runs: dict[str, StageRun] = Field(default_factory=dict)
    duration_seconds: float | None = None


class StagesResponse(BaseModel):
    stages: list[StageRun] = Field(default_factory=list)
