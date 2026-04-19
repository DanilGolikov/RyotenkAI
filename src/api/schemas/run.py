from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    run_id: str
    run_dir: str
    created_at: str
    created_ts: float
    status: str
    status_icon: str | None = None
    status_color: str | None = None
    attempts: int
    config_name: str
    mlflow_run_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    group: str = "(root)"


class RunsListResponse(BaseModel):
    runs_dir: str
    groups: dict[str, list[RunSummary]] = Field(default_factory=dict)


class LineageRefSchema(BaseModel):
    attempt_id: str
    stage_name: str
    outputs: dict[str, Any] = Field(default_factory=dict)


class RunDetail(BaseModel):
    schema_version: int
    logical_run_id: str
    run_directory: str
    config_path: str
    config_abspath: str | None = None
    active_attempt_id: str | None = None
    pipeline_status: str
    training_critical_config_hash: str = ""
    late_stage_config_hash: str = ""
    model_dataset_config_hash: str = ""
    root_mlflow_run_id: str | None = None
    mlflow_runtime_tracking_uri: str | None = None
    mlflow_ca_bundle_path: str | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    current_output_lineage: dict[str, LineageRefSchema] = Field(default_factory=dict)

    # derived
    status: str
    status_icon: str | None = None
    status_color: str | None = None
    running_attempt_no: int | None = None
    next_attempt_no: int = 1
    is_locked: bool = False
    lock_pid: int | None = None


class CreateRunRequest(BaseModel):
    run_id: str | None = Field(default=None, description="Explicit run id; auto-suggested when omitted")
    subgroup: str | None = Field(default=None, description="Optional subgroup relative to runs_dir")
