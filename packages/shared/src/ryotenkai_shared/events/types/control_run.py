"""Control-domain run lifecycle events.

Four event types covering the full pipeline run lifecycle: started,
completed, failed, cancelled. Producer: ``ryotenkai_control`` orchestrator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent
from ryotenkai_shared.events.types.pod_training import Algorithm  # noqa: TC001 — Pydantic field type


class RunStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str
    algorithm: Algorithm
    model_id: str
    dataset_id: str
    config_hash: str


class RunStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.run.started"] = "ryotenkai.control.run.started"
    severity: Literal["info"] = "info"
    payload: RunStartedPayload


class RunCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    duration_s: float
    final_status: str
    mlflow_run_id: str | None = None


class RunCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.run.completed"] = (
        "ryotenkai.control.run.completed"
    )
    severity: Literal["info"] = "info"
    payload: RunCompletedPayload


class RunFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    failing_stage: str
    error_type: str
    message: str
    traceback_excerpt: str


class RunFailedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.run.failed"] = "ryotenkai.control.run.failed"
    severity: Literal["error"] = "error"
    payload: RunFailedPayload


class RunCancelledPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    cancelled_at_stage: str | None = None


class RunCancelledEvent(BaseEvent):
    kind: Literal["ryotenkai.control.run.cancelled"] = (
        "ryotenkai.control.run.cancelled"
    )
    severity: Literal["warning"] = "warning"
    payload: RunCancelledPayload


__all__ = [
    "RunCancelledEvent",
    "RunCancelledPayload",
    "RunCompletedEvent",
    "RunCompletedPayload",
    "RunFailedEvent",
    "RunFailedPayload",
    "RunStartedEvent",
    "RunStartedPayload",
]
