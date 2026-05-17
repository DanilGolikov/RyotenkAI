"""Control-domain stage lifecycle events.

Five event types per pipeline stage: started, completed, failed,
skipped, interrupted. Producer: each stage entry/exit + the signal
handler in the orchestrator.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class StageStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_name: str
    stage_index: int
    total_stages: int
    inputs_summary: dict[str, Any]


class StageStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.stage.started"] = (
        "ryotenkai.control.stage.started"
    )
    severity: Literal["info"] = "info"
    payload: StageStartedPayload


class StageCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_name: str
    duration_s: float
    outputs_summary: dict[str, Any]


class StageCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.stage.completed"] = (
        "ryotenkai.control.stage.completed"
    )
    severity: Literal["info"] = "info"
    payload: StageCompletedPayload


class StageFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_name: str
    error_type: str
    message: str
    traceback_excerpt: str
    retry_count: int = 0


class StageFailedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.stage.failed"] = (
        "ryotenkai.control.stage.failed"
    )
    severity: Literal["error"] = "error"
    payload: StageFailedPayload


class StageSkippedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_name: str
    reason: str


class StageSkippedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.stage.skipped"] = (
        "ryotenkai.control.stage.skipped"
    )
    severity: Literal["info"] = "info"
    payload: StageSkippedPayload


class StageInterruptedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_name: str
    signal: int
    cleanup_completed: bool


class StageInterruptedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.stage.interrupted"] = (
        "ryotenkai.control.stage.interrupted"
    )
    severity: Literal["warning"] = "warning"
    payload: StageInterruptedPayload


__all__ = [
    "StageCompletedEvent",
    "StageCompletedPayload",
    "StageFailedEvent",
    "StageFailedPayload",
    "StageInterruptedEvent",
    "StageInterruptedPayload",
    "StageSkippedEvent",
    "StageSkippedPayload",
    "StageStartedEvent",
    "StageStartedPayload",
]
