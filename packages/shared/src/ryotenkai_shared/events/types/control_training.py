"""Control-domain training-monitor stage events.

Two event types covering the training_monitor stage's lifecycle on the
control side: start (with the pod endpoint and poll cadence) and the
"no events from pod for too long" timeout.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic field type
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class TrainingMonitorStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pod_endpoint: str
    poll_interval_s: float


class TrainingMonitorStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.training.monitor_started"] = (
        "ryotenkai.control.training.monitor_started"
    )
    severity: Literal["info"] = "info"
    payload: TrainingMonitorStartedPayload


class TrainingMonitorTimeoutPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    last_event_at: datetime
    timeout_s: float


class TrainingMonitorTimeoutEvent(BaseEvent):
    kind: Literal["ryotenkai.control.training.monitor_timeout"] = (
        "ryotenkai.control.training.monitor_timeout"
    )
    severity: Literal["error"] = "error"
    payload: TrainingMonitorTimeoutPayload


__all__ = [
    "TrainingMonitorStartedEvent",
    "TrainingMonitorStartedPayload",
    "TrainingMonitorTimeoutEvent",
    "TrainingMonitorTimeoutPayload",
]
