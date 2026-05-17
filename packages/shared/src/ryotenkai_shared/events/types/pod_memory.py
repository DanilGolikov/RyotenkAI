"""Pod-domain GPU memory events.

Four event types covering manual/scheduled cache clears, OOM detection,
threshold-based pressure warnings, and threshold-action gates. Producer:
``ryotenkai_pod.trainer`` (memory manager + OOM detector).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent

CacheClearTrigger = Literal["scheduled", "threshold", "manual"]


class MemoryCacheClearedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    before_bytes: int
    after_bytes: int
    trigger: CacheClearTrigger


class MemoryCacheClearedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.memory.cache_cleared"] = (
        "ryotenkai.pod.memory.cache_cleared"
    )
    severity: Literal["info"] = "info"
    payload: MemoryCacheClearedPayload


class MemoryOOMDetectedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    allocated_bytes: int
    reserved_bytes: int
    step: int | None = None


class MemoryOOMDetectedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.memory.oom_detected"] = (
        "ryotenkai.pod.memory.oom_detected"
    )
    severity: Literal["error"] = "error"
    payload: MemoryOOMDetectedPayload


class MemoryPressureWarningPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    utilization_pct: float
    threshold_pct: float


class MemoryPressureWarningEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.memory.pressure_warning"] = (
        "ryotenkai.pod.memory.pressure_warning"
    )
    severity: Literal["warning"] = "warning"
    payload: MemoryPressureWarningPayload


class MemoryThresholdReachedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    metric: str
    value: float
    threshold: float
    action_taken: str


class MemoryThresholdReachedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.memory.threshold_reached"] = (
        "ryotenkai.pod.memory.threshold_reached"
    )
    severity: Literal["warning"] = "warning"
    payload: MemoryThresholdReachedPayload


__all__ = [
    "CacheClearTrigger",
    "MemoryCacheClearedEvent",
    "MemoryCacheClearedPayload",
    "MemoryOOMDetectedEvent",
    "MemoryOOMDetectedPayload",
    "MemoryPressureWarningEvent",
    "MemoryPressureWarningPayload",
    "MemoryThresholdReachedEvent",
    "MemoryThresholdReachedPayload",
]
