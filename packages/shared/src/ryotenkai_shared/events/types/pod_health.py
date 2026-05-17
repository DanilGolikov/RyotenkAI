"""Pod-domain host/process health events.

Three event types: periodic snapshot, idle detection, and the
max-lifetime watchdog. Producer: runner heartbeat task.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic field type
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class GPUSnapshot(BaseModel):
    """One per attached GPU in :class:`HealthSnapshotPayload`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    utilization_pct: float
    memory_used_bytes: int
    memory_total_bytes: int
    temperature_c: float | None = None


class HealthSnapshotPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_pct: float
    ram_bytes: int
    gpu: list[GPUSnapshot]
    disk_free_bytes: int


class HealthSnapshotEvent(BaseEvent):
    """Periodic resource snapshot. Severity=debug so consumers can gate."""

    kind: Literal["ryotenkai.pod.health.snapshot"] = "ryotenkai.pod.health.snapshot"
    severity: Literal["debug"] = "debug"
    payload: HealthSnapshotPayload


class HealthIdleDetectedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idle_duration_s: float
    last_activity_at: datetime


class HealthIdleDetectedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.health.idle_detected"] = (
        "ryotenkai.pod.health.idle_detected"
    )
    severity: Literal["warning"] = "warning"
    payload: HealthIdleDetectedPayload


class HealthMaxLifetimeReachedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    started_at: datetime
    max_lifetime_s: float


class HealthMaxLifetimeReachedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.health.max_lifetime_reached"] = (
        "ryotenkai.pod.health.max_lifetime_reached"
    )
    severity: Literal["warning"] = "warning"
    payload: HealthMaxLifetimeReachedPayload


class HealthMaxLifetimeExceededPayload(BaseModel):
    """Replacement for the legacy free-form ``max_lifetime`` trigger.

    Carries enough state (start time, configured cap, observed runtime)
    that operators can decide post-facto whether the cap was too low
    or whether the trainer genuinely deserved to be stopped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    started_at: datetime
    max_lifetime_s: float
    actual_runtime_s: float


class HealthMaxLifetimeExceededEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.health.max_lifetime_exceeded"] = (
        "ryotenkai.pod.health.max_lifetime_exceeded"
    )
    severity: Literal["warning"] = "warning"
    payload: HealthMaxLifetimeExceededPayload


__all__ = [
    "GPUSnapshot",
    "HealthIdleDetectedEvent",
    "HealthIdleDetectedPayload",
    "HealthMaxLifetimeExceededEvent",
    "HealthMaxLifetimeExceededPayload",
    "HealthMaxLifetimeReachedEvent",
    "HealthMaxLifetimeReachedPayload",
    "HealthSnapshotEvent",
    "HealthSnapshotPayload",
]
