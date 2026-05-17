"""Pod-domain lifecycle events.

Covers runner startup/shutdown, job submission acknowledgement, and the
trainer subprocess fork-and-wait cycle. Producer: the pod runner
HTTP/loopback layer.

Six event types — see the plan's "Event taxonomy" table for the source
of truth on field semantics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class RunnerStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    git_sha: str
    gpu_count: int


class RunnerStartedEvent(BaseEvent):
    """Emitted once when the runner process has finished bootstrapping."""

    kind: Literal["ryotenkai.pod.lifecycle.runner_started"] = (
        "ryotenkai.pod.lifecycle.runner_started"
    )
    severity: Literal["info"] = "info"
    payload: RunnerStartedPayload


class RunnerShutdownPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    graceful: bool


class RunnerShutdownEvent(BaseEvent):
    """Emitted when the runner is exiting (graceful or signal-driven)."""

    kind: Literal["ryotenkai.pod.lifecycle.runner_shutdown"] = (
        "ryotenkai.pod.lifecycle.runner_shutdown"
    )
    severity: Literal["info"] = "info"
    payload: RunnerShutdownPayload


class JobSubmittedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    config_hash: str
    image_tag: str


class JobSubmittedEvent(BaseEvent):
    """Runner accepted a job spec from control."""

    kind: Literal["ryotenkai.pod.lifecycle.job_submitted"] = (
        "ryotenkai.pod.lifecycle.job_submitted"
    )
    severity: Literal["info"] = "info"
    payload: JobSubmittedPayload


class TrainerSpawnedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pid: int
    cmdline: str
    cwd: str


class TrainerSpawnedEvent(BaseEvent):
    """Runner forked the trainer subprocess; pid known."""

    kind: Literal["ryotenkai.pod.lifecycle.trainer_spawned"] = (
        "ryotenkai.pod.lifecycle.trainer_spawned"
    )
    severity: Literal["info"] = "info"
    payload: TrainerSpawnedPayload


class TrainerSpawnFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    exit_code: int | None = None


class TrainerSpawnFailedEvent(BaseEvent):
    """Runner attempted to fork the trainer but the spawn itself failed."""

    kind: Literal["ryotenkai.pod.lifecycle.trainer_spawn_failed"] = (
        "ryotenkai.pod.lifecycle.trainer_spawn_failed"
    )
    severity: Literal["error"] = "error"
    payload: TrainerSpawnFailedPayload


class TrainerExitedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exit_code: int
    signal: int | None = None
    duration_s: float


class TrainerExitedEvent(BaseEvent):
    """Trainer subprocess terminated; carries final disposition."""

    kind: Literal["ryotenkai.pod.lifecycle.trainer_exited"] = (
        "ryotenkai.pod.lifecycle.trainer_exited"
    )
    severity: Literal["info"] = "info"
    payload: TrainerExitedPayload


class StopRequestedPayload(BaseModel):
    """Operator-initiated stop request landed on the supervisor.

    ``grace_seconds`` mirrors the request_stop argument: how long the
    supervisor will wait between SIGTERM and SIGKILL escalation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    grace_seconds: float


class StopRequestedEvent(BaseEvent):
    """Runner accepted a stop request and dispatched SIGTERM."""

    kind: Literal["ryotenkai.pod.lifecycle.stop_requested"] = (
        "ryotenkai.pod.lifecycle.stop_requested"
    )
    severity: Literal["info"] = "info"
    payload: StopRequestedPayload


class PluginsUnpackedPayload(BaseModel):
    """Result of the per-job plugin bundle expansion.

    ``installed`` and ``skipped`` are sorted lists of plugin slugs so the
    payload is stable under round-trip and easy to diff in reports.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    installed: list[str]
    skipped: list[str]
    total_bytes: int


class PluginsUnpackedEvent(BaseEvent):
    """Job's plugin bundle was extracted before the trainer spawned."""

    kind: Literal["ryotenkai.pod.lifecycle.plugins_unpacked"] = (
        "ryotenkai.pod.lifecycle.plugins_unpacked"
    )
    severity: Literal["info"] = "info"
    payload: PluginsUnpackedPayload


__all__ = [
    "JobSubmittedEvent",
    "JobSubmittedPayload",
    "PluginsUnpackedEvent",
    "PluginsUnpackedPayload",
    "RunnerShutdownEvent",
    "RunnerShutdownPayload",
    "RunnerStartedEvent",
    "RunnerStartedPayload",
    "StopRequestedEvent",
    "StopRequestedPayload",
    "TrainerExitedEvent",
    "TrainerExitedPayload",
    "TrainerSpawnFailedEvent",
    "TrainerSpawnFailedPayload",
    "TrainerSpawnedEvent",
    "TrainerSpawnedPayload",
]
