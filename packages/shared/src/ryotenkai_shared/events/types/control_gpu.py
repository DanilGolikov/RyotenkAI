"""Control-domain GPU provider events.

Nine event types covering the gpu_deployer / provider preflight + cleanup
cycle: deployment lifecycle, preemption, ssh provisioning, code sync,
and post-run cleanup (started / completed / failed).

The cleanup events close a visibility gap surfaced by the post-Phase-10
research pass: ``cleanup_after_run`` (single_node) and ``cleanup_pod``
(RunPod) raise typed errors on failure but never emitted envelopes,
leaving orphaned containers / pods invisible to the unified event
timeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class GPUDeploymentStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    gpu_type: str
    gpu_count: int
    region: str | None = None


class GPUDeploymentStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.deployment_started"] = (
        "ryotenkai.control.gpu.deployment_started"
    )
    severity: Literal["info"] = "info"
    payload: GPUDeploymentStartedPayload


class GPUDeploymentCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_id: str
    endpoint: str
    provision_duration_s: float
    cost_per_hour_usd: float | None = None


class GPUDeploymentCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.deployment_completed"] = (
        "ryotenkai.control.gpu.deployment_completed"
    )
    severity: Literal["info"] = "info"
    payload: GPUDeploymentCompletedPayload


class GPUDeploymentFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    provider_error_code: str | None = None


class GPUDeploymentFailedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.deployment_failed"] = (
        "ryotenkai.control.gpu.deployment_failed"
    )
    severity: Literal["error"] = "error"
    payload: GPUDeploymentFailedPayload


class GPUPreemptedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_id: str
    preemption_reason: str


class GPUPreemptedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.preempted"] = (
        "ryotenkai.control.gpu.preempted"
    )
    severity: Literal["error"] = "error"
    payload: GPUPreemptedPayload


class GPUSSHProvisionedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str
    key_fingerprint: str


class GPUSSHProvisionedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.ssh_provisioned"] = (
        "ryotenkai.control.gpu.ssh_provisioned"
    )
    severity: Literal["info"] = "info"
    payload: GPUSSHProvisionedPayload


class GPUCodeSyncedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    local_sha: str
    remote_sha: str
    bytes_transferred: int


class GPUCodeSyncedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.code_synced"] = (
        "ryotenkai.control.gpu.code_synced"
    )
    severity: Literal["info"] = "info"
    payload: GPUCodeSyncedPayload


# ---------------------------------------------------------------------------
# Cleanup events (post-Phase-10 visibility gap close).
#
# ``provider`` is the canonical provider id — only ``"runpod"`` and
# ``"single_node"`` are emitted today; widening requires a schema bump.
#
# ``reason`` discriminates the upstream trigger so consumers can render
# "natural shutdown" differently from "cancelled mid-run" or "forced
# cleanup after orchestrator panic":
#   * ``natural``   — training completed normally; cleanup runs as the
#                     last step of the happy path.
#   * ``cancelled`` — user / SIGINT triggered cleanup before completion.
#   * ``failed``    — training failed and cleanup runs as part of the
#                     error path.
#   * ``forced``    — orchestrator finally-block / safety-net cleanup
#                     (typically when the normal flow already raised).
# ---------------------------------------------------------------------------

# Canonical provider id literal; matches ``ryotenkai_shared.constants``
# values but kept inline here so the event package stays leaf-free.
CleanupProvider = Literal["runpod", "single_node"]
CleanupReason = Literal["natural", "cancelled", "failed", "forced"]


class GpuCleanupStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: CleanupProvider
    instance_id: str | None = None
    reason: CleanupReason


class GpuCleanupStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.cleanup_started"] = (
        "ryotenkai.control.gpu.cleanup_started"
    )
    severity: Literal["info"] = "info"
    payload: GpuCleanupStartedPayload


class GpuCleanupCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: CleanupProvider
    instance_id: str | None = None
    duration_s: float
    # Open-ended bag of "what was freed" counters. Conventional keys:
    # ``containers`` (number of docker containers removed),
    # ``workspace_bytes`` (bytes freed on disk), ``pods`` (RunPod pods
    # terminated). Empty dict is a valid signal for an idempotent re-run
    # ("nothing to clean").
    resources_freed: dict[str, int]


class GpuCleanupCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.cleanup_completed"] = (
        "ryotenkai.control.gpu.cleanup_completed"
    )
    severity: Literal["info"] = "info"
    payload: GpuCleanupCompletedPayload


class GpuCleanupFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: CleanupProvider
    instance_id: str | None = None
    error_type: str
    message: str
    # True iff cleanup made forward progress before failing (e.g. removed
    # the container but failed to verify, or freed the workspace bytes
    # but a stop-pod call failed). Operators use this to triage the
    # severity of "leak vs. nothing-cleaned".
    partial_cleanup: bool


class GpuCleanupFailedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.gpu.cleanup_failed"] = (
        "ryotenkai.control.gpu.cleanup_failed"
    )
    severity: Literal["error"] = "error"
    payload: GpuCleanupFailedPayload


__all__ = [
    "CleanupProvider",
    "CleanupReason",
    "GPUCodeSyncedEvent",
    "GPUCodeSyncedPayload",
    "GPUDeploymentCompletedEvent",
    "GPUDeploymentCompletedPayload",
    "GPUDeploymentFailedEvent",
    "GPUDeploymentFailedPayload",
    "GPUDeploymentStartedEvent",
    "GPUDeploymentStartedPayload",
    "GPUPreemptedEvent",
    "GPUPreemptedPayload",
    "GPUSSHProvisionedEvent",
    "GPUSSHProvisionedPayload",
    "GpuCleanupCompletedEvent",
    "GpuCleanupCompletedPayload",
    "GpuCleanupFailedEvent",
    "GpuCleanupFailedPayload",
    "GpuCleanupStartedEvent",
    "GpuCleanupStartedPayload",
]
