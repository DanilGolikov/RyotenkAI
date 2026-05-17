"""Control-domain inference deployment events.

Phase 5 (coverage-gap migration, 2026-05-16): the original four event
types (deployment_started, health_check_completed, deployed,
deactivated) are joined by two helpers that close gaps in the previous
shape:

* :class:`InferenceHealthCheckStartedEvent` — fires once when the
  inference_deployer enters its readiness loop. Producers can correlate
  this with the existing ``health_check_completed`` envelope to derive
  cold-start latency directly from the event stream rather than from
  log scraping. Severity is ``debug`` because most users only care
  about the terminal completion event; the start envelope is
  diagnostic.
* :class:`InferenceDeploymentFailedEvent` — terminal failure envelope
  that the legacy ``MLflowEventLog`` pathway used to surface as a
  string-typed error log. Splitting it out gives reports / alerts a
  typed signal to match on without parsing the failure message.

The producers (``ryotenkai_control.pipeline.stages.inference_deployer``
and ``managers.deployment_manager``) emit envelopes via the unified
:class:`IEventEmitter`; the MLflow legacy path remains in place until
Phase 6 retires ``InferenceEventLogger`` Protocol.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent

InferenceTarget = Literal["vllm", "sglang", "hf_endpoint"]


class InferenceDeploymentStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: InferenceTarget
    model_path: str


class InferenceDeploymentStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.inference.deployment_started"] = (
        "ryotenkai.control.inference.deployment_started"
    )
    severity: Literal["info"] = "info"
    payload: InferenceDeploymentStartedPayload


class InferenceHealthCheckStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str
    timeout_s: float


class InferenceHealthCheckStartedEvent(BaseEvent):
    """Emitted once when inference_deployer enters its readiness loop.

    Severity is ``debug`` because the terminal
    :class:`InferenceHealthCheckCompletedEvent` is what most consumers
    surface; the started envelope is diagnostic / correlation-only.
    """

    kind: Literal["ryotenkai.control.inference.health_check_started"] = (
        "ryotenkai.control.inference.health_check_started"
    )
    severity: Literal["debug"] = "debug"
    payload: InferenceHealthCheckStartedPayload


class InferenceHealthCheckCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str
    latency_ms: float
    model_loaded: bool


class InferenceHealthCheckCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.inference.health_check_completed"] = (
        "ryotenkai.control.inference.health_check_completed"
    )
    severity: Literal["info"] = "info"
    payload: InferenceHealthCheckCompletedPayload


class InferenceDeployedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str
    api_key_ref: str | None = None
    model_id: str


class InferenceDeployedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.inference.deployed"] = (
        "ryotenkai.control.inference.deployed"
    )
    severity: Literal["info"] = "info"
    payload: InferenceDeployedPayload


class InferenceDeactivatedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: str
    reason: str


class InferenceDeactivatedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.inference.deactivated"] = (
        "ryotenkai.control.inference.deactivated"
    )
    severity: Literal["info"] = "info"
    payload: InferenceDeactivatedPayload


class InferenceDeploymentFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: InferenceTarget
    reason: str
    error_type: str


class InferenceDeploymentFailedEvent(BaseEvent):
    """Terminal failure envelope for ``InferenceDeployer.execute``.

    Replaces the legacy ``event_logger.log_event_error("...")`` MLflow
    string artifact: reports / alerts can now match on
    ``ryotenkai.control.inference.deployment_failed`` without parsing
    error messages.
    """

    kind: Literal["ryotenkai.control.inference.deployment_failed"] = (
        "ryotenkai.control.inference.deployment_failed"
    )
    severity: Literal["error"] = "error"
    payload: InferenceDeploymentFailedPayload


__all__ = [
    "InferenceDeactivatedEvent",
    "InferenceDeactivatedPayload",
    "InferenceDeployedEvent",
    "InferenceDeployedPayload",
    "InferenceDeploymentFailedEvent",
    "InferenceDeploymentFailedPayload",
    "InferenceDeploymentStartedEvent",
    "InferenceDeploymentStartedPayload",
    "InferenceHealthCheckCompletedEvent",
    "InferenceHealthCheckCompletedPayload",
    "InferenceHealthCheckStartedEvent",
    "InferenceHealthCheckStartedPayload",
    "InferenceTarget",
]
