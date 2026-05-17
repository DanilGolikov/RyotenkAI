"""Control-domain evaluation pipeline events.

Five event types: outer evaluation lifecycle (started, completed) plus
per-plugin lifecycle (plugin_started, plugin_completed, plugin_failed).
Producer: ``ryotenkai_control.pipeline.stages.model_evaluator`` and the
community evaluator plugin loader.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class EvaluationStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_names: list[str]
    model_path: str


class EvaluationStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.evaluation.started"] = (
        "ryotenkai.control.evaluation.started"
    )
    severity: Literal["info"] = "info"
    payload: EvaluationStartedPayload


class EvaluationPluginStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_name: str
    plugin_version: str


class EvaluationPluginStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.evaluation.plugin_started"] = (
        "ryotenkai.control.evaluation.plugin_started"
    )
    severity: Literal["info"] = "info"
    payload: EvaluationPluginStartedPayload


class EvaluationPluginCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_name: str
    metrics: dict[str, float]
    duration_s: float


class EvaluationPluginCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.evaluation.plugin_completed"] = (
        "ryotenkai.control.evaluation.plugin_completed"
    )
    severity: Literal["info"] = "info"
    payload: EvaluationPluginCompletedPayload


class EvaluationPluginFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_name: str
    error_type: str
    message: str


class EvaluationPluginFailedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.evaluation.plugin_failed"] = (
        "ryotenkai.control.evaluation.plugin_failed"
    )
    severity: Literal["error"] = "error"
    payload: EvaluationPluginFailedPayload


class EvaluationCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    aggregated_metrics: dict[str, float]
    total_duration_s: float


class EvaluationCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.evaluation.completed"] = (
        "ryotenkai.control.evaluation.completed"
    )
    severity: Literal["info"] = "info"
    payload: EvaluationCompletedPayload


__all__ = [
    "EvaluationCompletedEvent",
    "EvaluationCompletedPayload",
    "EvaluationPluginCompletedEvent",
    "EvaluationPluginCompletedPayload",
    "EvaluationPluginFailedEvent",
    "EvaluationPluginFailedPayload",
    "EvaluationPluginStartedEvent",
    "EvaluationPluginStartedPayload",
    "EvaluationStartedEvent",
    "EvaluationStartedPayload",
]
