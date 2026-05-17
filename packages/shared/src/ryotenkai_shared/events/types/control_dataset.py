"""Control-domain dataset validation events.

Six event types covering the dataset_validator stage:

* outer lifecycle — ``validation_started`` / ``validation_completed`` /
  ``validation_failed`` (one per stage invocation);
* per-plugin lifecycle — ``validation_plugin_started`` /
  ``validation_plugin_completed`` / ``validation_plugin_failed`` (one
  trio per plugin run, mirroring the evaluation stage's per-plugin
  emission so timeline consumers can render uniform progress).

The per-plugin events close a visibility gap surfaced by the
post-Phase-10 research pass: prior to this, per-plugin results lived
only in the ``dataset_validator_results.json`` artifact, invisible to
the unified event timeline and SSE consumers.

Producer: ``ryotenkai_control.pipeline.stages.dataset_validator``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class DatasetValidationStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_path: str
    validator_chain: list[str]


class DatasetValidationStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.dataset.validation_started"] = (
        "ryotenkai.control.dataset.validation_started"
    )
    severity: Literal["info"] = "info"
    payload: DatasetValidationStartedPayload


class DatasetValidationCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    num_samples: int
    num_rejected: int
    schema_version: str
    checks_passed: list[str]


class DatasetValidationCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.dataset.validation_completed"] = (
        "ryotenkai.control.dataset.validation_completed"
    )
    severity: Literal["info"] = "info"
    payload: DatasetValidationCompletedPayload


class DatasetValidationFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    failed_check: str
    details: str


class DatasetValidationFailedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.dataset.validation_failed"] = (
        "ryotenkai.control.dataset.validation_failed"
    )
    severity: Literal["error"] = "error"
    payload: DatasetValidationFailedPayload


# ---------------------------------------------------------------------------
# Per-plugin events (post-Phase-10 visibility gap close).
#
# Mirrors ``control_evaluation`` per-plugin emission shape so timeline
# consumers can render dataset-validation progress with the same
# "started -> completed | failed" UI primitive used for the evaluator
# stage.
#
# ``num_checked`` / ``num_passed`` / ``num_failed`` are derived from
# :class:`ValidationResult`'s ``error_groups`` / ``passed`` /
# ``metrics`` triplet at the emit site (see
# ``pipeline.stages.dataset_validator.plugin_runner.PluginRunner``) —
# the underlying dataclass does not surface them directly because
# different plugins (size, format, diversity) compute "checked" against
# different denominators (samples vs. characters vs. ngrams).
# ---------------------------------------------------------------------------


class DatasetValidationPluginStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_name: str
    plugin_version: str
    dataset_path: str


class DatasetValidationPluginStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.dataset.validation_plugin_started"] = (
        "ryotenkai.control.dataset.validation_plugin_started"
    )
    severity: Literal["info"] = "info"
    payload: DatasetValidationPluginStartedPayload


class DatasetValidationPluginCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_name: str
    num_checked: int
    num_passed: int
    num_failed: int
    duration_s: float


class DatasetValidationPluginCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.dataset.validation_plugin_completed"] = (
        "ryotenkai.control.dataset.validation_plugin_completed"
    )
    severity: Literal["info"] = "info"
    payload: DatasetValidationPluginCompletedPayload


class DatasetValidationPluginFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_name: str
    error_type: str
    message: str
    # Truncated to ~2KB at emit time so the journal row stays bounded
    # (full tracebacks belong in logs / artifact, not the SSE timeline).
    traceback_excerpt: str


class DatasetValidationPluginFailedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.dataset.validation_plugin_failed"] = (
        "ryotenkai.control.dataset.validation_plugin_failed"
    )
    severity: Literal["error"] = "error"
    payload: DatasetValidationPluginFailedPayload


__all__ = [
    "DatasetValidationCompletedEvent",
    "DatasetValidationCompletedPayload",
    "DatasetValidationFailedEvent",
    "DatasetValidationFailedPayload",
    "DatasetValidationPluginCompletedEvent",
    "DatasetValidationPluginCompletedPayload",
    "DatasetValidationPluginFailedEvent",
    "DatasetValidationPluginFailedPayload",
    "DatasetValidationPluginStartedEvent",
    "DatasetValidationPluginStartedPayload",
    "DatasetValidationStartedEvent",
    "DatasetValidationStartedPayload",
]
