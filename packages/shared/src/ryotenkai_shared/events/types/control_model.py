"""Control-domain model retrieval events.

Two event types covering the model_retriever stage: retrieval started
and retrieval completed (with checksum for audit). The "failed" variant
is reused from :mod:`ryotenkai_shared.events.types.control_stage` (the
generic ``StageFailedEvent``) since the model retriever uses standard
stage-failure semantics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent


class ModelRetrievalStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str
    target_path: str


class ModelRetrievalStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.model.retrieval_started"] = (
        "ryotenkai.control.model.retrieval_started"
    )
    severity: Literal["info"] = "info"
    payload: ModelRetrievalStartedPayload


class ModelRetrievalCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bytes_transferred: int
    duration_s: float
    checksum: str


class ModelRetrievalCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.model.retrieval_completed"] = (
        "ryotenkai.control.model.retrieval_completed"
    )
    severity: Literal["info"] = "info"
    payload: ModelRetrievalCompletedPayload


class ModelMetricsBufferRetrievedPayload(BaseModel):
    """Outcome of best-effort buffered MLflow metrics replay.

    Emitted by :meth:`ModelRetriever._retrieve_and_replay_metrics_buffer`
    once the helper finishes one of: replay succeeded, buffer was missing
    on pod (healthy case — trainer's drain succeeded), or buffer was
    oversized / fetch failed. All branches surface counts so operators
    have data parity with the legacy ``on_metrics_buffer_retrieved``
    callback (Phase 4 callback → event migration).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    replayed_count: int
    """Number of metrics replayed into MLflow."""

    line_count: int
    """Total non-empty lines observed in the buffer file (or 0 when missing)."""

    size_bytes: int
    """Buffer file size in bytes as probed on the pod."""

    missing: bool
    """True when the buffer file did not exist on the pod (healthy)."""

    oversized: bool
    """True when the buffer file exceeded the safety cap and was skipped."""


class ModelMetricsBufferRetrievedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.model.metrics_buffer_retrieved"] = (
        "ryotenkai.control.model.metrics_buffer_retrieved"
    )
    severity: Literal["info"] = "info"
    payload: ModelMetricsBufferRetrievedPayload


class MetricsBufferOversizedPayload(BaseModel):
    """Surface a buffer-size escalation in the timeline.

    Emitted by :class:`MetricsBufferRetriever` when the remote buffer
    file exceeds the active threshold. Two distinct outcomes are
    encoded by :attr:`discarded`:

    * ``discarded=False`` — the retriever bumped the in-process
      threshold (still below the ultra-large hard cap) and proceeded
      with the download; operator sees a warning but no data is lost.
    * ``discarded=True`` — the file exceeded the ultra-large hard cap
      and was NOT downloaded; the retriever additionally raises
      :class:`MetricsBufferTooLargeError` so the stage fails honestly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    size_bytes: int
    """Remote buffer size in bytes as probed via ``stat``."""

    threshold_bytes: int
    """Threshold the buffer was compared against (env-tunable)."""

    pod_path: str
    """Absolute remote path of the buffer file (for forensics)."""

    discarded: bool
    """True when the buffer was skipped (file too large to download);
    False when the retriever bumped the threshold and still downloaded."""


class MetricsBufferOversizedEvent(BaseEvent):
    kind: Literal["ryotenkai.control.model.metrics_buffer_oversized"] = (
        "ryotenkai.control.model.metrics_buffer_oversized"
    )
    severity: Literal["warning"] = "warning"
    payload: MetricsBufferOversizedPayload


__all__ = [
    "MetricsBufferOversizedEvent",
    "MetricsBufferOversizedPayload",
    "ModelMetricsBufferRetrievedEvent",
    "ModelMetricsBufferRetrievedPayload",
    "ModelRetrievalCompletedEvent",
    "ModelRetrievalCompletedPayload",
    "ModelRetrievalStartedEvent",
    "ModelRetrievalStartedPayload",
]
