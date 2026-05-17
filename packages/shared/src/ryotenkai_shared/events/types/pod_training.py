"""Pod-domain training events.

Nine event types covering the full training lifecycle inside the
trainer subprocess: configuration snapshot at start, epoch/step
boundaries, periodic metric logs, evaluation, checkpointing, and
terminal disposition. Producer: ``ryotenkai_pod.trainer``.

The ``algorithm`` literal mirrors the strategies registered in
``ryotenkai_pod.trainer.strategies`` and is duplicated here intentionally
so the event schema does not develop a runtime import on trainer code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ryotenkai_shared.events.envelope import BaseEvent

Algorithm = Literal["sft", "cpt", "dpo", "grpo", "sapo"]


class TrainingStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int
    num_train_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    algorithm: Algorithm


class TrainingStartedEvent(BaseEvent):
    """Trainer has finished setup and entered the training loop."""

    kind: Literal["ryotenkai.pod.training.started"] = "ryotenkai.pod.training.started"
    severity: Literal["info"] = "info"
    payload: TrainingStartedPayload


class TrainingEpochStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int
    global_step: int


class TrainingEpochStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.training.epoch_started"] = (
        "ryotenkai.pod.training.epoch_started"
    )
    severity: Literal["info"] = "info"
    payload: TrainingEpochStartedPayload


class TrainingEpochCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int
    global_step: int
    mean_loss: float
    duration_s: float


class TrainingEpochCompletedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.training.epoch_completed"] = (
        "ryotenkai.pod.training.epoch_completed"
    )
    severity: Literal["info"] = "info"
    payload: TrainingEpochCompletedPayload


class TrainingStepPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    loss: float
    learning_rate: float
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None


class TrainingStepEvent(BaseEvent):
    """High-frequency per-step snapshot. Severity=debug; consumers gate."""

    kind: Literal["ryotenkai.pod.training.step"] = "ryotenkai.pod.training.step"
    severity: Literal["debug"] = "debug"
    payload: TrainingStepPayload


class TrainingLogPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    metrics: dict[str, float]


class TrainingLogEvent(BaseEvent):
    """Free-form metric log keyed by trainer ``logging_steps``."""

    kind: Literal["ryotenkai.pod.training.log"] = "ryotenkai.pod.training.log"
    severity: Literal["debug"] = "debug"
    payload: TrainingLogPayload


class TrainingEvalMetricsPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    metrics: dict[str, float]
    dataset_name: str


class TrainingEvalMetricsEvent(BaseEvent):
    """Evaluation metrics emitted on ``evaluation_strategy`` boundaries."""

    kind: Literal["ryotenkai.pod.training.eval_metrics"] = (
        "ryotenkai.pod.training.eval_metrics"
    )
    severity: Literal["info"] = "info"
    payload: TrainingEvalMetricsPayload


class TrainingCheckpointSavedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    local_path: str
    size_bytes: int
    is_best: bool


class TrainingCheckpointSavedEvent(BaseEvent):
    """A checkpoint was persisted to local storage."""

    kind: Literal["ryotenkai.pod.training.checkpoint_saved"] = (
        "ryotenkai.pod.training.checkpoint_saved"
    )
    severity: Literal["info"] = "info"
    payload: TrainingCheckpointSavedPayload


class TrainingCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    final_step: int
    mean_loss: float
    duration_s: float
    tokens_processed: int


class TrainingCompletedEvent(BaseEvent):
    """Trainer exited the loop normally."""

    kind: Literal["ryotenkai.pod.training.completed"] = (
        "ryotenkai.pod.training.completed"
    )
    severity: Literal["info"] = "info"
    payload: TrainingCompletedPayload


class TrainingFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str
    message: str
    traceback_excerpt: str
    step: int | None = None


class TrainingFailedEvent(BaseEvent):
    """Trainer aborted with an exception."""

    kind: Literal["ryotenkai.pod.training.failed"] = "ryotenkai.pod.training.failed"
    severity: Literal["error"] = "error"
    payload: TrainingFailedPayload


__all__ = [
    "Algorithm",
    "TrainingCheckpointSavedEvent",
    "TrainingCheckpointSavedPayload",
    "TrainingCompletedEvent",
    "TrainingCompletedPayload",
    "TrainingEpochCompletedEvent",
    "TrainingEpochCompletedPayload",
    "TrainingEpochStartedEvent",
    "TrainingEpochStartedPayload",
    "TrainingEvalMetricsEvent",
    "TrainingEvalMetricsPayload",
    "TrainingFailedEvent",
    "TrainingFailedPayload",
    "TrainingLogEvent",
    "TrainingLogPayload",
    "TrainingStartedEvent",
    "TrainingStartedPayload",
    "TrainingStepEvent",
    "TrainingStepPayload",
]
