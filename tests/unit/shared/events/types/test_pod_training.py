"""Unit tests: pod-training event types.

Nine concrete event classes. Pin round-trip, payload extra=forbid,
type Literal and severity defaults — including the high-frequency
``step`` and ``log`` events whose severity is debug (so consumers may
gate them out cheaply).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.pod_training import (
    TrainingCheckpointSavedEvent,
    TrainingCheckpointSavedPayload,
    TrainingCompletedEvent,
    TrainingCompletedPayload,
    TrainingEpochCompletedEvent,
    TrainingEpochCompletedPayload,
    TrainingEpochStartedEvent,
    TrainingEpochStartedPayload,
    TrainingEvalMetricsEvent,
    TrainingEvalMetricsPayload,
    TrainingFailedEvent,
    TrainingFailedPayload,
    TrainingLogEvent,
    TrainingLogPayload,
    TrainingStartedEvent,
    TrainingStartedPayload,
    TrainingStepEvent,
    TrainingStepPayload,
)


def _started() -> TrainingStartedEvent:
    return TrainingStartedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=0,
        payload=TrainingStartedPayload(
            max_steps=10,
            num_train_epochs=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            algorithm="sft",
        ),
    )


def _epoch_started() -> TrainingEpochStartedEvent:
    return TrainingEpochStartedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=1,
        payload=TrainingEpochStartedPayload(epoch=0, global_step=0),
    )


def _epoch_completed() -> TrainingEpochCompletedEvent:
    return TrainingEpochCompletedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=2,
        payload=TrainingEpochCompletedPayload(
            epoch=0, global_step=100, mean_loss=0.5, duration_s=30.0,
        ),
    )


def _step() -> TrainingStepEvent:
    return TrainingStepEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=3,
        payload=TrainingStepPayload(step=5, loss=0.3, learning_rate=1e-4),
    )


def _log() -> TrainingLogEvent:
    return TrainingLogEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=4,
        payload=TrainingLogPayload(step=5, metrics={"loss": 0.3}),
    )


def _eval() -> TrainingEvalMetricsEvent:
    return TrainingEvalMetricsEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=5,
        payload=TrainingEvalMetricsPayload(
            step=10, metrics={"acc": 0.7}, dataset_name="val",
        ),
    )


def _checkpoint() -> TrainingCheckpointSavedEvent:
    return TrainingCheckpointSavedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=6,
        payload=TrainingCheckpointSavedPayload(
            step=100, local_path="/ckpt/100", size_bytes=12345, is_best=True,
        ),
    )


def _completed() -> TrainingCompletedEvent:
    return TrainingCompletedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=7,
        payload=TrainingCompletedPayload(
            final_step=100, mean_loss=0.2, duration_s=600.0, tokens_processed=1_000_000,
        ),
    )


def _failed() -> TrainingFailedEvent:
    return TrainingFailedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=8,
        payload=TrainingFailedPayload(
            error_type="RuntimeError",
            message="OOM",
            traceback_excerpt="Traceback ...",
            step=50,
        ),
    )


_ALL = [
    _started,
    _epoch_started,
    _epoch_completed,
    _step,
    _log,
    _eval,
    _checkpoint,
    _completed,
    _failed,
]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_started_payload_rejects_unknown_algorithm(self) -> None:
        with pytest.raises(ValidationError):
            TrainingStartedPayload(  # type: ignore[arg-type]
                max_steps=1,
                num_train_epochs=1,
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                algorithm="kfac",  # not in Literal
            )

    def test_log_payload_rejects_non_float_value(self) -> None:
        with pytest.raises(ValidationError):
            TrainingLogPayload(step=1, metrics={"loss": "nope"})  # type: ignore[dict-item]

    def test_step_payload_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingStepPayload(  # type: ignore[call-arg]
                step=1, loss=0.5, learning_rate=1e-4, mystery_field=True,
            )


class TestInvariants:
    def test_step_severity_is_debug(self) -> None:
        # High-frequency events default to debug so consumers can gate.
        assert _step().severity == "debug"
        assert _log().severity == "debug"

    def test_failed_severity_is_error(self) -> None:
        assert _failed().severity == "error"

    def test_started_kind_pinned(self) -> None:
        assert _started().kind == "ryotenkai.pod.training.started"

    def test_all_kinds_unique(self) -> None:
        kinds = {factory().kind for factory in _ALL}
        assert len(kinds) == len(_ALL)
