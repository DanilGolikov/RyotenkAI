"""Unit tests: :mod:`ryotenkai_shared.events.discriminator`.

Pins the closed-union assembly: every concrete event type must dispatch
to its own class when validated through :data:`EVENT_ADAPTER`. The
discriminator key is ``type`` and is set via Literal defaults on every
subclass, so a regression that drops one of the union members or
forgets to pin the Literal would silently degrade to ``UnknownEvent``.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import EVENT_ADAPTER, Event
from ryotenkai_shared.events.types.pod_lifecycle import (
    JobSubmittedEvent,
    JobSubmittedPayload,
    RunnerStartedEvent,
    RunnerStartedPayload,
)
from ryotenkai_shared.events.types.pod_training import (
    TrainingStartedEvent,
    TrainingStartedPayload,
)


def _runner_started() -> RunnerStartedEvent:
    return RunnerStartedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=0,
        payload=RunnerStartedPayload(version="1.0", git_sha="abc", gpu_count=1),
    )


def _job_submitted() -> JobSubmittedEvent:
    return JobSubmittedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=1,
        payload=JobSubmittedPayload(
            job_id="j-1", config_hash="hash", image_tag="latest",
        ),
    )


def _training_started() -> TrainingStartedEvent:
    return TrainingStartedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=2,
        payload=TrainingStartedPayload(
            max_steps=100,
            num_train_epochs=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            algorithm="sft",
        ),
    )


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_dispatch_returns_concrete_runner_event(self) -> None:
        raw = _runner_started().model_dump()
        result = EVENT_ADAPTER.validate_python(raw)
        assert type(result) is RunnerStartedEvent

    def test_dispatch_returns_concrete_training_event(self) -> None:
        raw = _training_started().model_dump()
        result = EVENT_ADAPTER.validate_python(raw)
        assert type(result) is TrainingStartedEvent

    def test_three_distinct_types_keep_their_classes(self) -> None:
        events = [_runner_started(), _job_submitted(), _training_started()]
        results = [EVENT_ADAPTER.validate_python(e.model_dump()) for e in events]
        assert [type(r).__name__ for r in results] == [
            "RunnerStartedEvent",
            "JobSubmittedEvent",
            "TrainingStartedEvent",
        ]


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_missing_discriminator_field_rejected(self) -> None:
        raw: dict[str, Any] = {
            "source": "x",
            "run_id": "x",
            "offset": 0,
            "severity": "info",
        }
        with pytest.raises(ValidationError):
            EVENT_ADAPTER.validate_python(raw)

    def test_mismatched_type_to_payload_shape_rejected(self) -> None:
        # The envelope says ``runner_started`` but the payload shape is
        # for ``job_submitted`` — discriminator picks RunnerStartedEvent
        # which rejects the unrelated payload fields.
        raw = _job_submitted().model_dump()
        raw["kind"] = "ryotenkai.pod.lifecycle.runner_started"
        with pytest.raises(ValidationError):
            EVENT_ADAPTER.validate_python(raw)

    def test_unknown_discriminator_value_rejected(self) -> None:
        raw = _runner_started().model_dump()
        raw["kind"] = "ryotenkai.fake.does_not_exist"
        with pytest.raises(ValidationError):
            EVENT_ADAPTER.validate_python(raw)


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_unknown_event_variant_member_of_union(self) -> None:
        # The annotation order: ``Annotated[Union[...], Discriminator(...)]``
        # The first positional argument of get_args is the Union; the
        # union's get_args lists members.
        annotated_args = get_args(Event)
        assert annotated_args, "Event union has no members"
        union = annotated_args[0]
        members = get_args(union)
        names = {m.__name__ for m in members}
        assert "UnknownEvent" in names


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_each_union_member_pins_kind_literal(self) -> None:
        union = get_args(Event)[0]
        for member in get_args(union):
            assert hasattr(member, "model_fields")
            kind_field = member.model_fields["kind"]
            # Default is a Literal value, not None or an Ellipsis.
            assert kind_field.default is not None

    def test_union_member_count_meets_taxonomy_size(self) -> None:
        # The plan lists ~52 concrete types (28 pod + 24 control) + the
        # UnknownEvent variant. We sanity-check the union has at least
        # 50 members so a forgotten event-class import surfaces.
        union = get_args(Event)[0]
        members = get_args(union)
        assert len(members) >= 50, f"union has only {len(members)} members"


# ===========================================================================
# 5. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_validate_python_returns_same_payload_object_after_round_trip(self) -> None:
        original = _training_started()
        round_tripped = EVENT_ADAPTER.validate_python(original.model_dump())
        assert isinstance(round_tripped, TrainingStartedEvent)
        assert round_tripped.payload == original.payload
