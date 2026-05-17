"""Unit tests: pod-io event types (stdout / stderr passthrough)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.pod_io import (
    TrainerStderrEvent,
    TrainerStderrPayload,
    TrainerStdoutEvent,
    TrainerStdoutPayload,
)


def _stdout() -> TrainerStdoutEvent:
    return TrainerStdoutEvent(
        source="pod://r/runner",
        run_id="r",
        offset=0,
        payload=TrainerStdoutPayload(line="step 1"),
    )


def _stderr() -> TrainerStderrEvent:
    return TrainerStderrEvent(
        source="pod://r/runner",
        run_id="r",
        offset=1,
        payload=TrainerStderrPayload(line="warn"),
    )


_ALL = [_stdout, _stderr]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_stdout_stream_field_pinned_to_stdout_literal(self) -> None:
        # ``stream`` is a Literal — supplying a different value must fail.
        with pytest.raises(ValidationError):
            TrainerStdoutPayload(line="x", stream="stderr")  # type: ignore[arg-type]

    def test_stderr_stream_field_pinned_to_stderr_literal(self) -> None:
        with pytest.raises(ValidationError):
            TrainerStderrPayload(line="x", stream="stdout")  # type: ignore[arg-type]


class TestInvariants:
    def test_both_io_events_default_severity_debug(self) -> None:
        assert _stdout().severity == "debug"
        assert _stderr().severity == "debug"

    def test_stream_defaults_pin_per_class(self) -> None:
        assert _stdout().payload.stream == "stdout"
        assert _stderr().payload.stream == "stderr"
