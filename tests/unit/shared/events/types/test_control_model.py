"""Unit tests: control-model retrieval event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_model import (
    MetricsBufferOversizedEvent,
    MetricsBufferOversizedPayload,
    ModelMetricsBufferRetrievedEvent,
    ModelMetricsBufferRetrievedPayload,
    ModelRetrievalCompletedEvent,
    ModelRetrievalCompletedPayload,
    ModelRetrievalStartedEvent,
    ModelRetrievalStartedPayload,
)


def _started() -> ModelRetrievalStartedEvent:
    return ModelRetrievalStartedEvent(
        source="control://orchestrator/model_retriever",
        run_id="r",
        offset=0,
        payload=ModelRetrievalStartedPayload(
            source_path="s3://m/", target_path="/local/m",
        ),
    )


def _completed() -> ModelRetrievalCompletedEvent:
    return ModelRetrievalCompletedEvent(
        source="control://orchestrator/model_retriever",
        run_id="r",
        offset=1,
        payload=ModelRetrievalCompletedPayload(
            bytes_transferred=1_000_000, duration_s=10.0, checksum="abc",
        ),
    )


def _buffer_retrieved() -> ModelMetricsBufferRetrievedEvent:
    return ModelMetricsBufferRetrievedEvent(
        source="control://orchestrator/model_retriever",
        run_id="r",
        offset=2,
        payload=ModelMetricsBufferRetrievedPayload(
            replayed_count=2,
            line_count=2,
            size_bytes=200,
            missing=False,
            oversized=False,
        ),
    )


def _oversized() -> MetricsBufferOversizedEvent:
    return MetricsBufferOversizedEvent(
        source="control://orchestrator/model_retriever/metrics_buffer",
        run_id="r",
        offset=3,
        payload=MetricsBufferOversizedPayload(
            size_bytes=200 * 1024 * 1024,
            threshold_bytes=100 * 1024 * 1024,
            pod_path="/workspace/metrics_buffer.jsonl",
            discarded=True,
        ),
    )


_ALL = [_started, _completed, _buffer_retrieved, _oversized]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original


class TestNegative:
    def test_completed_payload_requires_checksum(self) -> None:
        with pytest.raises(ValidationError):
            ModelRetrievalCompletedPayload(  # type: ignore[call-arg]
                bytes_transferred=1, duration_s=1.0,
            )

    def test_buffer_retrieved_payload_rejects_extra_fields(self) -> None:
        # ``extra="forbid"`` — unknown keys must raise. Pins the
        # schema-level guarantee that an upstream typo on a buffer-replay
        # branch cannot silently smuggle in a new field.
        with pytest.raises(ValidationError):
            ModelMetricsBufferRetrievedPayload(  # type: ignore[call-arg]
                replayed_count=0,
                line_count=0,
                size_bytes=0,
                missing=False,
                oversized=False,
                spurious="oops",
            )

    @pytest.mark.parametrize(
        "missing_field",
        [
            "replayed_count",
            "line_count",
            "size_bytes",
            "missing",
            "oversized",
        ],
    )
    def test_buffer_retrieved_payload_fields_are_required(
        self, missing_field: str,
    ) -> None:
        kwargs: dict[str, object] = {
            "replayed_count": 0,
            "line_count": 0,
            "size_bytes": 0,
            "missing": False,
            "oversized": False,
        }
        kwargs.pop(missing_field)
        with pytest.raises(ValidationError):
            ModelMetricsBufferRetrievedPayload(**kwargs)  # type: ignore[arg-type]


class TestInvariants:
    def test_severities_are_info(self) -> None:
        assert _started().severity == "info"
        assert _completed().severity == "info"
        assert _buffer_retrieved().severity == "info"

    def test_oversized_severity_is_warning(self) -> None:
        assert _oversized().severity == "warning"

    def test_oversized_kind_pinned(self) -> None:
        assert (
            _oversized().kind
            == "ryotenkai.control.model.metrics_buffer_oversized"
        )

    def test_buffer_retrieved_kind_is_pinned(self) -> None:
        # Discriminator must match what the codec dispatches on.
        ev = _buffer_retrieved()
        assert ev.kind == "ryotenkai.control.model.metrics_buffer_retrieved"

    def test_buffer_retrieved_payload_is_frozen(self) -> None:
        payload = ModelMetricsBufferRetrievedPayload(
            replayed_count=1,
            line_count=1,
            size_bytes=10,
            missing=False,
            oversized=False,
        )
        with pytest.raises(ValidationError):
            payload.replayed_count = 99  # type: ignore[misc]
