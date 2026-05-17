"""Unit tests: pod-memory event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.pod_memory import (
    MemoryCacheClearedEvent,
    MemoryCacheClearedPayload,
    MemoryOOMDetectedEvent,
    MemoryOOMDetectedPayload,
    MemoryPressureWarningEvent,
    MemoryPressureWarningPayload,
    MemoryThresholdReachedEvent,
    MemoryThresholdReachedPayload,
)


def _cache() -> MemoryCacheClearedEvent:
    return MemoryCacheClearedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=0,
        payload=MemoryCacheClearedPayload(
            device="cuda:0", before_bytes=10_000_000_000, after_bytes=2_000_000_000,
            trigger="threshold",
        ),
    )


def _oom() -> MemoryOOMDetectedEvent:
    return MemoryOOMDetectedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=1,
        payload=MemoryOOMDetectedPayload(
            device="cuda:0", allocated_bytes=22_000_000_000, reserved_bytes=24_000_000_000, step=42,
        ),
    )


def _pressure() -> MemoryPressureWarningEvent:
    return MemoryPressureWarningEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=2,
        payload=MemoryPressureWarningPayload(
            device="cuda:0", utilization_pct=88.0, threshold_pct=85.0,
        ),
    )


def _threshold() -> MemoryThresholdReachedEvent:
    return MemoryThresholdReachedEvent(
        source="pod://r/trainer",
        run_id="r",
        offset=3,
        payload=MemoryThresholdReachedPayload(
            device="cuda:0",
            metric="utilization_pct",
            value=92.0,
            threshold=90.0,
            action_taken="cache_clear",
        ),
    )


_ALL = [_cache, _oom, _pressure, _threshold]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_cache_payload_rejects_unknown_trigger(self) -> None:
        with pytest.raises(ValidationError):
            MemoryCacheClearedPayload(  # type: ignore[arg-type]
                device="cuda:0",
                before_bytes=0,
                after_bytes=0,
                trigger="kernel_panic",  # not in Literal
            )

    def test_threshold_payload_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MemoryThresholdReachedPayload(  # type: ignore[call-arg]
                device="cuda:0",
                metric="x",
                value=1.0,
                threshold=1.0,
                action_taken="y",
                bonus_field=True,
            )


class TestInvariants:
    def test_oom_severity_is_error(self) -> None:
        assert _oom().severity == "error"

    def test_pressure_severity_is_warning(self) -> None:
        assert _pressure().severity == "warning"

    def test_cache_severity_is_info(self) -> None:
        assert _cache().severity == "info"

    def test_threshold_severity_is_warning(self) -> None:
        assert _threshold().severity == "warning"
