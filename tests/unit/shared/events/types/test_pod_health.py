"""Unit tests: pod-health event types."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.pod_health import (
    GPUSnapshot,
    HealthIdleDetectedEvent,
    HealthIdleDetectedPayload,
    HealthMaxLifetimeExceededEvent,
    HealthMaxLifetimeExceededPayload,
    HealthMaxLifetimeReachedEvent,
    HealthMaxLifetimeReachedPayload,
    HealthSnapshotEvent,
    HealthSnapshotPayload,
)


def _snapshot() -> HealthSnapshotEvent:
    return HealthSnapshotEvent(
        source="pod://r/runner",
        run_id="r",
        offset=0,
        payload=HealthSnapshotPayload(
            cpu_pct=25.0,
            ram_bytes=8_000_000_000,
            gpu=[GPUSnapshot(
                device="cuda:0",
                utilization_pct=80.0,
                memory_used_bytes=10_000_000_000,
                memory_total_bytes=24_000_000_000,
                temperature_c=65.0,
            )],
            disk_free_bytes=500_000_000_000,
        ),
    )


def _idle() -> HealthIdleDetectedEvent:
    return HealthIdleDetectedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=1,
        payload=HealthIdleDetectedPayload(
            idle_duration_s=300.0,
            last_activity_at=datetime(2026, 5, 16, tzinfo=UTC),
        ),
    )


def _maxlife() -> HealthMaxLifetimeReachedEvent:
    return HealthMaxLifetimeReachedEvent(
        source="pod://r/runner",
        run_id="r",
        offset=2,
        payload=HealthMaxLifetimeReachedPayload(
            started_at=datetime(2026, 5, 16, tzinfo=UTC),
            max_lifetime_s=7200.0,
        ),
    )


def _maxlife_exceeded() -> HealthMaxLifetimeExceededEvent:
    return HealthMaxLifetimeExceededEvent(
        source="pod://r/runner",
        run_id="r",
        offset=3,
        payload=HealthMaxLifetimeExceededPayload(
            started_at=datetime(2026, 5, 16, tzinfo=UTC),
            max_lifetime_s=7200.0,
            actual_runtime_s=7250.5,
        ),
    )


_ALL = [_snapshot, _idle, _maxlife, _maxlife_exceeded]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original
        assert type(restored) is type(original)


class TestNegative:
    def test_gpu_snapshot_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GPUSnapshot(  # type: ignore[call-arg]
                device="cuda:0",
                utilization_pct=1.0,
                memory_used_bytes=1,
                memory_total_bytes=2,
                temperature_c=70.0,
                extra="x",
            )

    def test_snapshot_payload_requires_gpu_list(self) -> None:
        with pytest.raises(ValidationError):
            HealthSnapshotPayload(  # type: ignore[call-arg]
                cpu_pct=1.0, ram_bytes=1, disk_free_bytes=1,
            )


class TestInvariants:
    def test_snapshot_severity_is_debug(self) -> None:
        assert _snapshot().severity == "debug"

    def test_idle_severity_is_warning(self) -> None:
        assert _idle().severity == "warning"

    def test_maxlife_severity_is_warning(self) -> None:
        assert _maxlife().severity == "warning"

    def test_maxlife_exceeded_severity_is_warning(self) -> None:
        assert _maxlife_exceeded().severity == "warning"

    def test_maxlife_exceeded_kind_pinned(self) -> None:
        assert (
            _maxlife_exceeded().kind
            == "ryotenkai.pod.health.max_lifetime_exceeded"
        )
