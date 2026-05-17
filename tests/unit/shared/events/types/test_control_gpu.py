"""Unit tests: control-GPU provider event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_gpu import (
    GPUCodeSyncedEvent,
    GPUCodeSyncedPayload,
    GPUDeploymentCompletedEvent,
    GPUDeploymentCompletedPayload,
    GPUDeploymentFailedEvent,
    GPUDeploymentFailedPayload,
    GPUDeploymentStartedEvent,
    GPUDeploymentStartedPayload,
    GPUPreemptedEvent,
    GPUPreemptedPayload,
    GPUSSHProvisionedEvent,
    GPUSSHProvisionedPayload,
    GpuCleanupCompletedEvent,
    GpuCleanupCompletedPayload,
    GpuCleanupFailedEvent,
    GpuCleanupFailedPayload,
    GpuCleanupStartedEvent,
    GpuCleanupStartedPayload,
)


def _started() -> GPUDeploymentStartedEvent:
    return GPUDeploymentStartedEvent(
        source="control://orchestrator/gpu_deployer",
        run_id="r",
        offset=0,
        payload=GPUDeploymentStartedPayload(
            provider="runpod", gpu_type="A100", gpu_count=1, region="us-east",
        ),
    )


def _completed() -> GPUDeploymentCompletedEvent:
    return GPUDeploymentCompletedEvent(
        source="control://orchestrator/gpu_deployer",
        run_id="r",
        offset=1,
        payload=GPUDeploymentCompletedPayload(
            instance_id="i-1", endpoint="ssh://host", provision_duration_s=60.0,
            cost_per_hour_usd=1.5,
        ),
    )


def _failed() -> GPUDeploymentFailedEvent:
    return GPUDeploymentFailedEvent(
        source="control://orchestrator/gpu_deployer",
        run_id="r",
        offset=2,
        payload=GPUDeploymentFailedPayload(reason="capacity"),
    )


def _preempted() -> GPUPreemptedEvent:
    return GPUPreemptedEvent(
        source="control://orchestrator/gpu_deployer",
        run_id="r",
        offset=3,
        payload=GPUPreemptedPayload(instance_id="i-1", preemption_reason="spot reclaim"),
    )


def _ssh() -> GPUSSHProvisionedEvent:
    return GPUSSHProvisionedEvent(
        source="control://orchestrator/gpu_deployer",
        run_id="r",
        offset=4,
        payload=GPUSSHProvisionedPayload(host="1.2.3.4", key_fingerprint="ed25519:abc"),
    )


def _code() -> GPUCodeSyncedEvent:
    return GPUCodeSyncedEvent(
        source="control://orchestrator/gpu_deployer",
        run_id="r",
        offset=5,
        payload=GPUCodeSyncedPayload(
            local_sha="abc", remote_sha="abc", bytes_transferred=1_000_000,
        ),
    )


def _cleanup_started() -> GpuCleanupStartedEvent:
    return GpuCleanupStartedEvent(
        source="control://orchestrator/single_node_provider",
        run_id="r",
        offset=6,
        payload=GpuCleanupStartedPayload(
            provider="single_node", instance_id="ryotenkai_training_r", reason="natural",
        ),
    )


def _cleanup_completed() -> GpuCleanupCompletedEvent:
    return GpuCleanupCompletedEvent(
        source="control://orchestrator/runpod_provider",
        run_id="r",
        offset=7,
        payload=GpuCleanupCompletedPayload(
            provider="runpod",
            instance_id="pod-abc",
            duration_s=4.2,
            resources_freed={"pods": 1, "containers": 0},
        ),
    )


def _cleanup_failed() -> GpuCleanupFailedEvent:
    return GpuCleanupFailedEvent(
        source="control://orchestrator/single_node_provider",
        run_id="r",
        offset=8,
        payload=GpuCleanupFailedPayload(
            provider="single_node",
            instance_id="ryotenkai_training_r",
            error_type="ProviderUnavailableError",
            message="docker rm reported success but container still listed",
            partial_cleanup=True,
        ),
    )


_ALL = [_started, _completed, _failed, _preempted, _ssh, _code]
_ALL_CLEANUP = [_cleanup_started, _cleanup_completed, _cleanup_failed]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL + _ALL_CLEANUP, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original


class TestNegative:
    def test_failed_payload_provider_error_code_optional(self) -> None:
        payload = GPUDeploymentFailedPayload(reason="x")
        assert payload.provider_error_code is None

    def test_started_payload_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GPUDeploymentStartedPayload(  # type: ignore[call-arg]
                provider="x", gpu_type="y", gpu_count=1, fake_field=1,
            )

    def test_cleanup_started_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GpuCleanupStartedPayload(  # type: ignore[call-arg]
                provider="runpod", reason="natural", bogus=True,
            )

    def test_cleanup_completed_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GpuCleanupCompletedPayload(  # type: ignore[call-arg]
                provider="runpod",
                duration_s=1.0,
                resources_freed={"pods": 1},
                rogue_field=42,
            )

    def test_cleanup_failed_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GpuCleanupFailedPayload(  # type: ignore[call-arg]
                provider="runpod",
                error_type="X",
                message="m",
                partial_cleanup=False,
                stray="x",
            )

    def test_cleanup_started_rejects_unknown_provider(self) -> None:
        # Literal["runpod","single_node"] — anything else is rejected
        # at the Pydantic boundary.
        with pytest.raises(ValidationError):
            GpuCleanupStartedPayload(provider="gcp", reason="natural")  # type: ignore[arg-type]

    def test_cleanup_started_rejects_unknown_reason(self) -> None:
        with pytest.raises(ValidationError):
            GpuCleanupStartedPayload(provider="runpod", reason="quarterly")  # type: ignore[arg-type]


class TestInvariants:
    def test_failed_and_preempted_are_errors(self) -> None:
        assert _failed().severity == "error"
        assert _preempted().severity == "error"

    def test_info_events_are_info(self) -> None:
        for ev in (_started(), _completed(), _ssh(), _code()):
            assert ev.severity == "info"

    def test_cleanup_kinds_pinned(self) -> None:
        assert _cleanup_started().kind == "ryotenkai.control.gpu.cleanup_started"
        assert _cleanup_completed().kind == "ryotenkai.control.gpu.cleanup_completed"
        assert _cleanup_failed().kind == "ryotenkai.control.gpu.cleanup_failed"

    def test_cleanup_severities(self) -> None:
        assert _cleanup_started().severity == "info"
        assert _cleanup_completed().severity == "info"
        assert _cleanup_failed().severity == "error"

    def test_cleanup_instance_id_optional(self) -> None:
        # ``instance_id`` is intentionally optional — single_node
        # cleanup may run before any container name is known (e.g.
        # very early SSH failure). Pin via constructor accepting None.
        payload_started = GpuCleanupStartedPayload(
            provider="single_node", instance_id=None, reason="forced",
        )
        assert payload_started.instance_id is None
        payload_completed = GpuCleanupCompletedPayload(
            provider="runpod", instance_id=None, duration_s=0.0, resources_freed={},
        )
        assert payload_completed.instance_id is None
