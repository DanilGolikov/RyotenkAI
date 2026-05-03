"""Phase 11.C-2 — :class:`AttemptController` pod metadata mutators.

The two new mutators (``set_pod_metadata`` + ``update_pod_status``)
are the producer-side of Phase 11.C: GPUDeployer writes the pod
identity to the active attempt; TrainingMonitor refreshes the
last-known status when the pod transitions.

7-category coverage. Tests use real ``PipelineStateStore`` against
``tmp_path`` so persistence semantics (atomic write, round-trip
deserialisation) are exercised end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.state.attempt_controller import AttemptController
from src.pipeline.state.models import (
    PipelineAttemptState,
    PipelineState,
    PodMetadata,
    StageRunState,
    utc_now_iso,
)
from src.pipeline.state.store import PipelineStateStore


def _make_state(*, with_attempt: bool = True) -> PipelineState:
    state = PipelineState(
        schema_version=1,
        logical_run_id="run-test",
        run_directory="/tmp/run-test",
        config_path="/tmp/test.yaml",
        active_attempt_id="att-1" if with_attempt else None,
        pipeline_status=StageRunState.STATUS_RUNNING,
        training_critical_config_hash="",
        late_stage_config_hash="",
    )
    if with_attempt:
        state.attempts.append(
            PipelineAttemptState(
                attempt_id="att-1",
                attempt_no=1,
                runtime_name="test",
                requested_action="fresh",
                effective_action="fresh",
                restart_from_stage=None,
                status=StageRunState.STATUS_RUNNING,
                started_at=utc_now_iso(),
            ),
        )
    return state


def _make_controller(tmp_path: Path) -> AttemptController:
    """Build an AttemptController with a real store, one active attempt."""
    store = PipelineStateStore(tmp_path)
    state = _make_state(with_attempt=True)
    store.save(state)

    controller = AttemptController(save_fn=store.save, run_ctx=None)
    controller.adopt_state(state)
    controller.register_attempt(state.attempts[-1])
    return controller


# ---------------------------------------------------------------------------
# 1. Positive — set_pod_metadata persists the dataclass + survives reload
# ---------------------------------------------------------------------------


class TestSetPodMetadataPositive:
    def test_set_creates_pod_metadata(self, tmp_path: Path) -> None:
        c = _make_controller(tmp_path)
        c.set_pod_metadata(
            pod_id="pod-abc",
            provider="runpod",
            last_known_status="running",
        )

        attempt = c.snapshot().attempts[-1]
        assert attempt.pod_metadata is not None
        assert attempt.pod_metadata.pod_id == "pod-abc"
        assert attempt.pod_metadata.provider == "runpod"
        assert attempt.pod_metadata.last_known_status == "running"
        assert attempt.pod_metadata.created_at  # auto-filled

    def test_set_persists_to_disk(self, tmp_path: Path) -> None:
        c = _make_controller(tmp_path)
        c.set_pod_metadata(pod_id="pod-1", provider="runpod")

        # Reload from disk — pod_metadata should round-trip.
        reloaded = PipelineStateStore(tmp_path).load()
        meta = reloaded.attempts[-1].pod_metadata
        assert meta is not None
        assert meta.pod_id == "pod-1"


# ---------------------------------------------------------------------------
# 2. Negative — empty pod_id ignored; no active attempt → no-op
# ---------------------------------------------------------------------------


class TestSetPodMetadataNegative:
    def test_empty_pod_id_is_ignored(self, tmp_path: Path) -> None:
        c = _make_controller(tmp_path)
        c.set_pod_metadata(pod_id="", provider="runpod")
        assert c.snapshot().attempts[-1].pod_metadata is None

    def test_no_active_attempt_is_silent(self, tmp_path: Path) -> None:
        # Build a controller but don't register any attempt.
        store = PipelineStateStore(tmp_path)
        state = _make_state(with_attempt=False)
        store.save(state)
        c = AttemptController(save_fn=store.save, run_ctx=None)
        c.adopt_state(state)
        # Must not raise.
        c.set_pod_metadata(pod_id="pod-x", provider="runpod")


# ---------------------------------------------------------------------------
# 3. Boundary — overwrite is fine
# ---------------------------------------------------------------------------


class TestSetPodMetadataIdempotency:
    def test_overwrite_with_same_pod_id(self, tmp_path: Path) -> None:
        c = _make_controller(tmp_path)
        c.set_pod_metadata(pod_id="pod-1", provider="runpod")
        c.set_pod_metadata(pod_id="pod-1", provider="runpod",
                           last_known_status="running")
        meta = c.snapshot().attempts[-1].pod_metadata
        assert meta is not None
        assert meta.last_known_status == "running"


# ---------------------------------------------------------------------------
# update_pod_status — 7-cat
# ---------------------------------------------------------------------------


class TestUpdatePodStatusPositive:
    def test_status_refreshes(self, tmp_path: Path) -> None:
        c = _make_controller(tmp_path)
        c.set_pod_metadata(
            pod_id="pod-1", provider="runpod",
            last_known_status="running",
        )
        c.update_pod_status(last_known_status="stopped")

        meta = c.snapshot().attempts[-1].pod_metadata
        assert meta is not None
        assert meta.last_known_status == "stopped"
        # Other fields preserved.
        assert meta.pod_id == "pod-1"
        assert meta.provider == "runpod"

    def test_persisted_to_disk(self, tmp_path: Path) -> None:
        c = _make_controller(tmp_path)
        c.set_pod_metadata(pod_id="pod-1", provider="runpod")
        c.update_pod_status(last_known_status="terminated")

        reloaded = PipelineStateStore(tmp_path).load()
        meta = reloaded.attempts[-1].pod_metadata
        assert meta is not None
        assert meta.last_known_status == "terminated"


class TestUpdatePodStatusNegative:
    def test_no_metadata_is_noop(self, tmp_path: Path) -> None:
        c = _make_controller(tmp_path)
        # No prior set_pod_metadata call — update should be silent.
        c.update_pod_status(last_known_status="stopped")
        assert c.snapshot().attempts[-1].pod_metadata is None

    def test_no_attempt_is_silent(self, tmp_path: Path) -> None:
        store = PipelineStateStore(tmp_path)
        state = _make_state(with_attempt=False)
        store.save(state)
        c = AttemptController(save_fn=store.save, run_ctx=None)
        c.adopt_state(state)
        # Must not raise.
        c.update_pod_status(last_known_status="stopped")


class TestUpdatePodStatusInvariants:
    def test_metadata_is_immutable_dataclass(self, tmp_path: Path) -> None:
        # update_pod_status replaces the PodMetadata with a new
        # instance (frozen dataclass) — pin the immutable contract.
        c = _make_controller(tmp_path)
        c.set_pod_metadata(pod_id="pod-1", provider="runpod",
                           last_known_status="running")
        first = c.snapshot().attempts[-1].pod_metadata
        assert first is not None

        c.update_pod_status(last_known_status="stopped")
        second = c.snapshot().attempts[-1].pod_metadata
        assert second is not None
        # New instance, not in-place mutation.
        assert first is not second
        assert second.last_known_status == "stopped"
