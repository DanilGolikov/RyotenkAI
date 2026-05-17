"""Post-Phase-10 (F-СРЕД) — typed cleanup events for RunPodProvider.

Mirrors the single_node ``test_cleanup_events.py``: pins the
visibility-gap close around ``cleanup_pod`` so leaked / orphaned
RunPod pods surface on the unified timeline.

Coverage:

* **Positive** — happy-path cleanup emits started + completed;
  ``duration_s >= 0``; ``resources_freed['pods'] == 1``.
* **Negative** — cleanup_manager raises → started + failed;
  ``partial_cleanup=False`` (RunPod's terminate call is atomic from
  the provider's POV — either it succeeded or no side-effect).
* **Boundary** — second cleanup of the same pod still emits the pair
  (idempotent retry semantics live in the cleanup_manager itself).
* **Regressions** — caller-supplied ``reason`` propagates; the
  ``_safe_cleanup_pod`` wrapper swallows the exception but still
  emits the typed failed envelope.
"""

from __future__ import annotations

from typing import Any

import pytest

from ryotenkai_providers.runpod.training.provider import RunPodProvider
from ryotenkai_shared.errors import ProviderUnavailableError

from tests._fakes.event_emitter import FakeEventEmitter


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCleanupManager:
    """Mirrors ``RunPodCleanupManager`` surface to the extent the
    provider's emit funnel touches it: only ``cleanup_pod(pod_id)``.

    Programmable via ``raise_on_cleanup`` for failure scenarios.
    """

    def __init__(self, raise_on_cleanup: BaseException | None = None) -> None:
        self.raise_on_cleanup = raise_on_cleanup
        self.calls: list[str] = []

    def cleanup_pod(self, pod_id: str) -> None:
        self.calls.append(pod_id)
        if self.raise_on_cleanup is not None:
            raise self.raise_on_cleanup


def _make_provider_with_cleanup(
    cleanup_manager: _FakeCleanupManager,
    emitter: FakeEventEmitter | None,
) -> RunPodProvider:
    """Construct a provider with the cleanup manager + emitter wired in.

    Uses the ``from_resume_metadata`` shortcut to skip the heavy
    GraphQL-client construction in ``__init__``; we only need the
    cleanup helpers exercised. ``_emitter`` / ``_emitter_missing_warned``
    are slots the factory doesn't initialise (see
    test_provider_events.py).
    """
    provider = RunPodProvider.from_resume_metadata(api_key="dummy")
    provider._emitter = emitter  # type: ignore[attr-defined]
    provider._emitter_missing_warned = False  # type: ignore[attr-defined]
    provider._cleanup_manager = cleanup_manager  # type: ignore[attr-defined]
    return provider


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_happy_path_emits_started_and_completed(self) -> None:
        cleanup = _FakeCleanupManager()
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        provider._emitting_cleanup_pod("pod-abc", reason="natural")  # noqa: SLF001

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_completed",
        ]

    def test_completed_payload_carries_provider_and_resources(self) -> None:
        cleanup = _FakeCleanupManager()
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        provider._emitting_cleanup_pod("pod-abc", reason="natural")  # noqa: SLF001
        completed = emitter.emitted[1]
        assert completed.payload.provider == "runpod"
        assert completed.payload.instance_id == "pod-abc"
        assert completed.payload.duration_s >= 0.0
        # RunPod cleanup is "one pod, one terminate" — no partial
        # accounting available, so ``resources_freed`` reports the
        # canonical ``{"pods": 1}`` on success.
        assert completed.payload.resources_freed == {"pods": 1}


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_cleanup_manager_raises_emits_started_then_failed(self) -> None:
        exc = ProviderUnavailableError(
            detail="terminate timed out", context={"reason": "RUNPOD_TIMEOUT"},
        )
        cleanup = _FakeCleanupManager(raise_on_cleanup=exc)
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        with pytest.raises(ProviderUnavailableError):
            provider._emitting_cleanup_pod("pod-abc", reason="failed")  # noqa: SLF001

        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_failed",
        ]
        failed = emitter.emitted[1]
        assert failed.payload.provider == "runpod"
        # RunPod terminate is atomic from the provider's POV: either
        # the API call landed or nothing happened. No intermediate
        # "freed some resources" state.
        assert failed.payload.partial_cleanup is False
        assert failed.payload.error_type == "ProviderUnavailableError"
        assert "terminate timed out" in failed.payload.message

    def test_stray_exception_still_emits_failed(self) -> None:
        # cleanup_manager is contracted to only raise RyotenkAIError,
        # but defence-in-depth: an unexpected exception type still
        # produces a complete (started, failed) pair before re-raising.
        cleanup = _FakeCleanupManager(raise_on_cleanup=RuntimeError("oops"))
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        with pytest.raises(RuntimeError):
            provider._emitting_cleanup_pod("pod-abc", reason="failed")  # noqa: SLF001
        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds[-1] == "ryotenkai.control.gpu.cleanup_failed"
        assert emitter.emitted[-1].payload.error_type == "RuntimeError"


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_idempotent_rerun_emits_two_pairs(self) -> None:
        cleanup = _FakeCleanupManager()
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        provider._emitting_cleanup_pod("pod-abc", reason="natural")  # noqa: SLF001
        provider._emitting_cleanup_pod("pod-abc", reason="forced")  # noqa: SLF001
        kinds = [ev.kind for ev in emitter.emitted]
        assert kinds == [
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_completed",
            "ryotenkai.control.gpu.cleanup_started",
            "ryotenkai.control.gpu.cleanup_completed",
        ]
        # Both reasons surface on the corresponding started envelopes.
        assert emitter.emitted[0].payload.reason == "natural"
        assert emitter.emitted[2].payload.reason == "forced"


# ---------------------------------------------------------------------------
# 4. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    @pytest.mark.parametrize("reason", ["natural", "cancelled", "failed", "forced"])
    def test_reason_propagates_to_started_payload(self, reason: str) -> None:
        cleanup = _FakeCleanupManager()
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        provider._emitting_cleanup_pod("pod-abc", reason=reason)  # noqa: SLF001
        assert emitter.emitted[0].payload.reason == reason

    def test_safe_cleanup_pod_swallows_but_emits_failed(self) -> None:
        # The orchestrator stop-chain calls ``_safe_cleanup_pod`` which
        # logs-and-swallows the exception. The typed failed envelope
        # MUST still fire — that's the whole point of this gap close.
        exc = ProviderUnavailableError(
            detail="auth denied", context={"reason": "RUNPOD_AUTH"},
        )
        cleanup = _FakeCleanupManager(raise_on_cleanup=exc)
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        # Must NOT raise — _safe_cleanup_pod has soft-fail semantics.
        provider._safe_cleanup_pod("pod-abc")  # noqa: SLF001

        kinds = [ev.kind for ev in emitter.emitted]
        assert "ryotenkai.control.gpu.cleanup_started" in kinds
        assert "ryotenkai.control.gpu.cleanup_failed" in kinds

    def test_safe_cleanup_pod_default_reason_is_failed(self) -> None:
        # Every call-site of ``_safe_cleanup_pod`` is an abort / retry
        # path, so the default reason is ``"failed"`` rather than
        # ``"natural"``. Pin the default so call-sites don't silently
        # mislabel forced cleanups as natural shutdowns.
        cleanup = _FakeCleanupManager()
        emitter = FakeEventEmitter()
        provider = _make_provider_with_cleanup(cleanup, emitter)
        provider._safe_cleanup_pod("pod-abc")  # noqa: SLF001
        assert emitter.emitted[0].payload.reason == "failed"

    def test_no_emitter_does_not_crash(self) -> None:
        # Legacy callers without an emitter wired stay unaffected.
        cleanup = _FakeCleanupManager()
        provider = _make_provider_with_cleanup(cleanup, emitter=None)
        # No exception — clean path; no envelope side-effects.
        provider._emitting_cleanup_pod("pod-abc", reason="natural")  # noqa: SLF001
        # And the manager was still invoked.
        assert cleanup.calls == ["pod-abc"]


__all__: list[Any] = []
