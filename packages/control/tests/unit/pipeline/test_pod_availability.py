"""Phase 11.C — :class:`PodAvailabilityProbe` + :func:`resume_pod_with_retry` tests.

7-category coverage. The probe is pure logic over a transport
callable; the resume-with-retry function exercises the
backoff + capacity-error path that's the entire point of Phase 11.C
(RunPod's "no GPU available right now" can resolve in 30-120s).

Tests use fake transports — no RunPod SDK or network access.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ryotenkai_control.pipeline.launch.pod_availability import (
    RESUME_BACKOFFS,
    PodAvailability,
    PodAvailabilityProbe,
    ProbeResult,
    ResumeResult,
    resume_pod_with_retry,
)
from ryotenkai_control.pipeline.state.models import PodMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta(pod_id: str = "pod-abc") -> PodMetadata:
    return PodMetadata(pod_id=pod_id, provider="runpod", created_at="2026-04-27T00:00:00Z")


# ---------------------------------------------------------------------------
# Probe — 7-cat coverage
# ---------------------------------------------------------------------------


class TestProbeRunning:
    """1. Positive — RUNNING status."""

    def test_runpod_running_maps_to_running(self) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "RUNNING"},
        )
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.RUNNING
        assert result.runpod_status == "RUNNING"
        assert result.is_recoverable

    def test_running_is_not_resume_needed(self) -> None:
        # Sanity: running means the pipeline can SSH right away.
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "RUNNING"},
        )
        result = probe.probe(_meta())
        assert not result.availability.is_resume_needed


class TestProbeSleeping:
    """2. EXITED / STOPPED / PAUSED → SLEEPING_RESUMABLE."""

    @pytest.mark.parametrize("status", ["EXITED", "STOPPED", "PAUSED"])
    def test_sleeping_states_map_to_resumable(self, status: str) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": status},
        )
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.SLEEPING_RESUMABLE
        assert result.is_recoverable
        assert result.availability.is_resume_needed

    def test_sleeping_message_mentions_resume(self) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "EXITED"},
        )
        result = probe.probe(_meta())
        assert "resume" in result.message.lower()


class TestProbeGone:
    """3. Boundary — TERMINATED → GONE; explicit error markers → GONE."""

    def test_terminated_maps_to_gone(self) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "TERMINATED"},
        )
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.GONE
        assert not result.is_recoverable

    def test_dead_maps_to_gone(self) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "DEAD"},
        )
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.GONE

    def test_not_found_error_marker_maps_to_gone(self) -> None:
        # No status field, but errors say "not found" → GONE.
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"errors": ["pod does not exist"]},
        )
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.GONE
        assert "terminated" in result.message.lower()

    def test_gone_message_suggests_run_restart(self) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "TERMINATED"},
        )
        result = probe.probe(_meta())
        assert "run restart" in result.message


class TestProbeInvariants:
    """4. Invariants — never raises; legacy metadata defaults."""

    def test_legacy_no_metadata_returns_running(self) -> None:
        # No pod_metadata ⇒ assume RUNNING and let the pipeline's
        # SSH connect step surface real errors. Avoids a false GONE
        # signal on legacy (pre-Phase-11.C) attempts.
        probe = PodAvailabilityProbe(query_pod=lambda _id: {})
        result = probe.probe(None)
        assert result.availability == PodAvailability.RUNNING
        assert result.pod_id == "<no-metadata>"

    def test_query_pod_raising_yields_probe_failed(self) -> None:
        def boom(_id: str) -> dict:
            raise RuntimeError("transport explosion")

        probe = PodAvailabilityProbe(query_pod=boom)
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.PROBE_FAILED
        assert "transport explosion" in result.message

    def test_non_dict_payload_yields_probe_failed(self) -> None:
        probe = PodAvailabilityProbe(query_pod=lambda _id: "not a dict")  # type: ignore[arg-type,return-value]
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.PROBE_FAILED


class TestProbeDependencyErrors:
    """5. Dependency errors — None transport, missing fields."""

    def test_no_transport_returns_probe_failed(self) -> None:
        probe = PodAvailabilityProbe(query_pod=None)
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.PROBE_FAILED
        assert "transport" in result.message.lower()

    def test_missing_status_field_yields_probe_failed(self) -> None:
        probe = PodAvailabilityProbe(query_pod=lambda _id: {"foo": "bar"})
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.PROBE_FAILED


class TestProbeRegressions:
    """6. Regressions — snake_case + camelCase status keys."""

    def test_snake_case_status_supported(self) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desired_status": "RUNNING"},
        )
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.RUNNING

    def test_lowercase_value_normalised(self) -> None:
        # RunPod sometimes returns lowercase; we normalise to upper.
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "running"},
        )
        result = probe.probe(_meta())
        assert result.availability == PodAvailability.RUNNING


class TestProbeLogicSpecific:
    """7. Logic-specific — unknown status → PROBE_FAILED (don't guess)."""

    def test_unknown_runpod_status_returns_probe_failed(self) -> None:
        probe = PodAvailabilityProbe(
            query_pod=lambda _id: {"desiredStatus": "PROVISIONING"},
        )
        result = probe.probe(_meta())
        # PROVISIONING isn't in our map — be defensive, don't pick
        # a wrong default.
        assert result.availability == PodAvailability.PROBE_FAILED


# ---------------------------------------------------------------------------
# resume_pod_with_retry — 7-cat coverage
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.asyncio


class _ScriptedClock:
    """Deterministic clock for budget tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestResumePositive:
    """1. Positive — first attempt succeeds → ok=True, attempts=1."""

    async def test_first_attempt_succeeds(self) -> None:
        async def _no_sleep(_: float) -> None:
            return

        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=lambda _id: True,
            sleep=_no_sleep,
        )
        assert result.ok is True
        assert result.attempts == 1
        assert result.pod_id == "pod-abc"

    async def test_async_resume_call_supported(self) -> None:
        async def _no_sleep(_: float) -> None:
            return

        async def async_call(_id: str) -> bool:
            return True

        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=async_call,
            sleep=_no_sleep,
        )
        assert result.ok is True


class TestResumeNegative:
    """2. Negative — non-capacity error fails fast."""

    async def test_fatal_error_fails_immediately(self) -> None:
        async def _no_sleep(_: float) -> None:
            return

        def boom(_id: str) -> bool:
            raise RuntimeError("auth failure")

        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=boom,
            is_capacity_error=lambda _msg: False,  # not a capacity error
            sleep=_no_sleep,
        )
        assert result.ok is False
        assert result.attempts == 1  # one shot, fail fast
        assert result.capacity_exhausted is False
        assert "auth failure" in result.error_message

    async def test_resume_call_returning_false_fails_fast(self) -> None:
        async def _no_sleep(_: float) -> None:
            return

        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=lambda _id: False,
            sleep=_no_sleep,
        )
        assert result.ok is False
        assert result.attempts == 1


class TestResumeBoundary:
    """3. Boundary — capacity errors trigger retries until budget."""

    async def test_capacity_then_success(self) -> None:
        sleeps_seen: list[float] = []

        async def _capture_sleep(s: float) -> None:
            sleeps_seen.append(s)

        attempts = {"n": 0}

        def call(_id: str) -> bool:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("no longer any instances available")
            return True

        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=call,
            is_capacity_error=lambda msg: "no longer any instances" in msg,
            sleep=_capture_sleep,
        )
        assert result.ok is True
        assert result.attempts == 2
        # Slept once between the two attempts, with backoffs[0].
        assert sleeps_seen == [RESUME_BACKOFFS[0]]

    async def test_all_attempts_capacity_yields_exhausted(self) -> None:
        async def _no_sleep(_: float) -> None:
            return

        def call(_id: str) -> bool:
            raise RuntimeError("rate limit")

        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=call,
            is_capacity_error=lambda _msg: True,
            sleep=_no_sleep,
        )
        assert result.ok is False
        assert result.capacity_exhausted is True
        assert result.attempts == len(RESUME_BACKOFFS) + 1
        assert "capacity unavailable" in result.error_message


class TestResumeInvariants:
    """4. Invariants — never raises; budget enforced; clock-aware."""

    async def test_budget_exhausted_short_circuits(self) -> None:
        clock = _ScriptedClock()
        # Capture the side-effect of sleep advancing the clock.
        async def _sleep_advance(s: float) -> None:
            clock.advance(s)

        # All capacity errors. Budget = 5s, backoffs sum > 5s.
        def call(_id: str) -> bool:
            raise RuntimeError("rate limit")

        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=call,
            is_capacity_error=lambda _msg: True,
            backoffs=(2.0, 4.0, 8.0),
            budget_seconds=5.0,
            sleep=_sleep_advance,
            clock=clock,
        )
        assert result.ok is False
        assert result.capacity_exhausted is True
        # Don't keep sleeping past budget; first sleep=2 (clock=2),
        # second probe at clock=2, attempt 3 fails, sleep=4 → clamped
        # by remaining=3, clock=5 → loop exits via budget check.
        assert result.elapsed_seconds <= 6.0  # generous CI budget


class TestResumeDependencyErrors:
    """5. Dependency errors — capacity-error detector missing."""

    async def test_no_capacity_detector_treats_all_as_fatal(self) -> None:
        async def _no_sleep(_: float) -> None:
            return

        def call(_id: str) -> bool:
            raise RuntimeError("any error")

        # is_capacity_error=None ⇒ never retry, fail fast.
        result = await resume_pod_with_retry(
            "pod-abc",
            resume_call=call,
            is_capacity_error=None,
            sleep=_no_sleep,
        )
        assert result.ok is False
        assert result.attempts == 1
        assert result.capacity_exhausted is False


class TestResumeRegressions:
    """6. Regressions — backoff sequence pinned."""

    def test_default_backoffs_match_plan(self) -> None:
        # Pin the sequence — Phase 11 plan § 7.2 specifies
        # 10s + 30s + 60s + 120s. Operator dashboards (and SLO
        # alerts) assume this shape.
        assert RESUME_BACKOFFS == (10.0, 30.0, 60.0, 120.0)

    def test_default_budget_is_300_seconds(self) -> None:
        from ryotenkai_control.pipeline.launch.pod_availability import (
            RESUME_RETRY_BUDGET_SECONDS,
        )
        assert RESUME_RETRY_BUDGET_SECONDS == 300.0


class TestResumeLogicSpecific:
    """7. Logic-specific — backoff clamping when budget is tight."""

    async def test_backoff_clamped_to_remaining_budget(self) -> None:
        clock = _ScriptedClock()
        sleeps: list[float] = []

        async def _sleep(s: float) -> None:
            sleeps.append(s)
            clock.advance(s)

        def call(_id: str) -> bool:
            raise RuntimeError("capacity")

        # Budget=10s, backoffs=20s,30s. First sleep should be
        # clamped to ~10s (remaining at that point).
        await resume_pod_with_retry(
            "pod-abc",
            resume_call=call,
            is_capacity_error=lambda _msg: True,
            backoffs=(20.0, 30.0),
            budget_seconds=10.0,
            sleep=_sleep,
            clock=clock,
        )
        # First sleep clamped (was 20, now ≤ 10).
        assert sleeps[0] <= 10.0


# ---------------------------------------------------------------------------
# PodMetadata serialization tests
# ---------------------------------------------------------------------------


class TestPodMetadataSerialization:
    """PodMetadata.to_dict / from_dict — graceful for legacy attempts."""

    def test_round_trip(self) -> None:
        meta = PodMetadata(
            pod_id="pod-abc",
            provider="runpod",
            created_at="2026-04-27T00:00:00Z",
            last_known_status="running",
        )
        restored = PodMetadata.from_dict(meta.to_dict())
        assert restored == meta

    def test_legacy_missing_pod_id_returns_none(self) -> None:
        # Empty dict / no pod_id ⇒ from_dict returns None (legacy).
        assert PodMetadata.from_dict({}) is None
        assert PodMetadata.from_dict({"provider": "runpod"}) is None

    def test_optional_fields_default_gracefully(self) -> None:
        # Just pod_id ⇒ other fields default.
        result = PodMetadata.from_dict({"pod_id": "pod-x"})
        assert result is not None
        assert result.pod_id == "pod-x"
        assert result.provider == "runpod"
        assert result.last_known_status is None
