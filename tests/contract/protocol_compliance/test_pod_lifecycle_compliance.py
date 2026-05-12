"""Compliance tests for :class:`IPodLifecycleClient`.

Parametrized over ``[fake, real]``. The ``real`` variant requires
``RYOTENKAI_LIVE=1`` (and a configured upstream); otherwise it
``pytest.skip``s.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.lifecycle import (
    IPodLifecycleClient,
    PodTerminalOutcome,
)
from tests._fakes.lifecycle import FakePodLifecycleClient, PodState
from tests._harness.wait import Eventually

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("IPodLifecycleClient"),
    pytest.mark.uses_fake("FakePodLifecycleClient"),
    pytest.mark.asyncio,
]


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
def lifecycle_client(request: pytest.FixtureRequest, manual_clock: Any) -> IPodLifecycleClient:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real IPodLifecycleClient requires RYOTENKAI_LIVE=1")
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            pytest.skip(
                "real IPodLifecycleClient compliance requires RUNPOD_API_KEY pointing at a "
                "RunPod sandbox; export it and re-run with RYOTENKAI_LIVE=1.",
            )
        # Real adapter — additive class re-exported from
        # ``ryotenkai_providers.runpod.lifecycle.adapter``. The chaos
        # helpers below (``_as_fake``) skip themselves on the real
        # variant; what's exercised here is the Protocol shape +
        # transport behaviour.
        from ryotenkai_providers.runpod.lifecycle.adapter import RunPodLifecycleAdapter

        return RunPodLifecycleAdapter(api_key=api_key)
    return FakePodLifecycleClient(provider_name="fake", clock=manual_clock)


def _as_fake(client: IPodLifecycleClient) -> FakePodLifecycleClient:
    if not isinstance(client, FakePodLifecycleClient):
        pytest.skip(
            "test exercises FakePodLifecycleClient-only chaos helpers; "
            "real-mode covers Protocol shape only",
        )
    return client


class TestPodLifecycleCompliance:
    async def test_isinstance_protocol(self, lifecycle_client: IPodLifecycleClient) -> None:
        assert isinstance(lifecycle_client, IPodLifecycleClient)
        assert lifecycle_client.provider_name

    # -- terminate ------------------------------------------------------

    async def test_terminate_running_returns_terminated(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-1", state=PodState.RUNNING)
        result = await lifecycle_client.terminate(resource_id="p-1")
        assert result.outcome == PodTerminalOutcome.TERMINATED
        assert result.attempts_made == 1
        assert fake.get_pod_state("p-1") == PodState.TERMINATED

    async def test_terminate_stopped_returns_terminated(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        # Spec invariant: terminate(STOPPED) -> terminated.
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-2", state=PodState.STOPPED)
        result = await lifecycle_client.terminate(resource_id="p-2")
        assert result.outcome == PodTerminalOutcome.TERMINATED

    async def test_terminate_idempotent(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-3", state=PodState.TERMINATED)
        result = await lifecycle_client.terminate(resource_id="p-3")
        assert result.outcome == PodTerminalOutcome.ALREADY_TERMINATED

    # -- pause / stop ---------------------------------------------------

    async def test_pause_running_returns_stopped(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-4", state=PodState.RUNNING)
        result = await lifecycle_client.pause(resource_id="p-4")
        assert result.outcome == PodTerminalOutcome.STOPPED

    async def test_pause_stopped_returns_already_stopped(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-5", state=PodState.STOPPED)
        result = await lifecycle_client.pause(resource_id="p-5")
        assert result.outcome == PodTerminalOutcome.ALREADY_STOPPED

    async def test_pause_terminated_returns_skipped(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        # Spec invariant: stop(TERMINATED) -> skipped.
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-6", state=PodState.TERMINATED)
        result = await lifecycle_client.pause(resource_id="p-6")
        assert result.outcome == PodTerminalOutcome.SKIPPED

    # -- resume ---------------------------------------------------------

    async def test_resume_stopped_goes_to_running(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-7", state=PodState.STOPPED)
        result = await lifecycle_client.resume(resource_id="p-7")
        assert result.outcome == "resumed"
        assert fake.get_pod_state("p-7") == PodState.RUNNING

    async def test_resume_running_returns_already_running(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-8", state=PodState.RUNNING)
        result = await lifecycle_client.resume(resource_id="p-8")
        assert result.outcome == "already_running"

    # -- chaos surface --------------------------------------------------

    async def test_chaos_inject_failure_on_terminate(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-fail", state=PodState.RUNNING)
        fake.inject_failure_on("p-fail", "terminate", count=1)
        first = await lifecycle_client.terminate(resource_id="p-fail")
        assert first.outcome == PodTerminalOutcome.FAILED
        assert first.last_error
        # Second attempt recovers.
        second = await lifecycle_client.terminate(resource_id="p-fail")
        assert second.outcome == PodTerminalOutcome.TERMINATED

    async def test_chaos_set_pod_state_round_trips(
        self, lifecycle_client: IPodLifecycleClient,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.set_pod_state("p-set", PodState.STOPPING)
        assert fake.get_pod_state("p-set") == PodState.STOPPING
        # Resume from STOPPING is not the spec — only STOPPED resumes.
        # We let the state machine treat STOPPING as a non-terminal
        # mid-transition state by transitioning into RUNNING (eventual).
        fake.set_pod_state("p-set", PodState.STOPPED)
        result = await lifecycle_client.resume(resource_id="p-set")
        assert result.outcome == "resumed"

    async def test_chaos_eventual_consistency_uses_clock(
        self,
        lifecycle_client: IPodLifecycleClient,
        manual_clock: Any,
    ) -> None:
        fake = _as_fake(lifecycle_client)
        fake.register_pod("p-eventual", state=PodState.RUNNING)
        fake.make_eventually_consistent("p-eventual", transition_delay_seconds=2.0)
        result = await lifecycle_client.terminate(resource_id="p-eventual")
        # WHY: outcome reports success immediately (RunPod-style fire-and-forget),
        # but the visible state still reads RUNNING until the clock advances.
        assert result.outcome == PodTerminalOutcome.TERMINATED
        assert fake.get_pod_state("p-eventual") == PodState.RUNNING
        # Halfway through the window — still in the eventual-consistency
        # gap; the caller must tolerate this.
        manual_clock.advance(1.0)
        assert fake.get_pod_state("p-eventual") == PodState.RUNNING
        # After the full window, state settles.
        manual_clock.advance(1.0)
        # Eventually wraps the post-advance assertion to keep the L4+
        # idiom (poll=0.0 because the clock is already past the deadline,
        # so the first check succeeds without blocking on a sleep).
        async def state_is_terminated() -> bool:
            return fake.get_pod_state("p-eventual") == PodState.TERMINATED
        await Eventually(state_is_terminated, timeout=0.0, poll=0.0, clock=manual_clock)


__all__: list[str] = []
