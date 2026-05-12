"""L2 component tests for :class:`HibernationDetector` using FakeRunPodAPI."""

from __future__ import annotations

import pytest

from ryotenkai_control.cleanup import HibernationDetector
from tests._fakes.runpod import FakeRunPodAPI

pytestmark = [
    pytest.mark.component,
    pytest.mark.uses_fake("FakeRunPodAPI"),
    pytest.mark.exercises_protocol("IRunPodAPI"),
    pytest.mark.asyncio,
]


@pytest.fixture
def fake_runpod(manual_clock) -> FakeRunPodAPI:
    return FakeRunPodAPI(clock=manual_clock)


class TestHibernationDetectorSweep:
    async def test_sweep_finds_hibernated_pods(self, fake_runpod: FakeRunPodAPI) -> None:
        fake_runpod.upsert_pod("p-running", desired_status="RUNNING")
        fake_runpod.upsert_pod("p-stopped", desired_status="EXITED")
        fake_runpod.upsert_pod("p-hib")
        fake_runpod.set_hibernation_mode("p-hib")
        sut = HibernationDetector(api=fake_runpod)

        report = await sut.sweep()

        assert report.inspected == 3
        assert len(report.hibernated) == 1
        assert report.hibernated[0].pod_id == "p-hib"
        assert report.hibernated[0].desired_status == "HIBERNATED"

    async def test_sweep_with_no_pods_returns_empty(
        self, fake_runpod: FakeRunPodAPI,
    ) -> None:
        # Boundary: empty fleet, no errors.
        sut = HibernationDetector(api=fake_runpod)
        report = await sut.sweep()
        assert report.inspected == 0
        assert report.hibernated == ()
        assert report.api_errors == ()

    async def test_sweep_surfaces_rate_limit_as_error(
        self, fake_runpod: FakeRunPodAPI,
    ) -> None:
        fake_runpod.upsert_pod("p-1")
        fake_runpod.inject_429(count=1)
        sut = HibernationDetector(api=fake_runpod)
        report = await sut.sweep()
        assert report.inspected == 0
        assert report.hibernated == ()
        assert len(report.api_errors) == 1
        assert "RunPodRateLimitedError" in report.api_errors[0]

    async def test_sweep_recovers_after_transient_error(
        self, fake_runpod: FakeRunPodAPI,
    ) -> None:
        # Negative path then positive: first sweep hits 429, second succeeds.
        fake_runpod.upsert_pod("p-stable", desired_status="RUNNING")
        fake_runpod.inject_429(count=1)
        sut = HibernationDetector(api=fake_runpod)

        first = await sut.sweep()
        assert first.inspected == 0

        second = await sut.sweep()
        assert second.inspected == 1
        assert second.hibernated == ()


class TestHibernationDetectorPerPod:
    async def test_is_pod_hibernated_returns_true_for_hibernated_pod(
        self, fake_runpod: FakeRunPodAPI,
    ) -> None:
        fake_runpod.upsert_pod("p-h")
        fake_runpod.set_hibernation_mode("p-h")
        sut = HibernationDetector(api=fake_runpod)
        assert await sut.is_pod_hibernated("p-h") is True

    async def test_is_pod_hibernated_returns_false_for_running_pod(
        self, fake_runpod: FakeRunPodAPI,
    ) -> None:
        fake_runpod.upsert_pod("p-r", desired_status="RUNNING")
        sut = HibernationDetector(api=fake_runpod)
        assert await sut.is_pod_hibernated("p-r") is False

    async def test_is_pod_hibernated_returns_none_for_missing_pod(
        self, fake_runpod: FakeRunPodAPI,
    ) -> None:
        # Negative path: missing pod should be ``None``, not ``False``.
        sut = HibernationDetector(api=fake_runpod)
        assert await sut.is_pod_hibernated("ghost") is None

    async def test_is_pod_hibernated_returns_none_on_api_error(
        self, fake_runpod: FakeRunPodAPI,
    ) -> None:
        fake_runpod.upsert_pod("p-flaky")
        fake_runpod.set_hibernation_mode("p-flaky")
        fake_runpod.inject_429(count=1)
        sut = HibernationDetector(api=fake_runpod)
        # First call hits the 429 — None.
        assert await sut.is_pod_hibernated("p-flaky") is None
        # Second call recovers and reports True.
        assert await sut.is_pod_hibernated("p-flaky") is True
