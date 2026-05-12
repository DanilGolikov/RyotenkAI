"""L2 component tests for :class:`BatchPodTerminator` using FakePodLifecycleClient."""

from __future__ import annotations

import pytest

from ryotenkai_control.cleanup import BatchPodTerminator
from ryotenkai_shared.infrastructure.lifecycle import PodTerminalOutcome
from tests._fakes.lifecycle import FakePodLifecycleClient, PodState

pytestmark = [
    pytest.mark.component,
    pytest.mark.uses_fake("FakePodLifecycleClient"),
    pytest.mark.exercises_protocol("IPodLifecycleClient"),
    pytest.mark.asyncio,
]


@pytest.fixture
def fake_lifecycle(manual_clock) -> FakePodLifecycleClient:
    return FakePodLifecycleClient(provider_name="fake", clock=manual_clock)


class TestBatchPodTerminator:
    async def test_terminates_running_pods(self, fake_lifecycle: FakePodLifecycleClient) -> None:
        for pid in ("p-1", "p-2", "p-3"):
            fake_lifecycle.register_pod(pid, state=PodState.RUNNING)
        sut = BatchPodTerminator(client=fake_lifecycle)

        report = await sut.terminate_all(["p-1", "p-2", "p-3"])

        assert report.total == 3
        assert report.terminated == 3
        assert report.already_terminated == 0
        assert report.failed == 0
        assert report.outcomes == {
            "p-1": PodTerminalOutcome.TERMINATED,
            "p-2": PodTerminalOutcome.TERMINATED,
            "p-3": PodTerminalOutcome.TERMINATED,
        }

    async def test_already_terminated_classified_separately(
        self, fake_lifecycle: FakePodLifecycleClient,
    ) -> None:
        fake_lifecycle.register_pod("p-live", state=PodState.RUNNING)
        fake_lifecycle.register_pod("p-dead", state=PodState.TERMINATED)
        sut = BatchPodTerminator(client=fake_lifecycle)

        report = await sut.terminate_all(["p-live", "p-dead"])

        assert report.terminated == 1
        assert report.already_terminated == 1
        assert report.failed == 0

    async def test_failure_does_not_abort_batch(
        self, fake_lifecycle: FakePodLifecycleClient,
    ) -> None:
        fake_lifecycle.register_pod("p-ok-1", state=PodState.RUNNING)
        fake_lifecycle.register_pod("p-bad", state=PodState.RUNNING)
        fake_lifecycle.register_pod("p-ok-2", state=PodState.RUNNING)
        # Inject failure for the middle pod's terminate call.
        fake_lifecycle.inject_failure_on("p-bad", "terminate", count=999)
        sut = BatchPodTerminator(client=fake_lifecycle)

        report = await sut.terminate_all(["p-ok-1", "p-bad", "p-ok-2"])

        assert report.total == 3
        assert report.terminated == 2
        assert report.failed == 1
        assert len(report.failures) == 1
        bad_id, _msg = report.failures[0]
        assert bad_id == "p-bad"

    async def test_empty_batch_returns_zero_report(
        self, fake_lifecycle: FakePodLifecycleClient,
    ) -> None:
        # Boundary: empty input = empty report (no calls to client).
        sut = BatchPodTerminator(client=fake_lifecycle)
        report = await sut.terminate_all([])
        assert report.total == 0
        assert report.terminated == 0
        assert report.outcomes == {}

    async def test_outcomes_map_preserves_pod_ids(
        self, fake_lifecycle: FakePodLifecycleClient,
    ) -> None:
        # Invariant: the outcomes dict keyed by pod_id must contain every input.
        fake_lifecycle.register_pod("p-x", state=PodState.RUNNING)
        fake_lifecycle.register_pod("p-y", state=PodState.TERMINATED)
        sut = BatchPodTerminator(client=fake_lifecycle)
        report = await sut.terminate_all(["p-x", "p-y", "p-z-missing"])
        # Note: p-z-missing isn't pre-registered. The fake auto-creates it
        # in RUNNING by default and the terminate succeeds.
        assert set(report.outcomes.keys()) == {"p-x", "p-y", "p-z-missing"}
