"""Compliance tests for :class:`IRunPodAPI`.

Parametrized over ``[fake, real]``. The ``real`` variant exercises
:class:`HTTPRunPodAPIAdapter` pointed at the fake-runpod sidecar (so
no real RunPod account is needed); it requires ``RYOTENKAI_LIVE=1``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from ryotenkai_shared.infrastructure.runpod_api import (
    IRunPodAPI,
    RunPodPartialResponseError,
    RunPodRateLimitedError,
    RunPodTransientError,
)
from tests._fakes.runpod import FakeRunPodAPI
from tests._harness.stack import Stack

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("IRunPodAPI"),
    pytest.mark.uses_fake("FakeRunPodAPI"),
    pytest.mark.asyncio,
]


@pytest_asyncio.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
async def runpod_api(
    request: pytest.FixtureRequest,
    manual_clock: Any,
) -> AsyncIterator[IRunPodAPI]:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real IRunPodAPI requires RYOTENKAI_LIVE=1")
        # Boot a per-test sidecar stack and point the real REST adapter
        # at it. This lets us exercise the adapter's parser + transport
        # without needing a RunPod account.
        from ryotenkai_providers.runpod.runpod_api_adapter import HTTPRunPodAPIAdapter

        stack = await Stack.boot(clock="manual")
        adapter = HTTPRunPodAPIAdapter(base_url=stack.runpod_url)
        # Wrap the adapter so test helpers (``_as_fake``) keep working:
        # we attach a reference to the underlying sidecar-fake via the
        # state-dump but the tests below check ``isinstance(FakeRunPodAPI)``
        # for chaos-injection helpers. The real-mode tests therefore skip
        # the chaos paths — the live lane covers Protocol shape only.
        try:
            yield adapter  # type: ignore[misc]
        finally:
            await stack.shutdown()
        return
    yield FakeRunPodAPI(clock=manual_clock)


def _as_fake(api: IRunPodAPI) -> FakeRunPodAPI:
    if not isinstance(api, FakeRunPodAPI):
        pytest.skip(
            "test exercises chaos-injection helpers only available on FakeRunPodAPI; "
            "real-mode covers Protocol shape only",
        )
    return api


class TestRunPodAPICompliance:
    async def test_isinstance_protocol(self, runpod_api: IRunPodAPI) -> None:
        assert isinstance(runpod_api, IRunPodAPI)

    async def test_find_pod_returns_none_when_missing(self, runpod_api: IRunPodAPI) -> None:
        result = await runpod_api.find_pod("nonexistent")
        assert result is None

    async def test_find_pod_returns_info_when_present(self, runpod_api: IRunPodAPI) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-1", desired_status="RUNNING", ssh_host="1.2.3.4", ssh_port=22000)
        info = await runpod_api.find_pod("p-1")
        assert info is not None
        assert info.pod_id == "p-1"
        assert info.desired_status == "RUNNING"
        assert info.ssh_host == "1.2.3.4"
        assert info.ssh_port == 22000

    async def test_list_pods_enumerates(self, runpod_api: IRunPodAPI) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-A")
        fake.upsert_pod("p-B")
        pods = await runpod_api.list_pods()
        ids = sorted(p.pod_id for p in pods)
        assert ids == ["p-A", "p-B"]

    async def test_terminate_pod_idempotent(self, runpod_api: IRunPodAPI) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-T", desired_status="RUNNING")
        first = await runpod_api.terminate_pod("p-T")
        assert first.outcome == "ok"
        second = await runpod_api.terminate_pod("p-T")
        assert second.outcome == "already_done"

    async def test_terminate_missing_pod_is_already_done(
        self, runpod_api: IRunPodAPI,
    ) -> None:
        # WHY: real RunPod's "not found" maps to idempotency markers; the
        # control-plane treats it as success.
        result = await runpod_api.terminate_pod("never-existed")
        assert result.outcome == "already_done"

    async def test_stop_then_resume_cycles_back_to_running(
        self, runpod_api: IRunPodAPI,
    ) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-cycle", desired_status="RUNNING")
        stop = await runpod_api.stop_pod("p-cycle")
        assert stop.outcome == "ok"
        info = await runpod_api.find_pod("p-cycle")
        assert info is not None and info.desired_status == "EXITED"
        resume = await runpod_api.resume_pod("p-cycle")
        assert resume.outcome == "ok"
        info_after = await runpod_api.find_pod("p-cycle")
        assert info_after is not None and info_after.desired_status == "RUNNING"

    async def test_resume_terminated_pod_fails(self, runpod_api: IRunPodAPI) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-dead", desired_status="TERMINATED")
        result = await runpod_api.resume_pod("p-dead")
        assert result.outcome == "failed"
        assert "terminated" in result.message

    # -- chaos: rate limit, transient, partial, hibernation -------------

    async def test_chaos_inject_429_then_recovers(self, runpod_api: IRunPodAPI) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-429", desired_status="RUNNING")
        fake.inject_429(count=2)
        with pytest.raises(RunPodRateLimitedError):
            await runpod_api.find_pod("p-429")
        with pytest.raises(RunPodRateLimitedError):
            await runpod_api.find_pod("p-429")
        # Third call recovers.
        info = await runpod_api.find_pod("p-429")
        assert info is not None

    async def test_chaos_inject_5xx_raises_transient(self, runpod_api: IRunPodAPI) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-5xx", desired_status="RUNNING")
        fake.inject_5xx(count=1)
        with pytest.raises(RunPodTransientError):
            await runpod_api.terminate_pod("p-5xx")
        # Recovers.
        result = await runpod_api.terminate_pod("p-5xx")
        assert result.outcome == "ok"

    async def test_chaos_inject_partial_response_raises(
        self, runpod_api: IRunPodAPI,
    ) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-part", desired_status="RUNNING")
        fake.inject_partial_response(count=1)
        with pytest.raises(RunPodPartialResponseError):
            await runpod_api.find_pod("p-part")
        # Recovery: full response next time.
        info = await runpod_api.find_pod("p-part")
        assert info is not None

    async def test_chaos_hibernation_visibility(self, runpod_api: IRunPodAPI) -> None:
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-hib", desired_status="RUNNING")
        fake.set_hibernation_mode("p-hib")
        info = await runpod_api.find_pod("p-hib")
        assert info is not None
        assert info.desired_status == "HIBERNATED"

    async def test_snapshot_includes_chaos_and_pods(self, runpod_api: IRunPodAPI) -> None:
        import json
        fake = _as_fake(runpod_api)
        fake.upsert_pod("p-snap", desired_status="RUNNING")
        fake.inject_429(count=3)
        snap = fake.snapshot()
        # Round-trip through JSON catches non-serializable internals.
        json.dumps(snap)
        assert snap["chaos"]["rate_limit_remaining"] == 3
        assert snap["pods"]["p-snap"]["desired_status"] == "RUNNING"


__all__: list[str] = []
