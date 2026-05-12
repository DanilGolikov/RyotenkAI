"""Compliance tests for :class:`IJobClient`."""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.job_client import (
    IJobClient,
    JobClientNetworkError,
    JobClientNotFoundError,
    JobClientRateLimitedError,
)
from tests._fakes.job_client import FakeJobClient

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("IJobClient"),
    pytest.mark.uses_fake("FakeJobClient"),
    pytest.mark.asyncio,
]


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
def job_client(request: pytest.FixtureRequest, manual_clock: Any) -> IJobClient:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real IJobClient requires RYOTENKAI_LIVE=1")
        runner_url = os.environ.get("RYOTENKAI_RUNNER_URL")
        if not runner_url:
            pytest.skip(
                "real IJobClient compliance requires RYOTENKAI_RUNNER_URL pointing at a "
                "live runner; the adapter wraps the production JobClient.",
            )
        from ryotenkai_shared.utils.clients.job_client_adapter import HTTPJobClientAdapter

        return HTTPJobClientAdapter(base_url=runner_url)
    return FakeJobClient(clock=manual_clock)


def _as_fake(client: IJobClient) -> FakeJobClient:
    if not isinstance(client, FakeJobClient):
        pytest.skip(
            "test exercises FakeJobClient-only chaos helpers; "
            "real-mode covers Protocol shape only",
        )
    return client


class TestJobClientCompliance:
    async def test_isinstance_protocol(self, job_client: IJobClient) -> None:
        assert isinstance(job_client, IJobClient)

    async def test_health_check_true_by_default(self, job_client: IJobClient) -> None:
        assert await job_client.health_check() is True

    async def test_submit_get_request_stop_round_trip(
        self, job_client: IJobClient,
    ) -> None:
        result = await job_client.submit_job(
            {"job_id": "j-A", "model": "Q"}, plugins_payload=b"",
        )
        assert result.job_id == "j-A"
        status = await job_client.get_status("j-A")
        assert status["job_id"] == "j-A"
        await job_client.request_stop("j-A", grace_seconds=5.0)
        post = await job_client.get_status("j-A")
        assert post["state"] == "stopping"

    async def test_get_status_unknown_raises(self, job_client: IJobClient) -> None:
        with pytest.raises(JobClientNotFoundError):
            await job_client.get_status("never-existed")

    async def test_request_stop_unknown_raises(self, job_client: IJobClient) -> None:
        with pytest.raises(JobClientNotFoundError):
            await job_client.request_stop("never-existed")

    async def test_send_heartbeat_counts(self, job_client: IJobClient) -> None:
        fake = _as_fake(job_client)
        for _ in range(3):
            assert await job_client.send_heartbeat() is True
        assert fake.heartbeat_count() == 3

    # -- chaos surface --------------------------------------------------

    async def test_chaos_inject_429(self, job_client: IJobClient) -> None:
        fake = _as_fake(job_client)
        fake.inject_429(count=1)
        with pytest.raises(JobClientRateLimitedError):
            await job_client.submit_job({"job_id": "x"})
        # Recovery.
        await job_client.submit_job({"job_id": "x"})

    async def test_chaos_inject_timeout(self, job_client: IJobClient) -> None:
        fake = _as_fake(job_client)
        fake.inject_timeout(count=1)
        with pytest.raises(JobClientNetworkError):
            await job_client.submit_job({"job_id": "y"})

    async def test_chaos_inject_404_next(self, job_client: IJobClient) -> None:
        fake = _as_fake(job_client)
        await job_client.submit_job({"job_id": "j-404"})
        fake.inject_404_next()
        with pytest.raises(JobClientNotFoundError):
            await job_client.get_status("j-404")

    async def test_chaos_network_partition(
        self, job_client: IJobClient, manual_clock: Any,
    ) -> None:
        fake = _as_fake(job_client)
        fake.inject_network_partition(duration_seconds=10.0)
        with pytest.raises(JobClientNetworkError):
            await job_client.submit_job({"job_id": "p"})
        # Advance clock past the partition deadline.
        manual_clock.advance(15.0)
        await job_client.submit_job({"job_id": "p"})

    async def test_chaos_set_unhealthy(self, job_client: IJobClient) -> None:
        fake = _as_fake(job_client)
        fake.set_unhealthy(True)
        assert await job_client.health_check() is False
        fake.set_unhealthy(False)
        assert await job_client.health_check() is True

    async def test_snapshot_is_json_serializable(self, job_client: IJobClient) -> None:
        import json
        fake = _as_fake(job_client)
        await job_client.submit_job({"job_id": "snap"})
        snap = fake.snapshot()
        json.dumps(snap)
        assert "snap" in snap["jobs"]


__all__: list[str] = []
