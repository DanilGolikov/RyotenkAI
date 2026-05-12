"""L6 — exercise the fake-runpod sidecar over real HTTP.

The production ``RunPodAPIClient`` is hardcoded to the RunPod GraphQL
SDK and isn't trivially redirectable to a REST URL — see Phase 2 spec
§D5 for the documented limitation. We fall back to ``httpx.AsyncClient``
so we still prove the sidecar's HTTP surface matches what a real REST
client would expect.
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack

pytestmark = [pytest.mark.stack, pytest.mark.asyncio]


async def test_list_terminate_via_real_http(stack: Stack) -> None:
    async with httpx.AsyncClient(base_url=stack.runpod_url, timeout=2.0) as client:
        # Initially empty.
        response = await client.get("/api/pods")
        assert response.status_code == 200
        assert response.json() == []

        # Register two pods through the control axis.
        for pod_id in ("p-alpha", "p-beta"):
            response = await client.post(
                "/control/register_pod",
                params={"pod_id": pod_id, "desired_status": "RUNNING"},
            )
            assert response.status_code == 200

        # List shows both.
        response = await client.get("/api/pods")
        assert response.status_code == 200
        listing = response.json()
        assert {p["pod_id"] for p in listing} == {"p-alpha", "p-beta"}

        # Find by id.
        response = await client.get("/api/pods/p-alpha")
        assert response.status_code == 200
        assert response.json()["pod_id"] == "p-alpha"

        # 404 on unknown pod (mirrors real RunPod's "pod gone" semantic).
        response = await client.get("/api/pods/unknown")
        assert response.status_code == 404

        # Terminate is idempotent.
        response = await client.post("/api/pods/p-alpha/terminate")
        assert response.status_code == 200
        assert response.json()["outcome"] == "ok"
        response = await client.post("/api/pods/p-alpha/terminate")
        assert response.status_code == 200
        assert response.json()["outcome"] == "already_done"


async def test_429_then_success(stack: Stack) -> None:
    async with httpx.AsyncClient(base_url=stack.runpod_url, timeout=2.0) as client:
        await client.post("/control/register_pod", params={"pod_id": "p-1"})
        await client.post("/control/inject_429", params={"count": 2})

        # First two reads get 429; third succeeds.
        response = await client.get("/api/pods")
        assert response.status_code == 429
        response = await client.get("/api/pods")
        assert response.status_code == 429
        response = await client.get("/api/pods")
        assert response.status_code == 200
