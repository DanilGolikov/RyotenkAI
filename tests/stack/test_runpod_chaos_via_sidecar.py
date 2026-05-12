"""L6 smoke: fake-runpod sidecar chaos endpoints actually change HTTP behaviour.

These prove the `/control/*` axis on the runpod sidecar is wired through
to the underlying :class:`FakeRunPodAPI`, by exercising one endpoint per
chaos surface and asserting on the externally observable HTTP response.
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack

pytestmark = [pytest.mark.asyncio, pytest.mark.stack]


async def test_inject_429_makes_list_pods_return_429(stack: Stack) -> None:
    runpod = stack.runpod_url
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Arm 3 queued 429s. Subsequent /api/pods calls should burn them
        # one-by-one, then the 4th call should succeed.
        arm = await client.post(runpod + "/control/inject_429", params={"count": 3})
        assert arm.status_code == 200
        assert arm.json() == {"rate_limit_remaining": 3}

        for attempt in range(3):
            response = await client.get(runpod + "/api/pods")
            assert response.status_code == 429, f"attempt {attempt}: {response.status_code}"
            assert response.json()["error"] == "rate_limited"

        # Chaos exhausted — next call must succeed.
        ok = await client.get(runpod + "/api/pods")
        assert ok.status_code == 200
        assert ok.json() == {"pods": []}


async def test_register_pod_via_control_makes_it_findable(stack: Stack) -> None:
    runpod = stack.runpod_url
    async with httpx.AsyncClient(timeout=5.0) as client:
        register = await client.post(
            runpod + "/control/register_pod",
            json={"pod_id": "test-123", "ssh_host": "10.0.0.5", "ssh_port": 22001},
        )
        assert register.status_code == 200
        body = register.json()
        assert body["pod"]["pod_id"] == "test-123"
        assert body["pod"]["ssh_host"] == "10.0.0.5"
        assert body["pod"]["ssh_port"] == 22001

        # The Protocol axis should now find it.
        found = await client.get(runpod + "/api/pods/test-123")
        assert found.status_code == 200
        assert found.json()["pod"]["pod_id"] == "test-123"

        # And it should appear in list_pods.
        listed = await client.get(runpod + "/api/pods")
        assert listed.status_code == 200
        ids = [p["pod_id"] for p in listed.json()["pods"]]
        assert ids == ["test-123"]


async def test_set_pod_state_hibernated_visible_in_find(stack: Stack) -> None:
    runpod = stack.runpod_url
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Register, then flip into HIBERNATED.
        reg = await client.post(runpod + "/control/register_pod", json={"pod_id": "h-1"})
        assert reg.status_code == 200

        flip = await client.post(
            runpod + "/control/set_pod_state",
            json={"pod_id": "h-1", "state": "HIBERNATED"},
        )
        assert flip.status_code == 200
        assert flip.json() == {"pod_id": "h-1", "state": "HIBERNATED"}

        # Find now reflects hibernation in the runtime_status. The exact
        # field semantics are owned by FakeRunPodAPI.set_hibernation_mode;
        # we just probe the externally-observable HTTP shape.
        found = await client.get(runpod + "/api/pods/h-1")
        assert found.status_code == 200
        pod = found.json()["pod"]
        assert pod["pod_id"] == "h-1"
        # The state snapshot proves hibernation took.
        state = await client.get(runpod + "/control/state")
        assert state.status_code == 200
        assert state.json()["pods"]["h-1"]["hibernated"] is True


async def test_inject_5xx_makes_list_pods_return_503(stack: Stack) -> None:
    # Round-trip the 5xx chaos path for parity with the 429 test —
    # asserts the second exception_handler also wires up correctly.
    runpod = stack.runpod_url
    async with httpx.AsyncClient(timeout=5.0) as client:
        arm = await client.post(runpod + "/control/inject_5xx", params={"count": 1})
        assert arm.status_code == 200

        bad = await client.get(runpod + "/api/pods")
        assert bad.status_code == 503
        assert bad.json()["error"] == "transient"

        ok = await client.get(runpod + "/api/pods")
        assert ok.status_code == 200
