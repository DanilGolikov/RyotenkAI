"""L6 smoke: the real control plane subprocess boots against the stack.

This is the headline test for L6 — it proves the orchestrator's
``control_plane()`` async context manager:

1. Spawns the real ``ryotenkai_control.api.main:create_app`` subprocess.
2. Wires it with env vars pointing at the four sidecars.
3. Reaches a green ``/api/v1/health`` before yielding.
4. Cleans up the subprocess on exit.

NOTE on wiring: the control plane today does **not** proxy MLflow
through its public API (per orchestrator.py comment: "control plane
today doesn't read these but we publish them so Phase 1+ code can opt
in incrementally"). So we verify the wiring two ways instead:

* The /api/v1/health endpoint is green while the stack is alive — proving
  the subprocess survived env-var injection (otherwise it would crash on
  import).
* The fake-mlflow sidecar remains responsive concurrently with the
  control plane being up — proving boot order doesn't deadlock and the
  port allocator didn't collide.
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack
from tests._harness.wait import Eventually

pytestmark = [pytest.mark.asyncio, pytest.mark.stack]


@pytest.mark.slow
async def test_control_plane_subprocess_boots_against_stack(stack: Stack) -> None:
    # control_plane() yields the base URL after /api/v1/health is 200.
    async with stack.control_plane() as control_url:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # WHY Eventually: wait_for_health already polled once before
            # yielding, but a second probe with Eventually guards against
            # any transient hiccup right after the yield.
            async def is_healthy() -> bool:
                try:
                    r = await client.get(control_url + "/api/v1/health")
                except httpx.HTTPError:
                    return False
                return r.status_code == 200

            await Eventually(is_healthy, timeout=10.0, poll=0.2)

            response = await client.get(control_url + "/api/v1/health")
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] in ("ok", "degraded")
            # runs_dir was injected by the orchestrator; the response must
            # echo it back so we can verify the env wiring took.
            assert "runs_dir" in payload
            assert payload["runs_dir_readable"] is True


@pytest.mark.slow
async def test_control_plane_coexists_with_fake_mlflow(stack: Stack) -> None:
    """Proves the env-var wiring (MLFLOW_TRACKING_URI etc.) is plumbed
    without booting two collisions on the same port.

    With the control plane up we hit fake-mlflow's REST surface directly
    — both subprocesses must be alive and responsive simultaneously.
    """
    async with stack.control_plane() as control_url:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Control plane is alive.
            cp_health = await client.get(control_url + "/api/v1/health")
            assert cp_health.status_code == 200

            # Concurrently, fake-mlflow is alive and serves its setup
            # endpoint with the tracking URI the control plane received
            # via env. (We can't easily introspect the subprocess env, but
            # the orchestrator's wiring guarantees MLFLOW_TRACKING_URI ==
            # stack.mlflow_url.)
            ml_health = await client.get(stack.mlflow_url + "/health")
            assert ml_health.status_code == 200

            setup = await client.post(stack.mlflow_url + "/api/setup")
            assert setup.status_code == 200
            assert setup.json()["is_active"] is True
