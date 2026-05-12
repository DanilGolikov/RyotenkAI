"""L6 — boot the real FastAPI control plane against the fake sidecars.

Phase 2 exit criterion: production code can talk to the fakes via real
HTTP. Today's control plane doesn't yet read RunPod/MLflow URLs from env
(see investigation findings in the PR description), so this test only
verifies that the control-plane subprocess starts and serves /health.
Phase 3+ will add a chaos run-through once env wiring is in place.
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack

pytestmark = [pytest.mark.stack, pytest.mark.slow, pytest.mark.asyncio]


async def test_control_plane_health_against_stack(stack: Stack) -> None:
    async with stack.control_plane() as base_url, httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{base_url}/api/v1/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in ("ok", "degraded")

    # State dump still works after control-plane teardown.
    state = await stack.state_dump()
    assert "runpod" in state
    assert "mlflow" in state
