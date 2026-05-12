"""L6 web smoke — Phase-2 skeleton.

One sanity test ("control plane health endpoint reachable through HTTP").
Real Playwright user-flow tests are Phase 5/6 work.
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack

# TODO(phase-5): real E2E user-flow tests via Playwright + browser_context fixture.

pytestmark = [pytest.mark.stack, pytest.mark.slow, pytest.mark.web_e2e, pytest.mark.asyncio]


async def test_control_plane_health_reachable(stack: Stack) -> None:
    async with stack.control_plane() as base_url, httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{base_url}/api/v1/health")
        assert response.status_code == 200
