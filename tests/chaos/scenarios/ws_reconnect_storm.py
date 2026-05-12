"""Scenario 15 — ``ws_reconnect_storm``.

Open + close many HTTP/WS connections back-to-back against the
sidecars. Sidecars must remain healthy + responsive (no zombie
connections, no kernel-socket leak). We model 10 rapid open/close
cycles against ``/health``.

Note:
    Phase 5 sidecars don't yet expose a real WS endpoint; this is a
    partial implementation that uses HTTP probes as the closest
    available proxy. Wiring against a true WebSocket endpoint is
    Phase 6 follow-up.
"""

from __future__ import annotations

from datetime import timedelta

import httpx

from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.wait import Eventually
from tests.chaos.scenarios._base import ScenarioBase


@register_scenario
class WSReconnectStorm(ScenarioBase):
    name = "ws_reconnect_storm"
    tags = ["websocket", "network"]
    recovery_window = timedelta(seconds=30)

    async def inject(self, ctx: ScenarioContext) -> None:
        url = ctx.stack.sidecars["runpod"].base_url + "/health"
        for _ in range(10):
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.get(url)
        ctx.debug_recorder.record("inject", "10_rapid_reconnects")

    async def steady_state(self, ctx: ScenarioContext) -> None:
        url = ctx.stack.sidecars["runpod"].base_url + "/health"

        async def _healthy() -> bool:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(url)
                    return resp.status_code == 200
            except Exception:  # noqa: BLE001
                return False

        await Eventually(
            _healthy,
            timeout=self.recovery_window.total_seconds(),
            poll=0.1,
            clock=ctx.clock,
        )
        ctx.debug_recorder.record("steady_state", "server_still_healthy")


__all__: list[str] = []
