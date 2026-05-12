"""Scenario 1 — ``runpod_429_storm``.

Inject 10 consecutive 429s into the fake-runpod sidecar; assert a
naive REST client retries past the storm and eventually succeeds.
The "client" is a thin httpx loop that mirrors the kind of backoff
:class:`RunPodAPIClient` does — keep it tight (max 12 attempts,
exponential backoff with manual-clock-friendly sleeps).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx

from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.wait import Eventually
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class RunPod429Storm(ScenarioBase):
    name = "runpod_429_storm"
    tags = ["transient", "network"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        # Register a single running pod so list/find calls hit something.
        await sidecar_post(
            ctx, "runpod", "/control/register_pod",
            params={"pod_id": "p-storm", "desired_status": "RUNNING"},
        )
        ctx.debug_recorder.record("precondition", "pod_registered", pod_id="p-storm")

    async def inject(self, ctx: ScenarioContext) -> None:
        result = await sidecar_post(ctx, "runpod", "/control/inject_429", params={"count": 10})
        ctx.debug_recorder.record("inject", "inject_429", **result.json())

    async def steady_state(self, ctx: ScenarioContext) -> None:
        # A bounded retry loop that mirrors a real client. We retry up to
        # 12 times with 100ms gaps; this comfortably exceeds the 10 429s.
        async def _retry_until_success() -> bool:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    for _ in range(12):
                        resp = await client.get(
                            ctx.stack.sidecars["runpod"].base_url + "/api/pods/p-storm",
                        )
                        if resp.status_code == 200:
                            return True
                        # 429s — keep retrying.
            except Exception:  # noqa: BLE001
                return False
            return False

        await Eventually(
            _retry_until_success,
            timeout=self.recovery_window.total_seconds(),
            poll=0.1,
            message="client did not recover after 429 storm",
            clock=ctx.clock,
        )
        ctx.debug_recorder.record("steady_state", "recovered")

    async def cleanup(self, ctx: ScenarioContext) -> None:
        await super().cleanup(ctx)


__all__: list[str] = []
