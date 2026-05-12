"""Scenario 17 — ``runpod_graphql_partial_response``.

The RunPod GraphQL endpoint returns truncated payloads. The fake
maps these to :class:`RunPodPartialResponseError` via
:meth:`inject_partial_response`. The parser must classify the
response as transient (retryable) rather than crashing.
"""

from __future__ import annotations

from datetime import timedelta

import httpx

from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.wait import Eventually
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class RunpodGraphqlPartialResponse(ScenarioBase):
    name = "runpod_graphql_partial_response"
    tags = ["parser"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(
            ctx, "runpod", "/control/register_pod",
            params={"pod_id": "p-partial", "desired_status": "RUNNING"},
        )

    async def inject(self, ctx: ScenarioContext) -> None:
        await sidecar_post(ctx, "runpod", "/control/inject_partial_response", params={"count": 2})
        ctx.debug_recorder.record("inject", "partial_response", count=2)

    async def steady_state(self, ctx: ScenarioContext) -> None:
        # A retry-aware client should hit 502 twice then succeed.
        seen_502 = 0
        ok = False

        async def _probe() -> bool:
            nonlocal seen_502, ok
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    ctx.stack.sidecars["runpod"].base_url + "/api/pods/p-partial",
                )
                if resp.status_code == 502:
                    seen_502 += 1
                    return False
                ok = resp.status_code == 200
                return ok

        await Eventually(
            _probe,
            timeout=self.recovery_window.total_seconds(),
            poll=0.1,
            clock=ctx.clock,
        )
        if seen_502 < 1:
            raise AssertionError(
                f"expected at least one 502, observed {seen_502}",
            )
        ctx.debug_recorder.record(
            "steady_state", "partial_classified_transient",
            seen_502=seen_502,
        )


__all__: list[str] = []
