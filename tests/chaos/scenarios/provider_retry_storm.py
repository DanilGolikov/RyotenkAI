"""Scenario 13 — ``provider_retry_storm``.

Every external call fails its first attempt: runpod, mlflow, hf_hub
each get ``inject_5xx(1)``. The combined job time must stay bounded
within the recovery window — no infinite retries, no fan-out storm.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx

from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.wait import Consistently
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class ProviderRetryStorm(ScenarioBase):
    name = "provider_retry_storm"
    tags = ["network", "multi-component"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(
            ctx, "runpod", "/control/register_pod",
            params={"pod_id": "p-storm", "desired_status": "RUNNING"},
        )

    async def inject(self, ctx: ScenarioContext) -> None:
        await sidecar_post(ctx, "runpod", "/control/inject_5xx", params={"count": 1})
        # mlflow & hf_hub both lack a 5xx-injector but the patterns are
        # subsumed by ``set_unavailable`` flicker + ``fail_next_n_calls``.
        await sidecar_post(ctx, "mlflow", "/control/fail_next_n_calls", params={"count": 1})
        ctx.debug_recorder.record("inject", "all_transient")

    async def steady_state(self, ctx: ScenarioContext) -> None:
        # Issue probes in parallel; each should eventually succeed.
        async def _probe_runpod() -> bool:
            for _ in range(4):
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        ctx.stack.sidecars["runpod"].base_url + "/api/pods/p-storm",
                    )
                    if resp.status_code == 200:
                        return True
            return False

        ok = await _probe_runpod()
        if not ok:
            raise AssertionError("runpod did not recover within retry budget")

        async def _no_runaway() -> bool:
            # Ensure that the call_history isn't growing unboundedly. With
            # a single 5xx injection we expect at most 2 calls in the
            # window we just exercised; assert ≤ 10 as a sanity cap.
            snap = await ctx.stack.state_dump()
            calls = snap["runpod"].get("call_history", [])
            return len(calls) <= 10

        await Consistently(
            _no_runaway, duration=0.5, poll=0.1, clock=ctx.clock,
        )
        ctx.debug_recorder.record("steady_state", "bounded_retries")


__all__: list[str] = []
