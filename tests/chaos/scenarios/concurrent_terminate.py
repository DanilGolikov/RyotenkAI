"""Scenario 8 — ``concurrent_terminate``.

Two ``terminate(pod_id)`` calls fired ~50ms apart; both must complete
without an error outcome. The expected combinations are:
``ok``/``already_done`` or ``already_done``/``already_done``. Never
``failed``.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from tests._harness.chaos import ScenarioContext, register_scenario
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class ConcurrentTerminate(ScenarioBase):
    name = "concurrent_terminate"
    tags = ["lifecycle", "race"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(
            ctx, "runpod", "/control/register_pod",
            params={"pod_id": "p-conc", "desired_status": "RUNNING"},
        )

    async def inject(self, ctx: ScenarioContext) -> None:
        # Two terminates in quick succession via the sidecar's REST surface.
        async def _terminate() -> dict:
            resp = await sidecar_post(ctx, "runpod", "/api/pods/p-conc/terminate")
            return resp.json()

        first, second = await asyncio.gather(_terminate(), _terminate())
        ctx.extras["first"] = first
        ctx.extras["second"] = second
        ctx.debug_recorder.record(
            "inject", "concurrent_terminate",
            first_outcome=first.get("outcome"),
            second_outcome=second.get("outcome"),
        )

    async def steady_state(self, ctx: ScenarioContext) -> None:
        outcomes = {ctx.extras["first"]["outcome"], ctx.extras["second"]["outcome"]}
        if "failed" in outcomes:
            raise AssertionError(
                f"concurrent terminate produced a failure: {outcomes!r}",
            )
        if not outcomes.issubset({"ok", "already_done"}):
            raise AssertionError(f"unexpected terminate outcomes: {outcomes!r}")
        ctx.debug_recorder.record("steady_state", "idempotent_terminate", outcomes=list(outcomes))


__all__: list[str] = []
