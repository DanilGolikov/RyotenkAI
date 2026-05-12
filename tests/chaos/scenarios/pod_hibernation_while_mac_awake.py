"""Scenario 5 — ``pod_hibernation_while_mac_awake``.

The Mac is awake; the pod transitions to ``HIBERNATED``. The detector
sees the new status and the operator gets the truth (no auto-resume
today; we just assert the state is reported faithfully).
"""

from __future__ import annotations

from datetime import timedelta

from tests._harness.chaos import ScenarioContext, register_scenario
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class PodHibernationWhileMacAwake(ScenarioBase):
    name = "pod_hibernation_while_mac_awake"
    tags = ["lifecycle"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(
            ctx, "runpod", "/control/register_pod",
            params={"pod_id": "p-hib", "desired_status": "RUNNING"},
        )

    async def inject(self, ctx: ScenarioContext) -> None:
        await sidecar_post(
            ctx, "runpod", "/control/set_pod_state",
            params={"pod_id": "p-hib", "status": "HIBERNATED"},
        )

    async def steady_state(self, ctx: ScenarioContext) -> None:
        snap = await ctx.stack.state_dump()
        pod = snap["runpod"]["pods"]["p-hib"]
        if pod["desired_status"] != "HIBERNATED":
            raise AssertionError(
                f"detector did not surface hibernation; got {pod['desired_status']!r}",
            )
        if not pod["hibernated"]:
            raise AssertionError("hibernated flag not set on pod entry")
        ctx.debug_recorder.record("steady_state", "hibernation_reported")


__all__: list[str] = []
