"""Scenario 4 — ``mac_sleep_plus_pod_crash``.

The Mac sleeps for several minutes; mid-sleep, the pod transitions to
``TERMINATED``. The control plane must observe the terminated state
when it wakes back up — not report the run as completed.
"""

from __future__ import annotations

from datetime import timedelta

from tests._harness.chaos import ScenarioContext, advance_clock_everywhere, register_scenario
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class MacSleepPlusPodCrash(ScenarioBase):
    name = "mac_sleep_plus_pod_crash"
    tags = ["clock", "lifecycle"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(
            ctx, "runpod", "/control/register_pod",
            params={"pod_id": "p-crash", "desired_status": "RUNNING"},
        )

    async def inject(self, ctx: ScenarioContext) -> None:
        # 5 minutes of "sleep" elapse, mid-way the pod is killed.
        await advance_clock_everywhere(ctx.stack, 150.0, in_process=ctx.clock)
        await sidecar_post(
            ctx, "runpod", "/control/set_pod_state",
            params={"pod_id": "p-crash", "status": "TERMINATED"},
        )
        ctx.debug_recorder.record("inject", "pod_terminated_during_sleep")
        await advance_clock_everywhere(ctx.stack, 150.0, in_process=ctx.clock)

    async def steady_state(self, ctx: ScenarioContext) -> None:
        snap = await ctx.stack.state_dump()
        pod = snap["runpod"]["pods"]["p-crash"]
        if pod["desired_status"] != "TERMINATED":
            raise AssertionError(
                f"pod state misreported after sleep: {pod['desired_status']!r}",
            )
        ctx.debug_recorder.record("steady_state", "terminated_reported_faithfully")


__all__: list[str] = []
