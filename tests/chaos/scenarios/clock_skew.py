"""Scenario 14 — ``clock_skew``.

The runner's clock is 60s ahead of the control plane. Because both
clocks in the hermetic stack are :class:`ManualClock` instances
broadcast by the orchestrator, "skew" is modelled by advancing only
the runpod sidecar's clock and leaving the others (mlflow/vllm/hf_hub)
behind. The assertion: state queries against either side still
succeed (no heartbeat-driven false cancellation).
"""

from __future__ import annotations

from datetime import timedelta

from tests._harness.chaos import ScenarioContext, register_scenario
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class ClockSkew(ScenarioBase):
    name = "clock_skew"
    tags = ["clock"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(
            ctx, "runpod", "/control/register_pod",
            params={"pod_id": "p-skew", "desired_status": "RUNNING"},
        )

    async def inject(self, ctx: ScenarioContext) -> None:
        # Advance ONLY the runpod sidecar clock — others stay behind.
        await sidecar_post(ctx, "runpod", "/control/advance_clock", params={"seconds": 60})
        ctx.debug_recorder.record("inject", "skewed_60s")

    async def steady_state(self, ctx: ScenarioContext) -> None:
        snap = await ctx.stack.state_dump()
        # Pod must still be reachable + RUNNING; no heartbeat-driven cancel.
        pod = snap["runpod"]["pods"]["p-skew"]
        if pod["desired_status"] != "RUNNING":
            raise AssertionError(
                f"clock skew triggered false cancellation: {pod['desired_status']!r}",
            )
        ctx.debug_recorder.record("steady_state", "heartbeat_tolerated_skew")


__all__: list[str] = []
