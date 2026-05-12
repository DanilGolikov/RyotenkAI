"""Scenario 3 — ``mac_sleep_during_run``.

Advance the manual clock by 10 simulated minutes while a "training
session" is in-flight (modelled as an MLflow run whose status stays
``RUNNING``). The session must still be observable as ``RUNNING``
afterwards — no false cancellation just because the clock jumped.
"""

from __future__ import annotations

from datetime import timedelta

from tests._harness.chaos import ScenarioContext, advance_clock_everywhere, register_scenario
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class MacSleepDuringRun(ScenarioBase):
    name = "mac_sleep_during_run"
    tags = ["clock", "scheduler"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(ctx, "mlflow", "/api/setup")
        resp = await sidecar_post(
            ctx, "mlflow", "/api/2.0/mlflow/runs/create",
            json={"run_name": "long-run"},
        )
        ctx.extras["run_id"] = resp.json()["run"]["info"]["run_id"]

    async def inject(self, ctx: ScenarioContext) -> None:
        # 10 simulated minutes pass.
        await advance_clock_everywhere(ctx.stack, 600.0, in_process=ctx.clock)
        ctx.debug_recorder.record("inject", "clock_advanced", seconds=600)

    async def steady_state(self, ctx: ScenarioContext) -> None:
        snap = await ctx.stack.state_dump()
        run = snap["mlflow"]["runs"][ctx.extras["run_id"]]
        if run["status"] != "RUNNING":
            raise AssertionError(
                f"run was cancelled by clock jump: status={run['status']!r}",
            )
        ctx.debug_recorder.record("steady_state", "run_still_active")


__all__: list[str] = []
