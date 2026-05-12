"""Scenario 18 — ``mlflow_double_finalization``.

Call ``end_run(id)`` twice on the same run. The second call must be
a no-op; final status must not change.
"""

from __future__ import annotations

from datetime import timedelta

from tests._harness.chaos import ScenarioContext, register_scenario
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class MlflowDoubleFinalization(ScenarioBase):
    name = "mlflow_double_finalization"
    tags = ["idempotency"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        await sidecar_post(ctx, "mlflow", "/api/setup")
        resp = await sidecar_post(
            ctx, "mlflow", "/api/2.0/mlflow/runs/create",
            json={"run_name": "double-fin"},
        )
        ctx.extras["run_id"] = resp.json()["run"]["info"]["run_id"]

    async def inject(self, ctx: ScenarioContext) -> None:
        run_id = ctx.extras["run_id"]
        await sidecar_post(
            ctx, "mlflow", "/api/2.0/mlflow/runs/update",
            json={"run_id": run_id, "status": "FINISHED"},
        )
        await sidecar_post(
            ctx, "mlflow", "/api/2.0/mlflow/runs/update",
            json={"run_id": run_id, "status": "FAILED"},
        )
        ctx.debug_recorder.record("inject", "end_run_called_twice")

    async def steady_state(self, ctx: ScenarioContext) -> None:
        run_id = ctx.extras["run_id"]
        snap = await ctx.stack.state_dump()
        run = snap["mlflow"]["runs"][run_id]
        if run["status"] != "FINISHED":
            raise AssertionError(
                f"second end_run was not a no-op: status={run['status']!r}",
            )
        ctx.debug_recorder.record("steady_state", "second_end_run_noop")


__all__: list[str] = []
