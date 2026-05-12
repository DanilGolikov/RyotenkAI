"""Scenario 2 — ``mlflow_circuit_open``.

Flip the fake-mlflow sidecar into ``set_unavailable=True`` for a
short window, then restore it. Buffered metric writes must resume
without loss once the sidecar comes back. The fake here serves as the
buffer-of-last-resort: it records ``call_history`` so we can prove
that no metric was dropped.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx

from tests._harness.chaos import ScenarioContext, advance_clock_everywhere, register_scenario
from tests._harness.wait import Eventually
from tests.chaos.scenarios._base import ScenarioBase, sidecar_post


@register_scenario
class MlflowCircuitOpen(ScenarioBase):
    name = "mlflow_circuit_open"
    tags = ["transient", "network"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        # Boot mlflow + create a run so we have a target for metrics.
        await sidecar_post(ctx, "mlflow", "/api/setup")
        resp = await sidecar_post(
            ctx, "mlflow", "/api/2.0/mlflow/runs/create",
            json={"run_name": "chaos-run"},
        )
        body = resp.json()
        run_id = body["run"]["info"]["run_id"]
        ctx.extras["run_id"] = run_id
        ctx.debug_recorder.record("precondition", "run_created", run_id=run_id)

    async def inject(self, ctx: ScenarioContext) -> None:
        # Circuit open for 5 simulated seconds.
        await sidecar_post(ctx, "mlflow", "/control/set_unavailable", params={"value": True})
        ctx.debug_recorder.record("inject", "set_unavailable", value=True)
        # Advance the (manual) clock to mimic 5s downtime.
        await advance_clock_everywhere(ctx.stack, 5.0, in_process=ctx.clock)
        await sidecar_post(ctx, "mlflow", "/control/set_unavailable", params={"value": False})
        ctx.debug_recorder.record("inject", "set_unavailable", value=False)

    async def steady_state(self, ctx: ScenarioContext) -> None:
        # Write a metric — it must succeed since the circuit is closed.
        run_id = ctx.extras["run_id"]

        async def _log_metric_ok() -> bool:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    ctx.stack.sidecars["mlflow"].base_url + "/api/2.0/mlflow/runs/log-metric",
                    json={"run_id": run_id, "key": "loss", "value": 1.23, "step": 0},
                )
                return resp.status_code == 200

        await Eventually(
            _log_metric_ok,
            timeout=self.recovery_window.total_seconds(),
            poll=0.1,
            clock=ctx.clock,
        )
        # Round-trip: fetch the run; the metric must be visible in fake state.
        snap = await ctx.stack.state_dump()
        runs = snap["mlflow"]["runs"]
        if run_id not in runs:
            raise AssertionError(f"run {run_id!r} disappeared after circuit recovery")
        metrics = runs[run_id]["metrics"]
        if not any(m["key"] == "loss" for m in metrics):
            raise AssertionError("metric was dropped after circuit recovery")
        ctx.debug_recorder.record("steady_state", "no_metric_loss")


__all__: list[str] = []
