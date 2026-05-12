"""Scenario 10 — ``trainer_callback_failure``.

Inject N callback failures into a :class:`FakeTrainerSpawner`. The
next event emission raises; the runner-shaped consumer must surface
the error, not silently swallow it.
"""

from __future__ import annotations

from datetime import timedelta

from ryotenkai_shared.infrastructure.trainer_spawner import TrainerSpawnError, TrainerSpec
from tests._fakes.trainer import FakeTrainerSpawner
from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.clock import ManualClock
from tests.chaos.scenarios._base import ScenarioBase


@register_scenario
class TrainerCallbackFailure(ScenarioBase):
    name = "trainer_callback_failure"
    tags = ["trainer", "callback"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        clock = ManualClock()
        fake = FakeTrainerSpawner(clock=clock)
        spec = TrainerSpec(job_id="job-cb", command=("python",))
        handle = await fake.spawn(spec)
        ctx.extras.update({"fake": fake, "handle": handle})

    async def inject(self, ctx: ScenarioContext) -> None:
        fake: FakeTrainerSpawner = ctx.extras["fake"]
        fake.inject_callback_failure(count=3)
        ctx.debug_recorder.record("inject", "callback_failure", count=3)

    async def steady_state(self, ctx: ScenarioContext) -> None:
        fake: FakeTrainerSpawner = ctx.extras["fake"]
        handle = ctx.extras["handle"]
        # Each of the next three emissions must raise.
        failures = 0
        for _ in range(3):
            try:
                fake.emit_event(handle.trainer_id, "step", {"i": 0})
            except TrainerSpawnError:
                failures += 1
        if failures != 3:
            raise AssertionError(
                f"expected 3 callback failures, observed {failures}",
            )
        # Subsequent emit must succeed (failures consumed).
        fake.emit_event(handle.trainer_id, "step", {"i": 0})
        ctx.debug_recorder.record("steady_state", "failures_surfaced", count=failures)

    async def cleanup(self, ctx: ScenarioContext) -> None:
        ctx.extras.clear()
        await super().cleanup(ctx)


__all__: list[str] = []
