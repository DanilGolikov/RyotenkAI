"""Scenario 11 — ``oom_killed_trainer``.

The trainer is OOM-killed (exit_code=137 / SIGKILL). The
:class:`FakeTrainerSpawner` models this with
:meth:`inject_oom_next_spawn`; the next ``spawn`` lands in FAILED
with exit_code 137. The control plane must classify this as a
``failed`` terminal state, not a transient error.
"""

from __future__ import annotations

from datetime import timedelta

from ryotenkai_shared.infrastructure.trainer_spawner import TrainerSpec, TrainerStatus
from tests._fakes.trainer import FakeTrainerSpawner
from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.clock import ManualClock
from tests.chaos.scenarios._base import ScenarioBase


@register_scenario
class OomKilledTrainer(ScenarioBase):
    name = "oom_killed_trainer"
    tags = ["trainer", "signal"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        clock = ManualClock()
        fake = FakeTrainerSpawner(clock=clock)
        ctx.extras["fake"] = fake

    async def inject(self, ctx: ScenarioContext) -> None:
        fake: FakeTrainerSpawner = ctx.extras["fake"]
        fake.inject_oom_next_spawn()
        ctx.debug_recorder.record("inject", "oom_armed")
        spec = TrainerSpec(job_id="job-oom", command=("python",))
        handle = await fake.spawn(spec)
        ctx.extras["handle"] = handle

    async def steady_state(self, ctx: ScenarioContext) -> None:
        fake: FakeTrainerSpawner = ctx.extras["fake"]
        handle = ctx.extras["handle"]
        status = await fake.status(handle.trainer_id)
        if status != TrainerStatus.FAILED:
            raise AssertionError(
                f"OOM trainer not classified as FAILED; got {status!r}",
            )
        entry = fake.get_entry(handle.trainer_id)
        if entry.exit_code != 137:
            raise AssertionError(
                f"OOM exit_code not 137; got {entry.exit_code!r}",
            )
        ctx.debug_recorder.record("steady_state", "oom_classified_failed")

    async def cleanup(self, ctx: ScenarioContext) -> None:
        ctx.extras.clear()
        await super().cleanup(ctx)


__all__: list[str] = []
