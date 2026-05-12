"""Scenario 6 — ``journal_rotation_race``.

Modelled with an in-process :class:`FakeTrainerSpawner`: events are
appended monotonically; we simulate rotation by snapshotting the
current event log, asserting a concurrent reader reads the full
history without loss across the rotation boundary.

The "rotation" here is a logical bookmark — the fake emits N events,
the reader bookmarks offset, then more events arrive after the
bookmark; the reader's resumed read covers them all.
"""

from __future__ import annotations

from datetime import timedelta

from tests._fakes.trainer import FakeTrainerSpawner
from tests._harness.chaos import ScenarioContext, register_scenario
from tests._harness.clock import ManualClock
from tests.chaos.scenarios._base import ScenarioBase

from ryotenkai_shared.infrastructure.trainer_spawner import TrainerSpec


@register_scenario
class JournalRotationRace(ScenarioBase):
    name = "journal_rotation_race"
    tags = ["journal", "race"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        clock = ManualClock()
        fake = FakeTrainerSpawner(clock=clock)
        spec = TrainerSpec(job_id="job-rot", command=("python", "-m", "trainer"))
        handle = await fake.spawn(spec)
        ctx.extras.update({"fake": fake, "handle": handle})

    async def inject(self, ctx: ScenarioContext) -> None:
        fake: FakeTrainerSpawner = ctx.extras["fake"]
        handle = ctx.extras["handle"]
        # Emit a burst, take snapshot offset (rotation bookmark), emit more.
        for i in range(5):
            fake.emit_event(handle.trainer_id, "step", {"i": i})
        rotation_offset = fake.get_entry(handle.trainer_id).event_counter
        ctx.extras["rotation_offset"] = rotation_offset
        for i in range(5, 10):
            fake.emit_event(handle.trainer_id, "step", {"i": i})
        ctx.debug_recorder.record(
            "inject", "rotation_simulated",
            rotation_offset=rotation_offset,
        )

    async def steady_state(self, ctx: ScenarioContext) -> None:
        fake: FakeTrainerSpawner = ctx.extras["fake"]
        handle = ctx.extras["handle"]
        rotation_offset = ctx.extras["rotation_offset"]
        # Resume from rotation_offset; must see exactly the post-rotation events.
        post = await fake.read_events(handle.trainer_id, since=rotation_offset)
        step_events = [e for e in post if e.kind == "step"]
        observed_is = sorted(int(e.payload["i"]) for e in step_events)
        if observed_is != list(range(5, 10)):
            raise AssertionError(
                f"event loss across rotation; observed {observed_is!r}",
            )
        ctx.debug_recorder.record("steady_state", "no_event_loss")

    async def cleanup(self, ctx: ScenarioContext) -> None:
        ctx.extras.clear()
        await super().cleanup(ctx)


__all__: list[str] = []
