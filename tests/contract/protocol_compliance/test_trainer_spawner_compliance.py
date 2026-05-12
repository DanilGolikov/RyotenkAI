"""Compliance tests for :class:`ITrainerSpawner`.

Fake-only by default. ``real`` is parametrized but ``pytest.skip``s
until Phase 5+ wires a :class:`Supervisor`-adapter into the Protocol.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.trainer_spawner import (
    ITrainerSpawner,
    TrainerSpawnError,
    TrainerSpec,
    TrainerStatus,
)
from tests._fakes.trainer import FakeTrainerSpawner

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("ITrainerSpawner"),
    pytest.mark.uses_fake("FakeTrainerSpawner"),
    pytest.mark.asyncio,
]


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
def trainer_spawner(request: pytest.FixtureRequest, manual_clock: Any) -> ITrainerSpawner:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real ITrainerSpawner requires RYOTENKAI_LIVE=1")
        pytest.skip("real ITrainerSpawner not yet wired into compliance suite")
    return FakeTrainerSpawner(clock=manual_clock)


def _as_fake(spawner: ITrainerSpawner) -> FakeTrainerSpawner:
    assert isinstance(spawner, FakeTrainerSpawner)
    return spawner


def _spec(job_id: str = "j-1") -> TrainerSpec:
    return TrainerSpec(job_id=job_id, command=("python", "-m", "trainer", "--smoke"))


class TestTrainerSpawnerCompliance:
    async def test_isinstance_protocol(self, trainer_spawner: ITrainerSpawner) -> None:
        assert isinstance(trainer_spawner, ITrainerSpawner)

    async def test_spawn_returns_handle_and_running_status(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        handle = await trainer_spawner.spawn(_spec())
        assert handle.trainer_id
        status = await trainer_spawner.status(handle.trainer_id)
        assert status == TrainerStatus.RUNNING

    async def test_double_spawn_raises(self, trainer_spawner: ITrainerSpawner) -> None:
        await trainer_spawner.spawn(_spec("j-A"))
        with pytest.raises(TrainerSpawnError):
            await trainer_spawner.spawn(_spec("j-B"))

    async def test_unknown_trainer_status_raises(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        with pytest.raises(TrainerSpawnError):
            await trainer_spawner.status("never-existed")

    async def test_read_events_returns_spawn_chain(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        handle = await trainer_spawner.spawn(_spec())
        events = await trainer_spawner.read_events(handle.trainer_id)
        kinds = [e.kind for e in events]
        # ``job_submitted`` and ``trainer_spawned`` must both fire on
        # a successful spawn — mirrors :class:`Supervisor.submit_and_spawn`.
        assert "job_submitted" in kinds
        assert "trainer_spawned" in kinds

    async def test_send_signal_to_unknown_trainer_raises(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        with pytest.raises(TrainerSpawnError):
            await trainer_spawner.send_signal("never-existed", 15)

    async def test_send_sigterm_transitions_to_stopping(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        handle = await trainer_spawner.spawn(_spec())
        await trainer_spawner.send_signal(handle.trainer_id, 15)
        status = await trainer_spawner.status(handle.trainer_id)
        assert status == TrainerStatus.STOPPING

    async def test_send_sigkill_lands_in_cancelled(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        handle = await trainer_spawner.spawn(_spec())
        await trainer_spawner.send_signal(handle.trainer_id, 9)
        status = await trainer_spawner.status(handle.trainer_id)
        assert status == TrainerStatus.CANCELLED

    async def test_wait_returns_terminal_status(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        fake = _as_fake(trainer_spawner)
        handle = await trainer_spawner.spawn(_spec())
        fake.force_terminal(handle.trainer_id, TrainerStatus.COMPLETED, exit_code=0)
        status = await trainer_spawner.wait(handle.trainer_id, timeout=0.0)
        assert status == TrainerStatus.COMPLETED

    # -- chaos surface --------------------------------------------------

    async def test_chaos_oom_lands_in_failed(self, trainer_spawner: ITrainerSpawner) -> None:
        fake = _as_fake(trainer_spawner)
        fake.inject_oom_next_spawn()
        handle = await trainer_spawner.spawn(_spec())
        status = await trainer_spawner.status(handle.trainer_id)
        assert status == TrainerStatus.FAILED
        events = await trainer_spawner.read_events(handle.trainer_id)
        assert any(e.kind == "trainer_exited" and e.payload.get("exit_code") == 137 for e in events)

    async def test_chaos_callback_failure_blocks_event_emit(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        fake = _as_fake(trainer_spawner)
        fake.inject_callback_failure(count=1)
        with pytest.raises(TrainerSpawnError):
            # The first emit happens inside ``spawn`` (``job_submitted``);
            # injection triggers there.
            await trainer_spawner.spawn(_spec())

    async def test_chaos_signal_ignored_no_op(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        fake = _as_fake(trainer_spawner)
        handle = await trainer_spawner.spawn(_spec())
        fake.inject_signal_ignored()
        await trainer_spawner.send_signal(handle.trainer_id, 15)
        status = await trainer_spawner.status(handle.trainer_id)
        assert status == TrainerStatus.RUNNING

    async def test_snapshot_is_json_serializable(
        self, trainer_spawner: ITrainerSpawner,
    ) -> None:
        import json
        fake = _as_fake(trainer_spawner)
        handle = await trainer_spawner.spawn(_spec())
        snap = fake.snapshot()
        json.dumps(snap)
        assert handle.trainer_id in snap["trainers"]


__all__: list[str] = []
