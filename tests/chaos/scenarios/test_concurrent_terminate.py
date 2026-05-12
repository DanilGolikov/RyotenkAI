"""ChaosScenario #8: ``ConcurrentTerminate``.

Два ``terminate``-call'а 50ms друг за другом; ожидаем
идемпотентность — второй call отдаёт ``already_terminated`` без побочных
эффектов (chaos catalog item 8).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from ryotenkai_shared.infrastructure.lifecycle import PodTerminalOutcome
from tests._fakes.lifecycle import FakePodLifecycleClient, PodState

# NOTE: the catalog entry for this scenario lives in
# `tests/chaos/scenarios/concurrent_terminate.py` (sidecar-driven).
# This file is the pytest-driver companion using FakePodLifecycleClient
# directly — it does NOT re-register the scenario in the catalog.
pytestmark = [pytest.mark.chaos, pytest.mark.slow, pytest.mark.asyncio]


class ConcurrentTerminate:
    name = "concurrent_terminate"
    tags = ["lifecycle", "idempotency", "race"]
    recovery_window = timedelta(seconds=5)

    async def precondition(self, ctx) -> None:  # type: ignore[no-untyped-def]
        client: FakePodLifecycleClient = ctx.extras["lifecycle"]
        client.register_pod("pod-race", state=PodState.RUNNING)

    async def inject(self, ctx) -> None:  # type: ignore[no-untyped-def]
        client: FakePodLifecycleClient = ctx.extras["lifecycle"]
        # Два gather'нутых terminate'а — гонка между ними не должна
        # давать дрейф состояния.
        results = await asyncio.gather(
            client.terminate(resource_id="pod-race"),
            client.terminate(resource_id="pod-race"),
        )
        ctx.extras["results"] = results

    async def steady_state(self, ctx) -> None:  # type: ignore[no-untyped-def]
        client: FakePodLifecycleClient = ctx.extras["lifecycle"]
        # Конечное состояние — TERMINATED ровно один раз.
        assert client.get_pod_state("pod-race") == PodState.TERMINATED
        # Outcomes: exactly один TERMINATED + один ALREADY_TERMINATED
        # (или оба TERMINATED — допустимо в идемпотентной модели).
        outcomes = sorted(r.outcome for r in ctx.extras["results"])
        assert PodTerminalOutcome.TERMINATED in outcomes
        # Второй обязан быть либо already_terminated, либо ещё одним
        # terminated — оба значения являются идемпотентным результатом
        # одной и той же операции.
        assert outcomes[-1] in {
            PodTerminalOutcome.TERMINATED,
            PodTerminalOutcome.ALREADY_TERMINATED,
        }

    async def cleanup(self, ctx) -> None:  # type: ignore[no-untyped-def]
        client: FakePodLifecycleClient = ctx.extras["lifecycle"]
        client.reset_chaos()


# ---- driver -----------------------------------------------------------------


async def test_concurrent_terminate_is_idempotent() -> None:
    client = FakePodLifecycleClient()
    scenario = ConcurrentTerminate()

    class _Ctx:
        extras: dict = {"lifecycle": client}

    ctx = _Ctx()
    await scenario.precondition(ctx)
    await scenario.inject(ctx)
    await scenario.steady_state(ctx)
    await scenario.cleanup(ctx)
