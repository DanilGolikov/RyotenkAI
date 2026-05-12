"""ChaosScenario #2: ``MlflowCircuitOpen``.

fake-mlflow в режиме unavailable. Ожидаемое поведение клиента —
буферизация метрик и продолжение работы без потери данных (chaos
catalog item 2).

Здесь мы реализуем proof-of-pattern: ``FakeMLflowManager.set_unavailable``
зашёлкивает все call'ы; SUT-уровень (например ``BufferedMetricsReplay``)
должен это обработать. Полный multi-component pipeline переедет в
Phase 5.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests._fakes.mlflow import FakeMLflowManager, MLflowUnavailableError
from tests._harness.chaos import register_scenario

pytestmark = [pytest.mark.chaos, pytest.mark.slow, pytest.mark.asyncio]


@register_scenario
class MlflowCircuitOpen:
    name = "mlflow_circuit_open"
    tags = ["network", "mlflow", "buffering"]
    recovery_window = timedelta(seconds=60)

    async def precondition(self, ctx) -> None:  # type: ignore[no-untyped-def]
        mgr: FakeMLflowManager = ctx.extras["mlflow"]
        mgr.setup()
        with mgr.start_run() as run:
            ctx.extras["run_id"] = run.info.run_id

    async def inject(self, ctx) -> None:  # type: ignore[no-untyped-def]
        mgr: FakeMLflowManager = ctx.extras["mlflow"]
        mgr.set_unavailable(True)

    async def steady_state(self, ctx) -> None:  # type: ignore[no-untyped-def]
        mgr: FakeMLflowManager = ctx.extras["mlflow"]
        # Контракт: пока цепь "open", все операции бросают unavailable —
        # SUT-консьюмер должен ловить и буферизовать локально.
        raised = False
        try:
            mgr.adopt_existing_run(ctx.extras["run_id"])
        except MLflowUnavailableError:
            raised = True
        assert raised, "circuit was supposed to be open but call succeeded"

    async def cleanup(self, ctx) -> None:  # type: ignore[no-untyped-def]
        mgr: FakeMLflowManager = ctx.extras["mlflow"]
        mgr.reset_chaos()


# ---- driver -----------------------------------------------------------------


async def test_mlflow_circuit_open_blocks_writes() -> None:
    mgr = FakeMLflowManager()
    scenario = MlflowCircuitOpen()

    class _Ctx:
        extras: dict = {"mlflow": mgr}

    ctx = _Ctx()
    await scenario.precondition(ctx)
    await scenario.inject(ctx)
    await scenario.steady_state(ctx)
    await scenario.cleanup(ctx)

    # После cleanup всё работает.
    info = mgr.adopt_existing_run(ctx.extras["run_id"])
    assert info.run_id == ctx.extras["run_id"]
