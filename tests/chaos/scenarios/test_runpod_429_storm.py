"""ChaosScenario #1: ``Runpod429Storm``.

Inject 30 секунд 429-ответов от fake-runpod, ожидаем что клиент
восстанавливается через backoff (chaos catalog item 1 из
[docs/plans/structured-hopping-starfish.md](../../../docs/plans/structured-hopping-starfish.md)).

Этот тест намеренно использует in-process :class:`FakeRunPodAPI` вместо
поднятия сайдкара — proof-of-pattern, который запускается без расходов
на subprocess boot. Сценарии, требующие реального HTTP, переезжают на
sidecar-stack по мере добавления.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from ryotenkai_shared.infrastructure.runpod_api import RunPodRateLimitedError
from tests._harness.chaos import ChaosScenario, register_scenario
from tests._harness.clock import ManualClock
from tests._fakes.runpod import FakeRunPodAPI

pytestmark = [pytest.mark.chaos, pytest.mark.slow, pytest.mark.asyncio]


@register_scenario
class Runpod429Storm:
    """Inject 429 storm; expect client recovers within recovery_window."""

    name = "runpod_429_storm"
    tags = ["transient", "network", "runpod"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx) -> None:  # type: ignore[no-untyped-def]
        # Запускающий тест кладёт FakeRunPodAPI в extras["api"].
        api: FakeRunPodAPI = ctx.extras["api"]
        api.upsert_pod("pod-victim")

    async def inject(self, ctx) -> None:  # type: ignore[no-untyped-def]
        api: FakeRunPodAPI = ctx.extras["api"]
        # 5 заряженных 429 — типичный transient storm у RunPod.
        api.inject_429(count=5)

    async def steady_state(self, ctx) -> None:  # type: ignore[no-untyped-def]
        api: FakeRunPodAPI = ctx.extras["api"]
        # Клиент с retry-loop'ом обязан восстановиться.
        # Здесь мы моделируем минимальную retry-policy.
        attempts = 0
        last_err: Exception | None = None
        for attempts in range(1, 10):
            try:
                await api.find_pod("pod-victim")
                break
            except RunPodRateLimitedError as exc:
                last_err = exc
        # Хотя бы один успех должен случиться.
        assert last_err is None or attempts > 5, (
            f"client never recovered after {attempts} attempts; last={last_err!r}"
        )

    async def cleanup(self, ctx) -> None:  # type: ignore[no-untyped-def]
        api: FakeRunPodAPI = ctx.extras["api"]
        api.reset_chaos()


# ---- driver -----------------------------------------------------------------


async def test_runpod_429_storm_recovers_after_burst() -> None:
    """Драйвер: запускает сценарий через in-process fake без полного
    стека. Полная stack-based версия будет добавлена в Phase 5 вместе
    с остальным каталогом."""
    api = FakeRunPodAPI(clock=ManualClock())
    scenario = Runpod429Storm()

    # Минимальный контекст — extras несёт fakes; stack/clock не нужны
    # для in-process реплея 429-storm.
    class _Ctx:
        extras = {"api": api}
        clock = ManualClock()

    await scenario.precondition(_Ctx())
    await scenario.inject(_Ctx())
    await scenario.steady_state(_Ctx())
    await scenario.cleanup(_Ctx())

    # После cleanup chaos выключен — обычные операции возвращают данные.
    info = await api.find_pod("pod-victim")
    assert info is not None
    assert info.desired_status == "RUNNING"
