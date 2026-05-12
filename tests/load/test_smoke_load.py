"""Smoke load test: 5 параллельных attempts × 3 stages.

Цель — proof-of-pattern для RunLoader. Реальные L10 нагрузочные
сценарии (10×30, p99 budget'ы) поедут в weekly-lane после Phase 5.

В этом тесте каждый "attempt" — это поток операций над
:class:`FakePodLifecycleClient`. Проверяем что:

* все attempts завершаются успешно
* нет orphan pods (каждый зарегистрированный pod — terminated в конце)
* p99 latency в разумных пределах для in-process fake
"""

from __future__ import annotations

import asyncio

import pytest

from tests._fakes.lifecycle import FakePodLifecycleClient, PodState
from tests.load.runloader import RunLoader, RunLoaderConfig

pytestmark = [pytest.mark.load, pytest.mark.slow, pytest.mark.asyncio]


async def test_smoke_load_no_orphan_pods() -> None:
    """5 concurrent attempts × 3 stages: после прогона ни одного pod
    в state RUNNING/STOPPED не должно остаться."""
    client = FakePodLifecycleClient()
    pod_ids: list[str] = []

    async def _one_attempt(stages: int) -> None:
        attempt_id = id(asyncio.current_task())
        pid = f"pod-{attempt_id}"
        pod_ids.append(pid)
        client.register_pod(pid, state=PodState.RUNNING)
        # Несколько "stage"-операций
        for _ in range(stages):
            # имитируем работу — в реальном RunLoader тут было бы
            # ``await attempt_controller.next_stage()``.
            await asyncio.sleep(0)
        # В конце attempt'а pod обязан быть terminated.
        await client.terminate(resource_id=pid)

    cfg = RunLoaderConfig(concurrency=5, stages_per_attempt=3, timeout_seconds=10.0)
    result = await RunLoader(cfg).run(_one_attempt)

    assert result.attempts_total == 5
    assert result.attempts_succeeded == 5
    assert result.attempts_failed == 0
    # Никаких orphan pods — все terminated
    for pid in pod_ids:
        assert client.get_pod_state(pid) == PodState.TERMINATED, (
            f"orphan pod after load: {pid}"
        )
    # P99 sanity — in-process fake должен укладываться в 1с
    assert result.p99_ms < 1000, f"p99 too high: {result.p99_ms}ms"
