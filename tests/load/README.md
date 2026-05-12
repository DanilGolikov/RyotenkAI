# L10 — Load / scale tests (RunLoader)

Уровень нагрузочного тестирования: "10 параллельных attempts × 30
stages each" из плана
[docs/plans/structured-hopping-starfish.md](../../docs/plans/structured-hopping-starfish.md)
строка L10. Цель — поймать регрессии параллелизма, контеншена,
протекающих ресурсов, нелинейного роста latency.

## RunLoader

Лёгкий драйвер ([`runloader.py`](runloader.py)) с двумя примитивами:

* :class:`RunLoaderConfig` — concurrency, stages_per_attempt, timeout.
* :class:`RunLoader` — запускает N параллельных attempts через
  ``asyncio.gather``; собирает p50/p99 latency.

Сценарий — это просто async callable ``attempt_fn(stage_count)``,
который выполняет один полный attempt. RunLoader сам параллелит.

## Текущее наполнение

| Файл | Что тестит |
|---|---|
| `test_smoke_load.py` | 5 × 3 — no orphan pods через FakePodLifecycleClient |

## CI scheduling

L10 — **weekly lane** в [docs/plans/structured-hopping-starfish.md](../../docs/plans/structured-hopping-starfish.md):

| Lane | Trigger | Что запускает |
|---|---|---|
| `presubmit-fast` | every push | НЕ запускает L10 |
| `weekly` | cron Mon 03:00 UTC | полный L10 + p99 budget'ы |
| `release-gate` | tag push | smoke load на каждый релиз |

## Как добавить нагрузочный сценарий

1. Положить новый файл `tests/load/test_<scenario>_load.py`.
2. Пометить тест `@pytest.mark.load + @pytest.mark.slow`.
3. Описать ``attempt_fn`` — одна attempt — это, грубо говоря, полный
   жизненный цикл runs'а (provision → train → retrieve → cleanup).
4. Использовать canonical fakes из ``tests/_fakes/`` — НЕ реальный
   RunPod/MLflow. Цель — поймать contention в нашем коде, а не у
   провайдеров.

Пример скелета:

```python
import pytest
from tests._fakes.lifecycle import FakePodLifecycleClient, PodState
from tests.load.runloader import RunLoader, RunLoaderConfig

pytestmark = [pytest.mark.load, pytest.mark.slow, pytest.mark.asyncio]


async def test_my_load_scenario() -> None:
    client = FakePodLifecycleClient()

    async def _attempt(stages: int) -> None:
        ...

    result = await RunLoader(RunLoaderConfig(concurrency=10, stages_per_attempt=30)).run(_attempt)
    assert result.attempts_failed == 0
    assert result.p99_ms < TARGET_P99_BUDGET
```

## P99 budget guidance

* In-process fakes: <100 ms p99 для тривиальных операций.
* Sidecar stack (Phase 5): <500 ms p99 для одной HTTP-round-trip.
* End-to-end attempt (Phase 5): <30 s p99 на полный 30-stage attempt.

Budget жёсткий — превышение блокирует merge. Любое расширение бюджета
требует объяснения в PR description.

## Хранение результатов

JSON-сериализуемый ``LoadResult`` складывается в
``tests/.load_results/<run-id>.json`` (TODO: пока не реализовано —
ждёт интеграции с pytest-plugin аналогичным `telemetry.py`).

## TODO

* `test_pipeline_attempt_load.py` — 10 × 30 на полный attempt
  через sidecar stack.
* `test_runpod_api_burst_load.py` — burst-нагрузка на ``find_pod`` +
  list_pods через FakeRunPodAPI.
* `runloader/scenarios/` — переиспользуемые ``attempt_fn``-шаблоны.
* `LoadResult` → JSON-dump в pytest hook.
* Weekly workflow `.github/workflows/load.yml`.
* P99 regression detector (compare against last 10 runs).
