# L9 — Chaos / fault-injection scenarios

Уровень chaos-инженерии: каждый сценарий определяет одну инъекцию
сбоя (429-storm, mlflow-down, partition, OOM) и формальные ожидания
по восстановлению. Каталог из плана —
[docs/plans/structured-hopping-starfish.md](../../docs/plans/structured-hopping-starfish.md)
Decision 4.

## Фреймворк

Реализован в [`tests/_harness/chaos.py`](../_harness/chaos.py):

* :class:`ChaosScenario` — Protocol с четырьмя async-методами:
  ``precondition``, ``inject``, ``steady_state``, ``cleanup``.
* :func:`register_scenario` — декоратор регистрации в каталоге.
* :class:`ScenarioContext` — расшаренное состояние (stack, clock,
  extras для fakes).
* :class:`ScenarioRunner` / :func:`run_chaos_scenario` — драйвер,
  пишет debug-bundle при падении.

## Каталог сценариев (план, 18 штук)

| # | Имя | Что инжектится | Статус |
|---|-----|---------------|--------|
| 1 | `Runpod429Storm` | fake-runpod 429 30с; expect JobClient backoff | ✅ implemented |
| 2 | `MlflowCircuitOpen` | fake-mlflow дропает; expect circuit breaker open | ✅ implemented |
| 3 | `MacSleepDuringRun` | ManualClock прыгает 10 мин; expect heartbeat re-establishment | TODO |
| 4 | `MacSleepPlusPodCrash` | sleep + pod crash; expect faithful reporting | TODO |
| 5 | `PodHibernationWhileMacAwake` | fake-runpod транзитит HIBERNATED; expect detector + auto-resume | TODO |
| 6 | `JournalRotationRace` | rotate `events.NNN.jsonl` при чтении; expect no event loss | TODO |
| 7 | `StaleSshControlMaster` | kill ssh master socket; expect transparent reconnection | TODO |
| 8 | `ConcurrentTerminate` | два cancel call'а 50мс; expect idempotent cancellation | ✅ implemented |
| 9 | `PipelineContextCorruption` | garbage в byte N; expect detection + safe abort | TODO |
| 10 | `TrainerCallbackFailure` | fake-trainer raises; expect runner reports error | TODO |
| 11 | `OOMKilledTrainer` | fake-trainer exits 137; expect classified as Failure | TODO |
| 12 | `DiskFullOnPod` | ENOSPC; expect graceful degradation | TODO |
| 13 | `ProviderRetryStorm` | каждый external call fails first attempt | TODO |
| 14 | `ClockSkew` | runner clock 60с впереди; expect heartbeat tolerance | TODO |
| 15 | `WSReconnectStorm` | FE WS drops 10x за 30с; expect FE reconnect | TODO |
| 16 | `OpenAPIDriftMidFlight` | control reloaded с мутированной schema | TODO |
| 17 | `RunpodGraphqlPartialResponse` | truncate; expect parser doesn't crash | TODO |
| 18 | `MlflowDoubleFinalization` | два `end_run`; expect idempotency | TODO |

Реализовано: **3/18**.

## Конвенции

* Каждый сценарий — отдельный модуль под `tests/chaos/scenarios/`.
* `@register_scenario` декоратор обязателен; имя уникально.
* `name`, `tags`, `recovery_window` — required class attributes.
* `@pytest.mark.chaos` + `@pytest.mark.slow` на всех тестах.
* **Не использовать `time.sleep`** — `ManualClock` + ``Eventually``.
* **Cleanup всегда вызывается**, даже если steady_state упал.

## In-process vs sidecar stack

Сценарии делятся на два режима:

* **In-process** (текущие 3) — используют canonical fakes напрямую,
  без `Stack.boot()`. Быстрый запуск, проверяют логику chaos-surface'а
  fakes. Драйвер тривиален: ``ctx.extras`` несёт fakes.
* **Sidecar-stack** — нужны для сценариев, где SUT — HTTP-клиент,
  ходящий через сеть (toxiproxy и т.п.). Драйвер использует
  ``await Stack.boot()`` и ``run_chaos_scenario(stack, scenario)``.

В Phase 5 большинство сценариев переедет на sidecar-stack, поскольку
там можно достоверно тестировать таймауты соединений.

## Запуск

```bash
# Все scenarios локально (без --slow они скипаются по дефолту в CI)
pytest tests/chaos/ -m chaos

# Один конкретный
pytest tests/chaos/scenarios/test_runpod_429_storm.py

# Только collection — sanity check
pytest tests/chaos/ --collect-only
```

CI lanes:

* `nightly` — random subset N=5 сценариев каждые сутки.
* `chaos-deep` — weekly full catalog × 5 random seeds (~2-4 часа).

## Debug bundles

При падении в `steady_state` фреймворк дампит JSON-bundle в
``tests/.debug_bundles/<scenario>-<ts>.chaos.json``. Содержит:

* pre/post snapshot всех fakes/sidecars
* timeline шагов сценария
* traceback на момент падения

## TODO (per scenario)

* Доделать оставшиеся 15 сценариев каталога по плану.
* Sidecar-mode driver для каждого сценария (сейчас только in-process).
* Random-seed loader `pytest tests/chaos/ --seeds=5`.
* `tests/chaos/test_catalog.py` — discovery + parametrize всех
  зарегистрированных сценариев.
