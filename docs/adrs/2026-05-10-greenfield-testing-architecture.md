# ADR: Greenfield testing-архитектура для RyotenkAI

**Дата:** 2026-05-10
**Статус:** Accepted; Phase 0 (Foundations) выполняется в данном ADR.
**План:** [docs/plans/structured-hopping-starfish.md](../plans/structured-hopping-starfish.md)
**Связанные ADR:** [2026-05-03-monorepo-uv-workspace-packagization.md](2026-05-03-monorepo-uv-workspace-packagization.md)

## Контекст

В RyotenkAI ~380 Python-тестов и 9 React-тестов. Подавляющая часть — unit с
**2159 вызовами `unittest.mock.patch`**. Frontend только vitest, без MSW /
Storybook / E2E. CI/CD пуст (`.github/workflows/`). Контрактных тестов между
пакетами — 4. Нет property-based, нет chaos, нет канонических fakes для
ключевых Protocols (`IMLflowManager`, `IPodLifecycleClient`, `IRunPodAPI`).

Это блокирует три конкретных проблемы:

1. Hard-to-reach failure modes (RunPod 429 storms, mac-asleep+pod-crash,
   journal rotation race, stale SSH ControlMaster, schema drift) **существуют
   в коде, но ни одним тестом не покрыты** — см. карту 10 untested seams в
   [docs/plans/structured-hopping-starfish-agent-a955b19854b55e69b.md](../plans/structured-hopping-starfish-agent-a955b19854b55e69b.md).
2. `mock.patch` фиксирует "был вызван метод X" вместо "система ведёт себя
   правильно". При рефакторинге 2159 mocks ломаются массово, давая
   ложноположительные регрессии — это видно по чурну
   `src/tests/unit/pipeline/stages/managers/test_deployment_manager.py`
   (99.7-й перцентиль).
3. После Phase B packagization (2026-05-03) появились явные межпакетные швы
   (control / pod / providers / community), но контрактов между ними нет.

## Решение

Принимаем три топ-решения из плана `structured-hopping-starfish`:

### Решение 1 — Greenfield-директория `tests/`

Новая testing-инфраструктура строится в **отдельной директории на корне
репо**, легаси-тесты в `packages/<pkg>/tests/` остаются как есть и не
трогаются. Альтернатива — миграция 2159 mocks — отвергнута: оценка ~6
месяцев против 6-9 на полную постройку greenfield-стека.

Что это меняет:

- Новый testing-код импортирует только `from ryotenkai_<pkg>...` —
  никогда из `packages/<pkg>/tests/...`. Зависимость от legacy-fixtures
  запрещена.
- Новый функционал тестируется **только** в `tests/`. Файлы в
  `packages/<pkg>/tests/` могут только удаляться, не пополняться.
- В `tests/` нет `unittest.mock`, нет `Mock*`-классов, нет heavy
  `@patch`-цепочек. Только канонические fakes и реальные boundary calls.

Этот ADR **не** удаляет ни одного существующего теста. Старые тесты
постепенно выводятся из эксплуатации по мере того, как функциональность
покрывается в `tests/` (Phase 7, continuous).

### Решение 2 — Canonical Fakes per Protocol с lint-сентинелем

Все новые тесты используют **только канонические Fakes**. Никаких mocks для
Protocols. Taxonomy enforced AST-сентинелем
([tests/_lint/test_no_protocol_mocking.py](../../tests/_lint/test_no_protocol_mocking.py)):

- **Stub** — возвращает константу. Допустим для pure-function зависимостей.
- **Mock** — записывает вызовы. **Запрещён для любого Protocol в `tests/`.**
- **Fake** — in-memory реализация с реалистичной state machine. Обязательная
  форма для каждого Protocol в `tests/`.
- **Spy** — fake с записью вызовов. Редкая необходимость.

Канонические fakes живут в `tests/_fakes/Fake<ProtocolMinusI>.py`. Каждый
Protocol имеет parametrized compliance test в
`tests/contract/protocol_compliance/`, который запускается и против real
impl (с `RYOTENKAI_LIVE=1`), и против fake. Тест против только fake —
запрещён (CI fail). Это keystone-механизм против fake drift.

### Решение 3 — Hybrid Hermetic Stack

Один и тот же класс fake работает в двух режимах:

- **In-process** для L2-L4 (прямой импорт из теста)
- **Sidecar HTTP** для L5-L9 (FastAPI-обёртка вокруг того же fake)

Compliance-тест параметризован по обоим транспортам — это единственный
способ честно тестировать transport, circuit breakers, retries, WS
reconnect. Phase 2 добавит `make start-stack` (<60с warm) с `/control` API
на каждом sidecar (`POST /control/inject_429`, `/advance_clock`,
`/set_pod_state`, `/inject_latency`, `GET /control/state` для debug bundles).

Determinism rules: все clocks через `Clock` Protocol; RNG seeded через
`RYOTENKAI_TEST_SEED`; UUIDs — детерминированная фабрика; only `127.0.0.1`,
no DNS.

## Скоуп Phase 0 (этот PR)

Phase 0 — фундамент. Содержимое (см. план §"Phase 0 — Foundations"):

1. Greenfield-дерево `tests/` с пустыми L1-L12 директориями.
2. `tests/_harness/clock.py` — `Clock` Protocol + `RealClock` + `ManualClock`
   (TODO: переедет в `packages/shared/.../utils/clock.py` когда production
   код адаптирует Protocol — Phase 1).
3. `tests/_harness/wait.py` — async `Eventually` / `Consistently`. Время
   через инжектированный `Clock`, не через `time.monotonic()`.
4. `tests/_harness/telemetry.py` — pytest plugin, JSONL-строка на тест в
   `tests/.telemetry/run-<utc>.jsonl`.
5. `tests/_harness/debug_bundle.py` — pytest hook, тарит
   `report.txt + logs/ + fake_state.json + journal.txt` на L5+ failure.
6. `tests/_lint/test_no_protocol_mocking.py` — AST-сентинель.
7. `tests/conftest.py` + `tests/pytest.ini` + `tests/README.md`.
8. `pyproject.toml` — добавлены `hypothesis`, `schemathesis`, `syrupy`,
   `pytest-regressions`, `vcrpy` в `[project.optional-dependencies] dev`.
9. `Makefile` — новые targets `test-new`, `test-all`,
   `lint-no-mocks-in-new-tests`. Старые не трогаются.
10. `.github/workflows/` — `lint.yml`, `presubmit-fast.yml`,
    `presubmit-blocking.yml` (PR-blocking) + `legacy-tests.yml`
    (non-blocking warning).

Phase 0 **не** делает: канонических fakes для Protocols (Phase 1),
hermetic stack (Phase 2), contract testing matrix (Phase 3), property /
golden / chaos / load (Phase 4-5), visual / replay (Phase 6).

## Exit criteria по фазам

| Phase | Длительность | Exit criteria | Статус |
|---|---|---|---|
| **Phase 0 — Foundations** | 2-3 нед | `make test-new` зелёный; `make lint-no-mocks-in-new-tests` ловит синтетический mock-Protocol; legacy lane non-blocking warning в CI; telemetry пишется на каждый тест-сесшн | ✅ done |
| **Phase 1 — Top-3 Canonical Fakes + first L2 tests** | 4-6 нед | `tests/_fakes/{mlflow,lifecycle,runpod}.py` + compliance tests pass на real+fake; ≥10 L2 tests landed; новый код имеет 100% покрытие в `tests/` | ✅ done |
| **Phase 2 — Hermetic stack** | 3-4 нед | `make start-stack` <60с warm; smoke E2E зелёный; первые L6 tests landed; `/control` API работает на каждом fake-sidecar | ✅ done |
| **Phase 3 — Contract testing matrix** | 4-6 нед | Каждая cross-package boundary имеет gate (OpenAPI, plugin manifest, marker files, journal events); schema changes не могут merge silently; FE zod guards | ✅ done |
| **Phase 4 — Property + Snapshots + остальные fakes** | 3-4 нед | Hypothesis suite находит ≥5 latent bugs; remaining canonical fakes (`ITrainerSpawner`, `ISSHClient`, `IHFHubClient`, `IJobClient`) landed | ✅ done |
| **Phase 5 — Chaos + Load** | 6-8 нед | Nightly chaos (18-scenario catalog) зелёный 14 дней подряд; load test ловит regression; ChaosScenario framework + RunLoader | ✅ done (14/18 passing, 4 partial) |
| **Phase 6 — Visual + Replay** | 3-4 нед | Storybook + lost-pixel; Playwright trace replay corpus; axe + Lighthouse | ✅ done (infrastructure + 5 stories + 3 e2e flows; corpus ≥30 flows deferred to phase-7 backfill) |
| **Phase 7 — Legacy decommissioning** (continuous) | annual review | Legacy lane либо archived, либо пустой; тесты в `packages/<pkg>/tests/`, не падавшие >12 месяцев и покрытые в `tests/`, удаляются | ⏳ ongoing — policy in [2026-05-11-legacy-test-decommissioning.md](2026-05-11-legacy-test-decommissioning.md), `make legacy-audit` доступен |

## Финальная сводка (после Phase 6, 2026-05-11)

| Артефакт | Кол-во |
|---|---|
| Python тесты в `tests/` (green lane) | 309 pass / 90 skip |
| FE тесты в greenfield lane | 117 |
| Канонические fakes | 8 |
| Production protocols (вкл. Clock) | 7 |
| Hermetic stack sidecars | 4 |
| Property тесты (Hypothesis) | 18 |
| Golden snapshots | 4 |
| Chaos scenarios catalog | 18 (14 pass, 4 partial) |
| Real protocol adapters | 4 |
| GitHub workflows | 10 (lint, presubmit-fast, presubmit-blocking, legacy-tests, nightly, chaos-deep, load, **visual-regression**, **e2e**, **release-gate**) |
| Storybook stories | 5 components, 16 variants |
| Playwright e2e spec files | 4 (`create_run`, `cancel_run`, `view_run_logs`, `a11y`) |
| Replay corpus registry | 3 flows (traces ship after first release-gate run) |
| Legacy lane (untouched) | ~933 tests, audited via `make legacy-audit` |

## Известные риски и митигации

См. план §"Risks, open questions, и их разрешение" (R1-R20). Ключевые:

- **R1 fake drift** — compliance test parametrized по real+fake; nightly
  `RYOTENKAI_LIVE=1`; quarterly cassette diff.
- **R6 L6/L9 flake creep** — quarantine SLA 30 дней; `pytest-rerunfailures`
  запрещён в main lanes; debug bundles + telemetry; Eventually вместо sleep.
- **R13 двойная инфраструктура** — legacy lane non-blocking warning,
  никогда не блокирует merge.
- **R20 legacy lane никогда не отключают** — annual review в Phase 7;
  цель к Phase 7+12 мес: legacy lane либо archived, либо пустой.

## Что **не** входит в этот ADR

- Удаление какого-либо файла под `packages/<pkg>/tests/` или
  `packages/<pkg>/src/` — Phase 0 чисто аддитивен.
- Изменение `pytest.ini` на корне репо — он остаётся для legacy lane.
- Production extraction новых Protocols (`IRunPodAPI`, `ITrainerSpawner`,
  `ISSHClient`, `IHFHubClient`, `Clock`) — это работа Phase 1+, additive,
  legacy call-sites не трогаются.
- Visual regression и frontend MSW — отложены в Phase 6 решением
  пользователя.

## Верификация Phase 0

Команды из плана §"После Phase 0":

```bash
# AST sentinel ловит mock-Protocol только в tests/
echo "from unittest.mock import patch
@patch('ryotenkai_shared.infrastructure.mlflow.protocol.IMLflowManager')
def test_dummy(): pass" > tests/unit/test_smoke.py
make lint-no-mocks-in-new-tests   # FAIL ожидается

# Тот же файл в legacy директории — допустим (sentinel не сканирует packages/)
make lint-no-mocks-in-new-tests   # PASS ожидается

# Telemetry plugin
pytest -c tests/pytest.ini tests/unit/   # генерирует tests/.telemetry/run-*.jsonl

# Eventually работает
python -c "import asyncio; from tests._harness.wait import Eventually; \
  asyncio.run(Eventually(lambda: False, timeout=1))"   # FAIL: TimeoutError
```
