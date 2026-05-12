# Testing Architecture для RyotenkAI — Greenfield Strategy

> Архитектурное решение по тестированию ML-pipeline монорепы. **Greenfield-подход:** новая testing-инфраструктура строится в отдельной директории `tests/` на корне репо, легаси-тесты в `packages/<pkg>/tests/` не трогаем. Слоёный пирог из 12 уровней с каноническими fakes, гибридным hermetic-стеком, контрактным тестированием. Выполнение — 6-9 месяцев фокусированно.

---

## Context

Сейчас в RyotenkAI ~380 Python-тестов и 9 React-тестов. Подавляющая часть — unit с 2159 вызовами `unittest.mock.patch`. Frontend только vitest, без MSW/Storybook/E2E. CI/CD отсутствует (`.github/workflows/` пустой). Контрактных тестов между пакетами 4. Нет property-based, нет chaos, нет fakes для критических Protocols (`IMLflowManager`, `IPodLifecycleClient`, `IRunPodAPI`).

**Почему это не работает дальше:**

1. Невозможно дотянуться до hard-to-reach фич — RunPod 429 storms, mac-asleep+pod-crash, journal rotation race, stale SSH ControlMaster, schema drift между control и web. Они существуют в коде, но не покрыты ни одним тестом ([docs/plans/structured-hopping-starfish-agent.md](docs/plans/structured-hopping-starfish-agent.md) — детальная карта 10 untested seams).
2. `mock.patch` фиксирует "был вызван метод X" вместо "система ведёт себя правильно". При рефакторинге 2159 mocks ломаются массово, давая ложноположительные "regressions".
3. После Phase B packagization (2026-05-03) появились явные межпакетные швы (control/pod/providers/community), но контрактов между ними нет.
4. ML-pipeline по характеру похож на K8s controller + Terraform provider + Grafana plugin host — у этих систем есть отработанные testing playbooks, которые мы не применяем.

**Greenfield-подход (выбор пользователя):** новая testing-инфраструктура строится в отдельной директории. Легаси-тесты в `packages/<pkg>/tests/` остаются как есть, не трогаются. Это убирает 6 месяцев миграционной работы — фокус сразу на правильной архитектуре. Старые тесты постепенно удаляются по мере того, как новые покрывают функциональность.

**Желаемый исход:** инженер пишет фичу → быстрый внутренний цикл (`make test-new` <90с) гарантирует логику через канонические fakes → PR-blocking lane (<12 мин) гарантирует контракты и интеграцию → nightly chaos+load+property ловят hard-to-reach failure modes → flake-quarantine не даёт зелёному CI лгать. Тесты не зависят от живого RunPod/MLflow/HF Hub. Рефакторинг внутри пакета не ломает тесты.

**Bench-mark:** аналоги в индустрии — Kubernetes (`test/e2e_node`+`test/e2e`, kind, ChaosMesh, Prow flake-board), Argo Workflows (`make start PROFILE=mysql`, retry-policy fixtures), Grafana plugin SDK contract harness, Terraform/Pulumi providers (VCR cassettes, providertest gRPC replay), Unreal automation (Gauntlet session orchestrator, screenshot comparison). Стратегия — синтез этих паттернов.

**Соответствие best-practices** (2024-2026): Shai Yallin "Fake, Don't Mock" + Fowler "Mocks Aren't Stubs" подтверждают предпочтение fakes для классов с состоянием; Test Trophy (Kent C. Dodds) и Honeycomb (Spotify) обосновывают перевес L4-L6 над глубокими unit-тестами; schemathesis + Pact — стандарт для contract testing OpenAPI; Storybook + Chromatic/lost-pixel — стандарт для visual regression.

---

## Greenfield directory layout

Новая testing-инфраструктура полностью изолирована от легаси:

```
RyotenkAI/
├── packages/
│   └── <pkg>/
│       ├── src/                    # ← production code (не трогаем)
│       └── tests/                  # ← LEGACY tests, остаются как есть
├── tests/                          # ← НОВАЯ testing-инфраструктура (greenfield)
│   ├── _fakes/                     # канонические Fake* для всех Protocols
│   │   ├── mlflow.py               # FakeMLflowManager
│   │   ├── lifecycle.py            # FakePodLifecycleClient
│   │   ├── runpod.py               # FakeRunPodAPI
│   │   ├── trainer.py              # FakeTrainerSpawner
│   │   ├── ssh.py                  # FakeSSHClient
│   │   ├── hf_hub.py               # FakeHFHubClient
│   │   ├── job_client.py           # FakeJobClient
│   │   └── clock.py                # ManualClock
│   ├── _harness/                   # тестовая инфраструктура
│   │   ├── wait.py                 # Eventually/Consistently
│   │   ├── telemetry.py            # тест-телеметрия
│   │   ├── debug_bundle.py         # debug bundle hook
│   │   └── stack/                  # hermetic stack
│   │       ├── docker-compose.yaml
│   │       ├── orchestrator.py     # Gauntlet-style harness
│   │       └── sidecars/
│   │           ├── runpod_server.py
│   │           ├── mlflow_server.py
│   │           ├── vllm_server.py
│   │           └── hf_hub_server.py
│   ├── unit/                       # L1 — pure logic тесты для нового кода
│   ├── component/                  # L2 — class + fakes
│   ├── contract/                   # L3 — boundary contracts
│   │   ├── protocol_compliance/    # compliance tests за каждый Protocol
│   │   ├── openapi_drift/          # OpenAPI gate
│   │   ├── plugin_manifest/        # plugin TOML schema
│   │   ├── markers/                # marker-files schema
│   │   └── journal/                # journal events schema
│   ├── integration/                # L4 — multi-component
│   ├── e2e/                        # L5 — subsystem e2e
│   ├── stack/                      # L6 — full hermetic stack
│   ├── property/                   # L7 — Hypothesis
│   ├── golden/                     # L8 — snapshots
│   ├── chaos/                      # L9 — ChaosScenario catalog
│   │   └── scenarios/
│   ├── load/                       # L10 — RunLoader
│   ├── visual/                     # L11 — visual regression (Phase 6)
│   ├── replay/                     # L12 — replay regression
│   ├── conftest.py                 # shared fixtures
│   ├── pytest.ini                  # отдельный pytest config для tests/
│   └── README.md                   # как писать тесты в этой директории
├── pytest.ini                      # legacy config — не трогаем
└── Makefile
    ├── test                        # legacy lane: pytest packages/ community/
    ├── test-new                    # new lane: pytest tests/
    └── test-all                    # оба
```

**Принципы greenfield-директории:**

1. **Никакой legacy-инфраструктуры.** В `tests/` нет `unittest.mock`, нет `Mock*` классов, нет heavy `@patch`-чейнов. Только канонические fakes и реальные boundary calls.
2. **Импорты только production code.** Тесты в `tests/` импортируют `from ryotenkai_control.pipeline import ...`, никогда из `packages/<pkg>/tests/...`. Зависимость от легаси-fixtures запрещена.
3. **Новый код тестируется только в `tests/`.** ADR: после внедрения этого плана новые фичи в `packages/<pkg>/src/` тестируются только в `tests/`. Файлы в `packages/<pkg>/tests/` могут только удаляться, не пополняться.
4. **Параллельный CI.** Legacy lane и new lane живут параллельно. Legacy lane может стать optional после Phase 4.

---

## Architecture: 12-слойный пирог тестирования

Каждый тест принадлежит ровно одному слою. Каждый слой имеет одну цель, чёткое правило изоляции и бюджет времени.

| L | Цель | Инструменты | Где живёт | Бюджет | PR-blocking? |
|---|------|-------------|-----------|--------|--------------|
| **L0 Static gates** | Найти ошибки до запуска тестов | ruff, mypy --strict, importlinter, sentinel-AST, openapi-diff, json-schema validate, eslint, tsc, custom `make lint-no-mocks-in-new-tests` | top-level + `.pre-commit-config.yaml` + `.github/workflows/lint.yml` | <60с | Да |
| **L1 Unit** | Чистая логика без I/O | pytest, pytest-asyncio, vitest | `tests/unit/` | <2с/тест, <90с total | Да |
| **L2 Component** | Один класс, все коллабораторы — fakes (никогда mocks) | pytest + canonical fakes | `tests/component/` | <500мс/тест | Да |
| **L3 Contract** | Boundary contracts: HTTP, WS, Protocol, plugin TOML, CLI parity, marker files, journal | schemathesis, hypothesis-jsonschema, pydantic, Protocol-compliance harness | `tests/contract/` | <30с total | Да |
| **L4 Integration** | Multi-component внутри одного процесса | pytest со собранным fake-стеком | `tests/integration/` | <5с/тест | Да (subset) |
| **L5 Subsystem e2e** | Runner отдельно, control отдельно, web отдельно — реальные boundaries к fakes | pytest+subprocess, vitest+msw, runner-collocated | `tests/e2e/` | <30с/тест | Да (smoke), Nightly (full) |
| **L6 Stack e2e (hermetic)** | Полный стек через `make start-stack` | pytest-orchestrator, Playwright | `tests/stack/` | <2 мин/тест | Smoke на PR, full nightly |
| **L7 Property / fuzz** | Инварианты на множестве входов | hypothesis, schemathesis stateful, atheris | `tests/property/` | nightly, 5-10 мин | Nightly |
| **L8 Snapshot / golden** | Run reports, ML-артефакты, OpenAPI, plugin manifests, generated TS-types | syrupy, pytest-regressions, vitest snapshot | `tests/golden/` | <5с/тест | Да (regen — review) |
| **L9 Chaos / fault injection** | 429 storms, partitions, OOM, mac-sleep, pod-crash, journal-rotation race | `--chaos` flag на fakes, toxiproxy-py | `tests/chaos/` | nightly + on-demand | Нет |
| **L10 Load / RunLoader** | "10 параллельных attempts × 30 stages each" | locust или pytest-driven генератор | `tests/load/` | weekly + release | Нет |
| **L11 Visual regression** | UI-компоненты, дашборды | Storybook + lost-pixel + Playwright trace | `tests/visual/`, `web/.storybook/` | <5 мин на PR | Phase 6+ (низкий приоритет) |
| **L12 Replay regression** | Записанные user-flows воспроизводятся на каждый release | Playwright traces stored as artifacts | `tests/replay/` | release-gate | Release |

**Целевое распределение** в новой директории: ~70% L1-L2, ~15% L3-L4, ~10% L5-L6, ~5% L7-L12. Legacy `packages/<pkg>/tests/` остаётся как есть, не считается в это распределение.

---

## Decision 1: Canonical Fakes (зафиксировано пользователем)

Все новые тесты используют только канонические Fakes. Никаких mocks для Protocols.

### Taxonomy (enforced by lint только в `tests/`)

- **Stub** — возвращает константу. Допустим для pure-function зависимостей (`Clock`).
- **Mock** — записывает вызовы. **Запрещён для любого Protocol в `tests/`.** В legacy `packages/<pkg>/tests/` — допустим (не трогаем).
- **Fake** — in-memory реализация с реалистичной state machine. **Обязательная форма** для каждого Protocol в `tests/`.
- **Spy** — fake, обёрнутый для записи вызовов. Редкая необходимость.

### Канонические fakes — naming & location

Все fakes живут в `tests/_fakes/`. Naming: `Fake<ProtocolMinusI>`. Импорт: `from tests._fakes.mlflow import FakeMLflowManager`.

| Protocol | Production файл | Fake | Реалистичная state-machine |
|---|---|---|---|
| `IMLflowManager` | [packages/shared/src/ryotenkai_shared/infrastructure/mlflow/protocol.py](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/protocol.py) | `tests/_fakes/mlflow.py` | experiments + runs + params + metrics; time-travel; chaos modes |
| `IPodLifecycleClient` | [packages/shared/src/ryotenkai_shared/infrastructure/lifecycle/protocol.py](packages/shared/src/ryotenkai_shared/infrastructure/lifecycle/protocol.py) | `tests/_fakes/lifecycle.py` | `PROVISIONING → RUNNING → STOPPING → STOPPED → TERMINATED` |
| `IRunPodAPI` (новый Protocol) | вытащить из `packages/providers/src/ryotenkai_providers/runpod/` | `tests/_fakes/runpod.py` | pods registry, GraphQL stub, 429-mode, hibernation |
| `ITrainerSpawner` | вытащить из `packages/pod/src/ryotenkai_pod/runner/supervisor.py` | `tests/_fakes/trainer.py` | journal events на детерминированном расписании |
| `ISSHClient` | `packages/shared/.../utils/ssh*.py` | `tests/_fakes/ssh.py` | virtual filesystem, programmable connect/exec/copy |
| `IHFHubClient` (новый) | новый Protocol для HF API | `tests/_fakes/hf_hub.py` | model registry, upload state, error injection |
| `IJobClient` | `packages/shared/.../utils/clients/job_client.py` | `tests/_fakes/job_client.py` | синхронизирован с FakeTrainerSpawner |
| `Clock` (новый) | новый Protocol | `tests/_fakes/clock.py` | детерминированное время с `advance()` |

### Контракт-of-fake (keystone механизм)

Для каждого Protocol — `tests/contract/protocol_compliance/test_<protocol>_compliance.py`. Параметризован по реализациям:

```python
@pytest.fixture(params=[real_impl_factory, fake_impl_factory])
def lifecycle_client(request, env): ...

class TestIPodLifecycleClientCompliance:
    def test_terminate_idempotent(self, lifecycle_client): ...
    def test_resume_after_stop_returns_running(self, lifecycle_client): ...
```

`real_impl_factory` гейтится `RYOTENKAI_LIVE=1` (nightly + on-demand); `fake_impl_factory` — каждый PR. **Тест против только fake — запрещён** (CI fail).

### Lint enforcement (только для `tests/`)

`tests/_lint/test_no_protocol_mocking.py`:
- проходит по дереву `tests/` (НЕ `packages/<pkg>/tests/`)
- падает, если `mock.patch` / `MagicMock(spec=...)` / `create_autospec(...)` нацелен на любой Protocol
- не нужен allowlist — directory greenfield, никогда не было mocks
- legacy `packages/<pkg>/tests/` не сканируется

---

## Decision 2: Hybrid Hermetic Stack (зафиксировано пользователем)

Один и тот же класс fake работает в двух режимах: in-process для L2-L4, sidecar HTTP для L5-L9.

### Топология `make start-stack`

| Сервис | Process model | Назначение |
|---|---|---|
| `control-plane` | Real, in-process uvicorn | SUT |
| `runner` | Real, subprocess | SUT |
| `web` | Real, vite preview | SUT |
| `fake-runpod` | Sidecar HTTP (FastAPI) | RunPod GraphQL stub, программируется через `/control` |
| `fake-mlflow` | Sidecar HTTP | MLflow protocol |
| `fake-vllm` | Sidecar HTTP | OpenAI-compatible inference stub |
| `fake-hf-hub` | Sidecar HTTP | model registry, file serving |
| `fake-ssh-pod` | Real sshd in container | SSH ControlMaster, journal, marker files |
| `clock` | In-process module | ManualClock |

### Гибридный режим — one class, two transports

```python
# tests/_fakes/runpod.py
class FakeRunPodAPI:                # in-process для L2-L4
    async def list_pods(self): ...

# tests/_harness/stack/sidecars/runpod_server.py
class FakeRunPodHTTPServer:          # обёртка для L5-L9
    def __init__(self, fake: FakeRunPodAPI):
        self._fake = fake
        self._app = FastAPI()
```

Compliance test параметризован: тестирует одно и то же поведение через прямой импорт и через HTTP-клиент.

### `/control` API на каждом sidecar

- `POST /control/inject_429?count=5`
- `POST /control/advance_clock?seconds=300`
- `POST /control/set_pod_state?id=X&state=HIBERNATED`
- `POST /control/inject_latency?ms=500`
- `GET /control/state` — snapshot для debug bundles

### Determinism rules

- Все clocks через `Clock` Protocol; default — `monotonic`, в стеке — `ManualClock`.
- RNG seeded через `RYOTENKAI_TEST_SEED`; логируется на failure.
- UUIDs — детерминированная фабрика.
- Network only `127.0.0.1`; no DNS.
- `make start-stack` boot <60с warm, <120с cold; `make state-dump` сохраняет JSON для debug.

---

## Decision 3: Contract testing matrix

Все contract-тесты живут в `tests/contract/`.

| Граница | Артефакт | Gate | Conformance |
|---|---|---|---|
| Control HTTP API | [web/src/api/openapi.json](web/src/api/openapi.json) | расширить `make verify-api-sync`: PR fail при hash drift без regen `schema.d.ts` | schemathesis stateful в `tests/contract/openapi_drift/` |
| Control WebSocket events | новый `packages/control/src/.../ws/events_schema.json` | JSON-schema validation в CI | property test в `tests/contract/markers/` |
| Plugin manifest TOML v5 | Pydantic StrictBaseModel + golden corpus | round-trip property test | `tests/contract/plugin_manifest/` |
| Runner internal HTTP `/internal/events` | OpenAPI fragment + JSON schema | как control HTTP | `tests/contract/runner_api/` |
| Protocol compliance | сама Protocol-definition | mypy --strict | `tests/contract/protocol_compliance/` |
| CLI ↔ API parity | новый `tests/contract/cli_api/test_parity.py` (замена legacy) | CI fail на divergence | parametrized по каждой команде |
| Marker files | JSON schema | schema test | `tests/contract/markers/` |
| Journal events | JSON schema per event kind | schema validation в writer и reader | `tests/contract/journal/` |
| FE→BE runtime guard | `zod` schemas из openapi.json | codegen `gen:api:zod` | runtime validation каждого API response |

OpenAPI gate один ловит большинство сегодняшнего silent type drift.

---

## Decision 4: ChaosScenario framework

```python
# tests/_harness/chaos.py
class ChaosScenario(Protocol):
    name: str
    def precondition(self, stack): ...
    def inject(self, stack): ...
    def steady_state(self, stack) -> Assertion: ...
    recovery_window: timedelta
```

**Каталог в `tests/chaos/scenarios/`:**

1. `Runpod429Storm` — fake-runpod 429 30с; expect JobClient backoff
2. `MlflowCircuitOpen` — fake-mlflow дропает; expect circuit breaker open, no metric loss
3. `MacSleepDuringRun` — ManualClock прыгает 10 мин; expect heartbeat re-establishment
4. `MacSleepPlusPodCrash` — sleep + pod crash; expect faithful reporting
5. `PodHibernationWhileMacAwake` — fake-runpod transitions HIBERNATED; expect detector + auto-resume
6. `JournalRotationRace` — rotate `events.NNN.jsonl` при чтении; expect no event loss
7. `StaleSshControlMaster` — kill ssh master socket; expect transparent reconnection
8. `ConcurrentTerminate` — два cancel calls 50мс; expect idempotent cancellation
9. `PipelineContextCorruption` — garbage в byte N; expect detection + safe abort
10. `TrainerCallbackFailure` — fake-trainer raises; expect runner reports error
11. `OOMKilledTrainer` — fake-trainer exits 137; expect classified as Failure
12. `DiskFullOnPod` — ENOSPC; expect graceful degradation
13. `ProviderRetryStorm` — каждый external call fails first attempt; expect bounded total time
14. `ClockSkew` — runner clock 60с впереди; expect heartbeat tolerance
15. `WSReconnectStorm` — FE WS drops 10x за 30с; expect FE reconnect
16. `OpenAPIDriftMidFlight` — control reloaded with mutated schema; expect FE detects
17. `RunpodGraphqlPartialResponse` — truncate; expect parser doesn't crash
18. `MlflowDoubleFinalization` — два `end_run`; expect idempotency

Запуск: nightly subset (random N=5), weekly full catalog, on-demand `pytest tests/chaos/`.

---

## Decision 5: Observability-as-tests

L4+ тесты долгие. Не sleep — Eventually/Consistently.

- `tests/_harness/wait.py`: `Eventually(condition, timeout=30s, poll=100ms)`, `Consistently(condition, duration=2s, poll=100ms)`
- **Progress reports** — pytest plugin в `tests/_harness/progress.py`: SIGUSR1 dumps current test, current Eventually condition, last 50 lines of fake state
- **Test telemetry** — `tests/_harness/telemetry.py`: каждый тест emits JSONL event в `tests/.telemetry/run-<id>.jsonl`. Aggregated в SQLite/duckdb per CI run
- **Debug bundles** — `tests/_harness/debug_bundle.py`: pytest hook tars `{logs/, fake-state/, journal/, screenshot.png}` на L5+ failure

---

## Decision 6: Flake management

- `@pytest.mark.flaky` и Playwright `test.describe.flaky()` → отдельная non-blocking lane
- **Quarantine SLA: 30 дней.** CI fails merge если test flaky >30 дней
- **No silent retry.** `pytest-rerunfailures` запрещён в main lanes
- **Failure fingerprinting** — assertion message + first 5 stack frames hashed
- **Quarantine dashboard** — live page

---

## Decision 7: Frontend pillar (отложен в Phase 6)

Зафиксировано пользователем: visual regression — низкий приоритет.

| Концерн | Инструмент | Где | Phase |
|---|---|---|---|
| Component logic | vitest + RTL + **MSW** (handlers генерятся из openapi.json) | `web/src/**/*.test.tsx` (новые тесты — рядом с компонентом) | Phase 3 |
| Visual regression | Storybook + lost-pixel (self-hosted) | `web/.storybook/`, `web/src/**/*.stories.tsx` | Phase 6 |
| E2E | Playwright против hermetic stack | `tests/stack/web/` | Phase 2 (smoke) → Phase 5 (full) |
| API contract | openapi-typescript + zod | `web/src/api/zod.ts` (generated) | Phase 3 |
| Accessibility | `@axe-core/playwright`, jest-axe | colocated | Phase 6 |
| Performance | Lighthouse CI | `web/lighthouserc.json` | Phase 6 |

---

## Decision 8: CI/CD architecture

`.github/workflows/`:

| Lane | Trigger | Что запускает | Бюджет |
|---|---|---|---|
| `presubmit-fast` | every push | L0 + `make test-new` (только L1) | <3 мин |
| `presubmit-blocking` | PR + main | L0 + L1-L4 в `tests/` + smoke L5-L6 | <12 мин |
| `legacy-tests` | PR + main, **non-blocking warning** | `make test` (legacy lane) | <10 мин |
| `nightly` | cron 02:00 UTC | L5-L12 + L7 + L9 | ~60 мин |
| `chaos-deep` | weekly | Full chaos catalog × 5 random seeds | 2-4 часа |
| `load` | weekly | L10 RunLoader | 1-2 часа |
| `release-gate` | tag push | All layers | 90 мин |
| `flake-quarantine` | every push, **non-blocking** | quarantined только | best-effort |
| `live-protocol-compliance` | nightly | RYOTENKAI_LIVE=1 against real RunPod sandbox | 30 мин |

Caching: `uv` keyed by `uv.lock`; `pnpm` keyed by lockfile; Docker layer cache. Aggregation: GitHub Pages site из telemetry-store.

**Legacy lane изначально non-blocking warning** — если кто-то сломал legacy mock, это не блокирует merge, но видно в PR. После Phase 4 решаем — оставить как warning, заархивировать или окончательно удалить.

---

## Migration roadmap (6-9 месяцев — упрощено благодаря greenfield)

Без миграции 2159 mocks roadmap значительно короче:

| Phase | Длительность | Scope | Exit criteria |
|---|---|---|---|
| **Phase 0 — Foundations** | 2-3 нед | Создать `tests/` дерево; `tests/_harness/` (Eventually/Consistently/telemetry/debug bundles); `tests/_lint/test_no_protocol_mocking.py`; CI skeleton с двумя lanes (legacy + new) | Каждый новый тест ships with telemetry; new lane PR-blocking; legacy lane non-blocking warning |
| **Phase 1 — Top-3 Canonical Fakes + first L2 tests** | 4-6 нед | `tests/_fakes/{mlflow,lifecycle,runpod}.py` + compliance tests; первые L1-L2 tests для нового кода; начинаем писать новые фичи only-in-tests | Compliance tests pass на real+fake; ≥10 L2 tests landed; новый код имеет 100% покрытие в `tests/` |
| **Phase 2 — Hermetic stack** | 3-4 нед | `tests/_harness/stack/`; sidecar fakes + Gauntlet orchestrator + ManualClock + `/control` API + Playwright | `make start-stack` <60с; smoke E2E green; первые L6 tests landed |
| **Phase 3 — Contract testing matrix** | 4-6 нед | Schemathesis в `tests/contract/`; JSON schemas для events/markers/journal; FE zod guards; OpenAPI drift gate; plugin manifest property tests; MSW для FE | Каждая cross-package boundary имеет gate; schema changes не могут merge silently |
| **Phase 4 — Property + Snapshots + остальные fakes** | 3-4 нед | Hypothesis в `tests/property/`; syrupy в `tests/golden/`; remaining canonical fakes (ITrainerSpawner, ISSHClient, IHFHubClient, IJobClient) | Property suite находит ≥5 latent bugs |
| **Phase 5 — Chaos + Load** | 6-8 нед | ChaosScenario framework; 18-scenario catalog; RunLoader; Playwright full E2E | Nightly chaos зелёный 14 дней подряд; load test ловит regression |
| **Phase 6 — Visual + Replay** | 3-4 нед | Storybook + lost-pixel; Playwright trace replay corpus; axe + Lighthouse | Visual diffs blocking; replay corpus ≥30 flows |
| **Phase 7 — Legacy decommissioning (optional)** | continuous | Legacy `packages/<pkg>/tests/` тесты постепенно удаляются по мере того, как новые покрывают функциональность; legacy lane может быть archived | Legacy lane либо archived либо пустой |

Phases 0-2 покрывают ~70% durable value за первые 2-3 месяца (быстрее чем при migration-подходе благодаря отсутствию legacy-touchа).

---

## Critical files to be modified / created

**Существующие, требующие минимального расширения:**
- [pytest.ini](pytest.ini) — НЕ трогаем, пусть live для legacy
- новый [tests/pytest.ini](tests/pytest.ini) — отдельный config для greenfield
- [Makefile](Makefile) — новые targets `test-new`, `test-all`, `start-stack`, `stop-stack`, `state-dump`, `lint-no-mocks-in-new-tests`, `regen-openapi`, `regen-zod`
- [packages/shared/src/ryotenkai_shared/infrastructure/mlflow/protocol.py](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/protocol.py) — стабилизировать Protocol
- [packages/shared/src/ryotenkai_shared/infrastructure/lifecycle/protocol.py](packages/shared/src/ryotenkai_shared/infrastructure/lifecycle/protocol.py) — то же
- [scripts/sync_openapi.py](scripts/sync_openapi.py) — добавить hash-check для drift gate
- [pyproject.toml](pyproject.toml) — добавить deps: hypothesis, schemathesis, syrupy, pytest-regressions, vcrpy
- [.pre-commit-config.yaml](.pre-commit-config.yaml) — добавить openapi-diff, json-schema validate

**Новые protocols extracted** (additive только, легаси-call sites не трогаем):
- `IRunPodAPI` — extract из RunPod GraphQL client
- `ITrainerSpawner` — extract из Supervisor
- `ISSHClient` — extract из SSH utils
- `IHFHubClient` — новый
- `IJobClient` — extract из существующего класса
- `Clock` — новый

**Новые директории/файлы (полностью greenfield):**
- `tests/_fakes/{mlflow,lifecycle,runpod,trainer,ssh,hf_hub,job_client,clock}.py`
- `tests/_harness/{wait,telemetry,debug_bundle,progress,chaos}.py`
- `tests/_harness/stack/{docker-compose.yaml,orchestrator.py,sidecars/{runpod,mlflow,vllm,hf_hub}_server.py}`
- `tests/_lint/test_no_protocol_mocking.py`
- `tests/{unit,component,contract,integration,e2e,stack,property,golden,chaos,load,visual,replay}/`
- `tests/conftest.py`, `tests/pytest.ini`, `tests/README.md`
- `.github/workflows/{lint,presubmit-fast,presubmit-blocking,legacy-tests,nightly,chaos-deep,load,release-gate,flake-quarantine,live-compliance}.yml`
- `web/src/api/zod.ts` (generated)
- `web/src/api/msw_handlers.ts` (generated)
- `docs/adrs/2026-05-10-greenfield-testing-architecture.md` — этот план как ADR

**Reusable patterns to leverage:**
- Существующие sentinel-тесты ([packages/control/tests/sentinel/test_no_pod_imports.py](packages/control/tests/sentinel/test_no_pod_imports.py)) — паттерн для нового `tests/_lint/test_no_protocol_mocking.py`
- [packages/engines/tests/contract/test_engine_protocol_parity.py](packages/engines/tests/contract/test_engine_protocol_parity.py) — паттерн для compliance test
- [packages/control/tests/contract/test_cli_api_parity.py](packages/control/tests/contract/test_cli_api_parity.py) — паттерн для CLI/API parity gate (переписать в `tests/contract/cli_api/`)
- importlinter (уже подключён) — для архитектурных правил

---

## Risks, open questions, и их разрешение

### R1 — Fake drift (fake расходится с real impl)
**Mitigation:** compliance test class параметризован по real+fake; nightly `RYOTENKAI_LIVE=1` lane против sandbox RunPod; quarterly cassette-diff vs real responses.

### R2 — Maintenance cost of fakes
**Mitigation:** fakes — first-class код, owned by Protocol's package, reviewed как production. Бюджет на поддержку учтён в Phase 7.

### R3 — Fake nondeterminism
**Mitigation:** ManualClock, deterministic UUID factory, seeded RNG, запрет thread-pools в fakes.

### R4 — Snapshot churn
**Mitigation:** scrubbers для timestamps/IDs/paths; regenerate-with-review workflow; сегментация snapshots.

### R5 — Hermetic-stack drift from production
**Mitigation:** nightly `RYOTENKAI_LIVE=1` chaos run; cassettes diffed quarterly; auto-flag re-record после 90 дней.

### R6 — L6/L9 flake creep
**Mitigation:** quarantine SLA 30 дней, no rerunfailures, debug bundles + telemetry, Eventually вместо sleep.

### R7 — CI cost (chaos+load)
**Mitigation:** nightly scheduling; self-hosted runner если GH Actions free-tier лимиты; budget alarm.

### R8 — Legacy code теряет покрытие со временем
**Открытый вопрос:** при рефакторинге легаси-кода под extracted Protocols, мы можем сломать существующий functional behavior, не покрытый легаси-тестами (которые мы не трогаем). **Ответ:** Phase 1 делает только additive Protocol extraction — legacy-call sites не трогаются, новые Protocol-классы — параллельные. Legacy-code продолжает работать с оригинальной зависимостью; новый code use Protocol+Fake. Когда legacy-код переписывается под Protocol — обязательное требование 100% покрытие в `tests/` ДО удаления legacy-теста.

### R9 — Property-test combinatorial explosion
**Mitigation:** `--hypothesis-profile=ci` bounded examples (~50); `--hypothesis-profile=nightly` extended (~5000); falsifying examples checked-in.

### R10 — Protocol explosion
**Mitigation:** Protocol только на TRUE seams — external system, package boundary. Список ограничен ~10-15 штук.

### R11 — Cassette rot
**Mitigation:** cassette age tracked; auto-flag re-record после 90 дней; quarterly diff job.

### R12 — Determinism vs realism trade-off
**Mitigation:** TimedClock variant с compressed ratios; test-author выбирает Clock тип per test.

### R13 — Greenfield двойная инфраструктура
**Открытый вопрос:** у нас будут два набора тестов — legacy и new. Не приведёт ли это к путанице, дублированию работы, забытым багам в legacy? **Ответ:** legacy lane non-blocking warning — он либо ловит регрессию (хорошо), либо ничего не делает (нейтрально), но никогда не блокирует. Дублирование не нужно — новый функционал тестируется только в `tests/`. Legacy `packages/<pkg>/tests/` обслуживает только тот код, который не трогали с Phase 0.

### R14 — ML pipeline determinism limits
**Mitigation:** tolerance bands; FakeTrainerSpawner эмитит детерминированные events не зависящие от ML libs.

### R15 — Hermetic stack as bottleneck
**Mitigation:** каждый L6+ тест поднимает свой ephemeral instance стека (port allocator ensures isolation). docker-compose только для dev workflow.

### R16 — Schema drift gate too rigid
**Mitigation:** pre-commit hook auto-regen; CI gate fail только если разработчик обошёл pre-commit.

### R17 — Plugin sandbox security
**Mitigation:** Phase 5 chaos catalog добавит `MaliciousPluginEscape`; adversarial plugin fixture в `community/tests/adversarial/`.

### R18 — Replay corpus rot
**Mitigation:** Replay corpus regeneration на major UI versions с diff review; staleness budget 90 дней.

### R19 — Существующий code не testable без extraction Protocol
**Открытый вопрос:** если в legacy-code есть функции, которые делают direct call к RunPod без Protocol-абстракции, как мы их тестируем в `tests/`? **Ответ:** два варианта — (a) extract Protocol additively без изменения call sites (legacy продолжает работать, новый Path использует Protocol); (b) если код стабилен и не меняется — оставляем legacy-тесты как есть, в `tests/` не дублируем. ADR требует, что любая новая фича + любой существенный refactor проходит через Protocol.

### R20 — Legacy lane никогда не отключают
**Открытый вопрос:** есть риск, что legacy lane живёт вечно, даже когда покрытие переехало в `tests/`. **Ответ:** в Phase 7 (continuous) проводится annual review legacy-lane: если тест в legacy-lane не падал >12 месяцев и его функционал покрыт в `tests/`, тест удаляется. Цель к Phase 7+12мес — legacy lane либо archived либо пустой.

---

## Verification (как проверить, что план работает)

### После Phase 0 (Foundations)
```bash
# Lint должен ловить mock(Protocol) ТОЛЬКО в tests/
git checkout -b test-flag-mock-protocol
echo "from unittest.mock import patch
@patch('ryotenkai_shared.infrastructure.mlflow.protocol.IMLflowManager')
def test_dummy(): pass" > tests/unit/test_smoke.py
make lint-no-mocks-in-new-tests  # должен упасть

# Тот же файл в legacy директории — допустим
echo "..." > packages/control/tests/test_smoke.py
make lint-no-mocks-in-new-tests  # должен пройти (не сканирует packages/)

# Telemetry plugin работает
pytest tests/unit/  # должен сгенерить tests/.telemetry/run-*.jsonl

# Eventually helper
python -c "from tests._harness.wait import Eventually; Eventually(lambda: False, timeout=1)"
```

### После Phase 1 (Top-3 Fakes)
```bash
# Compliance test parameterized
RYOTENKAI_LIVE=0 pytest tests/contract/protocol_compliance/test_lifecycle_compliance.py -v
RYOTENKAI_LIVE=1 pytest tests/contract/protocol_compliance/test_lifecycle_compliance.py -v --runxfail

# Новый тест нового функционала
pytest tests/component/test_my_new_feature.py  # использует FakeRunPodAPI
```

### После Phase 2 (Hermetic stack)
```bash
make start-stack  # <60с
curl http://localhost:18080/health
curl http://localhost:18081/control/inject_429?count=3
pytest tests/stack/test_smoke_e2e.py
make state-dump > /tmp/dump.json && jq . /tmp/dump.json
make stop-stack
```

### После Phase 3 (Contract matrix)
```bash
# OpenAPI drift gate
sed -i 's/"version": "0.1.0"/"version": "0.2.0"/' packages/control/src/.../api/main.py
make verify-api-sync  # должно упасть
make regen-openapi regen-zod
make verify-api-sync  # должно пройти

# Schemathesis stateful
pytest tests/contract/openapi_drift/ --hypothesis-show-statistics

# Plugin manifest property
pytest tests/contract/plugin_manifest/ --hypothesis-seed=0
```

### После Phase 4 (Property + Snapshots)
```bash
pytest tests/property/ --hypothesis-profile=nightly
pytest tests/golden/ --snapshot-update
git diff --stat
```

### После Phase 5 (Chaos + Load)
```bash
pytest tests/chaos/ --chaos-seed=42  # 18/18 sceanrios
python tests/load/runloader.py --concurrency=10 --duration=10m
```

### После Phase 6 (Visual + Replay)
```bash
pnpm storybook:test  # baselines
pytest tests/replay/test_user_flow_create_run.py
```

### Continuous верификация
- Burn-down dashboard показывает рост `tests/` coverage
- Flake board <5 entries, no test >30 days в quarantine
- Legacy lane stability — если падает много раз без причины, кандидат на archival
- Telemetry SQLite query: `SELECT layer, COUNT(*), AVG(duration_ms) FROM tests GROUP BY layer` — ratio близок к target 70/15/10/5 в `tests/`

---

## Top 3 архитектурных решения (зафиксированы пользователем)

1. **Greenfield testing-инфраструктура в `tests/`, легаси `packages/<pkg>/tests/` не трогаем.**
   *Стоимость:* короче — нет миграции 2159 mocks; параллельная инфраструктура.
   *Польза:* фокус сразу на правильной архитектуре; Phase 0-2 даёт результат за 2-3 месяца вместо 4.
   ✅ Зафиксировано пользователем.

2. **Canonical Fakes per Protocol с lint-сентинелем для `tests/`.**
   *Стоимость:* ~6 fakes в Phase 1-4 + ongoing maintenance.
   *Польза:* единственный механизм, ограничивающий long-term test debt при росте системы.
   ✅ Зафиксировано пользователем (full migration в новой директории).

3. **Hybrid Hermetic Stack — один Fake-класс работает in-process и как Sidecar HTTP.**
   *Стоимость:* ~30с overhead на L6 тесты + `/control` API на каждом fake.
   *Польза:* единственный способ честно тестировать transport, circuit breakers, retries, WS reconnect.
   ✅ Зафиксировано пользователем.

**Visual regression** отложен в Phase 6.
**Timeline:** 6-9 месяцев фокусированно (~70% allocation одного инженера). Phase 0-2 даёт ~70% durable value за первые 2-3 месяца. Сокращение с 9-12 → 6-9 благодаря отсутствию миграции legacy mocks.

**Cross-package contracts как first-class versioned artifacts с PR-blocking drift gates** (Decision 3) подразумевается выбором greenfield + canonical fakes — без отдельного approval.
