# `tests/` — greenfield testing-инфраструктура RyotenkAI

> **План:** [docs/plans/structured-hopping-starfish.md](../docs/plans/structured-hopping-starfish.md)
> **ADR:** [docs/adrs/2026-05-10-greenfield-testing-architecture.md](../docs/adrs/2026-05-10-greenfield-testing-architecture.md)

## Зачем эта директория

Старая testing-инфраструктура в `packages/<pkg>/tests/` содержит ~380 Python-тестов и 2159 вызовов `unittest.mock.patch`. При рефакторинге mock-цепочки ломаются массово, давая ложноположительные регрессии. Гreenfield-подход (зафиксирован пользователем): новая инфраструктура строится в этой директории, **legacy не трогается**. Старые тесты постепенно удаляются по мере того, как функциональность покрывается здесь.

**Ключевые правила:**

1. **Никакого `unittest.mock` для Protocols.** Линт-сентинель (`tests/_lint/test_no_protocol_mocking.py`) валит CI при попытке.
2. **Только канонические fakes** из `tests/_fakes/` (Phase 1+).
3. **Импорты только production-кода** (`from ryotenkai_control...`); никаких legacy-fixture.
4. **Eventually вместо sleep.** Любой L4+ опрос — через `tests/_harness/wait.py`.
5. **Время через `Clock` Protocol.** `RealClock` в проде, `ManualClock` в тестах.

## Слойный пирог (12 уровней)

| L | Цель | Где живёт | Бюджет | PR-blocking? |
|---|------|-----------|--------|--------------|
| L0 Static gates | ruff, mypy, importlinter, sentinel-AST | top-level | <60с | Да |
| L1 Unit | чистая логика | `tests/unit/` | <2с/тест | Да |
| L2 Component | один класс + fakes | `tests/component/` | <500мс/тест | Да |
| L3 Contract | boundary contracts | `tests/contract/` | <30с total | Да |
| L4 Integration | multi-component | `tests/integration/` | <5с/тест | Да (subset) |
| L5 Subsystem e2e | runner/control/web e2e | `tests/e2e/` | <30с/тест | Smoke на PR |
| L6 Hermetic stack | full stack via `make start-stack` | `tests/stack/` | <2 мин/тест | Smoke на PR |
| L7 Property | Hypothesis | `tests/property/` | nightly | Nightly |
| L8 Golden | snapshots | `tests/golden/` | <5с/тест | Да |
| L9 Chaos | fault injection | `tests/chaos/` | nightly | Нет |
| L10 Load | RunLoader | `tests/load/` | weekly | Нет |
| L11 Visual | Storybook + lost-pixel | `tests/visual/` | <5 мин на PR | Phase 6+ |
| L12 Replay | recorded user flows | `tests/replay/` | release-gate | Release |

Целевое распределение: ~70% L1-L2, ~15% L3-L4, ~10% L5-L6, ~5% L7-L12.

## Как написать новый тест

### L1 (unit) — pure logic

```python
# tests/unit/test_my_feature.py
import pytest

def test_pure_function() -> None:
    assert my_pure_function(1, 2) == 3
```

### L2 (component) — class + fakes

```python
# tests/component/test_pipeline_orchestrator.py
import pytest

from tests._fakes.mlflow import FakeMLflowManager  # Phase 1+

@pytest.mark.uses_fake("FakeMLflowManager")
@pytest.mark.exercises_protocol("IMLflowManager")
def test_orchestrator_starts_run(manual_clock) -> None:
    fake = FakeMLflowManager(clock=manual_clock)
    sut = PipelineOrchestrator(mlflow=fake, clock=manual_clock)
    sut.start_run("exp-1")
    assert fake.runs_for("exp-1") == 1
```

### L3 (contract) — protocol compliance

```python
# tests/contract/protocol_compliance/test_lifecycle_compliance.py
import pytest

@pytest.fixture(params=["fake", "real"])
def lifecycle_client(request):
    if request.param == "real" and os.environ.get("RYOTENKAI_LIVE") != "1":
        pytest.skip("real impl gated by RYOTENKAI_LIVE=1")
    return _build(request.param)

class TestIPodLifecycleClientCompliance:
    def test_terminate_idempotent(self, lifecycle_client) -> None:
        ...
```

### L4 (integration) — multi-component

```python
# tests/integration/test_run_lifecycle.py
import pytest
from tests._harness.wait import Eventually

@pytest.mark.asyncio
async def test_run_completes(integration_stack):
    run = await integration_stack.start_run(...)
    await Eventually(
        lambda: integration_stack.run_status(run.id) == "succeeded",
        timeout=10.0,
        clock=integration_stack.clock,
    )
```

## Куда **нельзя** добавлять тесты

- `packages/<pkg>/tests/` — read-only после Phase 0. Любой PR, добавляющий туда тест, должен быть отклонён.
- В этой директории не размещаются legacy-fixture или mock-helpers — таксономия запрещает Mock над Protocols.

## Как запускать

```bash
make test-new                      # вся новая lane
pytest tests/unit/                 # L1 only
pytest tests/component/            # L2 only
pytest tests/contract/             # L3 only
pytest -m "uses_fake(FakeMLflowManager)" tests/   # выборка по маркеру
make lint-no-mocks-in-new-tests    # AST-сентинель
make test-all                      # legacy + new в одной команде
```

Per-test telemetry пишется в `tests/.telemetry/run-<utc>.jsonl`. Debug-bundles
для упавших L5+ — в `tests/.debug_bundles/`. Оба директория в `.gitignore`.

## Fake/mock taxonomy

| Тип | Допустим в `tests/`? | Когда уместен |
|-----|----------------------|---------------|
| **Stub** | да | возвращает константу для pure-function зависимостей |
| **Fake** | да (обязательно для Protocol) | in-memory state machine; первоклассный код |
| **Spy** | да (редко) | fake, обёрнутый записью вызовов |
| **Mock** | **нет** для Protocol; sentinel ловит | только не-Protocol зависимости |

Канонические fakes живут в `tests/_fakes/<protocol_name>.py`, naming
`Fake<ProtocolMinusI>`. Каждый fake имеет parametrized compliance test в
`tests/contract/protocol_compliance/test_<protocol>_compliance.py`,
который запускается и против real impl (с `RYOTENKAI_LIVE=1`), и против fake.
Тест против только fake — запрещён.

## L7-L12 поддиректории (Phase 4-6)

### L7 Property (`tests/property/`, Phase 4)

Hypothesis-based property tests. 18 properties за ~0.8с на полном запуске
(`pytest tests/property/`). Каждый файл задаёт `@given(...)` и утверждает
инвариант над случайно сгенерированным входом. Profile `nightly`
(10× shrinks) — в ночном CI, по умолчанию профиль `dev`. Counter-examples
фиксируются под `.hypothesis/examples/` и коммитятся вместе с тестом —
следующий запуск повторно проверит ту же patологию.

### L8 Golden (`tests/golden/`, Phase 4)

Snapshot-тесты на `syrupy`. 4 базовых золотых: журнал событий, marker
file, OpenAPI sample, JSON schema rendered output. Регенерация снэпшотов —
`make regen-snapshots`; ручной diff виден в PR.

### L9 Chaos (`tests/chaos/`, Phase 5)

`ChaosScenario` framework + catalog из 18 сценариев (14 passing, 4 partial).
Каждый сценарий — сидованная деривация (`pytest --chaos-seed=42`),
воспроизводящая конкретный fault-injection профиль: RunPod 429 storm,
journal rotation race, stale SSH ControlMaster и т.д. Catalog лежит в
`tests/chaos/scenarios/`.

### L10 Load (`tests/load/`, Phase 5)

`RunLoader` skeleton — параметризованный нагрузочный тест, поднимает
N concurrent runs и измеряет деградацию метрик. Бюджет — weekly CI lane
(`load.yml`).

### L11 Visual (`tests/visual/` + `web/.storybook/`, Phase 6)

Visual regression через Storybook + lost-pixel. Stories живут
колоцированно (`web/src/components/*.stories.tsx`), пять
репрезентативных компонентов покрыты как infrastructure-baseline:
StatusPill, RunRow, DeleteProjectModal, StalePluginsBanner, Icons demo.

CI workflow `visual-regression.yml` — **non-blocking warning** первые
6 месяцев per Decision 7. Маски волатильных регионов в
[web/lost-pixel.config.ts](../web/lost-pixel.config.ts).

### L12 Replay (`tests/replay/`, Phase 6)

Корпус Playwright traces, записанный из E2E-сценариев в
`web/e2e/*.spec.ts`. Каждая запись хранится в `tests/replay/corpus/`
как `.zip` и проверяется на интегритет в каждом запуске
([test_replay_corpus.py](replay/test_replay_corpus.py)).

Полный replay против текущего билда — `RYOTENKAI_REPLAY_FULL=1`
release-gate workflow only; для daily-loop достаточно базовой проверки
"трейс — валидный zip с `*.trace`".

### A11y (Phase 6)

`web/e2e/a11y.spec.ts` — Playwright + axe-core против 3 ключевых
страниц, hard-gate на critical/serious violations. Компонент-уровень —
пример в [web/src/components/__tests__/a11y-example.test.tsx](../web/src/components/__tests__/a11y-example.test.tsx).

### Lighthouse (Phase 6)

Бюджеты — `accessibility ≥ 0.95` (hard error), остальные категории
warning. Lighthouse гоняет против static Storybook build (proxy для
SPA performance). Запуск — `npm run lighthouse` в `web/`.

## Phase 7 — legacy decommissioning

`packages/<pkg>/tests/` декомиссионится по политике из
[docs/adrs/2026-05-11-legacy-test-decommissioning.md](../docs/adrs/2026-05-11-legacy-test-decommissioning.md).
Quarterly audit — `make legacy-audit` (read-only stats).
