# ADR: Legacy `packages/<pkg>/tests/` decommissioning policy

Date: 2026-05-11
Статус: принято

> **Контекст:** Phase 7 в [structured-hopping-starfish.md](../plans/structured-hopping-starfish.md)
> объявлен "continuous". Этот ADR фиксирует процесс — кто, когда и как
> удаляет легаси-тесты по мере того, как покрытие переезжает в
> `tests/` (greenfield).

## Цель

Снизить total cost of ownership testing-инфраструктуры. Легаси-лейн
сегодня — ~933 теста с ~2159 вызовами `unittest.mock.patch`; стоимость
поддержки = 5-10% времени каждого refactor PR. Greenfield (`tests/`)
к концу Phase 5 покрывает большую часть critical paths. Phase 6 даёт
visual / replay / a11y / lighthouse. Время начинать удалять.

## Решение

Принимаем три механизма (a) PR-trigger, (b) quarterly audit, (c)
hard exit.

### (a) PR-trigger — удаление в момент изменения

**Правило:** если PR трогает файл `packages/<pkg>/src/F.py` и при этом
greenfield-покрытие этого файла в `tests/` адекватно (acceptance criteria
ниже), легаси-тесты для `F.py` в `packages/<pkg>/tests/test_F.py` (или
аналогичных) удаляются **в том же PR**.

**Acceptance criteria для "адекватного" покрытия в `tests/`:**

1. По коду пакета `F.py` есть как минимум один L2 (component) тест в
   `tests/component/` или один L4 (integration) тест в
   `tests/integration/`.
2. Сценарии, покрытые легаси-тестами, явно перечислены в коммит-сообщении
   (или PR description) с маппингом `packages/.../test_X.py::test_foo`
   → `tests/.../test_Y.py::test_bar`. Не obligатory один-к-одному; один
   tests/ может покрывать N легаси.
3. Greenfield-покрытие включает property/golden там, где это уместно
   (Phase 4 уже дал инструменты), а не дублирует структурный mock
   через подмену.

**Что НЕ удаляем по PR-trigger:**

- Тесты, которые проверяют публичные API из API-стабильного пакета и не
  имеют greenfield-аналога — оставляем до отдельного спецификационного
  PR.
- Тесты на устаревший legacy-flow, который ещё используется (например,
  старый CLI), — флагируем `@pytest.mark.legacy_only` и оставляем до
  явного удаления самого flow.

**Метрика prozеса:**

- Total count тестов в `packages/<pkg>/tests/` падает **monotonically**
  per quarter — никаких прироста. Increase допустим только в течение
  квартала и должен быть откатан до конца квартала.

### (b) Quarterly audit — proactive prune

Раз в квартал (1 марта, 1 июня, 1 сентября, 1 декабря) проводится
audit легаси-лейна. Запуск:

```bash
make legacy-audit
```

`make legacy-audit` обёртывает [scripts/legacy_test_audit.py](../../scripts/legacy_test_audit.py),
который печатает:

- total count тестов в `packages/<pkg>/tests/` (suite-level и per-package);
- сколько из них **не падали 12 месяцев** подряд (определяем по
  `tests/.telemetry/` + git log mtime + flake-board snapshot);
- сколько из них **flake-quarantined** за последние 90 дней (потенциально
  ненадёжные, кандидаты на удаление как noise);
- сколько имеют greenfield-двойника (heuristic: ищем тест с похожим
  именем в `tests/`).

**Действие audit:** инженер-owner лейна (по умолчанию — последний
`git blame` коммитер) делает PR `legacy-audit-<YYYY-Q>` с удалениями
по правилам (a). Audit не блокирует merge ничего другого — это
operational housekeeping.

### (c) Hard exit

Легаси-лейн **archived** (rm -rf `packages/*/tests/`, удаление
`legacy-tests.yml` workflow) при выполнении любого из:

1. **Quantity floor:** total count тестов в `packages/<pkg>/tests/`
   опустился ниже **100**. На таком масштабе цена параллельной
   инфраструктуры превышает её ценность.
2. **Stability ceiling:** 12-month rolling stability легаси-лейна =
   **100%** (ни одного нового failure не нашлось — лейн перестал ловить
   баги). На таком уровне он эквивалентен no-op.

Hard exit оформляется отдельным ADR `legacy-lane-archived`, который
содержит final stats и список того, что было удалено.

## Mapping мapping mapping — как тегировать миграцию

В commit-сообщении PR, удаляющего легаси-тесты, обязательная секция:

```
Legacy test decommissioning:
- packages/control/tests/pipeline/test_orchestrator.py::test_resume_flow
  -> tests/component/test_pipeline_orchestrator.py::test_resume_with_journal
- packages/control/tests/pipeline/test_orchestrator.py::test_stage_failure_propagates
  -> tests/integration/test_stage_failure_journal.py::test_failure_marker_persisted
```

Это нужно для будущего audit + для ретроспективной uncomfortable
exercise "а что мы потеряли?".

## Open вопросы / risks

- **R1.** Greenfield-покрытие может пропустить sneaky test side
  effect. Mitigation: PR-reviewer обязан проверить, что новые `tests/`
  закрывают **тот же сценарий** (acceptance, не структура).
- **R2.** Legacy lane может стать "permanently green" просто потому,
  что код мутировал и тесты больше не покрывают meaningful flow. Это
  именно то, что hard-exit ceiling ловит — но не быстро. Mitigation:
  quarterly audit явно отмечает "no failure in 12 months" как
  кандидат на удаление.
- **R3.** Annual review legacy-lane не обязательно совпадает с
  PR-trigger удалениями. Mitigation: оба пути ведут к одной цели —
  снижению count. quarterly audit — backstop для PR-trigger пропусков.

## Не-цели

- **NE.** Этот ADR не покрывает миграцию production-кода — только тесты.
- **NE.** Не делаем bulk-rm `packages/<pkg>/tests/` без сопоставления
  с greenfield. Бэкап нужен, но в рамках обычной git-истории, не
  отдельного снэпшота.

## Связанные документы

- План: [docs/plans/structured-hopping-starfish.md](../plans/structured-hopping-starfish.md), Phase 7.
- Архитектура: [docs/adrs/2026-05-10-greenfield-testing-architecture.md](2026-05-10-greenfield-testing-architecture.md).
- Скрипт audit: [scripts/legacy_test_audit.py](../../scripts/legacy_test_audit.py).
