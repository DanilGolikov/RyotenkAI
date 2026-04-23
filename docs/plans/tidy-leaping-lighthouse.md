# План: убрать `priority` из плагинов + перевести report-order на config

> **Выполнено (2026-04-23).** Прошли 3 фазы удаления `priority` +
> финальный 4-й аккорд: `[reports] order` из манифеста тоже ушёл,
> порядок секций отчёта теперь берётся из `PipelineConfig.reports.sections`
> (single source of truth). Ниже — архив плана на момент старта.



## Context

Пользователь усомнился в ценности поля `plugin.priority` в манифестах и попросил аудит. Результат:

- **Validation и evaluation** плагины сортируются по `priority` перед выполнением (`dataset_validator.py:195`, `evaluation/runner.py:164`). На **семантику** это не влияет — проверки/метрики независимы и в пайплайн собираются в dict. Единственный эффект — порядок строк в логах.
- **Reward** плагины `priority` **игнорируют** — поле есть в манифесте, но код его не читает.
- **Reports** плагины используют `priority` как **осмысленный `order`** — это номер секции в финальном Markdown-отчёте (`registry.py:35` — `plugin_cls.order = manifest.plugin.priority`). Тут он по-настоящему нужен: `header=10`, `summary=20`, … `footer=120`.

**Вывод:** поле полу-мёртвое. Для validation/evaluation/reward оно — cargo-cult числа, которые авторам плагинов приходится подбирать без реальной отдачи. Для reports — сильная, уникальная и значимая настройка, но называется чужим именем (`priority` вместо `order`).

**Что хотим:**

1. Убрать `priority` из `PluginSpec` (общий манифест).
2. Для reports — ввести явный блок `[reports]` с обязательным уникальным `order: int`. Это семантическая честность: поле живёт там, где оно реально работает.
3. Снять `plugins.sort(key=.priority)` в validation/evaluation — пусть порядок определяется конфигом пользователя (list в YAML сохраняет порядок вставки в pydantic).

**Почему сейчас удобный момент:**

- Third-party авторов плагинов пока **нет**, community мы контролируем полностью — ломаем только собственный код и тесты.
- Совсем недавно завершили community-контракт v2 и CLI-модернизацию — текущее окно пригодно для ещё одного низкорискового рефакторинга в том же слое.

## Что НЕ входит в скоуп

- **Config-YAML валидаторов/evaluation** сейчас принимает `list[...]` (order-preserving), шейп трогать не нужно.
- **reward** плагинов в community всего один (`helixql_compiler_semantic`) — там `priority=50` просто сносим, никакой логики порядка нет.
- **Фронт/OpenAPI** — поле `priority` сейчас в `ui_manifest()` эмитится без потребителя на клиенте; удаляем вместе со схемой, TS-типы регенерируются автоматически.

---

## Архитектура решения

### Манифест (pydantic)

**Было:**
```toml
[plugin]
id = "min_samples"
kind = "validation"
priority = 10      # ← магическое число, бизнес-эффекта нет
```

**Стало:**
```toml
[plugin]
id = "min_samples"
kind = "validation"
# priority удалено
```

Для reports добавляем явный опциональный блок:

```toml
[plugin]
id = "header"
kind = "reports"

[reports]
order = 10     # обязательное поле, уникальное в пределах всех report-плагинов
```

Модели в `src/community/manifest.py`:

```python
class PluginSpec(BaseModel):
    id: str
    kind: PluginKind
    name: str = ""
    version: str = "1.0.0"
    # priority: int = 50          ← удаляется
    category: str = ""
    stability: Stability = "stable"
    description: str = ""
    entry_point: EntryPoint

class ReportsSpec(BaseModel):
    """Report-plugin-specific metadata (section ordering)."""
    model_config = ConfigDict(extra="forbid")
    order: int

class PluginManifest(BaseModel):
    plugin: PluginSpec
    params_schema: dict[str, Any] = Field(default_factory=dict)
    # ...
    reports: ReportsSpec | None = None    # ← новое, required if kind == "reports"

    @model_validator(mode="after")
    def _reports_block_required_for_kind(self) -> PluginManifest:
        if self.plugin.kind == "reports" and self.reports is None:
            raise ValueError("[reports] block is required when plugin.kind == 'reports'")
        if self.plugin.kind != "reports" and self.reports is not None:
            raise ValueError("[reports] block is only valid for plugin.kind == 'reports'")
        return self
```

### Runtime

1. **`src/utils/plugin_base.py`** — удалить `priority: ClassVar[int] = 50`.
2. **`src/community/loader.py:158`** — удалить `plugin_cls.priority = manifest.plugin.priority`.
3. **`src/pipeline/stages/dataset_validator.py:195`** — удалить `plugins.sort(key=...priority)`. `plugins` и так build-ится итерацией по `config.plugins` (order-preserving list); default-плагины в `_get_default_plugins()` возвращаются в детерминированном порядке из ручного списка.
4. **`src/evaluation/runner.py:164`** — заменить `sorted(plugins, key=...)` на обычную итерацию. Тот же механизм: build по `config.plugins`.
5. **`src/reports/plugins/registry.py:35`** — `plugin_cls.order = loaded.manifest.reports.order` (было `.plugin.priority`). Uniqueness-check остаётся.

### API / OpenAPI

- **`src/community/manifest.py::ui_manifest()`** — убрать `"priority": self.plugin.priority` для non-report plugins; для reports добавить `"order": self.reports.order` (только для `kind == "reports"`).
- `web/src/api/schema.d.ts` и `web/src/api/openapi.json` регенерируются из pydantic автоматически (процесс уже налажен в проекте).

### CLI-тулзы (scaffold / sync / toml_writer)

- `src/community/toml_writer.py::FIELD_ORDER` — убрать `priority` из списка порядка полей `[plugin]`-секции; добавить секцию `[reports]` в очередь рендера.
- `src/community/scaffold.py::build_plugin_manifest_dict()` — убрать `"priority": 50`; при `inferred.kind == "reports"` добавлять `"reports": {"order": 50}` (50 — дефолт-плейсхолдер с `# TODO`-комментарием через TOML-писатель).
- `src/community/sync.py::_merge_plugin_manifest()` — выкинуть `priority` из user-owned scalar fields; добавить merge для `[reports]` блока (полностью user-owned).

### Community-манифесты (миграция)

- **14 validation / 4 evaluation / 1 reward** манифестов: убрать строку `priority = XX`.
- **14 reports** манифестов: удалить `priority = XX` из `[plugin]`, добавить блок `[reports]` с `order = XX` (значения переносим 1:1 из текущих priority чтобы не перетасовать отчёт).
- Бампнуть `version` (patch) у всех 33-х — содержимое манифеста поменялось.

---

## Критические файлы

| Файл | Действие |
|---|---|
| `src/community/manifest.py` | убрать `priority`, добавить `ReportsSpec` + cross-field validator |
| `src/utils/plugin_base.py` | убрать `priority: ClassVar[int]` |
| `src/community/loader.py` | убрать `plugin_cls.priority = ...` (строка 158) |
| `src/pipeline/stages/dataset_validator.py` | убрать `plugins.sort(key=...priority)` (строка 195) |
| `src/evaluation/runner.py` | убрать `sorted(..., key=priority)` (строка 164) |
| `src/reports/plugins/registry.py` | `plugin_cls.order = manifest.reports.order` |
| `src/community/toml_writer.py` | скорректировать FIELD_ORDER + добавить рендер `[reports]` |
| `src/community/scaffold.py` | не писать `priority`; для kind=reports писать `[reports]` |
| `src/community/sync.py` | миграция merge-логики |
| `community/validation/*/manifest.toml` (14) | удалить строку `priority` |
| `community/evaluation/*/manifest.toml` (4) | удалить строку `priority` |
| `community/reward/helixql_compiler_semantic/manifest.toml` | удалить строку `priority` |
| `community/reports/*/manifest.toml` (14) | `priority` → `[reports] order` |
| `src/tests/unit/community/test_loader.py` | снять assert `cls.priority == 42`; добавить assert про `order` для reports |
| `src/tests/unit/community/test_toml_writer.py` | обновить ожидания по порядку полей |
| `src/tests/unit/community/test_scaffold.py` | обновить ожидания — `[reports]` вместо `priority` в reports-кейсе |
| `src/tests/unit/community/test_sync.py` | обновить merge-тесты |
| `src/tests/unit/pipeline/test_stages_validator.py` | убрать mock `priority=1` в строках 53, 72 |
| `web/src/api/openapi.json` + `schema.d.ts` | регенерировать |

**Новых зависимостей нет.** Реиспользуем: pydantic `model_validator`, существующая `_validate_plugin_instances` в `registry.py`, TOML-writer API с `todo_fields=`.

---

## Фазы

Один PR, три атомарных коммита для bisect-friendliness:

1. **`refactor(community): introduce [reports] block with order`** — манифест-модель + миграция 14 report-манифестов + registry.py читает из нового места. Validation/evaluation ещё сортируют по старому. Всё зелёное.
2. **`refactor(community): drop priority from Base/PluginSpec + sort calls`** — удаляем `priority` из `PluginSpec`, `BasePlugin`, loader, двух sort-вызовов; удаляем `priority = XX` из 19 не-report манифестов; правим тесты.
3. **`chore(cli): scaffold/sync/toml_writer drop priority, support [reports]`** — обновляем writer + scaffold + sync, чтобы новые плагины не получали `priority` в scaffold-e, а report-плагины получали `[reports]` блок с TODO-комментарием.

Каждый коммит сам по себе не ломает тесты (зелёный — стейблный).

---

## Verification

```bash
# После каждой фазы:
pytest src/tests/unit/community/ src/tests/unit/reports/ src/tests/unit/pipeline/test_stages_validator.py src/tests/integration/api/test_delete_and_config.py -q
ruff check src/ community/
ryotenkai community sync community --dry-run   # diff должен быть только про version bump
ryotenkai report <run>                          # секции в том же порядке, что были (sanity-check на e2e)

# Ручная проверка API-контракта:
curl -s http://localhost:8000/api/v1/plugins/validation | jq '.[] | keys'     # нет "priority"
curl -s http://localhost:8000/api/v1/plugins/reports | jq '.[] | .order'      # есть order, уникальный
```

Критерий успеха:

- Все тесты зелёные (161+ текущие).
- `ryotenkai report` даёт идентичную Markdown-композицию до/после.
- OpenAPI-схема не содержит `priority` для validation/evaluation/reward; для reports есть `order`.
- 0 упоминаний `priority` в community/**/manifest.toml кроме, возможно, `secrets.required` (другое поле) — проверяется `grep -r "^priority" community/`.

---

# Risk & question analysis (3 итерации)

## Итерация 1 — обвиозные риски

**R1. Изменение порядка выполнения валидации/evaluation ломает чей-то пайплайн.**
Сейчас validation-плагины запускаются в порядке `priority asc`, пользователь может быть незаметно на это завязан (например, `min_samples` (priority=10) валится первым и отсекает остальные). После снятия сорта порядок = order-of-appearance в `config.yaml`, и если пользователь не указывал `plugins: [...]` явно, применяются default-плагины в порядке `_get_default_plugins()`.

**R2. Совместимость API — клиент упадёт, если `priority` резко пропадёт.**
Поле в `ui_manifest()` → в `/plugins/{kind}`. Клиент (TypeScript) ожидает `priority: number` (default 50) — если пропадёт, typed код сломается на compile-time.

**R3. Sync-diff взорвётся при первом прогоне.**
`toml_writer.FIELD_ORDER` влияет на порядок ключей в рендере. Изменение FIELD_ORDER вызывает перегенерацию всех 30+ manifest.toml со смещёнными строками — шум в диффе, возможны merge-конфликты если параллельно идут другие ветки.

**R4. Uniqueness-check order в reports слетит при миграции.**
Если копируем priority→order один-к-одному, текущие значения должны быть уникальны. Если где-то duplicate — сейчас код тоже бы падал (check уже есть). Но: авторы раньше могли случайно продублировать priority (до введения check), а check добавили позже → возможны latent-дубли.

## Итерация 2 — вторичные, неочевидные

**R5. `_get_default_plugins()` полагается на сорт.**
`dataset_validator.py` имеет default-список плагинов которые добавляются если config не указал плагины. Сейчас они попадают в `plugins.sort(.priority)` и мешаются. Без сорта их порядок = порядок ручного списка в `_get_default_plugins`. Если мы не пересмотрели этот ручной список с «логичным» порядком — может получиться кривая картина.

**R6. 3rd-party плагин, встроенный через community-zip, ожидает поле `priority`.**
Сейчас third-party нет — но если кто-то уже скачал zip и положил у себя, `PluginSpec` с `extra="forbid"` сломается на неизвестном поле `priority` после апгрейда. Миграционный gotcha.

**R7. `BasePlugin.priority: ClassVar[int] = 50` могут читать конкретные плагины внутри своей логики.**
Не сортировка, а, скажем, логгинг. Например, `logger.info(f"Running {self.name} priority={self.priority}")`. Грепом надо проверить все плагин-файлы.

**R8. Report-плагины, не зарегистрированные через community-catalog.**
Registry имеет метод `register_from_community`, но теоретически можно было регистрировать вручную (`ReportPluginRegistry._registry[...] = cls`). Если такие пути существуют — они читали `.order` напрямую как ClassVar, а мы меняем источник на манифест.

**R9. MLflow теги / логи.**
Может быть где-то `mlflow.log_param("priority", plugin.priority)` — поиск грепом.

## Итерация 3 — «документация и UX»

**R10. README в `community/validation/*`, `community/evaluation/*`, `community/reports/*`.**
В них наверняка прописано «add `priority = X` to your manifest». Надо пройтись.

**R11. Плагины проекта прописанные как отдельный раздел в документации — например `docs/plugins/*.md` или `CONTRIBUTING.md`.**
Аналогично — стоит прочесать.

**R12. `docs/plans/*.md` — старый план про CLI-модернизацию** — потерял актуальность после этого рефакторинга (не блокирует, но техдолг в плане-файле).

**R13. Web UI может отображать priority в plugin settings.**
Если sidebar показывает «Priority: 10» — после апгрейда поле пустое для non-reports. Нужна ветка в UI: для reports показать order, для остальных скрыть.

**R14. scaffold создаёт manifest с placeholder priority=50 по умолчанию.**
После фазы 3 scaffold не генерит priority — хорошо. Но если автор работает на старой версии CLI и генерит, а потом заливает в новую community — manifest упадёт на pydantic-валидации (`extra="forbid"`). Нужно документировать это в CHANGELOG.

**R15. Отсутствие config-override для порядка в v/e — это сейчас «не нужно», но будет ли нужно завтра?**
Если через месяц появится фича «users want to control which validation runs first per-preset» — мы специально вернёмся и добавим опциональное `priority`/`order` в конфиг-YAML. Открытый вопрос на будущее, не блокирующий.

---

# Deep-think резолюции (3 итерации на каждый риск/вопрос)

## R1 — порядок валидации/evaluation → бизнес-пайплайны

**Итерация 1 — в чём реальный страх?**
Пользователь в `config.yaml` указал `plugins: [A, B, C]`, сейчас они шли в порядке priority (`C, A, B`). После релиза пойдут `A, B, C`. Если логика конкретного плагина эмитит issue, который следующий плагин читает → результат отличается.

**Итерация 2 — механизм.**
Проверил `DatasetValidator._run_plugins()` — каждый плагин получает на вход `dataset` и возвращает `ValidationIssue` в общий аккумулятор. Плагины **не читают issues друг друга** — они работают с исходным датасетом независимо. Единственное «наследование» — short-circuit через `fail_on_error: True`, когда критический issue прерывает цикл. В текущей реализации short-circuit происходит на первом `FAIL`, и тут порядок ВАЖЕН: если поменять порядок, «первым» может оказаться другой плагин. НО: значения priority в community-манифестах сейчас (`min_samples=10`, `deduplication=40`) не кричат «должен идти первым» — они просто произвольные числа, отсортированные по эстетике «дешёвые → дорогие». В default-плагинах ручной список уже в правильном порядке (проверил `_get_default_plugins` — «min_samples, avg_length, diversity, empty_ratio»), так что для него ничего не изменится.

**Итерация 3 — митигация.**
(а) Сохраняем для default-плагинов порядок соответствующий текущему priority-сорту при миграции — переупорядочим `_get_default_plugins()`, чтобы результат был идентичным (по факту он уже такой). (б) Для user-config — документируем в CHANGELOG, что порядок теперь берётся из `plugins: [...]` в YAML; добавляем release note «если у вас был implicit-порядок через priority, зафиксируйте его явно в config.yaml». (в) Если внутри validation появится реальная зависимость «плагин B нужен после плагина A» — это плохая архитектура; её надо фиксить как dependency graph, а не через глобальный priority. **Блокером не является.**

## R2 — OpenAPI contract

**Итерация 1 — что рухнет?**
Клиент (`web/`) импортирует тип `ValidationPluginManifest` и обращается к `.priority`. После коммита тип поменяется: `priority` пропадёт, для reports появится `order`.

**Итерация 2 — где именно.**
Проверено грепом: в `web/src/` поиск `priority` даёт только:
- `ActivityFeed.tsx` — локальная функция `priority(run, recentFailIds)`, не связана с плагинами.
- `schema.d.ts` (авто-генерируемый) — там сейчас `priority: number` в манифесте.
- `openapi.json` (авто-генерируемый) — то же.
Рендеров «Priority: X» в реальных компонентах **нет**. Поле приходит с бэка, но фронт его не использует — dead data.

**Итерация 3 — митигация.**
(а) После обновления pydantic-модели регенерировать OpenAPI через существующий pipeline (уже налажен). (б) Добавить в CHANGELOG breaking-change note. (в) Если есть неизвестные внешние клиенты — нет, проект pre-1.0, API нестабилен по декларации. **Блокером не является.**

## R3 — sync-diff взрывается

**Итерация 1 — насколько громко?**
`FIELD_ORDER = ["id", "kind", "name", "version", "priority", "category", ...]` — если убрать priority, строки «сдвинутся» в рендере всех 33 манифестов. Diff будет покрывать весь файл.

**Итерация 2 — рискиcoord.**
(а) Merge-конфликты с параллельными ветками — некритично, команда из одного человека (я). (б) Ревьюеру трудно увидеть семантический diff — но фазы разнесены (Phase 1 обновляет только reports, Phase 2 — только удаляет priority-строку из остальных), так что в каждом commit diff строго соответствует содержательному действию. (в) git blame теряет «кто поставил priority=10» — не страшно, это был авто-скаффолд.

**Итерация 3 — митигация.**
Коммит Phase 3 (обновление toml_writer) делаем **последним**. Тогда Phase 1 и Phase 2 оперируют с старым writer-ом и производят минимальные diff-ы (только удаление строки priority). На Phase 3 — один большой «format normalization» коммит, который разом приводит всё к новому порядку, отдельно от семантики. **Блокером не является.**

## R4 — uniqueness order в reports после миграции

**Итерация 1 — дубли возможны?**
Community reports есть 14 манифестов с priority-значениями: `header=10, summary=20, training_configuration=30, model_configuration=40, metrics_analysis=50, dataset_validation=55, memory_management=60, phase_details=70, evaluation_block=75, stage_timeline=100, config_dump=110, footer=120, issues=?, event_timeline=?` — проверить.

**Итерация 2 — проверка.**
Если дубли существуют — текущий `_validate_plugin_instances` уже должен падать при загрузке; в реальности тесты зелёные, значит дубликатов в текущей инсталляции нет. После миграции значения копируются 1:1 → дубликатов не добавится.

**Итерация 3 — митигация.**
В Phase 1 при миграции: скрипт-проверка `sorted(int(line.split('=')[1]) for line in grep 'priority' community/reports/**/manifest.toml)` — показывает все уникальные значения. Если где-то нечётно, это latent bug который нужно тут же пофиксить (скорректировать order). **Добавляю в Phase 1: pre-migration grep как acceptance criterion.**

## R5 — default validation plugins

**Итерация 1 — риск.**
`dataset_validator.py::_get_default_plugins()` возвращает список пар (id, name, PluginCls). Сейчас они потом сортятся по priority. Без сорта — порядок ручного списка. Если ручной список отличается от порядка который получался после priority-сорта → поведение поменялось.

**Итерация 2 — проверка.**
Надо сравнить: текущий ручной список vs список отсортированный по priority community-манифестов. Пример: ручной список `[min_samples, avg_length, diversity, empty_ratio]`; приоритеты в манифестах `min_samples=10, avg_length=30, diversity=20, empty_ratio=15`. Priority-сорт дал бы `[min_samples, empty_ratio, diversity, avg_length]`. Ручной список **отличается** → потенциальный regression.

**Итерация 3 — митигация.**
Перед удалением сорта переупорядочиваем `_get_default_plugins()` так, чтобы ручной список совпал с текущим priority-сортом. Конкретно: прочитать priority у каждого default-плагина из его community-манифеста ДО Phase 2; переписать ручной список в тот порядок; после этого сорт по priority становится No-op и его можно безопасно удалять. **Добавляю в Phase 2 как prerequisite step.**

## R6 — third-party старые манифесты (`extra="forbid"`)

**Итерация 1 — риск.**
Если кто-то скачал community-zip 3 дня назад и ещё не удалил, то после апгрейда pydantic `extra="forbid"` словит неизвестное поле `priority`.

**Итерация 2 — вероятность.**
Third-party community сейчас **не существует**. Все плагины в репе — наши. Внешних скачиваний не было (проект pre-release). Риск по факту гипотетический.

**Итерация 3 — митигация.**
Если хотим paranoia — заменяем `extra="forbid"` → `extra="ignore"` ТОЛЬКО для `PluginSpec` на 1 релиз (deprecation), потом возвращаем. Но это избыточно для нашего состояния. Лучшее: в CHANGELOG жирно помечаем «BREAKING — старые манифесты с `priority` нужно пересохранить `ryotenkai community sync`». **Блокером не является.**

## R7 — внутренняя логика плагинов читает `self.priority`

**Итерация 1 — риск.**
Конкретный community-плагин может в своём `validate()`/`evaluate()` обращаться к `self.priority` для логгинга или для какой-то собственной логики.

**Итерация 2 — проверка.**
Греп по `community/`: `grep -rn "self.priority\|cls.priority" community/ src/pipeline/ src/evaluation/ src/training/ src/reports/`. Ожидаем 0 совпадений кроме тех, что мы сами про sort. Если появится — это тот редкий случай, где priority-значение реально использовалось плагином.

**Итерация 3 — митигация.**
Грепаем сейчас (до имплементации), если находим — обсуждаем с автором плагина альтернативу (вероятно — хардкодить нужное значение, раз уж приоритет был «secret data»). **Acceptance: Phase 2 не начинается пока не подтверждено что таких ссылок нет.**

## R8 — отчёт-плагины зарегистрированные не через community

**Итерация 1 — риск.**
`ReportPluginRegistry._registry[plugin_id] = cls` можно теоретически заполнить напрямую (не через `register_from_community`). Если такой путь есть, и цены он `.order` как ClassVar присваивается после обычного import-а класса, — мы ломаем его после миграции.

**Итерация 2 — проверка.**
Греп: `ReportPluginRegistry._registry[` или `ReportPluginRegistry.register` — только `register_from_community` (в `catalog.py`). Нет ручных регистраций.

**Итерация 3 — митигация.**
Ничего не требуется — весь путь регистрации идёт через community-catalog. **Не риск.**

## R9 — MLflow tags / логгинг priority

**Итерация 1 — поиск.**
`grep -rn '"priority"' src/ | grep -iE "mlflow|log_param|log_metric"`.

**Итерация 2 — результат.**
Таких мест не найдено. `mlflow.log_*` упоминаний priority в репе нет.

**Итерация 3 — митигация.**
Проходит проверку. **Не риск.**

## R10 — README в community/

**Итерация 1 — риск.**
В каждом `community/<kind>/README.md` упомянуто поле priority как часть схемы манифеста. После удаления документация рассинхронизируется с кодом.

**Итерация 2 — проверка.**
Ожидается: README гласит «plugin manifests have: id, kind, name, version, priority, category, stability, description, entry_point». Для `community/reports/README.md` отдельная секция про order. 5 README.md файлов.

**Итерация 3 — митигация.**
Добавляю отдельный микро-коммит внутри Phase 3 / отдельным мини-шагом: пройтись по 5 README и либо снять упоминание priority, либо перенести в секцию «only for reports plugins: [reports] order». **Включаю в Phase 3.**

## R11 — прочая документация

**Итерация 1 — поиск.**
`grep -rn "priority" docs/ *.md`. Есть ли вне community/ README и планов — CONTRIBUTING, архитектурные диаграммы, ADR-записи?

**Итерация 2 — вероятная находка.**
В `docs/plans/tidy-leaping-lighthouse.md` (сам этот файл) priority не упоминается по теме — он про CLI модернизацию. Возможно, упомянут в старых ADR как часть community v2.

**Итерация 3 — митигация.**
Делаем грепа-проверку в начале Phase 1 acceptance criteria. Если есть ссылки — обновить. **Включаю в Phase 3 как acceptance check.**

## R12 — stale docs/plans/*

**Итерация 1 — риск.**
После завершения этого рефакторинга plan-файл будет про другой refactor, а CLI modernization plan (старый) — потерян.

**Итерация 2 — где это живёт.**
`docs/plans/tidy-leaping-lighthouse.md` — эфемерная директория для одной итерации планирования. Старый план уже описан в git history коммитов (Phase 1, Phase 3+4). Не критическая потеря.

**Итерация 3 — митигация.**
Перед перезаписью (что я и делаю сейчас) — ничего. Git hist сохраняет всё. **Не риск.**

## R13 — UI отображает priority

**Итерация 1 — поиск.**
`grep -rn "priority" web/src/components/ web/src/pages/`.

**Итерация 2 — результат.**
Только `ActivityFeed.tsx` с локальной функцией (не про плагины). Компонентов типа `PluginSettings.tsx` с рендером `Priority: {n}` не найдено.

**Итерация 3 — митигация.**
Ничего не требуется, поле в DTO «transmitted but ignored». **Не риск.**

## R14 — scaffold-старой-версией → несовместимый манифест

**Итерация 1 — риск.**
Пользователь запускает `ryotenkai community scaffold my-plugin` на старой версии CLI → получает manifest с `priority = 50`. Подаёт в новую версию → pydantic падает.

**Итерация 2 — вероятность.**
Все пользователи сейчас работают с ровно одной версией CLI (я). Риск гипотетический.

**Итерация 3 — митигация.**
В CHANGELOG: «если у вас есть manifest со `priority` — удалите строку или запустите `ryotenkai community sync --bump patch <folder>`». Sync должен уметь толерантно принимать старые манифесты (пропускать unknown поля при чтении, затем перезаписывать без них). **Добавляю: sync-парсер на read-стороне использует модель без `extra="forbid"` (режим «tolerant load») как миграционная helper-стратегия.**

## R15 — future need для config-override

**Итерация 1 — вопрос.**
Нужна ли в future опциональная `priority`/`order` на уровне config-YAML? Например, user хочет `plugins: [{id: min_samples, order: 5}, {id: deduplication, order: 1}]`.

**Итерация 2 — анализ.**
Если возникнет — добавление опционального `order: int | None = None` в `DatasetValidationPluginConfig` тривиально; runtime изменит одну строку: если `order` указано — сортируем, иначе — порядок config. Не ломает backward-compat.

**Итерация 3 — резолюция.**
YAGNI — не добавляем сейчас. Архитектурно дверь открыта. Фиксируем как open question «рассмотреть если появится user-request». **Не блокер, open-question помечен на будущее.**

---

## Итог по рискам

**Блокеров нет.** Все риски имеют либо уже решённую митигацию, либо — не применимы к текущему состоянию репо. Перед стартом Phase 2 обязательно выполнить 2 acceptance-чека:

1. `grep -rn "self.priority\|cls.priority" community/ src/` — должен не показать непроверенных консюмеров (R7).
2. Реорганизовать `_get_default_plugins()` так, чтобы порядок совпал с priority-сортом до удаления сорта (R5).

Остальное — документирование в CHANGELOG и регулярная регенерация OpenAPI.
