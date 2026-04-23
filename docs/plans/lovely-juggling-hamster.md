# План: плагины — полный UX-overhaul (каталог в Settings, DnD, multi-instance, configure-modal)

## Context

Текущий UX плагинов:
- Во вкладке `Plugins` проекта — только checkbox-toggle для validation/evaluation/reward. **Плагины отчёта не отображаются**.
- Параметры и пороги показаны как read-only YAML-блоки — отредактировать можно только через сырую YAML в `ConfigBuilder`.
- Порядок плагинов нигде не редактируется руками — validation/evaluation идут в порядке YAML-листа, reports — в `reports.sections: list[str]`, но UI это не экспонирует.
- Каталога плагинов в общих Settings нет — в Settings сейчас только providers.
- **Нет способа иметь два инстанса одного плагина с разными параметрами** (хотя бэкенд `DatasetValidationPluginConfig.id` vs `plugin` это поддерживает — user об этом не знал).

Пользователь хочет единый, «конфиг-билдер-подобный» UX для плагинов: каталог в Settings, drag-and-drop из палитры в проект, multi-instance, configure-модалка с полями в стиле ConfigBuilder, поддержка дефолтов и required.

Результат — плагины ощущаются как first-class configuration objects, а не глобальные тумблеры.

---

## Ключевые архитектурные решения (сразу, не обсуждается в фазах)

### A. Manifest schema v3 — backward-extending, без миграции existing плагинов

В `src/community/manifest.py` расширяем формат `params_schema` / `thresholds_schema` опциональными полями (отсутствуют → получаем текущее поведение):

```toml
[params_schema.sample_size]
type = "integer"
min = 1
default = 10000
# новые опциональные поля:
title = "Sample size"
description = "How many rows to draw for the length check."
required = false            # по умолчанию True если есть default — False, иначе True
```

### B. `ui_manifest()` → трансформирует в JSON Schema

Добавляем функцию `_params_to_json_schema(fields: dict) -> dict` которая ремаппит:
- `type: integer` → `"type": "integer"`
- `min / max` → `"minimum" / "maximum"`
- `options` → `"enum"`
- `title`, `description`, `default` — 1:1
- Собирает `required: [keys…]` из полей без дефолта (или с `required=true`)

Результат — полноценный JSON Schema object, который `FieldRenderer` рендерит без модификаций.

### C. Multi-instance контракт (согласовано с пользователем)

- **Validation / evaluation**: multi-instance — уже работает через `PluginConfig.id` (инстанс) vs `plugin` (reference). UI просто использует.
- **Reward**: single-instance — остаётся в `StrategyPhaseConfig.params.reward_plugin`, бизнес-логика GRPO/SAPO не трогается. **Но** в `PluginsTab` вкладке reward-плагин отображается как обычный row — редактирование через ту же Configure-модалку, под капотом пишет в strategy.
- **Reports**: single-instance — `reports.sections: list[str]` остаётся. Только drag-reorder + палитра отфильтровывает уже добавленные.

### C2. Корреляция strategy ↔ reward plugin (новое требование)

Reward-плагины ОБЯЗАНЫ декларировать совместимые стратегии. `supported_strategies` — обязательное поле для kind=reward, запрещённое для остальных kind’ов. Никаких «пусто = любые» (магия) — автор явно перечисляет список или получает ошибку валидации манифеста.

```toml
[plugin]
id = "helixql_compiler_semantic"
kind = "reward"
supported_strategies = ["grpo", "sapo"]   # ОБЯЗАТЕЛЬНОЕ для kind=reward
```

Реализация:
- **Backend**:
  - `src/community/manifest.py::PluginSpec` — добавить `supported_strategies: list[str] = []`.
  - `ui_manifest()` — пробрасываем в API.
  - `src/config/training/strategies/phase.py` — добавить cross-validator: если `strategy_type ∈ {grpo, sapo, dpo}` и `reward_plugin` указан, проверяем что `reward_plugin.kind == "reward"` **и** его `supported_strategies` либо пустой, либо содержит `strategy_type`. Иначе ConfigCheck-ошибка.
- **Frontend**:
  - Палитра reward-плагинов фильтруется по **текущему** `strategy_type` проекта.
  - В Configure-модалке reward — read-only хинт «Совместимо со стратегиями: grpo, sapo».
  - При смене strategy_type, если несовместимый reward — `ValidationBanner` показывает ошибку с предложением заменить.
- **Миграция 1 существующего reward-манифеста** (`helixql_compiler_semantic`) — указываем `supported_strategies = ["grpo", "sapo"]` явно.

### D. DnD библиотека — `@dnd-kit/core` + `@dnd-kit/sortable`

Современная, accessible (keyboard-nav из коробки), не deprecated, ~20 KB gzip. React-dnd тяжелее и менее поддерживается.

### E. ViewSettings.Catalog = read-only обзор. ProjectTabs.Plugins = editable + drag-target

Два разных UX:
- **Settings/Catalog** — «что вообще есть», grid-карточки с info-кнопками, фильтр/поиск по kind. Читаемый «справочник».
- **Project/PluginsTab** — rows by kind, каждая строка = инстанс с configure/remove/drag-handle. Палитра (sidebar drawer) — источник DnD.

---

## Фазы реализации (4 PR, каждый standalone-зелёный)

### Фаза 1 — Backend: manifest v3 + JSON Schema transform + reward correlation

**Файлы:**
- `src/community/manifest.py`:
  - Добавить optional `title`, `description`, `required` в field-level schema (`ParamFieldSchema` model).
  - Добавить `supported_strategies: list[str] = []` в `PluginSpec`.
  - Функция `_to_json_schema(raw: dict) -> dict` — трансформирует текущий формат в JSON Schema.
- `src/community/manifest.py::ui_manifest()` — выдаёт `params_schema` / `thresholds_schema` **уже в JSON Schema виде**. Пробрасывает `supported_strategies`.
- `src/config/training/strategies/phase.py` — cross-validator: strategy_type ↔ reward_plugin.supported_strategies.
- `community/reward/helixql_compiler_semantic/manifest.toml` — добавить `supported_strategies = ["grpo", "sapo"]`.
- `src/tests/unit/community/test_manifest.py` — tests на transform (type/min/max/enum/default/required) и на supported_strategies.
- `src/tests/unit/config/validators/test_cross_validators.py` — test на корреляцию.
- Регенерация `web/src/api/openapi.json` + `schema.d.ts`.

**Risk:** UI-клиенты, читающие `params_schema` как dict (текущий PluginBrowser / PluginsTab), будут видеть теперь `{type, properties, required}` вложенный. **Фикс:** PluginBrowser обновить в той же фазе (он сейчас просто `JSON.stringify` показывает).

**Acceptance:**
- `curl /plugins/validation | jq '.plugins[0].params_schema'` → валидный JSON Schema.
- `curl /plugins/reward | jq '.plugins[0].supported_strategies'` → `["grpo", "sapo"]`.
- `ryotenkai validate <project>` с `strategy_type: dpo` и `reward_plugin: helixql_compiler_semantic` возвращает ошибку «reward plugin X is not compatible with strategy dpo».

---

### Фаза 2 — Frontend: Settings → Catalog tab + Info-modal

**Файлы:**
- `web/src/pages/Settings.tsx` — добавить sidebar entry «Catalog» (или переименовать общую страницу в «Catalog»: plugins + presets).
- `web/src/pages/Catalog.tsx` — **новая**. Секции: Validation / Evaluation / Reward / Reports / Presets. Поиск, фильтр. Каждая карточка — `PluginCatalogCard` с `i`-кнопкой.
- `web/src/components/PluginCatalogCard.tsx` — **новая**, re-use стиль из `ProviderCard` + chip-палитры из `PresetPickerModal.tsx`.
- `web/src/components/PluginInfoModal.tsx` — **новая**. Рендерит: описание, `params_schema` через `FieldRenderer` (read-only с дефолт-значениями), `thresholds_schema` аналогично, пример YAML. Секция «example usage».
- `web/src/api/hooks/usePlugins.ts` — расширить, чтобы опционально загружал все kind’ы разом (для Catalog).

**Acceptance:** `/settings/catalog` показывает все 4 kind’а плагинов + пресеты, info-кнопка открывает modal с деталями.

---

### Фаза 3 — Frontend: Project/Plugins tab + multi-instance + DnD reorder

**Файлы:**
- `package.json` — добавить `@dnd-kit/core`, `@dnd-kit/sortable`, `@dnd-kit/utilities`.
- `web/src/components/ProjectTabs/PluginsTab.tsx` — **переписать**:
  - 4 секции: Validation / Evaluation / **Reward (single-slot)** / Reports.
  - Для validation/evaluation/reports: список инстансов + «+ Add» / drag-from-palette.
  - Для reward: одна фиксированная «карточка» (пустая → «Выберите reward-плагин» → drop target). Под капотом пишет в `training.strategies[active].params.reward_plugin`.
  - Каждый row с: drag-handle, kind-icon, имя (`id` или автогенерация `avg_length #2`), `Configure`, `Remove`.
  - `@dnd-kit/sortable` context per-kind (4 независимых sortable-списка). Reward — просто droppable, без sortable.
  - На reorder для validation/evaluation — перезаписываем `plugins: [...]` в том же порядке. Для reports — перезаписываем `reports.sections`.
- `web/src/components/ConfigBuilder/PluginPaletteDrawer.tsx` — **новая**. Drawer/sidebar с группированными кнопками плагинов. DnD source.
  - Reward-секция палитры **фильтруется** по `training.strategies[active].strategy_type` — показываем только совместимые (`supported_strategies` пусто или содержит текущий strategy_type).
  - Если плагин несовместим — отображается полупрозрачным с tooltip «Несовместимо со стратегией sft, переключитесь на grpo».
  - При drop в целевой kind-список — создаётся новый инстанс с `id = <plugin_id>_<n>`, `params = suggested_params`, `thresholds = suggested_thresholds`.
- Логика генерации уникального instance-id: собирать существующие id в списке, прибавлять суффикс если занят.

**Risk:** Drag-between-kinds — не даём (валидационный плагин в reward не попадёт — `kind` разный). Ограничиваем `droppable` только своим kind.

**Risk:** Reports пока single-instance — палитра для reports-плагинов отфильтровывает уже добавленные.

**Risk:** Смена strategy_type в другом месте (ConfigBuilder) делает активный reward_plugin несовместимым. Backend cross-validator из Фазы 1 это ловит, UI подсвечивает через ValidationBanner.

**Acceptance:**
- DnD работает мышью и клавиатурой (Tab+Space), инстансы сохраняются в YAML.
- Порядок reports отражается в `reports.sections`.
- Для strategy_type=grpo в палитре видны только совместимые reward-плагины; для sft reward-секция палитры пустая с пояснением.

---

### Фаза 4 — Frontend: Configure modal с FieldRenderer

**Файлы:**
- `web/src/components/ConfigBuilder/PluginConfigModal.tsx` — **новая**. Три секции:
  1. **Instance settings** — `id`, `plugin` (read-only ссылка на каталог), `apply_to` для validation, `enabled` и т.д.
  2. **Parameters** — `<FieldRenderer root={pluginSchema} node={params_schema} value={instance.params} onChange={…} />`.
  3. **Thresholds** — то же для `thresholds_schema`.
- Импорт `FieldRenderer`, `ValidationProvider` из `web/src/components/ConfigBuilder/`. Schema передаём как `pluginManifest.params_schema` (теперь валидный JSON Schema после Фазы 1).
- «Reset to defaults» кнопка — заливает `suggested_params` / `suggested_thresholds` обратно в value.
- Required-поля помечены `*` автоматически (FieldRenderer уже умеет).
- Валидация ошибок через существующий `ValidationBanner` pattern — reuse как есть.

**Acceptance:** Кликнуть Configure → редактировать поля → Save → в YAML обновлены `plugins[i].params` и `plugins[i].thresholds`. Ошибки валидации показываются под полями.

---

## Критические файлы и функции для переиспользования

| Где живёт | Что переиспользуем |
|---|---|
| `web/src/components/ConfigBuilder/FieldRenderer.tsx` | `<FieldRenderer />` для рендера params/thresholds. Принимает JSON Schema. |
| `web/src/components/ConfigBuilder/ValidationContext.tsx` | `ValidationProvider` + `useFieldStatus` для подсветки ошибок. |
| `web/src/components/ConfigBuilder/HelpTooltip.tsx` | `?`-иконка рядом с сложными полями в info-modal. |
| `web/src/components/ConfigBuilder/PresetPickerModal.tsx` | Chip-палитра (`CHIP_VRAM`, `CHIP_MODEL`…), search-input, keyboard nav — в каталог и палитру. |
| `web/src/components/ui.tsx` | `SectionHeader`, `Card`, `EmptyState` — Settings-шаблон. |
| `src/community/manifest.py::PluginManifest` | Базовая модель; расширяем без breaking изменений. |

---

## Verification

### После каждой фазы
```bash
pytest src/tests/unit/community/ -q
ruff check src/ community/
npx tsc --noEmit -p web/tsconfig.json     # в фазах 2-4
```

### E2E за весь фичеплан
1. `http://localhost:5173/settings/catalog` — вижу все 4 kind’а плагинов + пресеты; info на карточке открывает модал с описанием полей.
2. `http://localhost:5173/projects/testproject/config` → вкладка Plugins:
   - Палитра слева.
   - Перетаскиваю `avg_length` в validation → появляется инстанс `avg_length`.
   - Перетаскиваю второй раз → появляется `avg_length_1` с теми же suggested params.
   - Configure → меняю `sample_size` с 10000 на 5000 → Save → YAML обновлён.
   - Drag-reorder reports секций → `reports.sections` в YAML в нужном порядке.
3. `ryotenkai validate <project>` — конфиг валиден, оба инстанса запускаются (видно в логах).

---

## Решения, согласованные с пользователем

- **Reports — single-instance**, `reports.sections: list[str]` остаётся без миграции.
- **Reward — single-instance per project**, но показывается в PluginsTab как row (редактируется через ту же Configure-модалку, пишет в `strategy.params.reward_plugin`).
- **Корреляция strategy ↔ reward** через `supported_strategies` в манифесте + cross-validator на бэке.
- **Scope — 4 отдельных PR по фазам**, каждая самодостаточна и mergeable независимо.

---

# Risk analysis (3 итерации обнаружения + 3 итерации deep-think на каждый)

## Итерация 1 — очевидные риски

**R1. Breaking change формата `params_schema` в OpenAPI.** Сейчас поля сериализуются «плоским» dict’ом (`{sample_size: {type: integer, min: 1, default: 10000}}`). После Фазы 1 — JSON Schema (`{type: object, properties: {sample_size: {type: integer, minimum: 1, default: 10000}}, required: [...]}`). Все текущие потребители (`PluginBrowser`, `PluginsTab`) сломаются.

**R2. FieldRenderer ожидает `PipelineJsonSchema` (с `$defs`/`definitions`).** Наш плагинный schema — standalone, без refs. `resolveRef()` может упасть на пустом `root.$defs`.

**R3. State-менеджмент 4 независимых sortable-списков + external DnD source (палитра).** `@dnd-kit` через `DndContext` поддерживает, но нужно аккуратно разводить collision-detection и drop-target фильтрацию по kind.

**R4. Race между локальным form-state при drag и async save.** Пользователь перетянул плагин → UI показывает инстанс → save не успел → пользователь тянет второй → генератор `id` смотрит на локальный state (правильно) или на серверный (неправильно)?

**R5. Info-modal для preset ≠ для plugin** — разный контракт (scope/requirements/placeholders vs params_schema). Нужно 2 раздельных компонента или один с вариантами.

## Итерация 2 — вторичные, неочевидные риски

**R6. Конфликт `required=true` и `default` в TOML.** JSON Schema: поле с `default` формально optional. Если в манифесте `required=true` + `default=10` — что это значит?

**R7. `reports` single-instance требует реактивной фильтрации палитры.** Палитра должна моментально скрывать/показывать плагин при add/remove из `reports.sections`.

**R8. Reward — per-strategy, а не per-project.** `StrategyPhaseConfig.params.reward_plugin` существует для каждой фазы. В многофазных стратегиях (SAPO = 2 фазы: SFT + preference) у каждой фазы может быть свой reward. User сказал «reward только 1», но schema допускает N.

**R9. Смена strategy_type при уже выбранном несовместимом reward.** Что делает UI? Авто-очистить, оставить с ошибкой, мигрировать — решение не согласовано.

**R10. `apply_to` для validation-плагинов — где редактируется в Configure-модалке?** Это instance-setting, не params/thresholds. Нужна отдельная секция.

**R11. «Reset to defaults» — поле или весь инстанс?** `params_schema.foo.default = 10` (per-field) vs `suggested_params = {foo: 10, bar: 20}` (whole set). Различие.

**R12. Settings/Catalog и Project/Palette — overlap.** Оба показывают список плагинов. Различать контексты — первый обзорный с info, второй DnD-source. Пользователь подтвердил: оба нужны, не дублировать код.

## Итерация 3 — документация, edge-cases, долгосрочные

**R13. `community/<kind>/README.md` описывает старый schema format.** Рассинхронизация документации с кодом после Фазы 1.

**R14. `ryotenkai community scaffold` генерирует `params_schema` в старом формате.** Новые плагины приходят «сломанными».

**R15. `toml_writer.FIELD_ORDER` при добавлении `supported_strategies`.** Изменение порядка полей в `[plugin]` переформатирует все 33 manifest.toml на ближайшем `community sync` — шум в диффе.

**R16. DnD и accessibility (keyboard nav, screen readers).** `@dnd-kit` поддерживает, но требует явной конфигурации `KeyboardSensor` и `aria-describedby`.

**R17. Stale plugin reference.** Плагин удалён из community/ → проект ссылается на несуществующий id. UI должен показать placeholder «Plugin not found» вместо падения.

**R18. FieldRenderer.path для плагин-модалки.** Path threads deep-link + кастомные рендереры. Какое значение ставить для synthetic плагин-контекста?

**R19. Instance-id uniqueness check на клиенте.** Pydantic валидирует при save. Но UX лучше, если модалка ловит дубликат до save.

**R20. Client-side vs round-trip валидация в configure modal.** Задержка vs покрытие cross-field validation.

**R21. Triggers cross-validator strategy↔reward.** В каких точках бэкенда он выполняется (load, save, validate command)?

---

# Deep-think резолюции (3 итерации на каждый)

## R1 — backward-compat ui_manifest()

**Итерация 1 — насколько критично?** Клиенты: `PluginBrowser.tsx` (line 18–19 читает `suggested_params`/`suggested_thresholds` — не `params_schema`!), `PluginsTab.tsx` (та же история: берёт `suggested_*` при enable, params_schema не трогает). **Вывод:** фактических потребителей сырого `params_schema` сейчас нет — поле эмитится, но не разбирается. Breaking на UI уровне ~нулевой.

**Итерация 2 — OpenAPI типы?** `PluginManifest.params_schema: Record<string, unknown>` — нестрогий тип. Смена содержимого не ломает TS-компиляцию. Но: если где-то внешний потребитель схему разбирает — сломаем.

**Итерация 3 — митигация.** Оставляем имя поля `params_schema`, но меняем содержимое на JSON Schema. В CHANGELOG помечаем. Добавляем новое поле `params_fields: list[str]` (keys для быстрого iteration без walking JSON Schema) — **опционально**, для DX. **Блокером не является.**

## R2 — FieldRenderer standalone для плагин-схемы

**Итерация 1 — что упадёт.** `resolveRef` вызывается только если node содержит `$ref`. Наша плагин-схема flat, без refs — `resolveRef` не вызывается. Безопасно.

**Итерация 2 — `PipelineJsonSchema` vs generic JsonSchemaNode.** Тип `PipelineJsonSchema extends JsonSchemaNode` — структурно совместим, TS-каст достаточен. Передаём `pluginSchema as PipelineJsonSchema`.

**Итерация 3 — CUSTOM_FIELD_RENDERERS сработают на плагин-путях?** Лукап по `path` (`training.provider` → `TrainingProviderField`). Для плагина path = `plugin_params.foo` — не пересекается с кастомными. **Не риск.** Митигация — просто передать уникальный префикс path (`__plugin_params__.foo`) для страховки. **Блокером не является.**

## R6 — required vs default

**Итерация 1 — формальная семантика.** JSON Schema: `required` — массив имён ключей, которые должны присутствовать в объекте. `default` — значение при отсутствии ключа. Можно одновременно: «required present, default if omitted by generator». На практике `required` означает «user must decide», `default` — «fallback».

**Итерация 2 — что нужно пользователю в манифесте?** Два разных намерения:
- (a) «Поле обязательное, без default, автор должен указать» → `required=true`, no default.
- (b) «Поле с разумным default, можно не указывать» → no required, has default.
- (c) «Рекомендованное значение, но пользователь должен его явно подтвердить» — неоднозначно, запутывает.

**Итерация 3 — резолюция.** Запрещаем в манифесте одновременно `required=true` + `default`. Pydantic валидатор в `ParamFieldSchema`: если оба — raise ValueError. В JSON Schema output: `required` включает только поля с `required=true` (no default implies optional). **Симпл и однозначно.**

## R8 — reward per-strategy vs per-project

**Итерация 1 — что говорит schema.** `StrategyPhaseConfig.params.reward_plugin: str | None`. Каждая фаза независима. SAPO в community имеет 2 стратегии (`strategies: [{strategy_type: sft}, {strategy_type: sapo, reward_plugin: ...}]`). **User сказал «1 reward», имея в виду 1 активная preference-fine-tuning фаза, а не 1 reward на весь проект.**

**Итерация 2 — UI фолдинг.** Показывать reward внутри row стратегии (per-strategy), а не отдельной секцией. Но тогда PluginsTab теряет цельность «все плагины в одном месте».

**Итерация 3 — компромисс.** PluginsTab Reward-секция показывает **список stratgies** из конфига, и для каждой — swap-slot с reward_plugin. `Strategy #1: SFT — no reward (not applicable)`, `Strategy #2: SAPO — [reward slot]`. Drag в slot пишет в `strategies[idx].params.reward_plugin`. В manifest cross-validator смотрит `supported_strategies` против `strategies[idx].strategy_type`. **Резолюция — scope Фазы 3:** renderим N reward-слотов (по числу стратегий с reward-enabled типом: grpo/sapo/dpo/ppo).

## R9 — смена strategy → несовместимый reward

**Итерация 1 — варианты.** Auto-clear / keep-with-error / migrate-to-default / block-change.

**Итерация 2 — что делают другие тулы.** ML Studio и Kubeflow: **keep-with-error** — пользователь сам видит и исправляет. Избегают auto-modification чтобы не терять работу пользователя.

**Итерация 3 — резолюция.** Keep-with-error. Меняю strategy → reward остаётся. ValidationBanner показывает ошибку «reward X не совместим со стратегией Y» со ссылкой «Open reward settings». В Configure-модалке reward — баннер «Incompatible with current strategy» с кнопкой «Remove reward plugin». **Предсказуемо, не удаляет работу.**

## R10 — apply_to (и другие instance-settings) в Configure modal

**Итерация 1 — что есть у разных kind’ов.**
- Validation: `id`, `plugin`, `apply_to: list[str]`, `enabled: bool`, `fail_on_error: bool`, `params`, `thresholds`.
- Evaluation: `id`, `plugin`, `enabled: bool`, `params`, `thresholds`.
- Reward: `reward_plugin: str`, `reward_params: dict`. (Нет id/enabled — он implicit per strategy.)
- Reports: `str` в `sections` — вообще нет per-instance settings.

**Итерация 2 — 3 разных UX?** Чрезмерно. Модалка должна унифицировать.

**Итерация 3 — резолюция.** Configure modal = 3 секции:
1. **Instance** — `id`, `enabled`, `apply_to` (если kind=validation), `fail_on_error` (если kind=validation). Для evaluation — `id`, `enabled`. Для reward — read-only `strategy_ref`. Для reports — только показывает id.
2. **Parameters** — FieldRenderer по `params_schema`.
3. **Thresholds** — FieldRenderer по `thresholds_schema` (если непустая).
Если секция пустая (нет полей) — скрывается. **Универсальный layout, адаптивное содержимое.**

## R11 — Reset to defaults

**Итерация 1 — что логично.** `suggested_params` в манифесте = «набор рекомендованных значений, который автор хочет видеть при attach». Это включает поля со значениями, отличными от `default` (автор выбрал что-то лучше для этого плагина).

**Итерация 2 — два уровня.**
- Per-field: рядом с полем кнопка «↺ default» (если текущее ≠ field.default).
- Per-instance: кнопка внизу модалки «Reset to recommended defaults» → заливает `suggested_params` / `suggested_thresholds` целиком.

**Итерация 3 — резолюция.** Делаем оба. Per-field на каждом поле с `default`. Per-instance внизу модалки — **«Reset to suggested»** (не «default»), чтобы не путать с per-field default. Подпись tooltip: «Restores values the plugin author recommends». **Разведённая семантика.**

## R12 — Settings/Catalog vs Project/Palette

**Итерация 1 — различие контекстов.** Settings/Catalog — обзорный «справочник»: что существует, как выглядит, что принимает. Project/Palette — «source for DnD»: плотный список с кнопками-тегами для перетаскивания.

**Итерация 2 — переиспользование кода.** Обе используют `usePlugins`, чипы из `PresetPickerModal`, info-модалку. Но layout разный (grid vs compact list) и функционал разный (info vs DnD).

**Итерация 3 — резолюция.** Два компонента, общие atoms:
- Общий: `PluginChipRow` (name + kind + category + info-icon).
- Catalog (`web/src/pages/Catalog.tsx`): grid, секции по kind, фильтр, info-clickable.
- Palette (`web/src/components/ConfigBuilder/PluginPaletteDrawer.tsx`): flat list, drag-enabled, tight spacing. Use `PluginChipRow` inside `<Draggable>`. **Shared atom, разные контейнеры.**

## R3 — DnD state с 4 sortable + palette

**Итерация 1 — архитектура @dnd-kit.** `DndContext` один на весь PluginsTab. Внутри 4 `SortableContext` (по kind). Палитра вне `SortableContext` — plain `useDraggable`. Drop-target — `useDroppable` на каждом kind-контейнере.

**Итерация 2 — collision detection.** `closestCorners` подходит для sortable. Для external (palette → kind container) — пригодится `pointerWithin`. Можно combine: `customCollision = args => pointerWithin(args) ?? closestCorners(args)`.

**Итерация 3 — резолюция.** Один `DndContext`, одна `handleDragEnd` функция, которая диспатчит по `over.data.current.kind`. Палитра-источник передаёт `{kind, plugin_id, action: 'add'}`, kind-контейнер — `{kind, instance_id, action: 'reorder'}`. Event handler:
```ts
if (active.data.current.action === 'add') addInstance(over.data.current.kind, active.data.current.plugin_id)
else if (active.data.current.action === 'reorder') reorderInstances(over.data.current.kind, active.id, over.id)
```
**Компактно, 1 место для state-mutation.**

## R4 — race между локальным state и save

**Итерация 1 — источник правды.** Реактивный `formState` в `ConfigTab` — единственный source of truth для UI. Save — просто persistence, не влияет на генерацию id.

**Итерация 2 — id generator.** `generateInstanceId(plugin_id, existing_ids) → string` смотрит на локальный список — корректно. `useSaveProjectConfig` — fire-and-forget mutation.

**Итерация 3 — резолюция.** Не риск. Локальный state source-of-truth, save идёт фоново через TanStack Query (debounce). Гарантии нужны только на save-side: сервер отвергнет дубликат id → `ValidationBanner` показывает, UI откатывает. **Стандартный optimistic pattern.**

## R5 — Info-modal для preset vs plugin

**Итерация 1 — различие данных.** Preset: scope, requirements, placeholders. Plugin: params_schema, thresholds_schema, supported_strategies, stability.

**Итерация 2 — общие поля.** id, name, description, version, category/kind.

**Итерация 3 — резолюция.** 1 общий компонент `<CatalogItemInfoModal />` с 2 варианта-секциями, переключается по `item.kind === 'preset' | 'plugin'`:
```tsx
<InfoModal item={selected}>
  <Header /> {/* общее */}
  {item.kind === 'preset' ? <PresetDetails /> : <PluginDetails />}
</InfoModal>
```
**Разделение на уровне внутренних секций, 1 shell.**

## R7 — реактивная фильтрация палитры для reports

**Итерация 1 — проблема.** Если плагин уже в `reports.sections`, палитра должен его скрыть. После remove — вернуть.

**Итерация 2 — реализация.** Palette component берёт `config.reports.sections` и `availableReports` и делает `availableReports.filter(p => !sections.includes(p.id))`. Реактивно через `useProjectConfig`. React перерендерит при изменении config.

**Итерация 3 — резолюция.** Стандартный reactive filter. Не риск. **Митигация уже встроена.**

## R13 — документация community/*/README

**Итерация 1 — влияние.** Внешний читатель следует старому формату → создаёт плагин → получает ворнинг «unknown field description» (Pydantic extra=forbid)? Нет — description у нас optional новое поле.

**Итерация 2 — скоп обновления.** 4 README.md файла: validation/evaluation/reward/reports. Секция «Writing manifest.toml» в каждом.

**Итерация 3 — резолюция.** Включаем в Фазу 1 как отдельный mini-commit: «docs(community): manifest.toml schema reference update». **Не блокер, низкая стоимость.**

## R14 — scaffold

**Итерация 1 — текущее состояние.** `src/community/scaffold.py::build_plugin_manifest_dict()` создаёт плоский `params_schema` с типами. Обновить под новый формат — включить `description`, `required`, `title` placeholders.

**Итерация 2 — обратная совместимость.** Старые манифесты без новых полей работают (optional). Scaffold генерит ставит TODO-комментарии «description = "TODO"» чтобы автор их заполнил.

**Итерация 3 — резолюция.** Включаем в Фазу 1. Один cohesive PR «backend schema v3» содержит: manifest.py, scaffold.py, toml_writer.py (FIELD_ORDER), examples, README. **Не блокер.**

## R15 — FIELD_ORDER churn

**Итерация 1 — шум.** Добавление `supported_strategies` в `FIELD_ORDER` переставит поля в всех 33 manifest.toml при первом `community sync`.

**Итерация 2 — митигация.** Порядок новых полей ставим в конце списка — existing manifests уже в старом порядке, при sync новое поле просто добавится снизу, а не вставится в середину.

**Итерация 3 — резолюция.** `FIELD_ORDER = [..., existing..., "supported_strategies"]` — append only. Существующие манифесты без этого поля при sync не перегенерируются (writer идёт по присутствующим полям). **Нулевой diff.** Не блокер.

## R16 — accessibility DnD

**Итерация 1 — поддержка @dnd-kit.** Из коробки: `KeyboardSensor`, `aria-describedby`, announcers. Требуется: `sensors = [useSensor(PointerSensor), useSensor(KeyboardSensor, {coordinateGetter: sortableKeyboardCoordinates})]`.

**Итерация 2 — тестирование.** Пройтись Tab → Space (lift) → Arrow (move) → Space (drop). Announcer читает «picked up plugin avg_length, row 1 of 3, moved to row 2».

**Итерация 3 — резолюция.** Стандартная accessibility-настройка @dnd-kit включается в Фазу 3 (Drag-handle на row + KeyboardSensor в DndContext). **Не блокер, но требует явного вызова.**

## R17 — stale plugin reference

**Итерация 1 — симптом.** Плагин удалён из community/ → `config.validation.plugins[0].plugin = "removed_plugin"`. `useManifestFor(pluginId)` → undefined. Configure button падает.

**Итерация 2 — детект.** В `PluginRow` компонент: если `!manifest` — render fallback «⚠️ Plugin '<id>' not found in catalog. Remove or reinstall.».

**Итерация 3 — резолюция.** Helper-хук `usePluginManifest(pluginId)` возвращает `manifest | null`. Row показывает warning-state. Configure disabled. Remove всё ещё работает (удаляет из YAML). **Error-path встроен, не блокер.**

## R18 — FieldRenderer path для плагинов

**Итерация 1 — зачем path.** Deep-link hash, кастомные рендереры, анкоры.

**Итерация 2 — synthetic path.** В плагин-модалке: `__plugin__.<plugin_id>.params.<field>`. Не конфликтует с config-paths (config не начинается с `__`). Кастомных рендереров под этот префикс нет — safe.

**Итерация 3 — резолюция.** В `PluginConfigModal` передаём `pathPrefix="__plugin__/<instanceId>/params"`. Deep-link внутри модалки не нужен (модалка закрывается). **Не блокер.**

## R19 — instance-id uniqueness на клиенте

**Итерация 1 — почему важно.** Ошибка на save раздражает — лучше не дать её допустить.

**Итерация 2 — реализация.** Configure-modal при blur `id`-инпута проверяет `existingIds.includes(newId) && newId !== originalId`. Подсвечивает поле + inline error «id already used».

**Итерация 3 — резолюция.** Client-side валидация через `useClientFieldValidation` + backend-guard. Двойная сетка. Include в Фазу 4. **Не блокер.**

## R20 — validation client vs round-trip

**Итерация 1 — trade-off.** Client-side = мгновенная фидбек-петля, не ловит cross-field. Round-trip = полное покрытие, задержка.

**Итерация 2 — как делают сейчас.** `ConfigTab` делает debounced validate на blur (300ms). Полный round-trip. Работает хорошо.

**Итерация 3 — резолюция.** Применяем тот же pattern в `PluginConfigModal`: onBlur → debounced `validatePluginConfig(draft)` → `field_errors` → подсветка. Save кнопка не зависит от валидации (пользователь может сохранить с ошибками, они останутся видимы). **Reuse существующего pattern.**

## R21 — когда запускается cross-validator strategy↔reward

**Итерация 1 — точки триггера.** `validate` command, `save_config` endpoint. Сейчас cross-validators в `src/config/validators/` запускаются обоими.

**Итерация 2 — достаточно ли.** Да. На save backend вернёт ошибку в `field_errors`. UI подсветит баннер. На validate CLI — то же.

**Итерация 3 — резолюция.** Добавляем новый validator в стандартный pipeline, ничего нового не нужно. **Не блокер.**

---

# Итог по рискам

**Блокеров нет.** Все 21 риск имеют понятную митигацию или резолюцию. Критические уточнения, переносимые в основной план:

1. **R6** — запретить `required=true` + `default` в манифесте (Pydantic-validator).
2. **R8** — reward не global-single, а per-strategy-phase; PluginsTab показывает N reward-слотов (по числу совместимых фаз).
3. **R9** — keep-with-error при смене strategy (не auto-clear).
4. **R10** — Configure modal = 3 секции (Instance settings / Parameters / Thresholds); пустые секции скрываются.
5. **R11** — 2 уровня reset (per-field default, per-instance «suggested»).
6. **R15** — append-only `FIELD_ORDER` для нулевого diff у существующих манифестов.
7. **R16** — явный `KeyboardSensor` в DnDContext.
8. **R17** — fallback-row для отсутствующих plugin-id.

Дополнительные под-задачи, которые теперь включены в фазы:

- **Фаза 1**: обновление README community/ (R13), обновление scaffold (R14), Pydantic-валидатор required∩default (R6).
- **Фаза 3**: N reward-слотов по числу phases (R8), fallback-row для stale plugin (R17), KeyboardSensor + aria (R16).
- **Фаза 4**: 3-секционная Configure-modal (R10), two-level reset (R11), client-side id uniqueness (R19), debounced validate on blur (R20).

---

# Senior-review уточнения (Frontend UX + MLOps architect перспективы)

## Frontend best-practices, пропущенные в первой редакции

### FE-1. Undo для деструктивных действий
Пользователь случайно удалил инстанс перетаскиванием или кнопкой Remove — работа (params, thresholds) потеряна. Реализация: после remove показать toast «Plugin removed. Undo?» с таймаутом 5 сек. Реализуется через `useProjectConfig.undoBuffer` — снэпшот предыдущего state.

### FE-2. Skeleton / empty states с CTA
- Пустая validation-секция: не просто «No plugins», а illustration + стрелка на палитру + текст «Drag a validation plugin from the palette or click Add».
- Loading skeleton на fetch `/plugins/all` (вместо spinner).

### FE-3. Focus management в Configure modal
- Открытие: focus на первом редактируемом поле (не на «Save»).
- Закрытие: focus возвращается на row в PluginsTab, с которой был открыт.
- Реализуется через `useRef` + `useEffect` при mount/unmount.

### FE-4. Screen-reader announcements
Не только DnD. Add/remove/configure — все объявлять через `aria-live="polite"` область. `@dnd-kit` даёт свой announcer; custom events добавляем вручную.

### FE-5. URL state для Catalog
Фильтр kind, поиск → в query params (`/settings/catalog?kind=validation&q=avg`). Shareable, bookmarkable, browser-back работает. Через `useSearchParams` (react-router).

### FE-6. Mobile/touch
Desktop-only — ок для dev-tools. Но минимальная проверка: DnD на touch пройдёт через @dnd-kit при установке `TouchSensor` с `activationConstraint: {delay: 200}`. Не блокер, phase 3 включает sensor.

### FE-7. Live preview в Configure modal (optional polish)
Для validation-плагина — «Run on current dataset sample» кнопка (использует endpoint `/plugins/{kind}/{id}/preview` с текущими params). Показывает issue-count. **Scope: Фаза 4 nice-to-have, НЕ блокер.**

### FE-8. Form dirty state
Configure modal со Save-кнопкой: enabled только при dirty. Close с dirty state — confirm dialog «Discard changes?».

### FE-9. PluginsTab ↔ ConfigBuilder синхронизация
Оба редактируют один и тот же YAML. Нужен общий хук `useProjectConfig` как single source of truth — любое изменение через mutation вызывает refetch везде. Уже так? Проверить в Фазе 3.

### FE-10. Loading state во время DnD-save
При drop → optimistic update UI → async save. Если save failed → rollback + toast. Через TanStack Query `onMutate`/`onError`.

## MLOps / архитектурные улучшения

### ARCH-1. Versioning — не добавляем
Согласовано: обратную совместимость не делаем, мёртвый код не плодим. Все 33 manifest.toml мигрируются в одном коммите Фазы 1. Поле `manifest_version` не вводим — это мёртвый функционал при отсутствии legacy-пути.

### ARCH-2. Plugin dependency declarations
Некоторые плагины требуют другие (например, evaluator требует конкретного validator). Будущая возможность:
```toml
[plugin.requires]
plugins = [{kind: "validation", id: "min_samples"}]
```
**Scope:** не в этом плане. Отмечаем как future work.

### ARCH-3. Secret handling в Configure modal
Manifest может помечать поле как secret (`[params_schema.api_key] secret = true`). UI:
- Рендер как `<input type="password">`.
- Автосгенерированный env-var-name (`PLUGIN_<plugin_id>_API_KEY`).
- Инпут сохраняет имя env-var, не значение.
- В YAML: `params: {api_key: "${PLUGIN_AVG_LENGTH_API_KEY}"}`.
**Scope:** include в Фазу 4 как расширение `ParamFieldSchema`. Один плагин (helixql_compiler_semantic) пока секретов не использует — тестируем на mock.

### ARCH-4. Plugin observability (MLflow integration)
При attach через UI: писать mlflow tag `plugin.<kind>.<instance_id>.version = <manifest.version>`. Уже есть в existing pipeline? Проверить. Если нет — добавить в Фазу 1.

### ARCH-5. API endpoint для Catalog
Сейчас `/plugins/{kind}` → 4 отдельных запроса для catalog-page. Добавить:
```
GET /plugins
→ { validation: [...], evaluation: [...], reward: [...], reports: [...] }
```
Один запрос, один cache-key. Включаем в Фазу 2.

### ARCH-6. Cross-validator strategy↔reward — где запускается
`src/config/validators/cross_validators.py::validate_reward_strategy_compatibility(config, catalog) -> list[ConfigCheck]`. Вызывается:
- При `ryotenkai validate` (уже).
- В `save_config` endpoint (POST /projects/{id}/config).
- В Pydantic `model_validator(mode="after")` на `PipelineConfig` — для моментальной ошибки при парсинге YAML.

Три точки входа — нужно тесты на все три. Tests в `src/tests/unit/config/validators/test_cross_validators.py` + `src/tests/integration/api/test_save_config.py`.

### ARCH-7. Plugin schema каталог — single-source-of-truth
Catalog лежит в `community/<kind>/<plugin_id>/manifest.toml`. Фронт получает через API. UI catalog не хранит state — просто render from API. Никаких client-side кэшей манифестов дольше 60 сек.

### ARCH-8. Observability
- **Logging:** в `DatasetValidator._run_plugins()` и `EvaluationRunner._run_evaluators()` уже логируется id/name плагина. Ничего нового не нужно.
- **MLflow tags:** при запуске пайплайна писать `plugin.<kind>.<instance_id>.plugin_id = <manifest.id>` + `plugin.<kind>.<instance_id>.version = <manifest.version>`. Проверить наличие — если нет, add в Фазу 1.
- **Метрика:** exception-rate per-plugin через существующую `PipelineMetrics` — никаких новых табло.

### ARCH-9. Security
- **Secrets** (ARCH-3): `secret=true` → UI рендерит `<input type="password">`, значения никогда не логируются, YAML хранит `${ENV_VAR}` reference, не plain value.
- **Manifest load**: `toml.load` безопасен. `importlib.util.spec_from_file_location` — загружает произвольный Python код. Это known-trust модель: плагины community/ считаются доверенными (в репе). Third-party маркетплейс — out-of-scope.
- **CORS, CSRF:** без изменений, UI на том же домене.

### ARCH-10. Reliability + Rollback
- **Optimistic UI с rollback** (FE-10): любое изменение (add/remove/reorder/configure) пишется в local form state, затем async save. Если backend возвращает ошибку — UI откатывает на предыдущий snapshot через `useProjectConfig.undoBuffer`.
- **Undo для destructive действий** (FE-1): toast с 5-сек таймаутом, snapshot хранится в memory до таймаута.
- **Config versions** существующие: `useConfigVersions` — версия до изменений всегда доступна для rollback через UI «Revert to previous version».

---

# Deep-think: контракт плагинов (пункт 10 ТЗ пользователя)

Пользователь явно попросил: «добавить контракты/интерфейсы плагинов, чтобы удобно настраивать параметры/тресхолды, настройки и задавать описание полям, по аналогии с обычным конфигом, но правильно — тут необходим дипсинк анализ текущего кода и предложить варианты».

## Принципы контракта (SOLID/KISS/YAGNI)

- **SOLID/SRP:** `ParamFieldSchema` описывает ОДНО поле. `ParamsSchema` описывает группу полей. Не смешиваем.
- **KISS:** один формат. Без hybrid/auto-detect/magic. Автор пишет по одной схеме — трансформер всегда работает одинаково.
- **YAGNI:** не добавляем conditional-fields, discriminated-unions, nested-objects. Никому из 33 наших плагинов это не нужно. Когда понадобится — расширим формально, не спекулятивно.
- **Explicit:** `supported_strategies` — обязательное поле для reward-плагинов. Никаких «пусто = любые»; автор обязан перечислить явно.

## Единственный формат (финальный)

Flat dict полей. Каждое поле — `ParamFieldSchema`:

```toml
[plugin]
id = "avg_length"
kind = "validation"

[params_schema.sample_size]
type = "integer"           # integer | number | string | boolean | enum
min = 1
max = 1_000_000
default = 10_000
title = "Sample size"
description = "Rows drawn for the length check. More → slower, more accurate."
required = false

[params_schema.validation_backend]
type = "enum"
options = ["compile", "semantic_only"]
default = "compile"
title = "Validation backend"
description = "`compile` runs the full HelixQL compiler; `semantic_only` skips codegen."
```

Pydantic-модель `ParamFieldSchema` с `extra="forbid"` — любое незнакомое поле = ошибка загрузки манифеста. Никакой «магической толерантности».

## Transform в JSON Schema — детерминированно

```python
def field_to_json_schema(f: ParamFieldSchema) -> dict:
    node = {"type": f.type if f.type != "enum" else "string"}
    if f.title: node["title"] = f.title
    if f.description: node["description"] = f.description
    if f.default is not None: node["default"] = f.default
    if f.min is not None: node["minimum"] = f.min
    if f.max is not None: node["maximum"] = f.max
    if f.options: node["enum"] = f.options
    return node

def params_to_json_schema(fields: dict[str, ParamFieldSchema]) -> dict:
    return {
        "type": "object",
        "properties": {k: field_to_json_schema(v) for k, v in fields.items()},
        "required": [k for k, v in fields.items() if v.required],
        "additionalProperties": False,
    }
```

Валидации на уровне `ParamFieldSchema`:
- `required=True` + `default` not None → ValueError (R6).
- `type="enum"` без `options` → ValueError.
- `min > max` → ValueError.
- `default` вне `options` / `[min, max]` → ValueError.

## Контракт — полная сводка

| Раздел | Поле | Тип | Обязательное | Описание |
|---|---|---|---|---|
| `[plugin]` | id | str | ✅ | уникальный id плагина |
| `[plugin]` | kind | enum | ✅ | validation/evaluation/reward/reports |
| `[plugin]` | name | str | ✅ | display-имя |
| `[plugin]` | version | str | ✅ | semver |
| `[plugin]` | description | str | ✅ | что плагин делает |
| `[plugin]` | supported_strategies | list[str] | ✅ если kind=reward, иначе запрещено | список strategy_type |
| `[params_schema.X]` | type | enum | ✅ | integer/number/string/boolean/enum |
| `[params_schema.X]` | title | str | ❌ | label в UI |
| `[params_schema.X]` | description | str | ❌ | tooltip-описание |
| `[params_schema.X]` | default | any | ❌ (но если нет — required автоматом) | значение по умолчанию |
| `[params_schema.X]` | min/max | number | ❌ (только для integer/number) | границы |
| `[params_schema.X]` | options | list | ✅ если type=enum | допустимые значения |
| `[params_schema.X]` | required | bool | ❌ (default: computed) | явно-обязательное |
| `[params_schema.X]` | secret | bool | ❌ (default: false) | скрывать значение в UI, пробрасывать через env-var (ARCH-3) |

`[thresholds_schema.X]` — та же схема `ParamFieldSchema`.

## Пункт 11 — дефолты + required в UI

- **Required field** без значения → красный `*` в label + error-ring + inline message (FieldRenderer уже так умеет).
- **Optional field** без значения → placeholder с текстом default, значение при save — `null` или default. 
- Tooltip рядом с меткой через `HelpTooltip` — рендерим `description` как help-text.

---

# Принципы и чеклист реализации

## SOLID / KISS / DRY / YAGNI / Boy Scout

| Принцип | Применение в плане |
|---|---|
| **SRP** | `ParamFieldSchema` — 1 поле. `PluginsTab` — отображение + мутации, но не бизнес-логика (она в `generatePluginInstance`, `reorderInstances` — чистые функции). `PluginInfoModal` отдельно от `PluginConfigModal`. |
| **OCP** | Добавление нового kind — новая секция в PluginsTab + entry в registry; без модификации существующего кода. |
| **ISP** | Configure-modal получает только нужные props (`pluginManifest`, `instance`, `onSave`, `onClose`), не весь project-state. |
| **DIP** | Модалка зависит от абстракции `onSave: (instance) => void`, не от `useSaveProjectConfig`. |
| **KISS** | Один формат схемы (flat dict), один transform, один DnD-context, один info-modal с двумя ветками. |
| **DRY** | `PluginChipRow` переиспользуется в Catalog + Palette. `FieldRenderer` — в Configure + ConfigBuilder. `ValidationBanner` + `useFieldStatus` — в обоих. |
| **YAGNI** | Нет conditional-fields, nested objects, plugin dependencies, third-party marketplace, manifest-version. |
| **Boy Scout** | В Фазе 1: обновить README, обновить scaffold (трогаем — приводим в порядок). Не плодим TODO-комментарии. |

## Frontend-принципы

| Принцип | Применение |
|---|---|
| **Composition over inheritance** | `<PluginChipRow />` внутри `<Draggable>` / `<Sortable>` / `<Link>` — поведение композируется, не наследуется. |
| **Controlled components** | Configure-modal — fully controlled, state в parent. Нет uncontrolled inputs. |
| **Accessibility first** | aria-label на всех кнопках, `role="dialog"`, focus-trap, KeyboardSensor для DnD, `aria-live` announcements. |
| **Progressive enhancement** | DnD — primary UX, но «+ Add from catalog» кнопка всегда доступна как fallback (нет DnD → работает). |
| **Predictable state** | TanStack Query — server state; `useState`/`useReducer` — UI state. Нет global store для локальных модалок. |
| **Fail loudly in dev** | Неизвестное `params_schema` поле → console.warn + fallback renderer. Неизвестный kind → fallback component. |
| **Performance** | `React.memo` на `PluginChipRow` (много элементов в Catalog). Virtualize только если >100 плагинов (not now). |
| **Design system reuse** | Переиспользуем atom’ы из `ui.tsx`, chip-палитру из `PresetPickerModal`. Не изобретаем новые. |

## Observability / reliability / rollback

- Manifest-валидация на старте сервера — уже есть в loader, не трогаем.
- Cross-validator strategy↔reward тестируется в 3 точках (R21).
- MLflow tags для инстансов плагинов (ARCH-8).
- Optimistic UI с rollback на save-error (ARCH-10, FE-10).
- Undo-toast на destructive actions (FE-1).
- Config-versions для глобального rollback — существующий механизм.

## Dead-code — удаляем, не плодим

- **НЕ вводим** `manifest_version` «на будущее» — YAGNI.
- **НЕ оставляем** fallback-парсеры для legacy raw-dict формата — breaking change, migrate all 33 manifest.toml в одном коммите Фазы 1.
- **Удаляем** `PluginToggleCard` (checkbox-toggle UI) в Фазе 3 — заменяется на `PluginInstanceRow`.
- **Удаляем** `PluginBrowser.tsx` если он не используется после переработки — проверить в Фазе 2.
- **Удаляем** все упоминания `priority` в `ui_manifest()` если ещё остались из прошлого refactor.

## Что НЕ в плане (explicit out-of-scope)

- Third-party plugin marketplace / signing.
- Nested / discriminated schema fields (YAGNI — никто из 33 плагинов не просит).
- Conditional field visibility.
- Plugin dependencies / load-order contracts.
- Migration framework для schema-эволюции.
- Mobile-first UX (desktop-tool; @dnd-kit TouchSensor — best-effort).
- Live preview «run plugin on sample» (FE-7) — nice-to-have, явно out-of-MVP.
