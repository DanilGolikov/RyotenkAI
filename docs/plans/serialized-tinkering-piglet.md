# Datasets feature — proper UX + backend integration

## Context (зачем это нужно)

Сейчас в проекте RyotenkAI **датасеты технически уже описываются в YAML конфиге** (`datasets.<key>` блок с `source_type: local|huggingface`, путями, валидациями), и `StrategyPhaseConfig.dataset` поле даёт привязку «стратегия → датасет». Но **на фронте нет ни одного UI для работы с датасетами** — пользователь вынужден руками править YAML, чтобы поменять путь, добавить плагин-валидатор или хотя бы посмотреть, как выглядят данные.

Цель — закрыть эту дыру: показать датасеты как первоклассный объект проекта (отдельный таб, master-detail, превью с пагинацией, on-demand валидация с подсветкой битых полей, привязка к стратегии через picker).

Просто говоря: пользователь должен иметь возможность открыть проект → вкладка Datasets → увидеть все датасеты, их пути, превью первых N строк, нажать «Validate» и сразу увидеть, где данные не подходят выбранной стратегии (например DPO ждёт `chosen` + `rejected`, а в файле только `text`). Сейчас единственный способ это узнать — запустить полный pipeline и подождать падения на стадии 0.

## Бизнес-требования и как они ложатся на текущую реализацию

| Требование | Что уже есть | Что добавить |
|---|---|---|
| Отдельная вкладка Datasets | — | Новый таб в `ProjectDetail.tsx`, master-detail layout |
| 1 стратегия → 1 датасет (sharing — ошибка) | Поле `StrategyPhaseConfig.dataset: str \| None` уже есть. Движок валидации flag-нет 1:N через `_check_dataset_format` | UI помечает коллизию красным до save |
| Auto-create dataset при добавлении стратегии | — | Хук в Training-вкладке: при `addInstance(strategy)` параллельно `addDataset({key: derived_name, source_type: 'local', source_local: {local_paths: {train: ''}}})` |
| Auto-delete dataset при удалении стратегии | — | Симметричный хук на удаление |
| Path validation (local + HF) | — | Endpoint `/datasets/{key}/path-check`: для local — `Path.exists() + line count`, для HF — пробный head-запрос HF API |
| Превью датасета (raw + structured) | `DatasetLoaderFactory` загружает целиком; есть HF streaming | Новый endpoint `/datasets/{key}/preview` с `offset+limit`. Front: 2 режима view (raw jsonl-строки моноспейс + structured table) |
| Подгрузка по скроллу | — | IntersectionObserver на bottom-sentinel + `useInfiniteQuery` |
| Кнопка валидации датасета | `DatasetValidator.execute()` есть, но требует pipeline context | Extract pure-функции из `_check_dataset_format` (line 516) и `_run_plugin_validations` (line 772) в `src/data/validation/standalone.py`. Новый endpoint `POST /datasets/{key}/validate` использует их без GPU/pipeline |
| Per-плагин tabs в Validation секции PluginsTab | Сейчас читается только из `firstDatasetKey()` | Tabbed UI: один таб на dataset key |
| Format-check (соответствие формата стратегии) | `_check_dataset_format` уже вызывает `strategy.validate_dataset(dataset)` | В endpoint `/validate` возвращаем отдельным полем `format_check`. Front рендерит банер «Strategy `dpo` ожидает `chosen, rejected` — отсутствует `rejected`» |
| Dataset selector в Strategy item конфига | Нет UI — поле есть в схеме | Кастомный renderer для пути `training.strategies[*].dataset`: `SelectField` со списком `parsed.datasets` keys + ссылка «Configure → Datasets/{key}» |
| Подсветка битых строк/полей в превью | Backend выдаёт `error_groups[*].sample_indices` | Front: при наличии validation result, помечаем строки превью accent-цветом по плагину; tooltip с «failed by plugin X: error_type Y» |
| Inline-edit одной строки | — | **Фаза B** только: `PATCH /datasets/{key}/row` для local jsonl. Atomic-write (`tmp + os.replace`) + `.bak` |
| Syntax highlight | YamlEditor использует CodeMirror 6 (`@codemirror/lang-yaml`) | **Фаза B**: добавить `@codemirror/lang-json` (lazy) для raw-режима |

## Архитектурные решения (зафиксированы с пользователем)

1. **Strict 1:1 strategy↔dataset.** Sharing — это ошибка валидации (движок уже это enforces в `_check_dataset_format`). Auto-derived имя датасета: `<strategy_id>_dataset` (или коллизия → `<strategy_id>_dataset_2` и тд).
2. **Edit row** — только local jsonl, atomic-write + single `.bak`. HF — read-only с чипом «remote, read-only».
3. **Virtualization** — IntersectionObserver + paginated load (50 rows/page), DOM растёт. Без новых dep.
4. **Phased rollout** — A → B → C в отдельных PR. Каждая фаза самодостаточна и не ломает предыдущие.

## Phased rollout

### Phase A — MVP (большой PR, ~основной объём работы)

#### Backend

**Новые файлы:**

1. `src/api/routers/datasets.py` — три endpoint'а:
   - `GET /api/v1/projects/{project_id}/datasets/{dataset_key}/preview?split=train&offset=0&limit=50`
     → `{rows: list[dict], total: int|null, has_more: bool, schema_hint: list[str]}`
   - `POST /api/v1/projects/{project_id}/datasets/{dataset_key}/validate?split=train` body `{max_samples: int|null}`
     → `{format_check: {strategy_id, ok, missing_fields, message}, plugin_results: list[ValidationResult-shape], plugins_grouped_issues: dict[plugin_id, list[ErrorGroup]], duration_ms: int}`
   - `GET /api/v1/projects/{project_id}/datasets/{dataset_key}/path-check`
     → `{train: {exists, line_count|null, size_bytes|null, error}, eval: {...}}`

2. `src/data/validation/standalone.py` — `LightweightDatasetValidator`:
   - `check_dataset_format(dataset, strategy_phases) → FormatCheckResult` (pure-функция, **переиспользует** `StrategyFactory.create_from_phase` + `strategy.validate_dataset`)
   - `run_plugins(dataset, plugin_configs, parallel=True, timeout_s=60) → list[ValidationResult]` (использует `ValidationPluginRegistry`, ловит exceptions per-plugin)

3. `src/data/preview/loader.py` — `DatasetPreviewLoader`:
   - `preview_local_jsonl(path, offset, limit) → (rows, total, has_more)` — `itertools.islice` + `mtime`-cached line count (LRU 32)
   - `preview_hf(repo_id, split, offset, limit) → (rows, total=None, has_more)` — `load_dataset(streaming=True).skip().take()`. Token берётся из существующего `Secrets.hf_token`

**Refactor (без поломки поведения):**

- `src/pipeline/stages/dataset_validator.py:516` (`_check_dataset_format`) — вынести pure-логику в `standalone.check_dataset_format`. Метод класса делегирует. Тесты pipeline должны остаться зелёными.
- `src/pipeline/stages/dataset_validator.py:772` (`_run_plugin_validations`) — то же самое для `standalone.run_plugins`.

**Регистрация:**

- `src/api/main.py:57-71` — добавить `app.include_router(datasets.router)`.
- `src/api/dependencies.py` — новый dep `resolve_dataset_key(project_id, dataset_key) → DatasetConfig`. Защита: `Path(dataset.source_local.local_paths.train).resolve()` должна быть **внутри** `project_root` ИЛИ внутри явного `allowed_dataset_dirs` whitelist (берём из настроек). Reuse паттерна `resolve_run_dir` (там же).

#### Frontend

**Новые компоненты** (`web/src/components/Datasets/`):

| Файл | Роль |
|---|---|
| `DatasetsTab.tsx` | Master-detail wrapper |
| `DatasetList.tsx` | Sidebar list (name, source-type chip, status pill) |
| `DatasetDetail.tsx` | Header (key, source-type, чип «auto-created»/«manual») + form + preview + validation panel |
| `DatasetSourceFields.tsx` | Toggle local vs HF + paths/repo input. Использует `LabelledRow` + `INPUT_BASE` (новый стиль) |
| `DatasetPreviewPane.tsx` | Mode switcher (raw/structured), IntersectionObserver pagination |
| `PreviewRowRaw.tsx` | Mono jsonl-строка с accent-классом для bad rows |
| `PreviewRowStructured.tsx` | Table cell row, schema_hint header |
| `FormatCheckBanner.tsx` | «Strategy `dpo` expects `chosen, rejected` — found ✓/✗» |
| `ValidationResultsPanel.tsx` | List плагин-результатов, expand → error_groups → click → scroll к row в превью |

**Новые hooks** (`web/src/api/hooks/`):

- `useDatasetsList(projectId)` — derived from `parsed.datasets`, не отдельный fetch (config уже cached в react-query)
- `useDatasetPathCheck(projectId, key)` — useQuery, refetch on focus + after path edit (debounced)
- `useDatasetPreview(projectId, key, split)` — `useInfiniteQuery`, key `['dataset-preview', projectId, key, split]`, `getNextPageParam` из `has_more`
- `useDatasetValidation(projectId, key)` — `useMutation`, on success — `invalidateQueries(['dataset-path-check', projectId, key])`, обновляет status pill в list

**Утилитарный hook (для auto-coupling):**

- `web/src/components/ProjectTabs/pluginInstances.ts` уже паттерн mutate parsed — расширить: `addStrategyAndDataset(parsed, strategyManifest) → next` и `removeStrategyAndDataset(parsed, strategyId) → next`. Чистые функции, тестируются юнитами. **Вызываются из `ConfigTab` после `applyFormChange`** при детектировании add/remove в `training.strategies[]`.

**Модификации существующих файлов:**

- `web/src/pages/ProjectDetail.tsx:15-22` — добавить `{to: 'datasets', label: 'Datasets'}` в TABS после `plugins`. Добавить `<Route path="datasets/*" element={<DatasetsTab projectId={project.id} />} />` в Routes (~259).
- `web/src/components/ConfigBuilder/FieldRenderer.tsx` (`CUSTOM_FIELD_RENDERERS` map ~38-47) — добавить `'training.strategies.*.dataset': DatasetSelectField` (новый компонент). Рендерит `SelectField` со списком keys из `parsed.datasets` + кнопку «Configure» (router push `/projects/{id}/datasets/{key}`).
- `web/src/api/openapi.json` — после генерации regenerate через `npm run gen:api`.

#### Schema

**Минимальное** (опциональное, backward-compat):

- `src/config/datasets/schema.py:DatasetConfig` — добавить `auto_created: bool = False` (metadata-флажок для UI: «можно безопасно удалить вместе со стратегией»). Default = False, старые YAML-ы не ломаются.

### Phase B — Edit + syntax highlight

#### Backend

- Новый endpoint `PATCH /api/v1/projects/{project_id}/datasets/{dataset_key}/row?split=train&offset=N` body `{value: dict, etag: string}`.
- `src/data/preview/editor.py` — `JsonlRowEditor.update_row(path, line_index, new_value, expected_etag)`:
  - ETag = `f"{mtime_ns}-{size}"` — optimistic concurrency
  - Atomic: write `path.with_suffix('.tmp')` → `os.replace(path, path + '.bak')` → `os.replace(tmp, path)` (single backup, перезаписывается)
  - Advisory lock (`fcntl.flock` или `filelock` lib)
  - **Reject HF datasets** → 412
  - На 409 (etag mismatch) фронт invalidate-ит preview и re-открывает edit-форму с актуальным значением

#### Frontend

- `PreviewRowStructured.tsx` — добавить «edit» mode: row становится формой (CodeMirror JSON minimal или textarea с JSON.parse-валидацией), Save/Cancel
- `useUpdateDatasetRow` mutation hook → optimistic update в react-query cache, на 409 — refetch + show conflict toast
- Lazy import `@codemirror/lang-json` — добавить в `web/package.json`. Wrap существующий `YamlEditor` в более общий `CodeEditor` с `language` prop (или дублировать минимально). Bundle hit ~150KB — load on demand при первом edit-mode mount

### Phase C — Per-dataset plugin tabs

#### Backend

Изменений нет — всё уже поддерживается через `apply_to` в `DatasetValidationPluginConfig`.

#### Frontend

- `web/src/components/ProjectTabs/PluginsTab.tsx` — добавить `activeDatasetKey: string` state в Validation section
- Новый компонент `DatasetKeyTabs.tsx` — tab-strip с keys из `parsed.datasets`. Стилистика как в `ProjectDetail` tabs (violet underline)
- `pluginInstances.ts` — все mutate-функции (validation kind) принимают `datasetKey: string` как arg. Существующий `firstDatasetKey()` остаётся как fallback для совместимости (single-dataset проектов)
- Migration: при mount если active не выбран → set `firstDatasetKey()`

## Risks & open questions (3 итерации deep-think)

### Iteration 1 — очевидное

| # | Risk | Mitigation |
|---|---|---|
| R1 | Heavy line count для JSONL >1GB блокирует UI | Async background count, `total=null` пока считается, UI показывает «counting…» pill. Cache по `(path, mtime)` LRU |
| R2 | HF streaming cold start 5-15s | Spinner + AbortController; in-process loader cache per `(repo_id, split)` TTL 5 min |
| R3 | Format-check требует strategy_obj | Reverse-lookup: пробежать `parsed.training.strategies[]`, найти phases где `phase.dataset == key`, передать их списком. Если 0 — skip + info-banner. Если N>1 c РАЗНЫМИ `strategy_type` — error «strict 1:1 violation» (uses business rule) |
| R4 | Validation медленная на полном датасете | `max_samples` cap (default 1000), UI checkbox «full validation» для полного прогона |
| R5 | Auto-coupling race: user меняет `strategy.dataset` name → старый dataset осиротел | UI: при rename strategy modal «Rename dataset key too? / Keep both?». Default — rename. Если keep both — оба остаются, на коллизию ссылок (другая strategy теперь без dataset) — banner |

### Iteration 2 — concurrency, edge cases, rollback

| # | Risk | Mitigation |
|---|---|---|
| R6 | Concurrent edit (Phase B) — два браузера, одна row | ETag = `mtime_ns-size`, 409 Conflict, invalidate preview, conflict toast |
| R7 | Edit corrupts file (power loss между tmp и replace) | `os.replace` atomic on POSIX. `.bak` хранится до следующего успешного edit. Future: «Restore from backup» endpoint (out of MVP) |
| R8 | Strategy delete cascades dataset delete + user undo | Parsed config — single source of truth. Undo через config history (existing pattern в Project state, через snapshot YAML files). Atomic patch parsed config |
| R9 | JSONL schema drift (row 1: `{a,b}`, row 50: `{a,c}`) | Structured view header — union первых 100 rows. «Other» column для unknown fields с JSON-rendered value |
| R10 | IntersectionObserver fast-scroll → offset > total | Backend `rows: []`, `has_more: false`. UI «end of dataset» footer |
| R11 | Validation plugin throws unhandled | `LightweightDatasetValidator.run_plugins` обёрнут в per-plugin try/except, returns ValidationResult с `passed: false, errors: ["Plugin crashed: …"]`. ThreadPoolExecutor future timeout 60s |
| R12 | HF private/gated repo | Catch `huggingface_hub.errors.GatedRepoError` → 403 «Configure HF token in Settings → Integrations» |

### Iteration 3 — observability, security, perf

| # | Risk | Mitigation |
|---|---|---|
| R13 | Path traversal: user указал `../../etc/passwd` в local_paths.train | `resolve_dataset_key` dependency: `Path(...).resolve()` должна быть под `project_root` или в `allowed_dataset_dirs`. Reuse pattern `resolve_run_dir` (`src/api/dependencies.py`) |
| R14 | HF token утекает в logs/responses | Scrub middleware на error responses; log filter redact `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` |
| R15 | Memory pressure: row 10MB nested object × 200 limit | Row size cap 1MB serialized; truncate с `__truncated: true` marker. UI «(truncated)» badge |
| R16 | Validation request hangs | `asyncio.wait_for` 60s на endpoint level, ThreadPoolExecutor `future.result(timeout=60)` |
| R17 | Нет observability для новых endpoints | Span/structured log: `dataset.preview.duration_ms`, `dataset.validate.plugin_count/error_count`. Через existing logger |
| R18 | Backup growth (Phase B) — нужен ли rotate? | Single `.bak` overwrite, document в UI tooltip. YAGNI: не делаем rotate в MVP |
| R19 | JSONL with BOM / CRLF | Open `encoding='utf-8-sig'`, strip `\r`. Test fixture |
| R20 | CodeMirror lang-json bundle 150KB (Phase B) | Lazy dynamic import только при первом mount edit-mode |
| R21 | Auto-derived name collisions | Strategy IDs unique enforced. `<strategy_id>_dataset` 1:1 → unique. Renaming strategy → коллизия проверяется UI до save |

### Open questions (need answer per phase, not blocking now)

- **OQ1**: При format-check, если `strategy.dataset == None` (default) — какой dataset считается «default»? Текущий `PipelineDatasetMixin.get_dataset_for_strategy` берёт primary. Для UI — нужно явное правило: показывать в Datasets tab «default» как отдельный sentinel? Решение в Phase A: если `dataset is None` → используем литерал `"default"` как key. Если в `parsed.datasets` нет ключа `default` → создаём при первой стратегии без явного `dataset`.
- **OQ2**: Плагины-валидаторы могут требовать секреты (`DTST_*` env). На фронте мы их не показываем сейчас. UI «Show secret refs» — отложено в Phase C+.
- **OQ3**: Multi-source dataset (один key с двумя файлами train + eval) — превью переключается split-табом (train/eval). MVP — да, уже учтено в endpoint.

## What MUST NOT break

- `ConfigBuilder` generic flow — special-cases для dataset selector добавляются через `CUSTOM_FIELD_RENDERERS` (уже есть pattern, см. `training.provider`)
- `PluginsTab` — Phase A не трогает; Phase C заменяет `firstDatasetKey()` на `activeDatasetKey ?? firstDatasetKey()` (backward-compat для single-dataset)
- `DatasetValidator` pipeline-stage — продолжает работать; Phase A только extract pure-функций, делегирование, существующие тесты pipeline должны быть зелёными
- Старые YAML-конфиги — `auto_created: bool = False` опционально, default = False
- `/api/v1` namespace — новые endpoints под `/projects/{id}/datasets/...`, изолированы

## Tests

### Backend unit (`src/tests/unit/`)
- `data/validation/test_standalone.py` — `check_dataset_format`, `run_plugins` (mock plugins throw/return)
- `data/preview/test_loader.py` — local jsonl pagination, edge cases (offset > total, empty file, BOM, CRLF, malformed JSON line, нюансы HF streaming через mocks)
- `data/preview/test_editor.py` (Phase B) — atomic write, backup creation, ETag conflict, lock contention
- `api/routers/test_datasets.py` — endpoint contract: preview, validate, path-check; path traversal blocked (returns 403); HF gated repo → 403 с user-friendly message
- `pipeline/stages/test_dataset_validator_refactor.py` — non-regression: после extraction `_check_dataset_format` и `_run_plugin_validations` поведение pipeline-стадии идентично

### Frontend unit (`web/src/**/__tests__/`)
- `Datasets/__tests__/DatasetPreviewPane.test.tsx` — IntersectionObserver triggers next page, error states, mode switch
- `Datasets/__tests__/FormatCheckBanner.test.tsx` — render missing fields
- `components/ProjectTabs/__tests__/pluginInstances.test.ts` — `addStrategyAndDataset` / `removeStrategyAndDataset`, auto-create on add, auto-delete on remove, rename collision

### E2E (manual чек-лист с preview MCP, MVP не требует Playwright)
- Создать dataset через Datasets tab → появление в Training strategy selector
- Add strategy в Training tab → автогенерация dataset entry в Datasets tab
- Run validation → bad rows highlighted в preview, click на error_group → scroll к row
- Phase B: edit row → save → reload preview, видно обновлённую row, `.bak` существует на диске

## Verification (как проверить end-to-end)

1. **Backend smoke**: `pytest src/tests/unit/data/validation/test_standalone.py src/tests/unit/data/preview/test_loader.py src/tests/unit/api/routers/test_datasets.py -v`
2. **Pipeline non-regression**: `pytest src/tests/unit/pipeline/stages/test_dataset_validator_comprehensive.py -v`
3. **Type check**: `cd web && npm run lint` (= `tsc --noEmit`)
4. **Visual end-to-end** через preview MCP:
   - Стартовать `web-frontend` + `web-backend`
   - Создать тестовый проект (POST `/api/v1/projects`)
   - Открыть `/projects/{id}/datasets` → пустой state
   - В Training tab добавить SFT strategy → dataset auto-created → виден в Datasets tab
   - В Datasets detail указать local jsonl path → preview раскрылся → IntersectionObserver подгружает страницы → нажать Validate → format_check banner показывает соответствие SFT-формату
   - Скриншоты для финального отчёта

## Critical files for implementation

### Backend new
- `src/api/routers/datasets.py`
- `src/data/validation/standalone.py`
- `src/data/preview/loader.py`
- `src/data/preview/editor.py` (Phase B)

### Backend modify
- `src/api/main.py:57-71` — register new router
- `src/api/dependencies.py` — `resolve_dataset_key` dep
- `src/pipeline/stages/dataset_validator.py:516,772` — extract pure-функции
- `src/config/datasets/schema.py` — optional `auto_created: bool = False`

### Frontend new
- `web/src/components/Datasets/*.tsx` (9 компонентов, см. таблицу выше)
- `web/src/api/hooks/useDatasetPreview.ts`
- `web/src/api/hooks/useDatasetValidation.ts`
- `web/src/api/hooks/useDatasetPathCheck.ts`

### Frontend modify
- `web/src/pages/ProjectDetail.tsx:15-22` — TABS + Routes
- `web/src/components/ConfigBuilder/FieldRenderer.tsx:38-47` — `CUSTOM_FIELD_RENDERERS` add `training.strategies.*.dataset`
- `web/src/components/ProjectTabs/pluginInstances.ts` — `addStrategyAndDataset`, `removeStrategyAndDataset` хуки
- `web/src/components/ProjectTabs/PluginsTab.tsx` — Phase C: tabs per dataset

## Phase split summary

- **Phase A PR** = всё backend + Datasets tab + auto-coupling + preview + one-off validation + format-check + bad-rows highlight. Большой, но необходимый минимум для пользы.
- **Phase B PR** = edit row + syntax highlight (Раз эту фичу можно отложить и она требует доп. зависимости — отдельный PR).
- **Phase C PR** = per-dataset tabs в PluginsTab. Изолированный фронт-only PR.
