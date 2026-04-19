# Project Workspace + Config Builder + Plugin Manifests (+ Stage Timeline polish)

## Context

Сейчас в web UI нет «места для эксперимента»: чтобы собрать pipeline run
нужно вручную править YAML, искать плагины по исходникам, не видно defaults
из Pydantic, нет истории config-ов, нет метаданных о плагинах. Вся работа
структурирована вокруг **runs**, а не вокруг **experiment**.

Задача двойная:

**Основное** — добавить вкладку **Projects**: workspace-концепт, который
централизует для эксперимента (a) config с визуальным builder'ом поверх
`PipelineConfig` Pydantic-схемы, (b) историю config-версий (snapshot на
каждый save), (c) список плагинов (validation / reward / evaluation) с
описаниями и suggested params/thresholds, (d) runs данного проекта. Проект
= директория, по умолчанию `~/.ryotenkai/projects/<id>/`, user может задать
абсолютный путь.

**Заодно** — мелкий polish `StageTimeline`: поднять высоту сегмента и
вывести status в центр, status-литералы централизовать в одном файле
(`statusConstants.ts`) — уберём разбросанные literal-строки из
`ActivityFeed.tsx`, `useKpis.ts`, `MiniStageTimeline.tsx`.

## Decisions (подтверждено)

- **Project location**: default `~/.ryotenkai/projects/<id>/`, но user может
  задать любой абсолютный путь при create.
- **Project registry**: `~/.ryotenkai/projects.json` — индекс известных
  проектов (id, name, path).
- **Config builder scope**: **все 7 top-level групп** `PipelineConfig`
  (model, datasets, training, providers, inference, evaluation,
  experiment_tracking), schema-driven из `PipelineConfig.model_json_schema()`.
  Advanced поля свёрнуты по-умолчанию.
- **Versioning**: snapshots per save — `configs/current.yaml` + история в
  `configs/history/<ISO8601>.yaml`. Нет git.
- **Plugin manifests**: добавить `MANIFEST` ClassVar к `BasePlugin` +
  `get_manifest()` classmethod. Seed 2-3 плагина per system с полным
  manifest'ом (остальные вернут минимальный).
- **Out of scope MVP**: dataset upload/preview, config diff viewer,
  plugin upload/packaging, project file export/import, destructive
  project-delete (только unregister).

## Task 1 — StageTimeline polish + status constants

### Files to create
- **web/src/lib/statusConstants.ts** — источник истины:
  - `STATUS_LABELS: Record<Status, string>` (из `StatusPill.LABEL`)
  - `STATUS_LABELS_SHORT: Record<Status, string>` — для compact/micro вариантов
  - `TERMINAL_STATUSES: ReadonlySet<Status>` (из `useKpis.ts`)

### Files to modify
- **web/src/components/StatusPill.tsx** — импорт `STATUS_LABELS` из constants
- **web/src/components/ActivityFeed.tsx** — заменить literal-compares на `TERMINAL_STATUSES`
- **web/src/components/MiniStageTimeline.tsx** — то же
- **web/src/api/hooks/useKpis.ts** — убрать локальный `TERMINAL_STATUSES`, импорт из constants
- **web/src/components/StageTimeline.tsx**:
  - сегмент: `h-8 → h-14` (56px)
  - layout внутри: `flex-col justify-center`, 2 строки — `stage_name` (text-2xs truncate, ink-1/90) и `status-label` (text-2xs, цвет semantic из `statusConstants`/semantic mapping)
  - убрать нижнюю legend (дублирующую)

## Task 2 — Project workspace

### 2.1 Storage layout

```
~/.ryotenkai/                       # created on demand
  projects.json                     # { "projects": [{id, name, path, created_at}] }
  projects/
    <project-id>/                   # default. user can override with absolute path
      project.json                  # { schema_version, id, name, description, created_at, updated_at }
      configs/
        current.yaml                # active working copy
        history/
          2026-04-19T10:30:45Z.yaml
      runs/                         # standard runs layout (reuse PipelineStateStore)
```

### 2.2 Backend — Python

**Create**
- `src/pipeline/project/__init__.py`
- `src/pipeline/project/models.py` — `ProjectMetadata` dataclass, `ProjectConfigVersion` dataclass.
- `src/pipeline/project/store.py` — `ProjectStore`: `create()`, `load()`, `save_config()` (атомарно + snapshot в history/), `list_versions()`, `read_version()`, `current_yaml_text()`.
- `src/pipeline/project/registry.py` — `ProjectRegistry`: load/save `~/.ryotenkai/projects.json`, `register()`, `list()`, `resolve(id)`.
- `src/api/schemas/project.py` — `ProjectSummary`, `ProjectDetail`, `CreateProjectRequest`, `SaveConfigRequest`, `ConfigVersion`, `SaveConfigResponse`.
- `src/api/schemas/plugin.py` — `PluginManifest`, `PluginListResponse`.
- `src/api/services/project_service.py` — бизнес-логика (reuse `config_service.validate_config` для validate).
- `src/api/services/plugin_service.py` — агрегирует manifests из 3 registry.
- `src/api/routers/projects.py` — endpoints (см. 2.4).
- `src/api/routers/plugins.py` — `GET /plugins/{kind}`.

**Modify**
- `src/utils/plugin_base.py` — добавить `MANIFEST: ClassVar[dict | None] = None` + `@classmethod get_manifest(cls) -> dict` (мерджит `name/version/description` с `MANIFEST`; отсутствующие поля — пустышки).
- `src/training/reward_plugins/registry.py`, `src/data/validation/registry.py`, `src/evaluation/plugins/registry.py` — `list_manifests() -> list[dict]` (вызывает discovery, мапит через `get_manifest`).
- `src/training/reward_plugins/plugins/helixql_compiler_semantic.py` — seed `MANIFEST` (пример).
- `src/data/validation/plugins/base/min_samples.py`, `avg_length.py` — seed `MANIFEST`.
- `src/evaluation/plugins/semantic/helixql_semantic_match.py` — seed `MANIFEST`.
- `src/api/main.py` — зарегистрировать `projects.router`, `plugins.router` (в правильном порядке до runs — они не конфликтуют).
- `src/api/dependencies.py` — `get_project_registry()`, `resolve_project(id)` через `Depends`.

### 2.3 Plugin `MANIFEST` format

```python
MANIFEST: ClassVar[dict] = {
  "description": "Compiles HelixQL samples and rewards on compile success.",
  "category": "semantic",                 # for grouping in UI
  "stability": "stable",                  # stable | beta | experimental
  "params_schema": {                      # JSON-Schema-light
    "timeout_seconds": {"type": "integer", "min": 1, "default": 10},
  },
  "thresholds_schema": {
    "min_pass_rate": {"type": "float", "min": 0, "max": 1, "default": 0.95},
  },
  "suggested_params": {"timeout_seconds": 10},
  "suggested_thresholds": {"min_pass_rate": 0.95},
}
```

`get_manifest()` в base-классе собирает финальный dict:
`{id: cls.name, version: cls.version, ...MANIFEST|defaults}`.

### 2.4 API endpoints (под `/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/projects` | список из registry |
| `POST` | `/projects` | body: `{name, path?, description?}` |
| `GET` | `/projects/{id}` | metadata + current config text |
| `DELETE` | `/projects/{id}` | unregister only (файлы не трогаем) |
| `GET` | `/projects/{id}/config` | `{yaml, parsed_json}` |
| `PUT` | `/projects/{id}/config` | save + snapshot |
| `POST` | `/projects/{id}/config/validate` | reuse `config_service.validate_config` на временном файле |
| `GET` | `/projects/{id}/config/versions` | list snapshots |
| `GET` | `/projects/{id}/config/versions/{filename}` | full YAML + parsed JSON |
| `POST` | `/projects/{id}/config/versions/{filename}/restore` | копирует в current.yaml (делает snapshot прежнего) |
| `GET` | `/projects/{id}/runs` | proxy `scan_runs_dir_grouped(project.runs_dir)` |
| `GET` | `/config/schema` | `PipelineConfig.model_json_schema()` + добавить `_required_marks` если нужно нормализовать |
| `GET` | `/plugins/{kind}` | kind ∈ {`reward`, `validation`, `evaluation`} |

### 2.5 Frontend

**Create pages**
- `web/src/pages/Projects.tsx` — grid of `ProjectCard` + "New project" modal.
- `web/src/pages/ProjectDetail.tsx` — tabs: `Config | Versions | Runs | Plugins`.

**Create components**
- `web/src/components/ConfigBuilder/ConfigBuilder.tsx` — root; принимает schema + value, рендерит groups; debounced POST /validate.
- `web/src/components/ConfigBuilder/FormGroup.tsx` — collapsible wrapper.
- `web/src/components/ConfigBuilder/FieldRenderer.tsx` — dispatch по `schema.type` и `$ref`.
  - string/number/boolean/enum → inputs
  - object → nested FormGroup
  - array → list editor
  - union (`anyOf` с discriminator) → type selector + conditional block
- `web/src/components/ConfigBuilder/ValidationBanner.tsx` — ok/warn/fail строки из `ConfigValidationResult`.
- `web/src/components/ConfigBuilder/PluginPicker.tsx` — drawer, список manifests, "Add to evaluation/validation/training block".
- `web/src/components/PluginBrowser.tsx` — grid cards: name, stability tag, description, params/thresholds defaults (readonly view в MVP).
- `web/src/components/ProjectCard.tsx`, `VersionList.tsx`, `NewProjectModal.tsx`.

**Hooks**
- `web/src/api/hooks/useProjects.ts`, `useProject.ts`, `useCreateProject.ts`, `useSaveProjectConfig.ts`, `useProjectConfigVersions.ts`.
- `web/src/api/hooks/useConfigSchema.ts` — cached, fetched once per session.
- `web/src/api/hooks/usePlugins.ts` — per kind.

**Routes**
- `/projects` → `Projects.tsx`
- `/projects/:id` → `ProjectDetail.tsx` (default tab: Config)
- `/projects/:id/:tab?` — tab из URL (config/versions/runs/plugins).
- `web/src/components/Sidebar.tsx` — add nav item «Projects» (иконка папки).

**Types** (append to `web/src/api/types.ts`):
- `ProjectSummary`, `ProjectDetail`, `ConfigVersion`, `PluginManifest`, `PluginKind`, `SaveConfigRequest/Response`.

### 2.6 Tests

**Create**
- `src/tests/integration/api/test_projects.py`:
  - create project at tmp_path → registry updated, dir exists, project.json valid.
  - save config (valid yaml) → current.yaml + history file created.
  - save config (invalid) → 422 с details.
  - versions endpoint → список snapshots.
  - delete → registry unregister, но директория не стёрта.
- `src/tests/integration/api/test_plugins.py`:
  - `/plugins/reward` → вернёт как минимум seed-плагин с полным manifest.
  - `/plugins/validation` → manifest только у seed-плагинов, остальные — baseline.

## Reuse Surface

| Что | Где | Как используем |
|---|---|---|
| `PipelineConfig.model_json_schema()` | Pydantic v2 built-in | Источник формы + required + descriptions |
| `config_service.validate_config(path)` | [src/api/services/config_service.py](../../src/api/services/config_service.py) | Re-use в projects validate endpoint |
| `scan_runs_dir_grouped(runs_dir)` | [src/pipeline/run_queries.py](../../src/pipeline/run_queries.py) | Для /projects/{id}/runs |
| `PipelineStateStore` layout | [src/pipeline/state/store.py](../../src/pipeline/state/store.py) | runs/ внутри project dir остаются совместимы |
| `atomic_write_json` | [src/pipeline/state/store.py](../../src/pipeline/state/store.py) | Для project.json / registry.json записи |
| `BasePlugin` + discovery | [src/utils/plugin_base.py](../../src/utils/plugin_base.py), `plugin_discovery.py` | Extend без breaking change |
| `StatusPill.LABEL` | [web/src/components/StatusPill.tsx](../../web/src/components/StatusPill.tsx) | Переносим в `statusConstants.ts` |
| `useKpis.TERMINAL_STATUSES` | [web/src/api/hooks/useKpis.ts](../../web/src/api/hooks/useKpis.ts) | Переносим туда же |
| `Card / FormGroup` tokens | [web/src/components/ui.tsx](../../web/src/components/ui.tsx), `globals.css` | Builder переиспользует `card`, `btn-primary`, `pill-*` |
| Launch flow | [web/src/components/LaunchModal.tsx](../../web/src/components/LaunchModal.tsx) | В Project runs tab можно кнопку «Launch from current config» → open modal pre-filled с project path |

## Implementation Order

1. **Status constants** + StageTimeline polish (Task 1) — 30 min, sanity-proof для последующих изменений.
2. **Plugin `MANIFEST` extension + registries + endpoints + seed plugins** — изолированно, backend-only; тесты добавить сразу.
3. **Project backend layer**: models, store, registry, service, router, tests.
4. **Frontend scaffolding**: types, hooks, routes, Sidebar entry, `Projects.tsx` + `NewProjectModal`.
5. **ProjectDetail + ConfigBuilder skeleton** (Config tab): top-level groups, enum/string/number fields only.
6. **ConfigBuilder advanced rendering**: union/discriminated, arrays, nested objects, PluginPicker drawer.
7. **Versions tab + restore**.
8. **Runs tab** (reuse existing `ActivityFeed` filtered to project).
9. **Plugins tab** (PluginBrowser).
10. **Commit per step**.

## Verification

1. **Type check**: `cd web && npx tsc --noEmit` → clean.
2. **Build**: `npm run build` → success.
3. **Backend**: `pytest src/tests/integration/api/ -q` — existing + new = green.
4. **Manual preview** (`make web-start`):
   - Click Projects → empty state.
   - Create project «demo» with default path → appears in list, `~/.ryotenkai/projects/demo/` создан с project.json + empty configs/current.yaml.
   - Open project → Config tab, builder рендерит все 7 групп, required-поля помечены.
   - Ввести invalid значение → inline ошибка.
   - Save → появилась запись в Versions, `configs/history/<ts>.yaml` на диске.
   - Plugins tab → видно seed плагины с description, suggested params/thresholds.
   - Runs tab → пусто (до первого launch), но страница грузится без 500.
   - StageTimeline на странице run → высокие сегменты, stage + status видны.
5. **Regression — `/runs/:id`, Overview, Launch** — без визуальных regressions (status pills продолжают работать).

## Phase 2 (out of scope)

- Dataset preview (JSONL/parquet peek).
- Diff viewer между snapshot-ами.
- Git-backed versioning (optional toggle per project).
- Plugin packages / file-uploads (true Grafana-style).
- Project export/import (zip).
- Multi-user / RBAC.
- Drag-to-reorder strategies / evaluators.
- Live syntax-highlighted YAML editor рядом с формой.
