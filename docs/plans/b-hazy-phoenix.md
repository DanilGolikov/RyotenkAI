# Variant B — kubectl-style CLI refactor (no backward-compat, atop RESEACRH)

> **Status: implemented end-to-end (Phase 0 → Phase 5).** Final test
> snapshot: 5051 passed, 9 pre-existing failed (helixql plugin
> registration — outside CLI scope), 121 skipped, coverage 83.48 %
> (gate `fail_under = 83`). All 56 `(noun, verb)` pairs render
> `--help` cleanly; 3 contract-tests assert CLI ↔ API parity for the
> read paths most likely to drift (`runs ls`, `runs inspect`, `preset ls`).


## Context

CLI (`src/main.py`, ~1300 LOC, 17 команд) родился под TUI как «толстый launchpad». TUI давно заменили полноценным Web (`src/api/` + `web/`), но CLI остался плоской свалкой команд с непоследовательным синтаксисом, частично сломанным JSON-режимом, обманными глобальными флагами и тяжёлой bash-обёрткой `run.sh`. Параллельный мир Web/API оперирует понятием `Project` — в CLI его нет; backend для plugin/preset apply в API уже есть, но из CLI недоступен; `scripts/batch_smoke.py` живёт мимо Typer.

**Базовая ветка — `RESEACRH`** (имя с опечаткой в репо). HEAD по состоянию на 2026-04-26 = `332daea` (268 коммитов поверх `main`). На ней уже произведён массивный refactor плагинов и большая структурная перестройка `src/`, которая **ускоряет план B**:

**Plugin platform (готово, переиспользуем):**
- `src/community/registry_base.py` — `PluginRegistry[T]` единый базовый класс для validation/evaluation/reward/reports.
- `src/community/preflight.py` (455 LOC) + `POST /plugins/preflight` — pre-launch гейт missing envs + instance-shape ошибок.
- `src/community/stale_plugins.py` — детекция мёртвых ссылок на плагины в YAML.
- `src/community/manifest.py` — `LATEST_SCHEMA_VERSION = 4`, `RequiredEnvSpec` single-source, snake_case enforcement, `inference.py` для auto-fill манифеста.
- `src/community/instance_validator.py` — валидация YAML инстансов плагинов против их JSON-Schema.
- `src/cli/plugin_scaffold.py` — **уже введён** `plugin_app` (Typer sub-app под `plugin scaffold`), задаёт направление к `ryotenkai plugin <verb>`.
- `src/cli/community.py` — добавлена команда `community sync-envs` (cross-check `REQUIRED_ENV` ClassVar ↔ manifest TOML).
- TUI окончательно удалён.

**Структурный refactor `src/` (свежие 39 коммитов, 2026-04-26):**
- **NEW namespace `src/workspace/`** — `projects/`, `providers/`, `integrations/` + generic `WorkspaceRegistry/Store` базы. Бывшие `src/pipeline/project`, `src/pipeline/settings/{providers,integrations}` переехали сюда.
- **NEW `src/api/presentation/`** — `STATUS_ICONS`, `STATUS_COLORS`, `format_duration`, `format_mode_label` (бывший `src/pipeline/presentation`). View-helpers для API/CLI.
- **NEW `src/api/state_cache.py`** — process-local mtime-keyed `StateSnapshot` cache (бывший `src/pipeline/state/cache.py`). Read-only, для частого опроса state.
- **NEW `src/api/ws/live_tail.py`** — `LiveLogTail` (offset-based file tail). Domain-agnostic, pure-python — **переиспользуем для `runs logs --follow` и `run start --tail`** вместо self-rolled threading.
- **`src/pipeline/launch_queries.py` → `src/pipeline/launch/restart_options.py`** (consolidated с `restart_points.py`). Все CLI-импорты `load_restart_point_options`/`pick_default_launch_mode`/`validate_resume_run`/`resolve_config_path_for_run` идут из нового пути.
- **`src/pipeline/_fs.py` → `src/utils/atomic_fs.py`** — `atomic_write_json` теперь в utils. CLI `cli_state/context_store.py` использует его для атомарной записи `cli-context.json`.
- **`DatasetValidator` (Wave 4)**: stage `pipeline/stages/dataset_validator.py` (985 LOC) разбит на пакет `dataset_validator/{stage,format_checker,plugin_loader,plugin_runner,split_loader,artifact_manager,constants}.py`. **Behaviour change** (commit e51c151): «drop hidden default plugins — validation must be explicit». Влияет на UX `dataset validate` (см. NR-04).
- **`TrainingDeploymentManager` (Wave 3)**: 1427 → ~30 LOC фасад + extracted `managers/deployment/{code_syncer,dependency_installer,file_uploader,training_launcher,provider_config,ssh_helpers}.py`. Не касается CLI напрямую.
- **`pyproject.toml`**: `[tool.coverage.report] fail_under = 83` (baseline 85.35%, 2 pp буфер). Каждая фаза рефактора **должна** оставаться выше порога.
- Прочее (внутреннее, без влияния на CLI): `executor/ → execution/`, `inference/engines/` collapse, `pipeline/providers/` shim drop, `pipeline/validation/` drop, `pipeline/domain/run_context` → `pipeline/state/run_context`, split `pipeline/constants.py` per-stage.

**Решения, утверждённые пользователем (2026-04-25 / 2026-04-26):**
- Скоуп: **полный B** — `runs rm`, `project ls/use/create/rm/run/env`, `plugin install` входят в первый рефактор.
- `run.sh`: **удаляем сразу**, README учит `pip install -e . && ryotenkai`.
- Полировка: `-o yaml` рендерер, `project use <id>` + `~/.ryotenkai/cli-context.json`, `--remote URL` (hidden, NotImplementedError-stub), contract-тесты CLI ↔ API parity.
- **БЕЗ обратной совместимости**: `src/cli/deprecated.py` НЕ создаём, `*_legacy.py` тестов НЕТ, старые имена команд удаляются вместе со старым кодом, `run.sh` и `scripts/batch_smoke.py` тоже удаляются полностью.

**Цель:** перевести CLI на kubectl-style `noun verb`, разнести команды по `src/cli/commands/<noun>.py`, чтобы CLI и API использовали один pure-python сервисный слой (`src/api/services/` уже pure — подтверждено `grep`-ом, см. resolved Q-08), включить project-as-first-class, расширить plugin/preset toolkit до полного жизненного цикла включая preflight/stale, удалить `run.sh`, перенести `batch_smoke.py` внутрь Typer, зафиксировать `--remote` под будущий remote-mode.

**Срок:** 3 недели разработки + 0.5 недели на documentation pass.

---

## Целевая поверхность CLI

```
ryotenkai run     start | resume | restart | interrupt | restart-points
ryotenkai runs    ls | inspect | logs | status | diff | report | rm
ryotenkai config  validate | show | explain | schema
ryotenkai dataset validate
ryotenkai project ls | show | use | create | rm | env | run
ryotenkai plugin  ls | show | scaffold | sync | sync-envs | pack | validate | install | preflight | stale
ryotenkai preset  ls | show | apply | diff
ryotenkai smoke   <dir> [--workers --idle-timeout --dry-run]
ryotenkai server  start | status | stop
ryotenkai version
```

**Глобальные флаги (root callback):** `-o text|json|yaml`, `-q/-v/-vv`, `--no-color`, `--remote URL` (hidden), `--project ID`, `--timeout`. `--dry-run` — только на write-командах.

**Удаляются** (без deprecation): `train`, `train-local`, `validate-dataset`, `config-validate`, `runs-list`, `inspect-run`, `run-status`, `run-diff`, `logs`, `report`, `list-restart-points`, `info`, `community {scaffold,sync,sync-envs,pack}` (переносится в `plugin`/`preset`), `serve` (переезжает в `server start`).

---

## Архитектура

### Структура каталогов

```
src/
  cli/
    app.py                       # NEW: Typer root, callback, регистрация sub-Typers
    commands/                    # NEW package, 10 файлов по 100-150 LOC
      __init__.py                # register_all(app)
      run.py        runs.py      config.py    dataset.py
      project.py    plugin.py    preset.py    smoke.py
      server.py     version.py
    common_options.py            # NEW: общие Annotated[Path, typer.Option(...)] (см. Q-25)
    context.py renderer.py errors.py style.py formatters.py version.py run_rendering.py
                                 # существующая инфра — НЕ трогать (renderer обогащается YamlRenderer)
    plugin_scaffold.py           # DELETE: содержимое мигрирует в commands/plugin.py
    community.py                 # DELETE: scaffold/sync/sync-envs/pack уезжают
                                 # в commands/plugin.py + commands/preset.py
  cli_state/
    context_store.py             # NEW: ~/.ryotenkai/cli-context.json (current_project_id)
  community/
    install.py                   # NEW (~150 LOC): clone git/unzip/copy → community/<kind>/
    validate_manifest.py         # NEW (~50 LOC): standalone manifest validation
                                 # (пере-использует Pydantic-модели manifest.py без import плагина)
  main.py                        # collapses to `from src.cli.app import app`
```

**Сервисный слой остаётся в `src/api/services/`.** Аргумент: модули pure-python (HTTPException живёт только в `src/api/routers/`, подтверждено `grep -rn "fastapi\|HTTPException\|Depends\|Request,\|BackgroundTasks" src/api/services/` — единственный «hit» это `LaunchRequest` Pydantic-схема, не FastAPI). Перемещение даст rename-diff без выгоды. CLI импортирует напрямую: `from src.api.services.run_service import list_runs`. Косметический rename `src/api/services/ → src/services/` отложен на post-cutover.

### Что переиспользуем (НЕ дублируем)

| CLI команда | Сервис / query (готово) |
|---|---|
| `runs ls` | [src/api/services/run_service.py](src/api/services/run_service.py) `list_runs()` + [src/pipeline/run_queries.py](src/pipeline/run_queries.py) `scan_runs_dir()` |
| `runs inspect` | `run_service.get_run_detail()` + `RunInspector.load()` |
| `runs status` / `runs diff` | `run_queries.effective_pipeline_status`, `diff_attempts` |
| `runs rm` | [src/api/services/delete_service.py](src/api/services/delete_service.py) `delete_run()` |
| `runs logs` | [src/api/services/log_service.py](src/api/services/log_service.py) `fetch_logs()` + `tail_lines()` |
| `runs logs --follow` / `run start --tail` | [src/api/ws/live_tail.py](src/api/ws/live_tail.py) `LiveLogTail` (offset-based; **переиспользуем**, не пишем своё) |
| `runs report` | [src/api/services/report_service.py](src/api/services/report_service.py) + `ExperimentReportGenerator` |
| `run start/resume/restart/interrupt/restart-points` | [src/api/services/launch_service.py](src/api/services/launch_service.py) + [src/pipeline/launch/restart_options.py](src/pipeline/launch/restart_options.py) (бывший `launch_queries.py`) |
| `config validate` | [src/api/services/config_service.py](src/api/services/config_service.py) `validate_config()` |
| `project *` | [src/api/services/project_service.py](src/api/services/project_service.py) (15 методов) → [src/workspace/projects/](src/workspace/projects/) `ProjectRegistry/Store` |
| Стилизация text-renderer | [src/api/presentation/](src/api/presentation/) `STATUS_ICONS`, `STATUS_COLORS`, `format_duration`, `format_mode_label` (общая палитра CLI ↔ Web) |
| Атомарная запись `cli-context.json` | [src/utils/atomic_fs.py](src/utils/atomic_fs.py) `atomic_write_json` |
| `plugin ls/show` | [src/community/catalog.py](src/community/catalog.py) `CommunityCatalog.plugins()/get()` |
| `plugin scaffold/sync/sync-envs/pack` | [src/cli/plugin_scaffold.py](src/cli/plugin_scaffold.py) (мигрировать) + [src/community/scaffold.py](src/community/scaffold.py), [sync.py](src/community/sync.py), [pack.py](src/community/pack.py) |
| `plugin preflight` | [src/community/preflight.py](src/community/preflight.py) (RESEACRH) |
| `plugin stale` | [src/community/stale_plugins.py](src/community/stale_plugins.py) `find_stale_plugins()` |
| `plugin validate` | NEW `src/community/validate_manifest.py` поверх `manifest.py` Pydantic |
| `plugin install` | NEW `src/community/install.py` |
| `preset ls/show/apply/diff` | `catalog.presets()` + [src/community/preset_apply.py](src/community/preset_apply.py) `apply_preset()` |

---

## Фазы реализации

### Phase 0 — Перебазирование на RESEACRH (S, ~0.5 дня)
**Действия:**
- `git fetch origin && git rebase RESEACRH` (HEAD = `332daea` на 2026-04-26; перед стартом фазы перепроверить `git log origin/RESEACRH | head -1`).
- Перезапустить тесты на чистой базе → подтвердить, что ничего не падает (особенно учитывая coverage-gate `fail_under=83`, см. NR-05).
- **Запретить мердж в `RESEACRH` других CLI-фич на время рефакторинга** (feature-freeze CLI, см. Q-16). Свежие 39 коммитов касались pipeline/workspace, не CLI — продолжать так же.
- Зафиксировать актуальный baseline coverage до рефактора: `pytest --cov=src --cov-report=term | grep TOTAL` → записать в начало Phase 1.

### Phase 1 — Core prep (S, ~1.5 дня)
**Файлы:**
- NEW [src/community/install.py](src/community/install.py) (~150 LOC) + unit-тесты:
  - `install_local(source: Path, kind, *, force=False)` — copy / move папки внутрь `community/<kind>/<id>/`, валидация манифеста.
  - `install_archive(zip_path: Path, kind, *, force=False)` — extract + validate.
  - `install_git(url: str, *, ref: str, kind, allow_untrusted=False)` — clone в tmp + checkout commit-sha (см. Q-22), копирование, validate. `--ref` обязателен; tag/branch без `--allow-untrusted` отклоняется.
- NEW [src/community/validate_manifest.py](src/community/validate_manifest.py) (~50 LOC) + тесты:
  - `validate_manifest_file(path) -> ValidationResult` — TOML-парс + Pydantic-валидация без import класса плагина (отделена от `loader.py`). Проверяет `schema_version <= LATEST_SCHEMA_VERSION` (v5+ → понятная ошибка, см. Q-13).
- EDIT [src/api/services/__init__.py](src/api/services/__init__.py) — re-export курированной публичной поверхности (CLI не лазит в private-функции).
- NEW [src/cli_state/context_store.py](src/cli_state/context_store.py) (~80 LOC) + тесты:
  - `get_current_project()`, `set_current_project(pid)`, `clear_current_project()`.
  - Файл `${RYOTENKAI_HOME:-~/.ryotenkai}/cli-context.json`, lazy load.
  - **Запись через `from src.utils.atomic_fs import atomic_write_json`** (см. NR-11) — никаких ad-hoc `Path.write_text(json.dumps(...))`.
- NEW [src/cli/common_options.py](src/cli/common_options.py) — переиспользуемые `Annotated[...]` константы (`ConfigOpt`, `RunDirArg`, `OutputOpt`, `ProjectOpt`, ...) (см. Q-25).

**Не трогаем CLI.** Существующий тест-suite остаётся зелёным.

### Phase 2 — Новая CLI-поверхность (L, ~6 дней)
**Файлы:**
- NEW [src/cli/app.py](src/cli/app.py) — root Typer с callback и глобальными флагами.
- NEW [src/cli/commands/](src/cli/commands/) — 10 файлов:
  - `run.py` — start/resume/restart/interrupt/restart-points.
  - `runs.py` — ls/inspect/logs/status/diff/report/rm.
  - `config.py` — validate/show/explain/schema.
  - `dataset.py` — validate.
  - `project.py` — ls/show/use/create/rm/env/run.
  - `plugin.py` — ls/show/scaffold/sync/sync-envs/pack/validate/install/**preflight**/**stale**.
  - `preset.py` — ls/show/apply/diff.
  - `smoke.py` — порт `scripts/batch_smoke.py` целиком в Typer (см. Phase 3).
  - `server.py` — start/status/stop (бывший `serve`).
  - `version.py`.
- EDIT [src/cli/renderer.py](src/cli/renderer.py) — добавить `YamlRenderer` (~30 LOC, симметрично `JsonRenderer` с буферизацией). `_yaml_default` хелпер для `Path/datetime`.
- EDIT [src/main.py](src/main.py) — collapse до `from src.cli.app import app`.
- DELETE [src/cli/community.py](src/cli/community.py), DELETE [src/cli/plugin_scaffold.py](src/cli/plugin_scaffold.py) — содержимое перенесено в `commands/plugin.py` + `commands/preset.py`.

**Поведение глобальных флагов:**
- `-v/-vv/-q` → `logging.getLogger().setLevel(...)` в callback'е (precedence: flag > env `LOG_LEVEL` > INFO).
- `--no-color` → существующий `CLIContext.use_color` ([src/cli/context.py:34](src/cli/context.py)) экспонируем как root option.
- `--remote URL` → hidden, при не-пустом значении сохраняется в `CLIContext.remote`, любая команда вызывает `raise NotImplementedError("remote mode lands in v1.2; see roadmap")`. Стаб-маркер.
- `--dry-run` → только write-команды: `run start/resume/restart/interrupt`, `runs rm`, `plugin install/sync/sync-envs/pack`, `preset apply`, `project create/rm`, `project use` (write контекст-файла).

**Все команды используют `Renderer` — `typer.echo` запрещён.** Это лечит сегодняшний дефект: `train`, `validate-dataset`, `report`, `serve`, `logs` не уважали `-o json`.

### Phase 3 — Удаление shell-обвязки (M, ~2 дня)
**Файлы:**
- DELETE [run.sh](run.sh).
- DELETE [scripts/batch_smoke.py](scripts/batch_smoke.py) — функционал переехал в `src/cli/commands/smoke.py` 1-в-1, ENV `RYOTENKAI_RUNS_DIR` сохранён (см. Q-10).
- EDIT [Makefile](Makefile) — `$(RYOTENKAI) config-validate` → `$(RYOTENKAI) config validate`; удалить мёртвый таргет `tui` (TUI выпилен в RESEACRH).
- DELETE существующих тестов на старые имена ([src/tests/unit/test_main_cli.py](src/tests/unit/test_main_cli.py), [src/tests/unit/cli/test_read_commands.py](src/tests/unit/cli/test_read_commands.py)) — без сохранения, без `*_legacy.py`.
- NEW тесты на новые имена в тех же файлах (или в `test_main_cli.py` как один большой набор).
- EDIT [src/tests/unit/community/test_cli_community.py](src/tests/unit/community/test_cli_community.py) — переписать под `plugin scaffold/sync/sync-envs/pack` + `preset apply/diff`.

**`tail -F pipeline.log`** (бывший трюк `run.sh:375`) → `run start --tail` (default ON когда stdout — TTY и не `-o json`). Реализация: orchestrator уже пишет в известный путь `runs/<id>/attempts/N/logs/pipeline.log` (через `LogLayout` — подтверждено в helixir-памяти про per-stage log files). CLI после старта подпроцесса спавнит `threading.Thread`, который в цикле зовёт `LiveLogTail(path).read_new_lines()` ([src/api/ws/live_tail.py](src/api/ws/live_tail.py)) с 200 ms sleep — без self-rolled file-poll логики (см. NR-03). В `finally` block'е поток останавливается через флаг + `join(timeout=1)`. Для `--detach` не тейлим, печатаем только run_id.

### Phase 4 — Документация (S, ~0.5 дня)
**Файлы:**
- EDIT [README.md](README.md), [i18n/*/README.md](i18n/) — таблица команд переписана на noun-verb. Удалить любые упоминания `./run.sh` и старых имён.
- NEW `docs/cli/` — отдельная страница на noun (run.md, runs.md, project.md, plugin.md, preset.md, config.md, dataset.md, smoke.md, server.md).
- EDIT [docs/web-ui.md](docs/web-ui.md) — обновить упоминание `inspect-run`.
- EDIT [CONTRIBUTING.md](CONTRIBUTING.md) — секция «authoring plugins» использует `ryotenkai plugin scaffold/sync/sync-envs/pack/validate`.

**Smoke-проверка:** `grep -rE "python -m src\.main|./run\.sh|ryotenkai (train|inspect-run|runs-list|run-status|run-diff|validate-dataset|config-validate|list-restart-points|community)" docs/ README.md i18n/` возвращает пусто.

### Phase 5 — Contract-тесты + project context + remote stub + smoke gate (M, ~2 дня)
**Файлы:**
- NEW `src/tests/contract/test_cli_api_parity.py` — для каждой пары:
  - `runs ls` ↔ `GET /runs`
  - `runs inspect` ↔ `GET /runs/{id}`
  - `config validate` ↔ `POST /config/validate`
  - `plugin ls --kind X` ↔ `GET /plugins/{kind}`
  - `plugin preflight <cfg>` ↔ `POST /plugins/preflight`
  - `preset ls` ↔ `GET /config/presets`

  Поднимать FastAPI in-process через `httpx.AsyncClient(transport=ASGITransport(app))`, прогонять CLI на одну `RYOTENKAI_RUNS_DIR` фикстуру, сравнивать через `_normalize(payload)` хелпер (приводит ISO → UTC, режет ms, сортирует list-of-dict по `id`, см. Q-23). ~200 LOC + общие фикстуры.
- NEW `src/tests/contract/conftest.py` — autouse фикстура `_clean_state_cache` (см. NR-02): `from src.api.state_cache import _cache; _cache.clear()` до и после каждого теста, чтобы process-local cache API не давал stale-hit относительно raw-чтения CLI.
- NEW `src/tests/smoke/test_cli_help.py` — параметризованный e2e-тест: каждая top-level команда + каждая её sub-command с `--help` → exit 0 (см. Q-30). Ловит import-time errors при росте `commands/`.
- VERIFY `--remote URL` stub: тест что любая команда с `--remote http://x` падает с `NotImplementedError` и понятным сообщением.
- VERIFY `project use` flow: `set_current_project → get_current_project → resolve_project(--project flag > env > stored)` (см. Q-19).

---

## Verification

### Локально
```bash
pip install -e .
ryotenkai --help                                          # 9 групп команд видно
ryotenkai version -o json
ryotenkai runs ls -o yaml                                  # global -o yaml работает
ryotenkai run start --config config/test.yaml --dry-run    # write+dry-run
ryotenkai project use my-proj && ryotenkai project run     # context flow
ryotenkai plugin ls --kind validation -o json
ryotenkai plugin preflight config/test.yaml                # exit 0/1/2/3 (Q-21)
ryotenkai plugin stale config/test.yaml --remove --dry-run
ryotenkai plugin install ./local-plugin --kind validation
ryotenkai plugin install --git https://... --ref <sha> --allow-untrusted --kind reward
ryotenkai preset apply 04-sft-quickstart --dry-run
ryotenkai smoke community/presets --workers 2 --dry-run
ryotenkai run start --config x.yaml --remote http://api   # NotImplementedError
```

### Тесты
```bash
pytest src/tests/unit/cli/                                  # переписанные unit
pytest src/tests/unit/community/                            # plugin/preset toolkit
pytest src/tests/contract/test_cli_api_parity.py            # zero-drift gate
pytest src/tests/smoke/test_cli_help.py                     # все --help pass
ruff check src/cli/ src/community/install.py src/community/validate_manifest.py
mypy src/cli/
```

### Documentation gate
```bash
! grep -rE "python -m src\.main|./run\.sh" docs/ README.md i18n/
! grep -rE "ryotenkai (train|inspect-run|runs-list|run-status|run-diff|validate-dataset|config-validate|community)" docs/ README.md i18n/
```

### Repowise update
После Phase 5: вызвать `repowise update` — wiki видит новый layout `src/cli/commands/`.

---

## Critical files to modify / delete / create

**Create:**
- `src/cli/app.py`, `src/cli/commands/{run,runs,config,dataset,project,plugin,preset,smoke,server,version}.py`
- `src/cli/common_options.py`
- `src/cli_state/context_store.py`
- `src/community/install.py`, `src/community/validate_manifest.py`
- `src/tests/contract/test_cli_api_parity.py`, `src/tests/smoke/test_cli_help.py`
- `docs/cli/*.md`

**Modify:**
- `src/main.py` — collapse to re-export.
- `src/cli/renderer.py` — `+ YamlRenderer`.
- `src/api/services/__init__.py` — public re-export.
- `Makefile`, `README.md`, `i18n/*/README.md`, `docs/web-ui.md`, `CONTRIBUTING.md`.

**Delete (no shim, no deprecation):**
- `run.sh`
- `scripts/batch_smoke.py`
- `src/cli/community.py`
- `src/cli/plugin_scaffold.py`
- `src/tests/unit/test_main_cli.py` (заменён на новый набор)

---

## Risks & resolutions (3 итерации анализа + 3 итерации дипсинка)

Формат: `Q-N` — risk/question, `A` — резолюция после дипсинка.

| # | Level | Risk / Question | Resolution |
|---|-------|------------------|------------|
| Q-01 | HIGH | Базовая ветка для рефакторинга — main или RESEACRH? | **A:** RESEACRH. Phase 0 = rebase. Backend плагинов уже сделан там, ребейз сэкономит 1+ неделю работы. Worktree пока чистая → ребейз без конфликтов. |
| Q-02 | HIGH | На RESEACRH уже есть `src/cli/plugin_scaffold.py` с `plugin_app`. Конфликт имён? | **A:** Нет конфликта. План удаляет `plugin_scaffold.py` и переносит его команду `scaffold` внутрь нового `commands/plugin.py`. `plugin_app` создаётся заново как `typer.Typer` в `commands/plugin.py`. |
| Q-03 | MED | Что с `community sync-envs` (RESEACRH-новинка)? | **A:** Становится `plugin sync-envs`. Семантика та же (cross-check `REQUIRED_ENV` ClassVar ↔ manifest TOML). Регистрируется отдельной command'ой, не подкомандой `sync` — она не bump'ит версию. |
| Q-04 | HIGH | `run.sh` пропадает — кто скриптовал `./run.sh`? | **A:** По требованию пользователя — без grace period. Обновляем README одновременно с удалением. CHANGELOG в Phase 4 явно перечислит миграцию. |
| Q-05 | HIGH | `scripts/batch_smoke.py` пропадает — кто его звал извне? | **A:** Тоже без shim. Если внешние CI используют — переключаются на `ryotenkai smoke <dir>` (одинаковый ENV `RYOTENKAI_RUNS_DIR` и аргументы). |
| Q-06 | MED | Тесты на старые имена нужно ли сохранить как regression? | **A:** Нет. Удаляем целиком, пишем заново на новые имена. Никаких `*_legacy.py`. |
| Q-07 | MED | Документация (README, i18n, docs) — частичная или полная переписка? | **A:** Полная — `grep` в Phase 4 не должен находить ни одного старого упоминания. |
| Q-08 | ✅ closed | Сервисы `src/api/services/` могут импортировать FastAPI. | **A:** Подтверждено grep'ом — не импортируют (единственный hit — Pydantic-схема `LaunchRequest`). CLI зовёт services напрямую. |
| Q-09 | LOW | Frontend (`web/src/`) ссылается на CLI команды? | **A:** Нет (проверено). Только `web/src/api/schema.d.ts` содержит OpenAPI operationId `runs-list_runs` — это API-роут, не CLI; не трогаем. |
| Q-10 | HIGH | `RYOTENKAI_RUNS_DIR` ENV-контракт нужен `web/scripts/_common.sh` и `src/config/runtime.py`. | **A:** ENV-имя сохраняется. `src/cli/commands/smoke.py` читает ту же переменную, остальные потребители (web scripts, config/runtime.py) не трогаются. Regression-тест в Phase 3. |
| Q-11 | MED | На RESEACRH есть `src/community/preflight.py` — нужна CLI-команда. | **A:** Добавлена `plugin preflight <config>` → зовёт `preflight.preflight_config()`. Exit codes: `0=OK`, `1=missing envs`, `2=instance shape errors`, `3=catalog/load errors`. `--strict` поднимает warnings до errors. |
| Q-12 | MED | На RESEACRH есть `src/community/stale_plugins.py` — нужна CLI-команда. | **A:** Добавлена `plugin stale <config>` → зовёт `find_stale_plugins()`. Default: read-only, печатает таблицу. `--remove` (write, требует `--dry-run`/подтверждение) — удаляет stale-ссылки из YAML. |
| Q-13 | LOW | `LATEST_SCHEMA_VERSION = 4`. Что если манифест на v5? | **A:** `validate_manifest_file()` явно сверяет `schema_version <= LATEST_SCHEMA_VERSION`, на v5+ возвращает `ValidationResult` с понятным «upgrade RyotenkAI». `plugin validate` — exit 4. |
| Q-14 | HIGH | `plugin install <git-url>` — supply-chain risk. | **A:** Локальные пути и `.zip` — без флага. Git URLs — обязательно `--git URL --ref <commit-sha>`. Tag/branch (а не sha) и любой git без `--allow-untrusted` отклоняем. Документировано в `docs/cli/plugin.md` security-блоке. |
| Q-15 | MED | In-process FastAPI ASGI для contract-тестов: lifespan/CORS не сломают? | **A:** `src/api/main.py:create_app(settings)` принимает `ApiSettings` — фикстура передаёт `runs_dir=tmp_path` без `os.environ` мутаций (которые делает `src/api/cli.py:run_server`). CORS middleware не мешает (origins не используются в transport-режиме). `httpx.AsyncClient(transport=ASGITransport(app))` поднимает приложение. |
| Q-16 | HIGH | Параллельные мерджи в RESEACRH во время рефакторинга. | **A:** Feature-freeze CLI на 3 недели через owner-уведомление. Любые not-CLI фичи мерджатся как раньше; CLI-PR — только этот рефактор. |
| Q-17 | MED | `train --tail` — надо подтвердить, что orchestrator пишет log в известный путь. | **A:** Подтверждено (helixir-память + `LogLayout`): `runs/<id>/attempts/N/logs/pipeline.log`. CLI поллит этот путь после старта подпроцесса. |
| Q-18 | LOW | `-o yaml` для `Path/datetime` нужен сериализатор. | **A:** `_yaml_default` параллельно `_json_default` ([src/cli/renderer.py:179](src/cli/renderer.py)) — те же правила (`__fspath__` → str, `isoformat` → str). |
| Q-19 | HIGH | `project use` vs Web URL-state — рассинхрон? | **A:** Не пытаемся синхронизироваться. Документируем: Web хранит контекст в URL, CLI — в `~/.ryotenkai/cli-context.json`. Это две независимых сессии; рассинхрон безопасен (`--project` flag всегда выигрывает). |
| Q-20 | MED | `project run` должен пробрасывать project env. | **A:** Зовёт `project_service.get_project_env(pid)` → результат передаёт в orchestrator через те же контракты, что `POST /launch` (без дублирования). Уже работает в API. |
| Q-21 | LOW | `plugin preflight` exit codes. | **A:** Зафиксированы (см. Q-11). Документированы в `docs/cli/plugin.md`. |
| Q-22 | MED | `plugin install --git --ref <branch>` vs commit-sha. | **A:** Только commit-sha без `--allow-untrusted`. Branch/tag разрешены явным флагом `--allow-untrusted`. Проверка через `git rev-parse <ref>` после clone — если выдаёт ту же sha, что передана, OK. |
| Q-23 | LOW | Contract-тесты CLI ↔ API — разные форматы datetime / порядок ключей. | **A:** Helper `_normalize(payload)`: ISO → UTC, ms режется, list-of-dict сортируется по `id`/`run_id`/`name`, `Path` → str. ~30 LOC, переиспользуется во всех contract-тестах. |
| Q-24 | MED | `community/registry_base.py` — singleton state в pytest. | **A:** Pytest-фикстура `clean_catalog()` зовёт `catalog.reload()` в `setUp`. Регистрируется как autouse в `src/tests/contract/conftest.py` и `src/tests/unit/cli/conftest.py`. |
| Q-25 | HIGH | Дублирование подписей опций (`--config -c` в 3+ командах). | **A:** `src/cli/common_options.py` — модуль с переиспользуемыми `Annotated[...]` константами: `ConfigOpt`, `RunDirArg`, `OutputOpt`, `ProjectOpt`, `KindOpt`. Каждая команда импортирует и применяет. |
| Q-26 | LOW | Typer subcommand discovery при росте `commands/`. | **A:** `commands/__init__.py:register_all(app)` явно `app.add_typer(run_app, name="run")` для каждого. Лучше явный реестр, чем magic-import. |
| Q-27 | MED | `ASGITransport` глобальный state (`os.environ` в `run_server`). | **A:** Contract-тесты используют `create_app(test_settings)` напрямую, минуя `run_server`. `os.environ` НЕ мутируется. |
| Q-28 | MED | `plugin install` для локальной папки vs git. | **A:** API: `install <path>` (auto-detect dir/zip), `install --git URL --ref SHA`. 90% случаев — локальная папка, без git дополнительных требований. |
| Q-29 | LOW | Манифесты на v3 schema — что показывает `plugin validate`? | **A:** v3 принимается без warning'а (loader всё равно его понимает, см. `manifest.py:LATEST_SCHEMA_VERSION` history). При желании — `--strict` поднимает v3 как warning «consider upgrading to v4». |
| Q-30 | MED | E2e smoke для CLI — как поймать import-time errors. | **A:** `src/tests/smoke/test_cli_help.py` — параметризованный по всем `(noun, verb)` парам, прогоняет `--help`, проверяет exit 0. Стоит дёшево, ловит регрессии при росте `commands/`. |

### Новые риски от свежего RESEACRH (39 коммитов, 2026-04-26)

| # | Level | Risk / Question | Resolution |
|---|-------|------------------|------------|
| NR-01 | HIGH | Импорт-пути в плане устарели после структурного refactor: `pipeline.launch_queries` / `pipeline.project` / `pipeline.settings.{providers,integrations}` / `pipeline.state.cache` / `pipeline.presentation` / `pipeline.run_inspector` / `pipeline.restart_points` / `pipeline._fs` — всё переехало или удалено. | **A:** Все ссылки в плане обновлены на новые пути: `pipeline.launch.restart_options`, `workspace.{projects,providers,integrations}`, `api.state_cache`, `api.presentation`, `utils.atomic_fs`. Удалённые модули (`run_inspector`, `restart_points`, `pipeline.validation`) больше не упоминаются. Перед стартом Phase 2 — `grep -rn "pipeline.launch_queries\|pipeline.project\|pipeline.settings\|pipeline.state.cache\|pipeline.presentation\|pipeline._fs" src/cli/` должен возвращать пусто. |
| NR-02 | MED | `src/api/state_cache.py` — process-local mtime-keyed LRU. CLI читает state через `PipelineStateStore` напрямую (без cache), API через `load_state_snapshot()` (с cache). Contract-тест на `runs ls/inspect` рискует получить cache-hit с устаревшим snapshot и ложно-падать на сравнении с raw-чтением CLI. | **A:** В `src/tests/contract/conftest.py` autouse-фикстура `_clean_state_cache` зовёт `_cache.clear()` до и после каждого теста. Также при использовании `httpx.ASGITransport(app)` — клиент работает в том же процессе, что и тесты, поэтому cache виден. Документировать в docstring conftest'а. |
| NR-03 | MED | План имел self-rolled `threading.Thread` + `time.sleep(0.2)` + ручной seek/readlines для `tail -F`. Дублирует уже готовый `LiveLogTail` (offset-based, в `src/api/ws/live_tail.py`). | **A:** В `commands/run.py:start --tail` и `commands/runs.py:logs --follow` импортируем `from src.api.ws.live_tail import LiveLogTail`. Threading-обёртка вокруг него (~30 LOC), без переписывания offset-механики. Один шаблон для CLI и Web WebSocket-стрима. |
| NR-04 | HIGH | Behaviour change на RESEACRH: `DatasetValidator` больше не подгружает hidden default plugins — «validation must be explicit» (commit e51c151). Если конфиг не объявил `[validation.plugins]`, команда `dataset validate` молчаливо отчитается «всё ок». Это вводит пользователя в заблуждение. | **A:** В `commands/dataset.py:validate` — pre-check: если `cfg.validation.plugins` пуст → exit 2 с сообщением «no validation plugins configured. Add `[[validation.plugins]]` in your config or run `ryotenkai plugin ls --kind validation` to discover available ones.» Документировать в `docs/cli/dataset.md` + добавить unit-тест на «пустой validation» → exit 2. |
| NR-05 | HIGH | `pyproject.toml` ввёл coverage-gate `fail_under = 83` (baseline 85.35%, 2 pp буфер). CLI рефактор без аккуратности легко опустит метрику ниже порога — особенно если новые `commands/<noun>.py` написаны до тестов. | **A:** Coverage-чеклист в каждой фазе: <br>• Phase 1: новые модули (`install.py`, `validate_manifest.py`, `context_store.py`) — каждый ≥ 80% покрытия unit-тестами до merge.<br>• Phase 2: каждая `commands/<noun>.py` команда — минимум 1 happy-path + 1 error-path тест.<br>• Phase 5: contract-тесты не считаются за unit-coverage, но дают зелёный E2E.<br>• Перед merge каждой фазы: `pytest --cov=src --cov-fail-under=83`.<br>Пользоваться pytest-cov diff-coverage'ом (если доступен) для проверки ровно новых строк. |
| NR-06 | LOW | Дубль `STATUS_ICONS` в `src/api/presentation/icons.py` (для UI/web text) и `src/cli/style.py:Icons` (для terminal Rich). | **A:** Не консолидируем сейчас — у них разные runtime-цели (UI palette vs terminal-aware). Если `text-renderer` CLI хочет «как в Web» — импортирует из `src/api/presentation/`. Полную унификацию (вынос в `src/utils/status_icons.py`) — в отдельный PR после рефактора (бэклог). |
| NR-07 | MED | `src/workspace/projects/__init__.py` re-exports `ProjectMetadata`, `ProjectRegistry`, `ProjectRegistryEntry`, `ProjectStore`. Errors живут в `registry.py` (`ProjectRegistryError`, `validate_project_id`) и `store.py` (`ProjectStoreError`). CLI команды `project *` должны импортировать оттуда, не из `src.pipeline.project` (которого больше нет). | **A:** `commands/project.py` импортирует только через `src.api.services.project_service` (фасад). Прямой доступ к `src.workspace.projects.*` запрещён в CLI — через сервис. Сервис уже мигрирован (см. diff `src/api/services/project_service.py`). |
| NR-08 | LOW | `src/pipeline/launch/__init__.py` экспортирует `LaunchRequest`, `spawn_launch_detached`, `interrupt_launch_process`, `LaunchMode/Status/Result` — удобно для `run start --detach` и `run interrupt`. | **A:** `commands/run.py:start --detach` зовёт `launch_service.launch(..., detached=True)` (через сервисный слой, не напрямую `spawn_launch_detached`). `run interrupt` зовёт `launch_service.interrupt_launch(run_dir)`. CLI не зависит от `pipeline.launch` напрямую. |
| NR-09 | MED | Decomposition `dataset_validator` stage (985 → 7 файлов) — внутреннее изменение orchestrator'а, CLI его не зовёт напрямую. | **A:** No impact на план: `dataset validate` зовёт orchestrator, который сам собирает stage. Регрессии — на стороне pipeline-тестов. |
| NR-10 | LOW | Decomposition `TrainingDeploymentManager` (Wave 3) — внутреннее. | **A:** No impact на план B. |
| NR-11 | MED | `src/pipeline/_fs.py` → `src/utils/atomic_fs.py`. Запись `cli-context.json` должна использовать `atomic_write_json` для consistent semantics (POSIX rename → нет corrupt-при-crash). | **A:** `src/cli_state/context_store.py:set_current_project()` импортирует и использует `from src.utils.atomic_fs import atomic_write_json`. Тест: симулировать `KeyboardInterrupt` посреди записи — файл должен остаться валидным или вообще не появиться (атомарность). |
| NR-12 | MED | `src/pipeline/state/run_context.py` (бывший `pipeline/domain/run_context`). Сервисы зависят от `RunContext`. | **A:** Импорты в сервисах уже обновлены автоматически в RESEACRH. CLI зовёт сервисы — RunContext в публичный CLI-API не утекает. |
| NR-13 | HIGH | 39 коммитов прилетели за 1 день — высокий темп изменений на RESEACRH. К моменту старта Phase 0 ветка может уйти ещё дальше. | **A:** Phase 0 = атомарная операция в один день (rebase + smoke). Если на момент старта Phase 0 RESEACRH сдвинется ещё → повторить fetch+rebase. Запрет CLI-PR на 3 недели работает только для CLI-зон (`src/main.py`, `src/cli/`, `run.sh`, `scripts/batch_smoke.py`); другие зоны идут как обычно. |
| NR-14 | LOW | `pipeline/inference/engines/` collapse, `pipeline/executor/ → execution/` — внутренние перестановки. | **A:** No impact на CLI. |
| NR-15 | MED | `pipeline/restart_points.py` (отдельный модуль, 64 LOC) удалён, его функциональность мигрирована в `pipeline/launch/restart_options.py` через `compute_restart_points` из `restart_rules.py`. | **A:** План B уже использует новый путь (`commands/run.py:restart-points` → `load_restart_point_options` из `pipeline.launch.restart_options`). Зафиксировано в таблице переиспользования. |

---

## Open questions (требуют решения до Phase 2)

Только то, что осталось после трёх дипсинк-итераций — узкие технические развилки, которые лучше зафиксировать перед стартом:

1. **Project storage location.** `~/.ryotenkai/cli-context.json` — это user-home или там же где ENV `RYOTENKAI_HOME`? **Предложение:** `${RYOTENKAI_HOME:-~/.ryotenkai}/cli-context.json`. Default — стандартный home.
2. **`runs rm` — soft vs hard delete.** API `delete_run(mode="local_and_mlflow")` поддерживает оба. **Предложение:** CLI default — `local_and_mlflow`, флаги `--local-only` / `--mlflow-only`.
3. **`server start` — daemon mode.** Сейчас `serve` запускает foreground uvicorn. **Предложение:** оставить foreground default; `--daemon` отложить (не входит в B-скоуп).
4. **`run start --tail` — что когда subprocess завершился раньше, чем log-файл создался.** **Предложение:** print «pipeline did not start; see startup output», скопировать stderr.
5. **Совместимость с MLflow в `runs report` --remote.** Когда `--remote` всё-таки заработает, MLflow tracking server должен быть достижим со стороны API, не CLI. **Предложение:** остаётся для v1.2 roadmap.
6. **`runs logs --follow` и `run start --tail` — общий хелпер?** Оба пути используют `LiveLogTail`. Стоит ли вынести общую обёртку (`src/cli/_log_follow.py: follow(path, *, on_line)`) или дублировать ~30 LOC в двух командах? **Предложение:** вынести в `src/cli/_log_follow.py` (~40 LOC) с явным `Stop` callback'ом — DRY + тестируется одной фикстурой.
7. **Где живёт `_normalize(payload)` для contract-тестов?** Используется во всех parity-тестах. **Предложение:** `src/tests/contract/_normalize.py` (внутренний хелпер, не публичный API).
