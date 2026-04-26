# План: Job Server — in-pod control plane для удалённого обучения

> Status: **APPROVED — в реализации**
> Author: daniil
> Date: 2026-04-26
> Worktree: `nice-jepsen-07d789` (на base `origin/RESEACRH` после fast-forward 67 commits)
> Supersedes: [`docs/plans/nice-jepsen-runner.md`](./nice-jepsen-runner.md) (research-документ v1)
> Updated: 2026-04-26 после изучения [`b-hazy-phoenix.md`](./b-hazy-phoenix.md) — kubectl-style CLI рефактор уже выполнен.

## Migration policy: NO BACKWARDS COMPATIBILITY

Утверждено пользователем: старый код заменяется в одно касание, без `RYOTENKAI_RUNNER_V2` или подобного feature-flag, без legacy-режима.

Конкретно это значит:
- В Phase 6 `TrainingLauncher` (генератор `start_training.sh` + nohup probes) **удаляется**, не дублируется. Новый код заменяет его in-place.
- `watchdog.sh`, `_touch_pipeline_heartbeat`, marker-probes в `TrainingMonitor` удаляются в той же фазе, что вводит замену (Phase 4 IdleDetector / Phase 6 events).
- Поля `image_name` / `docker_image` в Pydantic-схемах удаляются без deprecation alias — миграция конфигов делается одним прогоном.
- Документация и тесты обновляются вместе с кодом, без legacy-секций.

Цена этой политики — **скоординировать с активными SAPO/GRPO experiments** (см. § 13). Координация уже подтверждена пользователем: окно для миграции открыто.

## CLI noun-verb адаптация (новое знание после merge)

После merge `origin/RESEACRH` (67 коммитов) в репо появилась **готовая kubectl-style CLI** ([607eca9]):
- `src/cli/app.py` — root Typer + global flags (`-o text|json|yaml`, `--project`, **`--remote URL`** уже зарезервирован под v1.2)
- `src/cli/commands/<noun>.py` — один модуль на noun (run, runs, config, dataset, project, plugin, preset, server)
- `register_all(app)` в `src/cli/commands/__init__.py` — регистрация sub-Typers

**Готовые модули, которые становятся backbone для нашего PluginPacker/Unpacker:**
- `src/community/pack.py::pack_community_folder(...)` — упаковка `community/<kind>/<id>/` → ZIP с валидацией manifest. **Используем напрямую** в PluginPacker.
- `src/community/install.py::install_local(...)`, `install_git(...)`, `_extracted_archive(...)` — распаковка/установка плагинов. **Используем напрямую** в PluginUnpacker.
- `src/community/validate_manifest.py` — standalone manifest validator (без import плагина). Полезен на pod side при unpack.

**Адаптация плана:**
- Phase 7 CLI: вместо `src/cli/job.py` создаём `src/cli/commands/job.py` (новый noun рядом с run/runs/...). Регистрируем в `register_all`.
- `src/main.py` НЕ модифицируем — он 17-line re-export `from src.cli.app import app`.
- `--remote URL` global flag уже есть как stub. В v1.0 нашего job-server CLI работает локально (через `--project` контекст). В v1.2 (отдельный план) можно будет включить `--remote` для подключения к удалённому Mac-control-plane.

---

## 1. Context

Удалённое обучение сегодня работает на 3 независимых процессах (Python trainer, bash watchdog, sshd) общающихся через **9 marker-файлов** на `/workspace`. Mac поллит SSH каждые 5 секунд. Нет push-прогресса, нет interactive control, watchdog критическая логика на bash без тестов, sleep-Mac → watchdog убивает pod, plugin runtime закрыт от introspection.

Цель — единый **Job Server** (FastAPI на pod) + **JobClient** (Mac), HTTP/WS поверх SSH `-L` тоннеля, унифицированный docker image.

### Подтверждено пользователем
- Auth single-tenant через SSH-туннель, JWT — позже
- **Detach/reattach** клиента (не "истинный pause"): training продолжается, Mac уходит в сон, возвращается и догоняет события
- Один docker image, поля `image_name` / `docker_image` **прибиты гвоздями**
- Single_node = SSH к удалённому docker-host, никаких локальных запусков
- MLflow остаётся (job-server становится прокси-релеем)
- Транспорт Mac↔pod: WebSocket поверх `ssh -L` (туннель прозрачен)

---

## 2. Что изменилось в `origin/RESEACRH` — полная картина

61 коммит, +23k −6.4k строк, 260 файлов. Я прочитал **все** commit messages с body. Группы изменений:

### 2.1. Wave 3 — `TrainingDeploymentManager` декомпозирован (8 коммитов, 51e36ef → e848efc)

Был god-class 1426 LoC → теперь 97 LoC facade с публичным API из 5 методов:
- `__init__(config, secrets)`
- `workspace` property + `set_workspace(...)`
- `deploy_files(ssh_client, context)`
- `install_dependencies(ssh_client)`
- `start_training(ssh_client, context, provider=None)`

Новые компоненты (`src/pipeline/stages/managers/deployment/`):
- **`code_syncer.py::CodeSyncer`** — rsync `REQUIRED_MODULES` (нет `community/`!) + tar fallback + pycache cleanup
- **`file_uploader.py::FileUploader`** — config + datasets через tar-pipe + SCP fallback. DI: receives CodeSyncer
- **`dependency_installer.py::DependencyInstaller`** — runtime contract checker (`/opt/helix/runtime_check.py`), pull image для single_node
- **`training_launcher.py::TrainingLauncher`** ← владеет генерацией `.env`, `start_training.sh`, nohup-launch, probe loop. ~530 LoC. **Это единственный компонент, который мы заменяем**.
- `provider_config.py` — pure functions (`get_active_provider_name`, `is_single_node_provider`, `get_*_training_cfg`)
- `ssh_helpers.py::build_ssh_opts` — pure function
- `deployment_constants.py` — DEPLOYMENT_* константы переехали из общего `pipeline/constants.py`

Тесты переписаны как 6 файлов в `src/tests/unit/pipeline/stages/managers/deployment/`.

### 2.2. Wave 4 — `DatasetValidator` декомпозирован (5 коммитов, 5e6eaf7 → 90740aa)

`src/pipeline/stages/dataset_validator/` стал package:
- `stage.py::DatasetValidator` — facade (985 → 454 LoC)
- `format_checker.py::FormatChecker` — column-format checks
- `plugin_loader.py::PluginLoader` — instantiation + DTST_ secrets
- `split_loader.py::DatasetSplitLoader` — HF streaming + local file
- `plugin_runner.py::PluginRunner` — per-plugin loop + critical_failures threshold
- `artifact_manager.py::ValidationArtifactManager` — переехал из `pipeline/validation/`

**Хидден-default плагины удалены** (e51c151): пустой `validations.plugins` блок = NO plugins run. Validation теперь explicit.

### 2.3. Plugin Platform refactor (cozy-booping-walrus, ~17 коммитов, 5409584 → cf961bb)

#### Schema v4 + RequiredEnvSpec
- `manifest.toml` `schema_version=4`, `[secrets]` блок **удалён**, всё в `[[required_env]]` ([74beab6])
- Поля: `name`, `description`, `optional`, `secret`, `managed_by`
- `LATEST_SCHEMA_VERSION = 4`, future versions reject с upgrade hint ([5409584])
- snake_case identifier enforced для `params_schema`/`thresholds_schema` field names ([cf17dff])

#### PluginRegistry[T] (382b727)
Один Generic base class, заменил 4 разрозненных registry. Public API: `register_from_community / instantiate / get_class / manifest / list_ids / list_manifests / is_registered / clear`. Single secret-injection через `PluginSecretsResolver` (DTST_ / EVAL_ / RWRD_ / RPRT_).

#### Preflight gate (14c718c, cf9b2e8)
Mandatory **до launch**, через `src/pipeline/bootstrap/pipeline_bootstrap.py` step 1.5:
- `src/community/preflight.py::run_preflight(config, secrets, project_env) → PreflightReport(ok, missing_envs, instance_errors)`
- API: `POST /api/v1/plugins/preflight`
- `MissingEnv` (kind, plugin, instance, env name, secret, managed_by)
- `InstanceValidationError` через `Draft7Validator.iter_errors` (`src/community/instance_validator.py`)
- `LaunchAbortedError` собирает обе категории

#### REQUIRED_ENV cross-check (03f4382)
- `BasePlugin.REQUIRED_ENV: ClassVar[tuple[RequiredEnvSpec, ...]]`
- Loader делает cross-check Python ClassVar ↔ TOML manifest
- CLI `ryotenkai community sync-envs <plugin>` — переписывает manifest из ClassVar

#### Loader hardening (7b32d47, d3100d8)
- `LoadFailure` / `LoadResult` / `PresetLoadResult` — структурированные ошибки вместо silent skip
- `error_type`: `manifest_parse | kind_mismatch | import_error | missing_yaml | metadata_error`
- `COMMUNITY_STRICT=1` → fail-fast mode (CI / dev)
- API: `PluginListResponse.errors: list[PluginLoadError]`

#### Runtime helpers (337a4dd)
- `BasePlugin._env(name, default=None)` — `_injected_env` (project env.json) → `os.environ`. Empty string = unset.
- `BasePlugin._secret(name)` — required secret или raise. **Не падает на os.environ** — manifest должен быть source of truth.
- `priority` field удалён везде — execution order = config YAML order

#### Stale plugin detection (cf961bb)
- `src/community/stale_plugins.py::find_stale_plugins(config) → list[StalePluginRef]`
- `StalePluginRef`: kind, plugin_name, instance_id, dotted location
- API: `ConfigResponse.stale_plugins: list[StalePluginEntry]`
- UI: amber banner + "Remove from config" per row

#### Reward broadcast log (cf961bb)
В [src/training/reward_plugins/factory.py](src/training/reward_plugins/factory.py) при каждом instantiate:
```
[REWARD_PLUGIN] strategy=<...> plugin='<id>' params=[<key list>]
```
**Param values НЕ логируются** (могут содержать секреты).

#### Plugin scaffold CLI (fc1aeca)
`ryotenkai plugin scaffold <kind> <id>` — создаёт `community/<kind>/<id>/` с manifest@v4, plugin.py, README, tests/. Reward kind emits `supported_strategies = []` с TODO.

#### Reports refactor (10070f2, e6a966c)
`ReportPlugin.plugin_id` / `.title` теперь `@property` из `cls._community_manifest` (manifest single-source). `order` per-instance. 13 community/reports/* manifests refreshed.

#### Test matrix (d976d8f, dad32e5, 7343825, 69b37be)
- D2 fixtures: `tmp_community_root`, `make_plugin_dir`, `mock_catalog`, `fake_secrets`
- 7-category coverage: positive / negative / boundary / invariants / dependency-errors / regressions / logic-specific / combinatorial
- Frontend: Vitest+RTL harness, 112 component tests

### 2.4. Phase 6 — Workspace umbrella (8 коммитов, 96069d7 → 6f5754d)

`src/workspace/` финальное место для **user-scoped configuration** (не runtime!):
- `_registry_base.py::WorkspaceRegistry[EntryT]`, `WorkspaceStore[MetadataT, ConfigVersionT]` — Generic bases
- `projects/` — registry, store, models (per-project `configs/`, `env.json`, history)
- `providers/` — reusable provider credentials
- `integrations/` — HF/MLflow/custom integrations + encrypted token blob

`src/utils/atomic_fs.py` — переехал из `src/pipeline/_fs.py`. Утилиты: `atomic_write_json`, `atomic_write_text`, `utc_now_iso`, `snapshot_filename`, `unique_snapshot_path`, `created_at_from_filename`.

`src/api/ws/live_tail.py::LiveLogTail` — переехал из `pipeline/live_logs.py`. Offset-based byte cursor для tail файла.

### 2.5. Pipeline cleanup (множественные коммиты)

- `src/pipeline/launch/` — `RunLockGuard` (823d3b4) + `restart_options.py` (consolidated, 889f913) + `runtime.py`
- `src/pipeline/state/` — `PipelineStateStore`, `RunContext` (move from domain/, f329b87), `state/queries.py::first_unfinished_stage` (89314ed)
- `src/pipeline/execution/` — `executor/` влит сюда (201e086): `stage_planner`, `stage_registry`, `stage_execution_loop`, `restart_inspector`
- `src/pipeline/inference/` — `engines/` collapsed (15c1928): `vllm.py` напрямую
- `src/pipeline/stages/` — облегчён: `__init__` экспортит только `StageNames`, `CANONICAL_STAGE_ORDER`, `PipelineContextKeys` (861d010). Heavy modules больше не тянутся при CLI invocation.
- `src/pipeline/constants.py` — split per-stage: `deployment_constants.py`, `dataset_validator/constants.py`, `model_retriever/constants.py`. EVAL_* dead block удалён.
- `src/api/state_cache.py` — переехал (2563b34). Mtime-keyed cache for `pipeline_state.json` reads.
- `src/api/presentation/` — `icons.py` + `formatters.py` (eb238ee, 34edefc)

### 2.6. Datasets first-class (4a4a48d) — БОЛЬШОЙ

- Dedicated **Datasets tab** на проект: auto-coupling on strategy add (NOT on remove), preview master-detail, fullscreen mode, JSON syntax-highlight, "only errors" filter
- **Validation plugins переехали** из Plugins tab → DatasetDetail (под `datasets.<key>.validations.plugins[]`)
- API endpoints: `src/api/routers/datasets.py` — preview / validate / path-check
- `src/data/preview/loader.py` — paginated jsonl + HF streaming
- `src/data/validation/standalone.py` — pure `check_dataset_format` / `run_plugins` extracted (DatasetValidator stage делает delegate)
- `resolve_dataset_key` dependency с project-root path-traversal guard
- `_run_configured_plugins` merges manifest's `suggested_params` для backwards-compat

### 2.7. Required_env declarations (91d3503)

Все `community/` плагины обновили manifests с `[[required_env]]`. Включая:
- `cerebras_judge` — `EVAL_CEREBRAS_API_KEY`
- `helixql_compiler_semantic` — optional `HELIX_CLI_PATH`

---

## 3. Что это значит для миграции на Job Server (синтез)

### 3.1. Что **остаётся на Mac** (control plane)
- Все **stage 0** проверки: dataset preview, dataset validation (через `src/data/validation/standalone.py` + `DatasetValidator` stage), config validation
- **Preflight gate** (`run_preflight`) — обязательный pre-launch step через `pipeline_bootstrap` step 1.5
- **Stale plugin detection** (`find_stale_plugins`) — UI warning + блокировка на preflight
- **Evaluation плагины** (cerebras_judge etc.) — это post-training stage на Mac
- **Reports плагины** — render на Mac после полного pipeline
- **Preset apply** — до запуска (config_path уже final YAML)
- **Workspace storage** — `~/.ryotenkai/projects/`, `providers.json`, `integrations.json` остаются исключительно на Mac
- **MLflow tracking** — root run + parent run на Mac как сейчас. Job Server делает relay (async).

### 3.2. Что **уезжает на pod**
- `src/training/` (через CodeSyncer как сейчас)
- `src/utils/`, `src/config/`, `src/data/`, `src/infrastructure/`, `src/constants.py`, `src/__init__.py` (как сейчас)
- **Reward плагины** (только used) — через новый `PluginPacker` как `plugins_payload`
- Конфиг + датасеты (как сейчас, через `FileUploader`)
- **Job Server код** — пред-инсталлирован в docker image
- Secrets (через .env): HF_TOKEN, MLFLOW_*, RWRD_* secrets для used reward plugins, RUNPOD_* для self-stop

### 3.3. Что **переиспользуем** (НЕ изобретаем заново)

| Существующее | Где используем |
|---|---|
| `src/utils/atomic_fs.py::atomic_write_json/text` | `state.jsonl` persistence на pod, `attempts/<n>/job_*` на Mac |
| `src/api/ws/live_tail.py::LiveLogTail` | Fallback tail для `training.log` если WS оборвался; pattern для ring-buffer offset |
| `src/pipeline/state/store.py::PipelineStateStore` | Mac source-of-truth — добавляем `attempts/<n>/job_submission.json`, `job_events.jsonl` |
| `src/pipeline/launch/run_lock_guard.py::RunLockGuard` | Контракт "только один active run", остаётся как есть |
| `src/api/services/launch_service.py::launch()` | Точка входа для запуска. Внутри pipeline subprocess — новый flow tunnel+JobClient |
| `src/community/preflight.py::run_preflight` | **Mandatory** перед `JobClient.submit_job` |
| `src/community/stale_plugins.py::find_stale_plugins` | Job submission блокируется если есть stale refs |
| `src/community/loader.py` (LoadFailure, COMMUNITY_STRICT) | На pod-е Job Server включает `COMMUNITY_STRICT=1` (после распаковки plugins_payload) |
| `src/community/registry_base.py::PluginRegistry[T]` | Reward registry на pod после распаковки community/ |
| `src/training/reward_plugins/factory.py` (broadcast log) | Уже там, остаётся |
| `src/training/reward_plugins/secrets.py` (RWRD_ resolver) | Без изменений — Job Server передаёт env vars в trainer subprocess |
| `src/training/managers/mlflow_manager/resilient_transport.py::MLflowTransportCircuitBreaker` | MLflow relay в Job Server использует тот же circuit breaker |
| `src/providers/training/interfaces.py::TrainingScriptHooks` | RunPod провайдер для self-stop env vars (env_vars поле; pre/post bash hooks больше не используются) |
| `src/api/presentation/icons.py` | UI mapping для job статусов (Web UI Live Training tab) |
| `src/cli/commands/plugin.py` (Typer noun pattern) | Pattern для нового `ryotenkai job` (как noun-verb) |
| **`src/community/pack.py::pack_community_folder`** | Backbone PluginPacker (Mac side) |
| **`src/community/install.py::install_local`, `_extracted_archive`** | Backbone PluginUnpacker (pod side) |
| **`src/community/validate_manifest.py`** | Standalone валидация manifest при unpack |
| `src/cli/app.py::_check_remote_stub` | `--remote URL` global flag — оставляем как stub для v1.2 |
| `src/cli/context.py::CLIContext` | Контекст для job CLI команд |
| `src/cli/renderer.py::get_renderer` | Output формат (-o text\|json\|yaml) |

---

## 4. Целевая архитектура

```
┌──────────────────── Mac (Control Plane) ────────────────────┐
│                                                             │
│ Web UI (existing) ──────┐                                   │
│ FastAPI (src/api)       │                                   │
│  ├ routers/launch       │                                   │
│  ├ routers/datasets     │ (preview/validate, на Mac)        │
│  ├ routers/plugins      │ (preflight, stale)                │
│  ├ routers/projects     │ (workspace)                       │
│  └ ws/live_tail         │ (fallback file tail)              │
│ CLI (src/main.py + new) │                                   │
│  └ ryotenkai job ...    │ (NEW Typer subcommand)            │
│                         │                                   │
│ Pipeline subprocess (existing flow):                        │
│  ├ Stage 0: DatasetValidator (Wave 4)                       │
│  ├ Stage 1: GPUDeployer → создаёт pod, получает SSH         │
│  ├ Stage 2: TrainingDeploymentManager facade (Wave 3)       │
│  │    ├ FileUploader.deploy_files()                         │
│  │    ├ DependencyInstaller.install_dependencies()          │
│  │    └ start_training()  ← заменяется одной реализацией   │
│  │       ├ PluginPacker (NEW, R-2 fix)                      │
│  │       ├ SSHTunnelManager (NEW, отдельный CtrlMaster)     │
│  │       └ JobClient (NEW)                                  │
│  ├ Stage 3: TrainingMonitor (только WS subscribe, без SSH-poll) │
│  ├ Stage 4: ModelRetriever                                  │
│  ├ Stage 5: ModelEvaluator (Mac, evaluation плагины)        │
│  └ Stage 6: IntegrationTest                                 │
└─────────────────────────────────────────────────────────────┘
              │ HTTP+WebSocket поверх ssh -L
              ▼
┌──────────── pod (один docker container) ──────────────────────┐
│                                                               │
│ ENTRYPOINT: dumb-init →                                       │
│   ├ /usr/sbin/sshd                                            │
│   └ uvicorn ryotenkai_runner:app --host 127.0.0.1:8080        │
│                                                               │
│ Job Server (src/runner/, NEW package):                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ JobLifecycleFSM (state.jsonl persisted via atomic_fs)   │  │
│  │ EventBus (pub/sub + ring buffer, offset-based)          │  │
│  │ Supervisor (subprocess.Popen, process group, signals)   │  │
│  │ MLflowRelay (async + MLflowTransportCircuitBreaker)     │  │
│  │ IdleDetector (Python; pynvml + nvidia-smi fallback)     │  │
│  │ HealthReporter (GPU/RAM snapshots)                      │  │
│  │ PluginUnpacker (распаковка plugins_payload в community/)│  │
│  │ ArtifactIndex                                           │  │
│  └─────────────────────────────────────────────────────────┘  │
│           │ subprocess.Popen (одна process group)             │
│           ▼                                                   │
│ python -m src.training.run_training                           │
│  ├ RunnerEventCallback (NEW) ──HTTP loopback─────────────────►│
│  ├ MLflow callbacks (existing)                                │
│  ├ Reward plugin (loaded from community/<kind>/<id>/)         │
│  └ Faulthandler crash observability (existing)                │
└───────────────────────────────────────────────────────────────┘
```

## 5. Зоны ответственности

| Слой | Файлы (existing/NEW) | Знает | Не знает |
|---|---|---|---|
| **Control Plane Mac** | `src/api/services/launch_service.py` (existing), `src/api/services/tunnel_service.py` (NEW), `src/api/clients/job_client.py` (NEW), `src/pipeline/state/store.py` (existing) | Provider lifecycle, SSH endpoint, attempts layout, MLflow setup, preflight, stale-plugin checks, dataset validation | Внутренности trainer'а, GPU метрики |
| **Provider** | `src/providers/runpod/training/*` (existing, минимальные правки), `src/providers/training/interfaces.py` (existing) | API провайдера, как создать SSH endpoint, какие env vars для self-stop | Что внутри pod после SSH |
| **Job Server (pod)** | `src/runner/` (NEW package) | Lifecycle одной работы, события, MLflow relay, idle detection, plugin unpack | Какой провайдер, кто запустил |
| **Trainer** | `src/training/run_training.py` (existing, +`RunnerEventCallback`) | Как обучать, как использовать reward plugin (через PluginRegistry[T]) | Что есть Job Server |

---

## 6. Транспорт и IPC

| Связь | Протокол | Файлы | Существующее переиспользуем |
|---|---|---|---|
| Mac CLI ↔ Job Server | HTTP REST + WebSocket поверх `ssh -L 18080:127.0.0.1:8080` | `src/api/clients/job_client.py` (NEW) | — |
| Trainer ↔ Job Server | loopback HTTP `127.0.0.1:8080/internal/events` | `src/training/callbacks/runner_event_callback.py` (NEW) | TrainerCallback HF API |
| MLflow tracking | как сейчас + async relay через Job Server | `src/runner/mlflow_relay.py` (NEW) | `MLflowTransportCircuitBreaker` (existing) |
| Provisioning | как у провайдера | `src/providers/runpod/...` | Без изменений |
| Plugin payload | multipart `POST /jobs` (job_spec JSON + plugins_payload ZIP) | `JobClient.submit_job()` | `PluginRegistry[T]` (existing) |

---

## 7. Унифицированный Docker образ

`ryotenkai/training-runtime:VERSION` — расширяет существующий [`docker/training/Dockerfile.runtime`](docker/training/Dockerfile.runtime):

- CUDA 12.4 + PyTorch 2.5.1 + Python 3.12 (как сейчас)
- ML deps (как сейчас, `requirements.runtime.txt`) — `fastapi`, `uvicorn[standard]`, `websockets` уже в зависимостях основного `pyproject.toml`
- **NEW**: `dumb-init` (apt package, для PID 1)
- **NEW**: `COPY src/__init__.py + src/runner → /opt/ryotenkai/src/...` + `ENV PYTHONPATH=/opt/ryotenkai:${PYTHONPATH:-}`. Прямое копирование вместо `pip install -e` — нам не нужен ни отдельный pyproject под runner, ни editable mode. По мере появления зависимостей runner на `src/utils.atomic_fs`, `src/community.install` и т.д. — расширим `COPY` соответствующими модулями. Pipeline-driven `/workspace` остаётся первым в PYTHONPATH (выставляется в job_spec env), так что rsync-обновлённый `src/utils` overrides image baseline — by design.
- **NEW** (Phase 4+): образ выставляет `ENV COMMUNITY_STRICT=1` после первого вызова `PluginUnpacker` — strict-mode для catalog
- Updated `entrypoint.sh`: `dumb-init` → стартует sshd + uvicorn job-server (`127.0.0.1:8080`, **только loopback**)

Поля `image_name` / `docker_image` **удаляются** из конфига одним cutover'ом (no Pydantic alias, no deprecation warning):
- `src/config/providers/runpod/training.py` — поле `image_name` → удалить
- `src/config/providers/single_node/training.py` — поле `docker_image` → удалить
- Образ зашит в `src/runner/__about__.py::RUNTIME_IMAGE` и читается из `src/providers/runpod/training/api_client.py`
- Если в существующих project YAML есть эти поля — load выкинет PydanticValidationError на extra-forbid; пользователь правит YAML руками (one-shot, см. Phase 6.8)

---

## 8. FSM и persistence

```
                              POST /jobs (multipart)
                                  │
                                  ▼
                         [preparing]  ← unpacking plugins, validating
                              │   │
                       prep ok│   │ prep failed
                              ▼   ▼
                          [running] ──► [failed]
                              │
                ┌─────────────┼──────────────┐
        natural │             │stop request   │ idle / crash
        finish  │             │               │
                ▼             ▼               ▼
         [completed]    [stopping] ──►   [failed]
                              │
                              ▼
                         [completed | cancelled]
```

State persistence:
- **Pod side**: `/workspace/.ryotenkai/state.jsonl` (append-only через `atomic_fs.atomic_write_text`) + `state.json` (последнее состояние, atomic)
- **Mac side** (через PipelineStateStore):
  - `attempts/<n>/job_submission.json` — JobHandle, tunnel info
  - `attempts/<n>/job_events.jsonl` — догнанные с pod события
  - `pipeline_state.json` — high-level (status, started_at, finished_at)
- На startup Job Server делает `restore_or_init()`: читает `state.jsonl`. Если последнее состояние = preparing/stopping → переводит в failed (R-27).

Ring buffer событий: 10k × ~1KB ≈ 10MB RAM. Configurable через env `RYOTENKAI_EVENT_BUFFER_SIZE`.

Detach/reattach сценарий:
1. Mac: `ryotenkai run start config.yaml` → SSH-туннель + POST /jobs (multipart)
2. Mac уходит в сон → SSH рвётся → Job Server **продолжает**, копит события
3. Mac просыпается → `ryotenkai run resume <attempt>` → читает `pipeline_state.json` → провайдер.status() → SSH-туннель → GET `/jobs/{id}` → WS `/jobs/{id}/events?since=<offset>` → догнал

---

## 9. Plugin delivery (R-2 fix) — критичное

R-2 подтверждён открытым после Wave 3. `CodeSyncer.REQUIRED_MODULES` не содержит `community/`. Решение использует уже готовую инфраструктуру.

### 9.1. Mac: PluginPacker (использует существующий `pack_community_folder`)

`src/pipeline/stages/managers/deployment/plugin_packer.py` (NEW):

```python
@dataclass(frozen=True)
class PluginRef:
    kind: str          # 'reward' (на pod уезжают только reward)
    plugin_id: str
    source_path: Path  # community/reward/<id>/

class PluginPacker:
    def __init__(self, config: PipelineConfig): ...

    def determine_required_plugins(self) -> list[PluginRef]:
        """Walk config: для каждой phase в strategy chain взять
        params.reward_plugin (RL strategies). Только reward — validation
        и evaluation плагины остаются на Mac."""

    def pack(self, plugins: list[PluginRef]) -> bytes:
        """Делегирует на src.community.pack.pack_community_folder()
        для каждого плагина, склеивает в один tar/zip-bundle."""
```

**Переиспользуем готовое**:
- `src/community/pack.py::pack_community_folder(source: Path, ...)` — упаковка с валидацией manifest и exclude_patterns
- `src/community/validate_manifest.py` — standalone валидация перед упаковкой

Используется внутри `TrainingLauncherV2` перед вызовом `JobClient.submit_job(...)`.

### 9.2. Pod: PluginUnpacker (использует существующий `install_local`)

`src/runner/plugin_unpacker.py` (NEW):
- Принимает `plugins_payload` bytes из `POST /jobs`
- Делегирует на `src.community.install._extracted_archive(zip_path)` (context manager) → `install_local(source, kind, plugin_id, force, allow_untrusted=True)` для каждого плагина
- Цель: `/workspace/community/<kind>/<id>/`
- Если `COMMUNITY_STRICT=1` (включено в образе как ENV) — fail-fast при любой проблеме
- После распаковки `catalog.ensure_loaded()` находит reward plugin

**Переиспользуем готовое**:
- `src/community/install.py::install_local()`, `_extracted_archive()` — копирование folder с manifest validation
- `src/community/loader.py` (LoadFailure, COMMUNITY_STRICT) — для fail-fast detection
- Никакой логики копирования файлов или валидации manifest заново НЕ пишем

### 9.3. Mandatory preflight integration

До `JobClient.submit_job` Mac вызывает `run_preflight(config, secrets, project_env)`. Уже сейчас это часть `pipeline_bootstrap` step 1.5. Если есть `missing_envs` или `instance_errors` — `LaunchAbortedError` → submit не происходит.

Если в config есть stale references (`find_stale_plugins(config)` non-empty) — submit заблокирован с структурированной ошибкой.

---

## 10. Поэтапная реализация

> Учитывает уже декомпозированный TrainingDeploymentManager и Wave 4. Заменяем только `TrainingLauncher` → `TrainingLauncherV2`, остальные компоненты Wave 3 не трогаем.

### Phase 0 — инфраструктура (½ дня)
- 0.1 `src/runner/` skeleton: package layout, FastAPI app, тестовые фикстуры
- 0.2 `docker/training/Dockerfile.runtime` — добавить `dumb-init`, `pip install -e /opt/ryotenkai-runner`
- 0.3 `docker/training/entrypoint.sh` — заменить `tail -f /dev/null` на `dumb-init` → sshd + uvicorn
- 0.4 `docker/training/build_and_push.sh` — добавить копирование `src/runner/`
- 0.5 `src/tests/unit/runner/` scaffold + conftest

### Phase 1 — Job Server skeleton (1-2 дня)
- 1.1 `src/runner/state.py::JobLifecycleFSM` (transitions, persistence через `atomic_fs.atomic_write_text` для state.jsonl + state.json)
- 1.2 `src/runner/event_bus.py` — pub/sub, ring buffer (offset-based, паттерн `LiveLogTail`)
- 1.3 `src/runner/api/jobs.py` — REST endpoints:
   - `POST /jobs` (multipart: job_spec + plugins_payload)
   - `GET /jobs/{id}`
   - `POST /jobs/{id}/stop`
- 1.4 `src/runner/api/events.py` — WebSocket `/jobs/{id}/events?since=<offset>`
- 1.5 `src/runner/api/internal.py` — loopback `POST /internal/events`
- 1.6 `src/runner/main.py` — FastAPI app, lifespan (`restore_or_init` + dumb-init-friendly shutdown)
- 1.7 `src/runner/__about__.py` — `RUNTIME_IMAGE = "ryotenkai/training-runtime:VERSION"`
- 1.8 Unit tests: FSM transitions (positive/negative/boundary), ring buffer overflow + offset-correctness, restore semantics, multipart upload

### Phase 2 — Supervisor + signals (1 день)
- 2.1 `src/runner/supervisor.py`:
   - `subprocess.Popen` с `start_new_session=False` (одна process group)
   - Capture stdout/stderr через pipes → events (NOT raw text in events; `training.log` пишется как раньше для tail-fallback)
   - Two-phase shutdown: SIGTERM → wait timeout → SIGKILL on process group через `os.killpg`
   - Exit code parsing: `> 128` → `signal.Signals(rc - 128).name`
- 2.2 `src/training/signal_handlers.py` (existing pattern в `src/training/orchestrator/shutdown_handler.py` — расширить):
   - SIGTERM в trainer process → `TrainerControl.should_save = True` (FIRST) → `should_training_stop = True` (THEN)
   - Это закрывает R-23 (неполный checkpoint при stop)
- 2.3 Unit tests: graceful stop, force-kill timeout, native crash detection (rc>128 → signal name), orphan trainer prevention, dataloader workers killed via pgid

### Phase 3 — TrainerCallback (½ дня)
- 3.1 `src/training/callbacks/runner_event_callback.py` (NEW):
   - Event types: `training_started`, `epoch_start/end`, `step` (every N), `eval_metrics`, `checkpoint_saved`, `model_loading_progress`, `training_complete`
   - Local buffer (deque), flush every N steps (configurable, default 10) ИЛИ on level=ERROR/INFO|important
   - POST `127.0.0.1:8080/internal/events`
   - Conditional: only if `RYOTENKAI_RUNNER_URL` env установлен
   - Graceful degrade: если runner не отвечает 3 раза подряд — отключиться до конца сессии (training не падает)
- 3.2 Регистрация в [`src/training/run_training.py`](src/training/run_training.py) как один из callbacks (рядом с MLflow + system_metrics + gpu_metrics + training_events)
- 3.3 Unit tests: events flow, retry, degradation, buffer overflow

### Phase 4 — IdleDetector + HealthReporter + MLflowRelay (1 день)
- 4.1 `src/runner/idle_detector.py`:
   - Python замена `watchdog.sh`: те же пороги (`STARTUP_GRACE=300`, `IDLE_THRESHOLD=1200`, `MAX_LIFETIME=172800`)
   - `pynvml` (приоритет) → fallback `subprocess nvidia-smi`
   - Триггер на FSM stop вместо kill -9
- 4.2 `src/runner/health_reporter.py` — periodic GPU/RAM/CPU snapshot → events каждые 30с
- 4.3 `src/runner/mlflow_relay.py`:
   - Async forward через `asyncio.Queue` worker
   - Reuse `src/training/managers/mlflow_manager/resilient_transport.py::MLflowTransportCircuitBreaker`
   - Buffer offline события до восстановления соединения
- 4.4 ✅ **DONE** — Self-stop hook: `src/runner/pod_stopper.py` (Python replacement for `runpod_stop_pod.sh`). На FSM transition → `[completed | cancelled | failed]` Supervisor вызывает `terminal_hook` который делегирует в `stop_pod_on_terminal()`. Env-driven: `RUNPOD_AUTO_STOP=false` → `disabled`, missing creds → `skipped`, GraphQL ok → `stopped`, idempotent already-stopped → `already_stopped`, retry exhausted → `failed`. Outcome публикуется как `pod_stop_attempt` event. **Bash скрипт `runpod_stop_pod.sh` удаляется в Phase 6** вместе с остальным cleanup.
- 4.5 ✅ Unit tests: idle detection across thresholds (12 tests), MLflow circuit breaker integration (deferred), pod_stopper full matrix (19 tests: decision table + env short-circuits + GraphQL paths + retry + network errors + wrapper), supervisor terminal_hook (3 tests).

### Phase 5 — Mac: SSHTunnel + JobClient (1 день) ✅ DONE
- 5.1 ✅ `src/api/services/tunnel_service.py` — `SSHTunnelManager`:
   - `open()` запускает `ssh -fN -L <local>:127.0.0.1:8080` с `ExitOnForwardFailure=yes`
   - **Отдельный** ControlMaster socket в `~/.ssh/control_sockets/ryotenkai_runner/` — изолирован от `SSHClient`'s `~/.ssh/sockets/` (rsync/exec)
   - Auto-выбор свободного порта 18080-18099 (sequential, predictable for debugging)
   - Readiness probe через TCP-connect до `_TUNNEL_READY_TIMEOUT_SECONDS=10s`
   - `close()` — best-effort `ssh -O exit`, swallows errors
   - Idempotent open(); failed open резетит state
   - Async context manager support
- 5.2 ✅ `src/api/clients/job_client.py` — `JobClient`:
   - `httpx.AsyncClient` для HTTP + `websockets.connect` для WS
   - `health_check()`, `submit_job()` (multipart: form `job_spec` + file `plugins_payload`), `get_status()`, `request_stop()`
   - `subscribe_events(job_id, since=N)` — async generator с auto-reconnect, exponential backoff (1s → 30s + ±25% jitter), offset tracking для seamless resume
   - WS close-code translation: 4404 → `JobNotFoundError`, 4410 → `ReplayTruncatedError`, 4422 → `JobClientError`
   - `max_reconnect_attempts` cap (по умолчанию: ∞)
   - HTTP→WS URL scheme translation (`http://` → `ws://`, `https://` → `wss://`)
- 5.3 ✅ Unit tests (38 новых):
   - 21 JobClient unit tests (httpx.MockTransport + fake WS): health, submit, get_status, request_stop, subscribe_events с reconnect, close-code translation, URL scheme
   - 13 SSHTunnelManager tests (mock subprocess runner + mock port probe): argv shape, port allocation, lifecycle, readiness probe timeout, close best-effort, socket dir isolation
   - 4 contract tests (httpx.ASGITransport против реального runner app): submit/status/stop wire shape parity
- **Stream_log endpoint и `runs_resume` integration отложены в Phase 7** (CLI) — там же подключим `JobClient` к `ryotenkai run resume`

### Phase 6 — Pipeline integration + cutover (1-2 дня)

Эта фаза одновременно вводит новую реализацию **и** удаляет старую (no backwards compat).

- 6.1 `src/pipeline/stages/managers/deployment/plugin_packer.py` (NEW) — § 9.1
- 6.2 `src/pipeline/stages/managers/deployment/training_launcher.py` — **переписать целиком**:
   - Удалить `_create_env_file` (env vars теперь приходят через `POST /jobs` job_spec, не через `.env`)
   - Удалить `_start_training_cloud` / `_start_training_docker` / probe loop / nohup
   - Удалить генерацию `start_training.sh`
   - Новая публичная сигнатура `start_training(...)`: вызвать `run_preflight(...)` → собрать `plugins_payload` через `PluginPacker` → `SSHTunnelManager.open()` → `JobClient.submit_job(...)`
   - Имя класса остаётся `TrainingLauncher` (facade `TrainingDeploymentManager.start_training` НЕ меняется)
- 6.3 `src/pipeline/stages/training_monitor.py` (existing): убрать SSH-poll целиком, заменить на `JobClient.subscribe_events()`. `LiveLogTail` остаётся **только** как fallback для скачивания tail логов после завершения (offline forensics).
- 6.4 Удалить marker-based код в этой же фазе (single cutover):
   - `src/pipeline/stages/training_monitor._check_marker`, `_read_marker_content`, `_touch_pipeline_heartbeat`
   - `src/training/notifiers/marker_file.py` целиком (заменяется push events через `RunnerEventCallback`)
   - `src/providers/runpod/training/resources/watchdog.sh` (логика → `IdleDetector` Phase 4)
   - `src/providers/runpod/training/resources/runpod_stop_pod.sh` (вызывается из Python supervisor через provider hook — оставить как тонкую GraphQL-обёртку или переписать на Python; решим в реализации)
- 6.5 `CodeSyncer.REQUIRED_MODULES` — **НЕ расширяем** community/. Plugin payload идёт через multipart.
- 6.6 `src/config/providers/runpod/training.py` — удалить поле `image_name` (no Pydantic alias, no deprecation warning)
- 6.7 `src/config/providers/single_node/training.py` — удалить поле `docker_image`
- 6.8 Один-shot config-migration script для существующих project YAML — strip полей при чтении (warn, продолжить). Это для пользователя, не для legacy mode — после миграции YAML script не нужен.
- 6.9 Integration tests: e2e через локальный uvicorn (mock pod). Удалить старые тесты под marker-based полinging.

### Phase 7 — CLI + Web UI (1 день)
- 7.1 `src/cli/commands/job.py` (NEW noun, добавляется к существующим run/runs/config/dataset/...):
   - `ryotenkai job status <attempt>` — GET /jobs/{id}
   - `ryotenkai job logs <attempt> --follow [--tail N]` — chunked HTTP, переиспользует `LiveLogTail` для fallback
   - `ryotenkai job stop <attempt>` — POST /jobs/{id}/stop
   - `ryotenkai job events <attempt> --follow --since <offset>` — WS history+live
   - `ryotenkai job metrics <attempt>` — последние GPU/RAM events
   - Использует существующие `CLIContext`, `get_renderer`, `die`, `common_options` (как `runs`/`run`)
   - Поддерживает `-o text|json|yaml`, `--project`
- 7.2 `src/cli/commands/__init__.py::register_all()` — зарегистрировать `job_app` как `app.add_typer(job_app, name="job")`
- 7.3 `ryotenkai run resume <attempt>` (existing команда в `commands/run.py`) — обновить flow: открыть SSHTunnel + JobClient.subscribe_events(since=offset)
- 7.4 Web UI: новая страница "Live Training" в проекте — WS subscription через тот же тоннель. Использует `STATUS_ICONS` из `src/api/presentation/icons.py`. Vitest тесты по существующему паттерну (`web/src/components/__tests__/`).
- 7.5 Tests:
   - `src/tests/smoke/test_cli_help.py` — добавить (job, status), (job, logs), (job, stop), (job, events), (job, metrics) в smoke matrix (5 новых пар)
   - `src/tests/contract/test_cli_api_parity.py` — добавить parity-test для job status (CLI ↔ API)
   - WS reconnect tests
   - **`--remote URL` flag не трогаем** — оставляем stub для v1.2 как есть. Наш job-сервер работает через `--project` контекст

### Phase 8 — Final polish + docs (½ дня)

Большая часть cleanup сделана в Phase 6 одним cutover'ом. Финальная фаза только закрывает остатки.

- 8.1 `docs/runner-architecture.md` — финальная документация с диаграммой, sequence-flow, troubleshooting
- 8.2 README обновление — секция про job-server (как запустить, как подключиться по SSH `-L`)
- 8.3 Обновить `community/README.md` — упомянуть что reward плагины уезжают на pod, остальные kinds выполняются на Mac
- 8.4 Final regression: full pytest, coverage gate ≥ 83, smoke на RunPod

**Итого: 6-8 рабочих дней** (½ дня меньше за счёт удалённой Phase 8 cleanup-секции — она интегрирована в Phase 6).

---

## 11. Критические файлы (paths to modify / create)

### Новые

**`src/runner/` package:**
- `main.py` — FastAPI app, lifespan
- `__about__.py` — `RUNTIME_IMAGE` константа
- `state.py` — JobLifecycleFSM
- `event_bus.py` — pub/sub + ring buffer
- `supervisor.py` — subprocess.Popen + signals
- `idle_detector.py` — Python замена watchdog.sh
- `health_reporter.py` — GPU/RAM snapshots
- `mlflow_relay.py` — async forward + circuit breaker
- `plugin_unpacker.py` — распаковка plugins_payload
- `api/jobs.py`, `api/events.py`, `api/internal.py`

**Mac side:**
- `src/api/services/tunnel_service.py` — SSH `-L` менеджер
- `src/api/clients/job_client.py` — HTTP+WS клиент
- `src/pipeline/stages/managers/deployment/plugin_packer.py` — wrapper над `pack_community_folder`
- `src/training/callbacks/runner_event_callback.py`
- `src/cli/commands/job.py` — NEW Typer noun (рядом с существующими run/runs/...)
- `src/config/migrations/v6_drop_image_fields.py`
- `docs/runner-architecture.md`

### Модифицируем
- `docker/training/Dockerfile.runtime` — добавить dumb-init + runner install
- `docker/training/entrypoint.sh` — sshd + uvicorn вместо tail
- `docker/training/build_and_push.sh` — копирование `src/runner/`
- `src/pipeline/stages/managers/deployment_manager.py` (facade) — без diff, fasade перенаправляет на новый `TrainingLauncher`
- `src/pipeline/stages/managers/deployment/training_launcher.py` — **переписать целиком** (cutover, см. Phase 6.2)
- `src/pipeline/stages/training_monitor.py` — заменить SSH-poll на `JobClient.subscribe_events()`
- `src/training/run_training.py` — добавить `RunnerEventCallback`
- `src/training/orchestrator/shutdown_handler.py` — расширить SIGTERM handling под `should_save=True`
- `src/config/providers/runpod/training.py` — удалить `image_name` (no alias, no warning)
- `src/config/providers/single_node/training.py` — удалить `docker_image`
- `src/cli/commands/__init__.py` — зарегистрировать `job_app` через `register_all` (1-2 строки diff)

### Удаляем целиком (Phase 6 cutover)
- `src/training/notifiers/marker_file.py` — push events заменяют marker files
- `src/providers/runpod/training/resources/watchdog.sh` — `IdleDetector` (Python) заменяет
- (Опционально, решим в реализации) `src/providers/runpod/training/resources/runpod_stop_pod.sh` — переписать на Python в supervisor
- Старые тесты `test_training_launcher.py` под marker-flow — переписываются под новую сигнатуру

### Переиспользуем (без изменений)
- `src/utils/atomic_fs.py` (atomic_write_*)
- `src/api/ws/live_tail.py::LiveLogTail`
- `src/pipeline/state/store.py::PipelineStateStore`
- `src/training/managers/mlflow_manager/resilient_transport.py::MLflowTransportCircuitBreaker`
- **`src/community/pack.py::pack_community_folder`** (PluginPacker backbone)
- **`src/community/install.py::install_local`, `_extracted_archive`** (PluginUnpacker backbone)
- **`src/community/validate_manifest.py`** (standalone manifest validation)
- `src/community/preflight.py::run_preflight`
- `src/community/stale_plugins.py::find_stale_plugins`
- `src/community/loader.py` (LoadFailure, COMMUNITY_STRICT)
- `src/community/registry_base.py::PluginRegistry[T]`
- `src/training/reward_plugins/secrets.py` (RWRD_SecretsResolver)
- `src/training/reward_plugins/factory.py` (broadcast log)
- `src/providers/training/interfaces.py::TrainingScriptHooks` — `env_vars` поле для self-stop secrets (RUNPOD_API_KEY etc.); pre/post bash hooks больше не используются и удаляются из dataclass
- **`src/cli/app.py`, `src/cli/context.py`, `src/cli/renderer.py`, `src/cli/common_options.py`** (CLI infrastructure для job noun)

---

## 12. Top-7 critical risks (актуализированы)

| ID | Риск | S | L | Score | Mitigation |
|---|---|---:|---:|---:|---|
| R-2 | community/ delivery gap всё ещё открыт | 5 | 5 | **25** | `PluginPacker` + `plugins_payload` multipart (§ 9) |
| R-5 | SIGTERM не доходит до Python trainer | 5 | 4 | **20** | `dumb-init`, exec, process group, `os.killpg` |
| R-1 | SIGSEGV в bitsandbytes/flash-attn | 5 | 4 | **20** | Supervisor различает rc>128, читает `training.faulthandler.log` |
| R-11 | uvicorn PID 1 не пробрасывает signals | 4 | 4 | **16** | `dumb-init` как ENTRYPOINT (Phase 0) |
| R-22 | TrainerCallback POST overhead | 4 | 4 | **16** | Local buffer + flush каждые N шагов; raw stdout остаётся в `training.log` |
| R-15 | In-flight runs при cutover (SAPO experiments) | 5 | 3 | **15** | Скоординировать окно с пользователем (подтверждено — открыто) |
| R-7 | Slow dev cycle (rebuild image) | 3 | 5 | **15** | Bind-mount `src/runner/` в dev mode через `RYOTENKAI_DEV_MOUNT` |

Полный список 30 рисков — см. [`nice-jepsen-runner.md` § 17](./nice-jepsen-runner.md).

---

## 13. Open Questions (всё закрыто после изучения RESEACRH)

| OQ | Резолюция |
|---|---|
| OQ-1 (community/ delivery) | `PluginPacker` (Mac) → `plugins_payload` multipart → `PluginUnpacker` (pod) → `community/` распакован → `catalog.ensure_loaded()` находит. Только reward plugins (validation/evaluation/reports на Mac). |
| OQ-2 (state.jsonl path) | `/workspace/.ryotenkai/state.jsonl` — RunPod `volume_mount_path` persistent, переживёт container restart |
| OQ-3 (ring buffer size) | 10k × ~1KB ≈ 10MB. Configurable через `RYOTENKAI_EVENT_BUFFER_SIZE` |
| OQ-4 (MLflow relay sync vs async) | Async + `MLflowTransportCircuitBreaker` (existing) |
| OQ-5 (in-flight SAPO migration) | Cutover-стратегия (no backwards compat). Окно для миграции скоординировано с пользователем — открыто. Старые runs если есть — дожидаемся их завершения, потом cutover |
| **OQ-6 NEW** (datasets validation) | Validation plugins **остаются на Mac** (Wave 4 confirmed: они теперь под `datasets.<key>.validations.plugins[]`, выполняются в Stage 0 DatasetValidator). На pod уезжают только reward. |
| **OQ-7 NEW** (preflight на Mac vs pod) | Preflight работает только на Mac (через `pipeline_bootstrap` step 1.5), до отправки `submit_job`. На pod-е дополнительная проверка не нужна — config уже валиден. |
| **OQ-8 NEW** (image override для dev) | Жёстко прибито в `__about__.py::RUNTIME_IMAGE`. Override **только** через env `RYOTENKAI_RUNTIME_IMAGE_OVERRIDE` (для CI/dev, не для пользователей). |

---

## 14. Verification (как тестировать end-to-end)

### Unit
```bash
# Новые
pytest src/tests/unit/runner/ -v          # ~80+ tests (FSM/EventBus/Supervisor/IdleDetector/PluginUnpacker)
pytest src/tests/unit/api/clients/test_job_client.py -v
pytest src/tests/unit/api/services/test_tunnel_service.py -v
pytest src/tests/unit/training/callbacks/test_runner_event_callback.py -v
pytest src/tests/unit/pipeline/stages/managers/deployment/test_plugin_packer.py -v
pytest src/tests/unit/pipeline/stages/managers/deployment/test_training_launcher_v2.py -v
pytest src/tests/unit/cli/test_job.py -v

# Регрессия (Wave 3 + Wave 4 + plugin platform)
pytest src/tests/ --tb=short  # 4900+ tests, zero regressions, fail_under=83
pytest --no-header -q  # baseline 4939 passed на origin/RESEACRH
```

### Integration (mock pod через локальный uvicorn)
```bash
pytest src/tests/integration/runner/test_e2e_happy.py -v             # POST /jobs → events → completed
pytest src/tests/integration/runner/test_detach_reattach.py -v       # WS reconnect с offset
pytest src/tests/integration/runner/test_stop_with_checkpoint.py -v  # SIGTERM → save → cancelled
pytest src/tests/integration/runner/test_plugin_payload.py -v        # PluginPacker → Unpacker round-trip
pytest src/tests/integration/runner/test_preflight_blocks_submit.py -v
pytest src/tests/integration/runner/test_stale_plugin_blocks.py -v
```

### Manual (RunPod, end-to-end)
1. Build & push новый образ: `./docker/training/build_and_push.sh --bump minor`
2. Запустить SAPO run (старая команда теперь использует новую инфраструктуру по умолчанию):
   ```bash
   ryotenkai run start config/sapo_helixql.yaml
   ```
3. Проверить:
   - `ryotenkai job status <attempt>` → состояние `running`, виден reward broadcast log
   - `ryotenkai job events <attempt> --follow` → events стримятся
   - Закрыть лаптоп на 5 минут → открыть → `ryotenkai run resume <attempt>` → события догнали
   - `ryotenkai job stop <attempt>` → state → `stopping` → `cancelled` + последний checkpoint сохранён
4. Сравнить wall-clock time + correctness с прежним SAPO baseline (комит до cutover)

### Cleanup checks
```bash
# Watchdog.sh исчез
test ! -f src/providers/runpod/training/resources/watchdog.sh

# Marker probes исчезли из TrainingMonitor
! grep -n "_check_marker\|TRAINING_COMPLETE\|TRAINING_FAILED\b\|_touch_pipeline_heartbeat" src/pipeline/stages/training_monitor.py

# Поля image_name/docker_image удалены из providers
! grep -rn "^\s*image_name:\|^\s*docker_image:" src/config/providers/

# Coverage gate выдержан
pytest --cov=src --cov-fail-under=83
```

---

**STATUS: WAITING FOR USER APPROVAL**

После одобрения:
1. Скоординировать с активными SAPO/GRPO experiments (R-15) — окно для миграции
2. Стартовать Phase 0 (½ дня)
3. Параллельно держать публикацию обновлённого Docker image готовой
4. **Сначала влить `origin/RESEACRH` в текущий worktree** (61 коммит, fast-forward) — план писался под него
