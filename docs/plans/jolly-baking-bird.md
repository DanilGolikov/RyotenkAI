# Web Frontend + Backend for RyotenkAI Pipeline

## Context

Сейчас у RyotenkAI есть три интерфейса над одним и тем же движком `PipelineOrchestrator`:
CLI (Typer, [src/main.py](../../src/main.py)), TUI (Textual, [src/tui/apps.py](../../src/tui/apps.py))
и MLflow UI для метрик/артефактов. Нужен web-фронт для запуска/наблюдения/удаления
pipeline-ранов из браузера. TUI в перспективе уйдёт, но в MVP остаётся.

Архитектурное решение — **Kubernetes-way "Shared State"**:
- Движок (PipelineOrchestrator) — отдельный subprocess, как сейчас у TUI.
- Файловый state store (`runs/<id>/pipeline_state.json` + `run.lock`) — единственный источник правды.
- Backend (FastAPI) — **не оборачивает CLI** через subprocess для чтения state. Он sibling-клиент
  к тому же state store, что и TUI. Launch — это один subprocess.Popen, дальше state читается из файлов.
- Web UI, TUI и CLI — равноправные клиенты.

Это сохраняет инвариант "если backend упал — pipeline продолжает", и позволяет в Phase 2
переехать к Engine-as-Service (REST + gRPC) без переписывания клиентов.

## Decisions (подтверждено)

- **Frontend stack**: React + Vite + TypeScript + TanStack Query + Tailwind + shadcn/ui.
- **Frontend location**: `web/` в корне репо (monorepo). `web/dist/` монтируется через `StaticFiles` когда существует.
- **Launch module**: переместить [src/tui/launch.py](../../src/tui/launch.py) → `src/pipeline/launch.py`,
  оставить 1-строчный shim `from src.pipeline.launch import *` в старом месте.
- **TUI**: ничего не трогать, Web UI становится primary по документации.
- **Auth**: нет в MVP. Default bind `127.0.0.1`.

## Reuse Surface (переиспользуем, не пишем заново)

### Pure pipeline layer (backend импортирует напрямую)
- [src/pipeline/state/store.py](../../src/pipeline/state/store.py): `PipelineStateStore`, `acquire_run_lock`,
  `atomic_write_json`, `build_attempt_id`, `PipelineStateError/LoadError/LockError`.
- [src/pipeline/state/models.py](../../src/pipeline/state/models.py): `PipelineState`,
  `PipelineAttemptState`, `StageRunState`, `StageLineageRef` (все с `to_dict`/`from_dict`).
- [src/pipeline/run_queries.py](../../src/pipeline/run_queries.py): `scan_runs_dir`,
  `scan_runs_dir_grouped`, `RunInspector`, `tail_lines`, `effective_pipeline_status`, `diff_attempts`.
- [src/pipeline/launch_queries.py](../../src/pipeline/launch_queries.py):
  `load_restart_point_options`, `pick_default_launch_mode`, `validate_resume_run`,
  `resolve_config_path_for_run`, `derive_resume_stage`, `RestartPointOption`.
- [src/pipeline/artifacts/base.py](../../src/pipeline/artifacts/base.py) + `schemas.py` — артефакт-контракты.

### TUI adapters (backend импортирует — не copy-paste)
- [src/tui/adapters/delete_backend.py](../../src/tui/adapters/delete_backend.py): `TuiDeleteBackend` —
  MLflow tree traversal для безопасного удаления.
- [src/tui/adapters/presentation.py](../../src/tui/adapters/presentation.py): `STATUS_ICONS`,
  `STATUS_COLORS`, `format_duration`, `format_mode_label`.
- [src/tui/live_logs.py](../../src/tui/live_logs.py): `LiveLogTail` для offset-based tail.
- [src/tui/adapters/run_catalog.py](../../src/tui/adapters/run_catalog.py): `build_suggested_run_dir`.

### Launch механика (после переноса)
- `src/pipeline/launch.py` (новый, перенос): `LaunchRequest`, `LaunchResult`, `ActiveLaunch`,
  `build_train_command`, `execute_launch_subprocess`, `interrupt_launch_process`.

## Backend Architecture

### Layering
- **routers/**: тонкие HTTP-хендлеры, конвертируют path/query, вызывают service, мапят в Pydantic.
- **services/**: бизнес-логика, собирает pure-Python модули. Не знает про FastAPI.
- **schemas/**: Pydantic v2, `model_validate(dataclass.to_dict())`. Источник OpenAPI → TS.
- **dependencies.py**: `get_settings`, `get_runs_dir`, `get_state_store(run_id)`, ...
- **exceptions.py**: `PipelineStateLoadError` → 404, `PipelineStateLockError` → 409,
  `ValueError` из launch_queries → 422, `FileNotFoundError` → 404.

### Директории (создать)
```
src/api/
  main.py           # create_app() factory
  cli.py            # run_server() — вызывается из src/main.py::serve
  config.py         # ApiSettings(BaseSettings) — env prefix RYOTENKAI_API_
  dependencies.py
  exceptions.py
  schemas/          # common, run, attempt, stage, launch, log, delete, report, config_validate, health
  routers/          # runs, attempts, launch, logs, config, reports, health
  services/         # run_service, launch_service, log_service, delete_service, report_service, artifact_service, config_service
  ws/log_stream.py  # WebSocket endpoint + LiveLogTail driver
```

### HTTP API v1 (все под `/api/v1`)

| Method | Path | Reuse | Response |
|---|---|---|---|
| GET | `/runs` | `scan_runs_dir_grouped` | `RunsListResponse` (grouped) |
| POST | `/runs` | `build_suggested_run_dir` | `RunSummary` |
| GET | `/runs/{id}` | `RunInspector.load` + `find_running_attempt_no` + `predict_next_attempt_no` | `RunDetail` |
| DELETE | `/runs/{id}` | `TuiDeleteBackend.delete_target` | `DeleteResult` (409 если `run.lock` есть) |
| GET | `/runs/{id}/attempts/{n}` | `get_attempt_by_no` | `AttemptDetail` |
| GET | `/runs/{id}/attempts/{n}/stages` | `enabled_stage_names` + `stage_runs` | `StagesResponse` |
| GET | `/runs/{id}/attempts/{n}/artifacts/{stage}` | `artifact_service` (local JSON → MLflow fallback) | `StageArtifactResponse` |
| GET | `/runs/{id}/attempts/{n}/logs` | `LiveLogTail` (offset-based chunk) | `LogChunk` |
| WS | `/runs/{id}/attempts/{n}/logs/stream` | `LiveLogTail` poll + state mtime watch | `{type, ...}` events |
| GET | `/runs/{id}/restart-points` | `load_restart_point_options` | `RestartPointsResponse` |
| GET | `/runs/{id}/default-launch-mode` | `pick_default_launch_mode` | `{mode}` |
| POST | `/runs/{id}/launch` | `validate_resume_run` + `execute_launch_subprocess` | `LaunchResponse` (409 если уже lock) |
| POST | `/runs/{id}/interrupt` | `interrupt_launch_process` (pid из `run.lock`) | `InterruptResponse` |
| POST | `/config/validate` | рефактор `src/main.py::config_validate` | `ConfigValidationResult` |
| GET | `/config/default` | скан `examples/` | templates |
| GET | `/runs/{id}/report` | `ExperimentReportGenerator.generate` | `ReportResponse` |
| GET | `/health` | scan_runs_dir probe | `HealthStatus` |

### Launch execution (detached subprocess)
- Backend вызывает `execute_launch_subprocess(request, on_started=record_pid)` через
  `fastapi.concurrency.run_in_threadpool`, чтобы Popen не блокировал event loop.
- `start_new_session=True` — subprocess отвязан от API процесса.
- `run.lock` пишет **orchestrator** (как сейчас), backend только читает его для source-of-truth.
- **Нет** in-memory `ActiveLaunch` registry между реквестами — source of truth это `run.lock` + `pipeline_state.json`.
- Restart API → pipeline продолжает работать, реконсиляция при следующем чтении state.

### Interrupt
- Читаем pid из `run.lock`, вызываем `interrupt_launch_process(pid)`.
- Zombie detection: `os.kill(pid, 0)` → `ProcessLookupError` → удаляем stale lock, отвечаем `interrupted=false, reason="process_not_found"`.

## Frontend Architecture

### Директории (создать)
```
web/
  package.json, vite.config.ts, tsconfig.json, tailwind.config.ts
  src/main.tsx, App.tsx
  src/api/client.ts, queryKeys.ts, generated.ts (openapi-typescript output)
  src/api/hooks/  # useRuns, useRun, useAttempt, useRestartPoints, useLogs, useLogStream, useLaunch, useInterrupt, useDelete
  src/pages/      # RunsList, RunDetail, AttemptDetail, LogsView, ReportView
  src/components/ # Layout, NavBar, RunsTable, RunStatusBadge, AttemptsTable, StagesTable,
                  # StageArtifactCard, LaunchModal, LaunchFab, LogPanel, ConfirmDialog,
                  # DeleteMenu, ConfigValidateBanner, MlflowLinkBadge
  src/lib/        # format.ts (mirrors presentation.py), ws.ts, sse.ts
```

### Pages ↔ TUI screens (1:1 mapping)

| TUI screen | Web page | Poll |
|---|---|---|
| `runs_list.py` | `RunsList.tsx` | 5s |
| `run_detail.py` | `RunDetail.tsx` | 2s (running) / 10s (idle) |
| `attempt_detail.py` | `AttemptDetail.tsx` | 2s + WS logs |
| `launch_modal.py` | `LaunchModal.tsx` | on-demand (restart-points) |
| `file_preview_modal.py` | `LogsView.tsx` (full-screen) | WS |

### Networking
- OpenAPI → TypeScript через `openapi-typescript` (script `npm run gen:api`).
- Thin `client.ts` fetch wrapper с `ApiError` классом.
- TanStack Query — cache + invalidation + polling.
- WebSocket hook `useLogStream` с exponential backoff (0.5→8s), ring buffer 10k строк,
  pause-on-scroll + "jump to live" chip.

### MLflow integration
- MVP: ссылка в header `RunDetail.tsx` (X-Frame-Options обычно блокирует iframe).
- Phase 2: iframe с fallback на ссылку.

## CLI integration

В [src/main.py](../../src/main.py) добавить команду `serve` (после `ryotenkai_tui`, перед `version`):

```python
@app.command(name="serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    cors_origins: str = typer.Option("http://localhost:5173", "--cors-origins"),
    reload: bool = typer.Option(False, "--reload"),
    log_level: str = typer.Option("info", "--log-level"),
):
    """Run the FastAPI web backend."""
    from src.api.cli import run_server
    run_server(host=host, port=port, runs_dir=runs_dir,
               cors_origins=cors_origins.split(","), reload=reload, log_level=log_level)
```

`src/api/cli.py::run_server` → `uvicorn.run("src.api.main:create_app", factory=True, ...)`.

**Добавить в pyproject.toml**: `fastapi>=0.115`, `uvicorn[standard]>=0.32`, `websockets>=13`.

## Files to Create / Modify

### Create (Python)
- `src/api/__init__.py`, `main.py`, `cli.py`, `config.py`, `dependencies.py`, `exceptions.py`
- `src/api/schemas/*.py` (10 files)
- `src/api/routers/*.py` (7 files)
- `src/api/services/*.py` (7 files)
- `src/api/ws/{__init__,log_stream}.py`
- `src/pipeline/launch.py` (перенос из `src/tui/launch.py`)

### Modify (Python)
- [src/main.py](../../src/main.py): добавить `serve` command.
- [src/tui/launch.py](../../src/tui/launch.py): заменить содержимое на `from src.pipeline.launch import *` (shim).
- `pyproject.toml`: добавить зависимости.

### Create (Web)
- `web/` (см. структуру выше).

## Tests

### Unit (`src/tests/unit/api/`)
- `test_run_service.py` — fabricated `pipeline_state.json` в `tmp_path`, проверка list/detail/create.
- `test_launch_service.py` — monkeypatch `execute_launch_subprocess`, проверка 409/422.
- `test_log_service.py` — offset chunks, EOF.
- `test_delete_service.py` — mock `TuiDeleteBackend`, 409 при locked run.
- `test_schemas.py` — round-trip `PipelineState.to_dict()` ↔ Pydantic.

### Integration (`src/tests/integration/api/`)
- `test_runs_endpoints.py` — `TestClient(create_app(...))` + fixture runs/.
- `test_launch_endpoints.py` — фейковый orchestrator пишет state, проверяем reflection.
- `test_logs_ws.py` — `websocket_connect`, фоновый writer, проверка delivery.
- `test_delete_endpoint.py` — fixture без mlflow, проверка issue surfacing.

### E2E (`src/tests/e2e/api/test_full_launch_cycle.py`, mark `slow`)
1. Запуск API в `multiprocessing.Process`.
2. POST /runs → POST /launch (с `examples/smoke_cpu.yaml`).
3. Polling GET /runs/{id} до completed/failed (60s timeout).
4. Сравнение API state vs `pipeline_state.json` на диске — equal.
5. WS tail → >0 chunks.

## Verification

1. **Boot**: `ryotenkai serve --runs-dir runs` → `curl .../api/v1/health` = ok.
2. **Parity**: `curl .../api/v1/runs` vs `ryotenkai runs-list` — same data.
3. **Launch**: через web UI запустить pipeline, параллельно `tail -f runs/<new>/pipeline_state.json`, сверить с UI в течение 5 сек.
4. **WS tail**: `tail -f runs/<id>/attempts/attempt_1/pipeline.log` vs LogPanel — identical.
5. **Interrupt**: клик Interrupt в running stage → `pipeline_state.json` получает `interrupted` в течение 30 сек, lock удалён.
6. **API restart mid-run**: kill uvicorn, relaunch → `GET /runs/{id}` возвращает live state, subprocess жив (`ps aux | grep 'src.main train'`).
7. **Delete**: completed run → клик Delete в UI → `runs/<id>` удалён, MLflow children gone.
8. **Coexist**: `ryotenkai tui` в другом терминале видит ран из web UI.
9. **Schema drift**: `cd web && npm run gen:api && npx tsc --noEmit` pass.

## Implementation Sequencing

1. **Skeleton**: deps в pyproject, `src/api/main.py` с `/health`, `serve` command. Загрузиться.
2. **Read-only**: GET /runs, /runs/{id}, /attempts/{n}, /logs chunk. Unit+integration тесты.
3. **Frontend scaffold**: `web/` с Vite, `npm run gen:api`, RunsList + RunDetail + AttemptDetail read-only.
4. **Launch+Interrupt**: перенос `src/tui/launch.py` → `src/pipeline/launch.py` + shim.
   POST /launch, POST /interrupt. LaunchModal в UI.
5. **WebSocket logs**: /logs/stream + `useLogStream`. Plug в AttemptDetail.
6. **Delete / Report / Config-validate**: endpoints + UI controls.
7. **E2E**: `test_full_launch_cycle.py`.
8. **Docs**: `docs/web-ui.md`, note в README.

## MVP Scope vs Phase 2

**In MVP**: все endpoints из таблицы выше, React frontend с пятью pages, detached subprocess launch,
WS log streaming, TUI сосуществует, auth отсутствует (localhost bind), MLflow-ссылка (не iframe).

**Phase 2**: OAuth/JWT + RBAC, multi-user; MLflow iframe когда настроим `frame-ancestors`;
remote engine service (split filesystem ↔ object store); config YAML inline editor;
Playwright e2e; GPU utilization live charts; presence indicators; TUI removal checklist —
`src/tui/*`, `src/tests/unit/tui/*`, `textual` dep, `tui` Typer command.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Subprocess orphan при crash API | `start_new_session=True` + `run.lock` владеет orchestrator |
| Partial write `pipeline_state.json` | `atomic_write_json` уже tempfile+rename |
| MLflow недоступен | 503 с retry hint; Phase 2 — локальная копия envelope в `attempts/.../artifacts/` |
| Race: две launch-команды | `acquire_run_lock` O_CREAT|O_EXCL → вторая получает 409 |
| Config hash mismatch resume | `validate_resume_run` → 422; frontend показывает банер |
| TS ↔ backend schema drift | `npm run gen:api && git diff --exit-code` в CI |
| Log file unbounded | ring buffer 10k строк, `max_log_chunk_bytes` cap |

## Critical Files

- [src/pipeline/state/store.py](../../src/pipeline/state/store.py)
- [src/pipeline/state/models.py](../../src/pipeline/state/models.py)
- [src/pipeline/run_queries.py](../../src/pipeline/run_queries.py)
- [src/pipeline/launch_queries.py](../../src/pipeline/launch_queries.py)
- [src/pipeline/artifacts/base.py](../../src/pipeline/artifacts/base.py)
- [src/tui/launch.py](../../src/tui/launch.py) (будет перенесён)
- [src/tui/live_logs.py](../../src/tui/live_logs.py)
- [src/tui/adapters/delete_backend.py](../../src/tui/adapters/delete_backend.py)
- [src/tui/adapters/presentation.py](../../src/tui/adapters/presentation.py)
- [src/main.py](../../src/main.py) (добавить `serve`)
- [pyproject.toml](../../pyproject.toml) (добавить fastapi/uvicorn/websockets)
