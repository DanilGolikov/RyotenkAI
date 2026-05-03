# План: Mac↔pod transport unification — HTTP-only runtime, SSH bootstrap-only

> Status: **APPROVED scope = Phase 1 + 2 + 3 на той же feature branch** (user decision 2026-05-03). Open questions §13 resolved (см. ниже). Готов к реализации Phase 1 после явной команды.
> Author: daniil + claude (deep-think пасс, helixir + tavily research, code inventory)
> Worktree: `claude/dazzling-rosalind-b482fa`
> Trigger: запрос пользователя «всё взаимодействие через uvicorn после bootstrap», подтверждённый deep-think анализом 2026-05-03.
> Связанные планы:
> - [2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md](2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md) — pull-only ground-truth (IMPLEMENTED)
> - [2026-05-02-fail-fast-prevention-and-log-visibility.md](2026-05-02-fail-fast-prevention-and-log-visibility.md) — import gate + diagnostic grace (IMPLEMENTED, merged)
> - [2026-05-02-monitor-cleanup-and-control-plane-redesign.md](2026-05-02-monitor-cleanup-and-control-plane-redesign.md) — Phase 0 done, Phase 1-3 переопределяется этим планом

---

## 1. Контекст

### 1.1. Где мы сейчас

После Phase 0 (committed в `claude/dazzling-rosalind-b482fa`):
- ✅ Дубль push/pull tail убран
- ✅ `[OUT]/[ERR]` префиксы удалены из operator-facing логов
- ✅ Status line: `running` + Temp + VRAM в GB
- ✅ Code shipping: full `src/` tree

Остаётся **архитектурный долг**: 4 разных канала Mac↔pod с разными правилами/error handling/тестированием:

| Канал | Что несёт | Кто использует |
|---|---|---|
| **HTTP (uvicorn)** | submit/status/stop/heartbeat | Mac orchestrator → runner |
| **WebSocket (uvicorn)** | lifecycle/telemetry events | Mac orchestrator ← runner |
| **SSH `exec_command`** | dmesg, nvidia-smi, marker checks | Mac orchestrator → pod (raw shell) |
| **SSH `rsync`/`tar pipe`/`scp`** | code sync, config + dataset upload, log pull | Mac orchestrator ↔ pod (file transfer) |

Каждый канал — свой error contract, свои тесты, свой mental model.

### 1.2. Что предлагается (target architecture: «Variant C»)

**Сократить до 2 каналов**:

1. **HTTP+WS (uvicorn runner)** — всё runtime: submit/status/stop/heartbeat/diagnostics/resources/logs/file-upload/import-check + WS event stream.
2. **SSH (bootstrap-only, документировано)** — только то, что физически нужно до старта uvicorn:
   - `mkdir -p` workspace
   - `rsync src/` (доставка кода, включая runner)
   - `python -m uvicorn ...` (старт сервера)

После того как `/healthz` отвечает — SSH `exec_command` от Mac на pod **запрещён политикой** (linter rule в Phase 3). RunPod SDK остаётся для cloud-level операций (create/start/terminate pod) — это другая граница.

### 1.3. Real numbers (verified 2026-05-03)

| Артефакт | Размер | Канал сейчас | Канал предлагается |
|---|---|---|---|
| `src/` tree | 3.4 MB | SSH rsync (incremental) | **SSH rsync (bootstrap only)** ✅ keep |
| `pipeline_config.yaml` | 10 KB | scp tar pipe | **HTTP POST upload** |
| Dataset (HelixQL) | 2.5 MB | scp tar pipe | **HTTP POST upload** |
| Dataset (worst case ~10MB) | 10 MB | scp tar pipe | **HTTP POST streaming upload** |
| `trainer.stdio.log` | up to 100 MB per run | SCP scheduled pull (5s) | **HTTP GET range** (chunked) |
| `runner.log` | up to 5 MB per run | SCP scheduled pull (5s) | **HTTP GET range** (chunked) |
| `dmesg` output | <50 KB | SSH exec | **HTTP GET diagnostics** |
| `nvidia-smi` snapshot | <1 KB | SSH exec | **HTTP GET resources/diagnostics** |
| `runtime_check.py --check-source` | n/a | SSH exec | **HTTP POST runtime/import-check** |

**Итого SSH вызовов от Mac на pod**: было ~8 категорий, станет **3 (bootstrap)**.

### 1.4. Что НЕ задевается этим планом

- **RunPod SDK** (Mac → cloud): create_pod, start_pod, terminate_pod. Это API провайдера, не наш runner.
- **Web frontend WS** (`/runs/{id}/.../logs/stream`): отдельный WS на Mac API server, читает локальные файлы. Не пересекается с Mac↔pod transport.
- **Trainer ↔ runner loopback** (`POST /api/v1/internal/events`): уже HTTP, на 127.0.0.1 в pod'е.
- **MLflow tracking** (Mac → MLflow server): отдельная HTTP связь, не наша.

---

## 2. Целевая архитектура

```
┌──────────────────────── Mac orchestrator ──────────────────────────┐
│                                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐               │
│  │TrainingStage │  │PodDeployer │  │ PodRecovery  │               │
│  └──────┬───────┘  └──────┬─────┘  └──────┬───────┘               │
│         │                 │               │                         │
│         │                 │ bootstrap     │ cloud SDK (RunPod)      │
│         │                 │ (3 SSH calls) │ — NOT our runner        │
│         │                 │               │                         │
│         ▼                 ▼               ▼                         │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ JobClient (httpx + websockets, problem+json typed errors)    │ │
│  │   ── lifecycle ────────────                                  │ │
│  │     • submit_job        POST /api/v1/jobs                    │ │
│  │     • get_status        GET  /api/v1/jobs/{id}               │ │
│  │     • request_stop      POST /api/v1/jobs/{id}/stop          │ │
│  │     • subscribe_events  WS   /api/v1/jobs/{id}/events        │ │
│  │   ── operations ───────────                                  │ │
│  │     • send_heartbeat    POST /api/v1/control/heartbeat       │ │
│  │     • health_check      GET  /healthz                        │ │
│  │   ── data ─── (NEW Phase 2) ─                                │ │
│  │     • upload_file       POST /api/v1/files/upload  (chunked) │ │
│  │     • read_log          GET  /api/v1/logs/{name}?offset=     │ │
│  │   ── diagnostics ── (NEW Phase 2) ──                         │ │
│  │     • get_diagnostics   GET  /api/v1/diagnostics?include=    │ │
│  │     • get_resources     GET  /api/v1/resources               │ │
│  │   ── runtime ── (NEW Phase 2) ──                             │ │
│  │     • check_imports     POST /api/v1/runtime/import-check    │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────│────────────────────────────────────┘
                                 │ ssh -L 127.0.0.1:18080 → pod:8080
                                 │ (ОДИН SSH ControlMaster для всего)
                                 ▼
┌────────────────────────── Pod (uvicorn runner) ─────────────────────┐
│  /api/v1/jobs/*          (existing — submit/status/stop)            │
│  /api/v1/jobs/{id}/events (existing — WS lifecycle/telemetry)       │
│  /api/v1/control/heartbeat (existing)                               │
│  /api/v1/internal/events  (existing — trainer loopback)             │
│  /healthz, /readyz, /version (existing)                             │
│                                                                     │
│  /api/v1/diagnostics      ◄── PR-2.1 NEW (dmesg, gpu, kernel)       │
│  /api/v1/resources        ◄── PR-2.2 NEW (instant snapshot)         │
│  /api/v1/logs/{name}      ◄── PR-2.3 NEW (range GET)                │
│  /api/v1/files/upload     ◄── PR-2.4 NEW (multipart streaming)      │
│  /api/v1/runtime/import-check ◄── PR-2.5 NEW                        │
│                                                                     │
│  ProblemDetailsExceptionHandler ─→ application/problem+json (PR-1)  │
│  Pydantic schemas (single source of truth)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1. Bootstrap surface (что остаётся через SSH)

После Phase 2/3 SSH вызовов от Mac будет **ровно 3 категории**, всё bootstrap-only:

```
1. mkdir -p /workspace/runs/<run_id>/                       (1 ssh exec)
2. rsync src/ → pod                                         (1 rsync stream)
3. nohup python -m uvicorn src.runner.main:app ...          (1 ssh exec)
   + wait for /healthz                                      (HTTP, не SSH)
```

После того как `/healthz` отвечает 200 → **никаких SSH `exec_command` от Mac**. Любой такой вызов — bug, ловится import linter rule (Phase 3).

### 2.2. RunPod SDK граница (отдельная, не наша)

```
Mac ──HTTP──► RunPod cloud API
   create_pod(image, gpu, ...)
   start_pod(pod_id)         ← pod recovery после laptop sleep
   terminate_pod(pod_id)
   get_pod(pod_id)           ← status snapshot
```

Это HTTP, но не наш runner. Документируем как **provider boundary** в `SSH_SURFACE.md` (Phase 3.1).

---

## 3. Inventory: что переезжает / что остаётся

| Сейчас | После | Why |
|---|---|---|
| `ssh dmesg ...` | `GET /api/v1/diagnostics?include=dmesg` | structured response, problem+json errors, FastAPI middleware |
| `ssh nvidia-smi ...` | `GET /api/v1/diagnostics?include=gpu` или `GET /api/v1/resources` | то же |
| `ssh tail -n 30 ...trainer.stdio.log` | `GET /api/v1/logs/trainer.stdio?offset=N&limit=K` | range GET, бесшовно с polling 5s |
| `ssh stat -c%s ...` (size check) | `GET /api/v1/logs/trainer.stdio` returns `total_size` in body | один call вместо двух |
| SCP pull `trainer.stdio.log` | `GET /api/v1/logs/trainer.stdio` (chunked) | unified transport |
| SCP pull `runner.log` | `GET /api/v1/logs/runner` | unified transport |
| `ssh check_marker training_complete` | устарело — FSM events через WS | already done (Phase 6.3b) |
| `ssh tar pipe scp config.yaml` | `POST /api/v1/files/upload` (multipart) | structured, OpenAPI'd |
| `ssh tar pipe scp dataset.jsonl` | `POST /api/v1/files/upload` (multipart streaming) | structured |
| `ssh runtime_check.py --check-source` | `POST /api/v1/runtime/import-check` | structured response |
| `rsync src/` | **`rsync src/`** ✅ keep | bootstrap, до старта uvicorn |
| `ssh nohup uvicorn ...` | **`ssh nohup uvicorn ...`** ✅ keep | bootstrap, единственный способ |
| `mkdir -p workspace` | **`mkdir -p workspace`** ✅ keep | bootstrap |

---

## 4. Фазированный план

### Phase 1: Error contract foundation (1 PR, ~300 LOC)

**Цель**: единый transport для ошибок (RFC 9457 problem+json) **до** того как добавлять новые endpoints. Без него каждый новый endpoint унаследует ad-hoc 409/422 detail-shape от существующих.

#### PR-1: ProblemDetails + ErrorCode + FastAPI exception_handler

Уже спроектировано в [2026-05-02-monitor-cleanup-and-control-plane-redesign.md §4 Phase 1](2026-05-02-monitor-cleanup-and-control-plane-redesign.md#phase-1--error-contract-foundation-1-pr-300-loc). Без изменений.

**Файлы (NEW)**: `src/runner/api/errors.py`, `src/runner/api/exception_handlers.py`, `src/api/clients/problem_details.py`.

**Migration policy**: Phase 1 НЕ переписывает existing endpoints. Каждый новый endpoint в Phase 2 сразу использует `APIError(ErrorCode.X, ...)`. Existing endpoints (jobs.py, control.py) мигрируют постепенно (Phase 3.4).

---

### Phase 2: HTTP runtime endpoints (5 PR, ~1100 LOC)

Расширение Phase 2 из предыдущего плана плюс file upload + import check.

#### PR-2.1: `GET /api/v1/diagnostics` (~250 LOC)

**Что**:
- `src/runner/diagnostics/collectors.py` (NEW): `collect_dmesg`, `collect_nvidia_smi`, `collect_kernel_signals`.
- `src/runner/api/diagnostics.py` (NEW): router `GET /diagnostics?include=dmesg,gpu,kernel` → `DiagnosticsResponse`.
- `src/runner/api/schemas/diagnostics.py` (NEW): typed reports.

- `JobClient.get_diagnostics(include) -> DiagnosticsResponse`.
- `training_monitor._collect_death_diagnostics`: SSH probes (827-858) **deleted**, заменены HTTP вызовом.

**Errors**: `DIAGNOSTIC_FAILED` (502), `DIAGNOSTIC_TIMEOUT` (504), `DIAGNOSTIC_INVALID_INCLUDE` (422).

**Тесты**: 7-категорий — positive (all blocks), negative (invalid include), boundary (huge dmesg → truncated), invariants (one collector failure not affecting others), dependency-error (subprocess timeout), regression (postmortem still surfaces kernel info), combinatorial (`include_flags × subprocess_outcome × runner_state`).

#### PR-2.2: `GET /api/v1/resources` (~150 LOC)

**Что**:
- `src/runner/diagnostics/resources.py` (NEW): `collect_resource_snapshot()` returns full GPU/VRAM/CPU/RAM/Temp.
- `src/runner/api/resources.py` (NEW): router.
- `JobClient.get_resources()`.
- `training_monitor._maybe_log_status`: вместо ожидания WS `health_snapshot` — poll endpoint each tick (15s). Status line **гарантированно** появляется даже при early crash (was: только если trainer успел дожить до первого health_snapshot).

**Errors**: `RESOURCES_UNAVAILABLE` (502 — nvidia-smi отсутствует), `RUNNER_NOT_READY` (503).

#### PR-2.3: `GET /api/v1/logs/{name}` (~300 LOC)

**Что**:
- `src/runner/api/logs.py` (NEW):
  - `GET /logs/{name}?offset=&limit_bytes=` → `LogChunkResponse(content, total_size, next_offset, truncated)`.
  - `GET /logs/{name}/size` → `LogSizeResponse` (lightweight для poll).
- `src/runner/api/schemas/logs.py` (NEW): `LogName(StrEnum: TRAINER_STDIO, RUNNER)`, response models.
- `src/pipeline/stages/managers/log_fetcher.py` (NEW): replaces `LogManager`. Использует `JobClient.read_log` вместо SSH/SCP. **Парсит `[OUT]/[ERR]` префикс** при чтении (PR-0.1 logic moves here), Mac logger получает чистые строки.

**Errors**: `LOG_NAME_INVALID` (422 — unknown enum value), `LOG_NOT_AVAILABLE` (404 — file deleted), `LOG_OFFSET_OUT_OF_RANGE` (416 — offset > size).

**Удаляется**: `src/pipeline/stages/managers/log_manager.py` (старый SSH-based puller). Без backward compat — Phase 0 принципы.

**Streaming consideration**: `limit_bytes` hard cap 10 MB, `truncated=True` flag. Большие логи читаются range chunks.

#### PR-2.4: `POST /api/v1/files/upload` (~250 LOC)

**Что**:
- `src/runner/api/files.py` (NEW): `POST /files/upload` принимает `UploadFile` через FastAPI streaming. Записывает на disk через `aiofiles` chunked write (chunk_size=1MB) — memory bounded.
- `src/runner/api/schemas/files.py` (NEW): `FileUploadRequest` (target_path enum + filename), `FileUploadResponse` (bytes_written, sha256).
- Whitelist target paths: `config`, `dataset`, `community-plugins-zip` — anti-path-traversal.

**Mac side**:
- `src/pipeline/stages/managers/deployment/file_uploader.py` rewritten: tar-pipe-scp **deleted**, replaced with HTTP multipart through `JobClient.upload_file(target, local_path)`.

**Errors**: `FILE_TARGET_INVALID` (422 — unknown enum), `FILE_TOO_LARGE` (413 — > config max), `FILE_WRITE_FAILED` (502 — disk full).

**Размеры**: упомянутые в §1.3 datasets ≤10 MB. HTTP overhead через SSH tunnel ~100 ms — invisible vs 3-минутный cold-start.

**Streaming**: FastAPI `UploadFile` + `aiofiles.open(path, 'wb')` async chunked write — memory-safe для любых размеров (verified pattern, [Streaming File Uploads with FastAPI](https://python.plainenglish.io/streaming-file-uploads-and-downloads-with-fastapi-a-practical-guide-ee5be38fdd66)).

**Phase issue — chicken-and-egg**: file upload endpoint доступен только после старта uvicorn. Bootstrap order:
```
1. SSH: mkdir, rsync src/                  ← code shipped first
2. SSH: nohup uvicorn ...                  ← runner up
3. HTTP: wait for /healthz                 ← runner ready
4. HTTP: POST /runtime/import-check        ← validate code
5. HTTP: POST /files/upload (config)       ← deploy config
6. HTTP: POST /files/upload (dataset)      ← deploy data
7. HTTP: POST /files/upload (plugins.zip)  ← reward plugins
8. HTTP: POST /jobs (submit)               ← start trainer
```

#### PR-2.5: `POST /api/v1/runtime/import-check` (~150 LOC)

**Что**:
- `src/runner/api/runtime.py` (NEW): `POST /runtime/import-check` body=`ImportCheckRequest(modules: list[str])` → `ImportCheckReport(per_module: list[ImportResult])`.
- Subprocess isolation: `subprocess.run([sys.executable, "-c", "import X"], timeout=30)` per module — не загружает torch в runner process.
- `code_syncer._verify_importability` rewritten: SSH exec → HTTP call.

**Errors**: `IMPORT_CHECK_TIMEOUT` (504), `IMPORT_CHECK_TOO_MANY_MODULES` (422 — > 50).

**Phase ordering**: import-check теперь HAPPENS AFTER uvicorn started. Если import-check fail — pipeline halts at Stage 1 deployment с named module (как сейчас), просто через HTTP вместо SSH.

---

### Phase 3: Cleanup + enforcement (4 PR, ~600 LOC)

#### PR-3.1: SSH surface contract docs

- `docs/architecture/SSH_SURFACE.md` (NEW): allowlist разрешённых SSH вызовов:
  - `mkdir -p` (workspace bootstrap)
  - `rsync` (code shipping, bootstrap)
  - `nohup python -m uvicorn` (runner launch, bootstrap)
- Документирует что любой другой `ssh.exec_command(...)` от Mac на pod — это violation.

#### PR-3.2: importlinter rule + refactor known violations

- `setup.cfg` (or `.importlinter`): rule запрещающий
  - `src.providers.* → src.pipeline.*`
  - `src.providers.* → src.api.*`
  - `src.providers.* → src.cli.*`
  - `src.runner.* → src.pipeline.*` (runner runtime НЕ должен зависеть от Mac orchestration)
- Refactor существующих violations: `src.providers.runpod.training.provider` → переместить общий код в `src.utils` или `src.contracts`.
- CI gate.

#### PR-3.3: DTO mirror через shared package

- `src/contracts/runner_api/` (NEW): move all schemas из `src/runner/api/schemas/`. Both runner и Mac client импортируют отсюда.
- `scripts/sync_openapi.py` (NEW): regenerate `web/src/api/openapi.json` from FastAPI app, CI gate `test_openapi_freshness`.
- Round-trip contract test (`src/tests/contract/test_dto_round_trip.py`): для каждого endpoint sentence Mac DTO → wire → runner DTO даёт identical object.

#### PR-3.4: Migrate existing endpoints to APIError pattern

- `jobs.py`, `control.py`, `internal.py` — заменить все `HTTPException(detail={...})` на `raise APIError(ErrorCode.X, ...)`.
- Mac `job_client.py` — все методы используют `parse_problem_details` + raise typed exceptions.
- Update `web/src/api/openapi.json` (regenerate via PR-3.3 script).
- **Web frontend impact**: error response shape меняется с `{detail: {code, ...}}` на `{type, title, status, code, detail, ...}`. `web/src/api/client.ts` нужно обновить error parser. Это **separate PR в frontend track** — не блокирует backend release.

---

### Phase 4 (deferred — после Phase 1-3 stabilize): WS → SSE migration

Не начинаем сейчас. Решение принимается **после** того как Phase 1-3 поработают в проде. Если появится конкретный pain (auth middleware, debug сложность, web frontend pain) — открываем follow-up plan.

---

## 5. Frontend impact

### 5.1. Что НЕ задевается

- **Web `useLogStream.ts` (`/runs/{id}/.../logs/stream`)**: это **отдельный WebSocket** на Mac API server (НЕ runner). Читает уже-SCP'нутые локальные файлы. Refactor Mac↔pod transport не задевает.
- **React Query polling** (3-15s интервалы): тривиальный HTTP fetch, формат ответов (JSON) не меняется.
- **ConfigBuilder, Datasets, ActivityFeed**: REST CRUD на Mac API, нет touch'а.

### 5.2. Что задевается (отдельным PR)

- **Error parser**: при PR-3.4 (migrate existing endpoints to ProblemDetails) — web fetch'и получат новый body shape. `web/src/api/client.ts` нужен update.
- **OpenAPI regeneration**: `web/src/api/openapi.json` обновляется в PR-3.3, TypeScript types регенерируются (`web/src/api/schema.d.ts`).

### 5.3. Что МОЖНО добавить в web (out of scope этого плана, но хорошо иметь)

- **Diagnostics panel**: используя новый `GET /api/v1/diagnostics` показать live dmesg/gpu в attempt detail UI.
- **Resources widget**: `GET /api/v1/resources` для real-time GPU usage в TopBar.
- **Live log tail через range polling**: web уже tails локальные файлы, можно перейти на pod-direct read через runner's `GET /logs/{name}` если будет доступ к runner endpoint напрямую.

Это backlog для frontend track, **не блокирующее** этот transport refactor.

---

## 6. Error contract — полный реестр (расширен)

| ErrorCode | HTTP | Domain | Semantic |
|---|---|---|---|
| `JOB_NOT_FOUND` | 404 | jobs | `job_id` не существует или не активен |
| `JOB_STATE_INVALID` | 409 | jobs | FSM transition не разрешён в текущем state |
| `JOB_SPEC_INVALID` | 422 | jobs | Pydantic validation, JSON parse, `extra="forbid"` |
| `JOB_IN_PROGRESS` | 409 | jobs | Active non-terminal blocks submit |
| `PLUGIN_UNPACK_FAILED` | 422 | jobs | ZIP corrupt / community/ extraction error |
| `SPAWN_FAILED` | 422 | jobs | exec() failure: FileNotFoundError, OSError |
| `RUNNER_NOT_READY` | 503 | system | Startup not complete |
| `RUNNER_BUSY` | 409 | system | Supervisor занят |
| `DIAGNOSTIC_FAILED` | 502 | diagnostics | dmesg/nvidia-smi unavailable, permission denied |
| `DIAGNOSTIC_TIMEOUT` | 504 | diagnostics | subprocess.TimeoutExpired |
| `DIAGNOSTIC_INVALID_INCLUDE` | 422 | diagnostics | Unknown key в `?include=` |
| `RESOURCES_UNAVAILABLE` | 502 | resources | nvidia-smi missing / permission denied |
| `LOG_NAME_INVALID` | 422 | logs | Unknown name (не в LogName enum) |
| `LOG_NOT_AVAILABLE` | 404 | logs | Whitelisted name, нет файла |
| `LOG_OFFSET_OUT_OF_RANGE` | 416 | logs | Offset > total_size |
| `FILE_TARGET_INVALID` | 422 | files | Unknown target type |
| `FILE_TOO_LARGE` | 413 | files | Размер > config max |
| `FILE_WRITE_FAILED` | 502 | files | Disk full / permission denied |
| `FILE_HASH_MISMATCH` | 422 | files | Optional checksum verify failed |
| `IMPORT_CHECK_TIMEOUT` | 504 | runtime | Subprocess timeout |
| `IMPORT_CHECK_TOO_MANY_MODULES` | 422 | runtime | > 50 modules |
| `IMPORT_CHECK_INVALID_MODULE_NAME` | 422 | runtime | Module name doesn't match `[a-z_.][a-z_0-9.]*` |
| `WS_REPLAY_TRUNCATED` | (4410 ws) | events | Existing |
| `WS_INVALID_PARAMS` | (4422 ws) | events | Existing |
| `INTERNAL_ERROR` | 500 | system | Catch-all, traceback в logs не в body |

---

## 7. DTO inventory (target state)

### Existing (не меняются в Phase 1)
`JobSpec`, `JobSubmittedResponse`, `JobSnapshotResponse`, `JobStopAcceptedResponse`, `EventResponse`, `InternalEventRequest`, `ControlHeartbeatRequest`, `ControlHeartbeatResponse`.

### Phase 1 NEW
- `ProblemDetails(BaseModel)` — RFC 9457
- `ErrorCode(StrEnum)` — реестр кодов
- `APIError(Exception)` — runner-side
- `APIException(Exception)` — Mac client side

### Phase 2 NEW (по PR)
- **PR-2.1**: `DmesgReport`, `GpuReport`, `KernelSignalsReport`, `DiagnosticsResponse`, `DiagnosticsInclude(StrEnum)`
- **PR-2.2**: `ResourceSnapshot`
- **PR-2.3**: `LogName(StrEnum)`, `LogChunkResponse`, `LogSizeResponse`
- **PR-2.4**: `FileUploadTarget(StrEnum)`, `FileUploadResponse`
- **PR-2.5**: `ImportCheckRequest`, `ImportCheckReport`, `ImportResult`

### Phase 3
- All schemas → moved to `src/contracts/runner_api/`

---

## 8. Risk register (12 рисков, 3 итерации audit)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| RP1 | Web frontend (`web/src/api/openapi.json`) ломается на смене `detail:dict` → `problem+json` | High (после PR-3.4) | Medium | PR-3.3 регенерирует openapi; web update — separate PR в том же merge train |
| RP2 | `dmesg` требует CAP_SYSLOG, в RunPod может быть закрыт | Medium | Medium | Collector ловит `PermissionError` → блок `{error: "permission_denied"}`, не 500 |
| RP3 | HTTP file upload через SSH tunnel: SSH frame size limits | Low | High | FastAPI streaming через `UploadFile` chunked; tested pattern для multipart over tunnel; verify в PR-2.4 integration test |
| RP4 | Multipart upload memory blow-up на 5GB+ файле | Low (current datasets <10MB) | High | `aiofiles` async chunked write (1MB chunks), `MAX_FILE_SIZE` config cap |
| RP5 | Bootstrap chicken-and-egg для file upload (нужно сначала uvicorn) | Low | Low | Bootstrap order documented (§4 PR-2.4), tests проверяют sequence |
| RP6 | Замена SCP-pull на HTTP range увеличивает latency для log tail (5s polling vs SCP) | Low | Low | HTTP roundtrip ~50ms, SCP ~80ms — нейтрально. Range read tail-only — meaningfully faster |
| RP7 | `training_monitor.py` 99.9% bug-prone hotspot — каждый PR в Phase 2 риск регрессии | High | Medium | Каждый PR landит атомарно с зелёными тестами; не batch'ить. Полная регрессия после каждого |
| RP8 | importlinter rule (PR-3.2) ломает CI на existing violations не-трогаемых модулей | Medium | Low | First pass: только новые правила, refactor known violations в том же PR. Document любые grandfathered exceptions |
| RP9 | DTO move в `src/contracts/` (PR-3.3) ломает 50+ import sites | High | Medium | Two-PR approach: первый только move + alias re-exports; второй обновляет imports |
| RP10 | RunPod SDK errors не унифицированы с problem+json | Medium | Low | Out of scope (Mac↔cloud, не наш runner). Document в SSH_SURFACE.md |
| RP11 | `JobClient` transport errors (ssh tunnel down, network) не имеют `code` | High | Medium | Новый `TransportError(code=TRANSPORT_UNREACHABLE)` subclass — отличный от runner-issued errors |
| RP12 | Phase 1 catch-all `Exception` handler маскирует debug errors | Low | Medium | Traceback в logs (level=ERROR), response body 500 INTERNAL_ERROR без traceback (security). Grep test что caught traceback printed |

### Audit iteration 1 — ordering
- ✅ Phase 1 (errors) перед Phase 2 (endpoints) — иначе новые endpoints наследуют ad-hoc.
- ✅ PR-2.5 (import-check) после PR-2.4 (file upload) — потому что import-check happens after deployment, file upload — часть deployment.

### Audit iteration 2 — scope creep
- ✅ Web frontend changes — **отдельный track**, не блокирующий backend.
- ✅ WS → SSE migration — **Phase 4 deferred**, не делаем сейчас (рекомендация подтверждена user).
- ✅ RunPod SDK migration — **out of scope** (provider boundary).

### Audit iteration 3 — production failure modes
- ✅ Identified RP3 (SSH tunnel limits) — нужен integration test для multipart over tunnel.
- ✅ Identified RP9 (DTO move ripples) — split в two PR.
- ✅ Identified RP11 (transport errors) — new TransportError subclass.
- ✅ Verify: bootstrap order chain устойчив к partial failures (если import-check fails → no file upload → no spawn → clean abort). Test it.

---

## 9. Best practices alignment

### RFC 9457 (Problem Details for HTTP APIs)
`ProblemDetails`: `type`, `title`, `status`, `detail`, `instance` + extension `code`, `details`. Соответствует §3 спецификации. Sources: [Swagger Problem Details](https://swagger.io/blog/problem-details-rfc9457-api-error-handling/), [Codecentric deep dive](https://www.codecentric.de/en/knowledge-hub/blog/charge-your-apis-volume-19-understanding-problem-details-for-http-apis-a-deep-dive-into-rfc-7807-and-rfc-9457).

### Kubernetes API conventions
- Subresource pattern: `GET /resources`, `GET /diagnostics?include=...`, `GET /logs/{name}?offset=`.
- LogName enum whitelist vs raw paths — anti-path-traversal как `kubectl logs --container`.
- All state lower-case (matches `JobState.value`).

### Twelve-factor — logs as event streams
Pod-side `trainer.stdio.log` остаётся ground truth (Phase 0 `[OUT]/[ERR]` префиксы preserved on disk). Mac читает через HTTP range. Persistent storage = source of truth.

### FastAPI streaming uploads
PR-2.4 использует FastAPI `UploadFile` + `aiofiles` async chunked write — verified pattern для memory-bounded large file uploads ([Streaming File Uploads with FastAPI](https://python.plainenglish.io/streaming-file-uploads-and-downloads-with-fastapi-a-practical-guide-ee5be38fdd66), [FastAPI 5-method benchmark Reddit](https://www.reddit.com/r/Python/comments/1o04g6v/i_benchmarked_5_different_fastapi_file_upload/)).

### CODE_ERRORS.md project convention
`ErrorCode` ↔ `AppError.code` (one-to-one). UPPER_SNAKE_CASE с domain prefix (`JOB_*`, `DIAGNOSTIC_*`, `LOG_*`, `FILE_*`, `IMPORT_CHECK_*`). `ProblemDetails.code` несёт значение `ErrorCode`. Mac client raises `APIException(code: ErrorCode)`.

### Single transport principle (Kubernetes pattern)
After Phase 2/3:
- Bootstrap (immutable, документирован): SSH 3 calls.
- Runtime (everything observable/operational): HTTP + WS только.
- Provider lifecycle (Mac ↔ cloud): RunPod SDK HTTP, документировано как отдельная boundary.

This **точно mirror Kubernetes**: kubelet boot via static manifest (immutable), runtime через kube-apiserver HTTP, cloud-controller-manager отдельно для cloud lifecycle.

### Frontend best practices
- React Query polling (already used) — **не меняется**, формат ответов JSON остаётся.
- Web frontend WS — independent track, refactor backend transport не задевает.
- TypeScript types regenerated from OpenAPI on PR-3.3 → type safety preserved.

---

## 10. Что мы явно НЕ делаем (rejected alternatives)

1. **Bake runner в Docker image** — медленный dev cycle. Уже отвергнуто пользователем в прошлом плане (thin image migration сделана не зря).
2. **Удалять rsync `src/`** — это bootstrap, incremental sync 3.4 MB → 0 ms на repeat. Rsync benefit реальный.
3. **WS → SSE migration в этом плане** — deferred Phase 4. Не делаем сейчас.
4. **Bake `src/` в Docker volume вместо rsync** — то же что #1, just delayed by one boot cycle.
5. **Двухступенчатый bootstrap (SSH for runner + HTTP for everything else)** — два canonical paths для shipping кода. Хуже чем один.
6. **HTTP migrate RunPod SDK calls** — это external cloud API, не наш runner. Out of scope.
7. **Web frontend WS migration** — independent track, не блокирует backend.
8. **Backward compat для existing endpoints HTTPException → APIError** — Phase 0 принципы: «обратную совместимость не пилим».

---

## 11. Critical files (по фазам)

### Phase 1
```
src/runner/api/errors.py                     # NEW (~120 LOC)
src/runner/api/exception_handlers.py         # NEW (~80 LOC)
src/api/clients/problem_details.py           # NEW (~80 LOC)
src/runner/main.py                           # MODIFIED (~5 LOC)
src/tests/unit/runner/api/test_errors.py     # NEW
src/tests/unit/runner/api/test_exception_handlers.py # NEW
src/tests/unit/api/clients/test_problem_details.py   # NEW
```

### Phase 2 PR-2.1 (diagnostics)
```
src/runner/diagnostics/__init__.py           # NEW
src/runner/diagnostics/collectors.py         # NEW (~150 LOC)
src/runner/api/diagnostics.py                # NEW (~80 LOC)
src/runner/api/schemas/diagnostics.py        # NEW (~60 LOC)
src/api/clients/job_client.py                # MODIFIED (+get_diagnostics)
src/pipeline/stages/training_monitor.py      # MODIFIED (-SSH probes)
src/runner/main.py                           # MODIFIED (mount router)
src/tests/unit/runner/diagnostics/           # NEW dir
```

### Phase 2 PR-2.2 (resources)
```
src/runner/diagnostics/resources.py          # NEW (~80 LOC)
src/runner/api/resources.py                  # NEW (~40 LOC)
src/runner/api/schemas/resources.py          # NEW (~30 LOC)
src/api/clients/job_client.py                # MODIFIED (+get_resources)
src/pipeline/stages/training_monitor.py      # MODIFIED (poll vs WS)
```

### Phase 2 PR-2.3 (logs)
```
src/runner/api/logs.py                       # NEW (~120 LOC)
src/runner/api/schemas/logs.py               # NEW (~50 LOC)
src/pipeline/stages/managers/log_fetcher.py  # NEW (~150 LOC)
src/pipeline/stages/managers/log_manager.py  # DELETED
src/api/clients/job_client.py                # MODIFIED (+read_log/get_log_size)
src/pipeline/stages/training_monitor.py      # MODIFIED (LogManager → LogFetcher)
```

### Phase 2 PR-2.4 (files)
```
src/runner/api/files.py                      # NEW (~150 LOC)
src/runner/api/schemas/files.py              # NEW (~50 LOC)
src/pipeline/stages/managers/deployment/file_uploader.py  # REWRITTEN (~150 LOC)
src/api/clients/job_client.py                # MODIFIED (+upload_file)
```

### Phase 2 PR-2.5 (runtime check)
```
src/runner/api/runtime.py                    # NEW (~80 LOC)
src/runner/api/schemas/runtime.py            # NEW (~40 LOC)
src/pipeline/stages/managers/deployment/code_syncer.py  # MODIFIED (SSH→HTTP for verify)
src/api/clients/job_client.py                # MODIFIED (+check_imports)
docker/training/runtime_check.py             # MODIFIED (HTTP-callable extraction)
```

### Phase 3
```
docs/architecture/SSH_SURFACE.md             # NEW
.importlinter (or pyproject.toml [tool.importlinter])  # NEW
src/contracts/runner_api/                    # NEW dir (DTO move)
scripts/sync_openapi.py                      # NEW
src/tests/contract/test_openapi_freshness.py # NEW
src/tests/contract/test_dto_round_trip.py    # NEW
src/runner/api/jobs.py                       # MODIFIED (HTTPException → APIError)
src/runner/api/control.py                    # MODIFIED
src/runner/api/internal.py                   # MODIFIED
src/api/clients/job_client.py                # MODIFIED (typed exceptions)
web/src/api/client.ts                        # MODIFIED (problem+json parser)
web/src/api/openapi.json                     # REGENERATED
web/src/api/schema.d.ts                      # REGENERATED
```

---

## 12. Verification (по фазам)

### Phase 1
```bash
pytest src/tests/unit/runner/api/test_errors.py \
       src/tests/unit/runner/api/test_exception_handlers.py \
       src/tests/unit/api/clients/test_problem_details.py -v
ruff check src/runner/api/ src/api/clients/
mypy src/runner/api/ src/api/clients/
```

### Phase 2
```bash
# Per-PR unit
pytest src/tests/unit/runner/diagnostics/ -v        # PR-2.1
pytest src/tests/unit/runner/api/test_resources.py  # PR-2.2
pytest src/tests/unit/runner/api/test_logs.py       # PR-2.3
pytest src/tests/unit/runner/api/test_files.py      # PR-2.4
pytest src/tests/unit/runner/api/test_runtime.py    # PR-2.5

# Integration
pytest src/tests/integration/test_http_runtime_e2e.py -v
```

### Phase 3
```bash
pytest src/tests/contract/test_openapi_freshness.py
pytest src/tests/contract/test_dto_round_trip.py
importlinter --config setup.cfg
```

### Manual (after all phases)
1. Запустить pipeline на RunPod.
2. **Acceptance**:
   - В логе deployer ровно 3 SSH вызова (mkdir, rsync, uvicorn-launch). Дополнительный `ssh` exec — bug.
   - Postmortem dmesg/gpu приходит через HTTP (виден в `pipeline.log` как `[HTTP] GET /diagnostics`).
   - Status line caps `running |` появляется через 15s ВСЕГДА (даже при early crash), потому что poll-based, не event-based.
   - File upload (config + dataset) через HTTP — заметна в OpenAPI swagger UI на `http://localhost:18080/docs`.

---

## 13. User decisions (resolved 2026-05-03)

### Q1 RESOLVED — Phase 4 (WS→SSE) deferred

**User choice**: defer, не в этом цикле.

**Импликация**: Phase 4 удалён из текущего roadmap. WebSocket остаётся для events stream. Возвращаемся к WS→SSE migration когда появится конкретный триггер (auth middleware pain, frontend разработчик жалуется на debugability, и т.п.).

### Q2 RESOLVED — frontend update отдельным PR

**User choice**: PR-3.4 backend-only + separate `web/` PR.

**Импликация и risk mitigation**:
- PR-3.4 переписывает existing endpoints с `HTTPException(detail={...})` на `APIError(ErrorCode.X, ...)`. Body shape меняется.
- Web frontend (`web/src/api/client.ts` parser) **СЛОМАЕТСЯ** между мерджем PR-3.4 и frontend PR. Operator увидит broken error text в UI на time window.
- **Mitigation**:
  - Frontend PR должен быть готов **до** PR-3.4 merge (не после).
  - Coordinated merge: открыть оба PR одновременно, мерджить frontend первым (он будет gracefully fallback'ить на оба shape — old и new), потом backend.
  - Documentation в release notes.
- В плане PR-3.4 будет добавлена pre-condition «frontend PR ready и approved».

### Q3 RESOLVED — RunPod SDK документируется как boundary

**User choice (default)**: option A (документировать как provider boundary в `SSH_SURFACE.md`).

**Импликация**: RunPod SDK errors остаются как есть (RunPod-specific exceptions). Не оборачиваются в ProblemDetails. Документируется в SSH_SURFACE.md что Mac↔cloud — отдельный transport, не наш control.

### Q4 RESOLVED — см. Q1 (Phase 4 deferred)

### Q5 RESOLVED — importlinter

**User choice**: `importlinter`.

**Импликация**: PR-3.2 использует `import-linter` (PyPI package), config в `pyproject.toml` секции `[tool.importlinter]`. CI gate через `lint-imports` command. Standard, well-maintained, declarative.

### Q6 NOT ASKED but settled — alias mode SSH сохраняется

Текущий код поддерживает оба (`is_alias_mode` flag в SSHClient). Без изменений в этом плане.

### Scope decision (Q4) — продолжаем на feature branch

**User choice**: Phase 1 + 2 + 3 на той же `claude/dazzling-rosalind-b482fa` ветке. Все commits атомарны. Финальный merge train в RESEACRH когда вся работа landed.

**Risk note**: feature branch проживёт ~1-2 недели. Регулярно подтягиваем main/RESEACRH (если в RESEACRH что-то landит — merge into feature branch, чтобы конфликтов в финальный merge train избежать).

---

## 14. Audit trail (3 итерации)

**Iteration 1** (decomposition):
- ✅ Identified: пользователь явно сказал «всё через uvicorn после bootstrap». Это Variant C (rsync src/ + uvicorn = SSH bootstrap; всё runtime — HTTP).
- ✅ Identified: RunPod SDK — отдельная boundary (Mac↔cloud), не часть Mac↔pod.
- ✅ Identified: Web frontend WS — отдельный канал, не задевается.

**Iteration 2** (scope and risk balancing):
- ✅ File upload через HTTP — viable для текущих <10MB datasets. FastAPI streaming pattern verified.
- ✅ Logs streaming через HTTP — natural fit, replaces SCP scheduled pull.
- ✅ Phase 4 (WS→SSE) — defer. Текущий WS работает.

**Iteration 3** (production readiness):
- ✅ Identified RP3 (SSH tunnel limits) — нужен integration test multipart over tunnel.
- ✅ Identified RP9 (DTO move ripples) — split в two PR.
- ✅ Identified RP12 (catch-all Exception маскирует) — explicit logging at ERROR level.
- ✅ Web frontend impact ограничен — error parser update только при PR-3.4.

---

## 15. Total scope

| Phase | PRs | LOC | Risk | Closes |
|---|---|---|---|---|
| Phase 1 | 1 | ~300 | Low | foundation для error contract |
| Phase 2 | 5 | ~1100 | Medium | runtime через HTTP (diagnostics, resources, logs, files, import-check) |
| Phase 3 | 4 | ~600 | Medium | enforcement (linter, SSH docs, DTO mirror, existing endpoints migrate) |
| **Total** | **10** | **~2000** | **Low-Medium** | full HTTP runtime + bootstrap-only SSH + clean DTO surface |

**Ожидаемый результат**:
- SSH вызовов от Mac на pod: было ~8 категорий, станет **3 bootstrap-only**
- Endpoints HTTP: было 6, станет **11** (+5)
- Error contract: ad-hoc → RFC 9457 problem+json
- DTO source of truth: разнесён → `src/contracts/runner_api/`
- CI gates: openapi freshness + importlinter rules + dto round-trip

---

## 16. Best practices conformance check (post-write)

После написания плана я ещё раз проверил соответствие community best practices:

✅ **Kubernetes API conventions** — subresource patterns, enum whitelist для path params, PR-2.3 уровень-аналог `kubectl logs --since-time`.
✅ **RFC 9457 problem+json** — single error contract, `type`/`title`/`status`/`detail`/`instance` + extensions.
✅ **Twelve-factor logs** — pod-side ground truth + Mac читает чанки, не push'ит весь файл.
✅ **FastAPI multipart streaming** — `UploadFile` + `aiofiles` chunked, memory-bounded для любых размеров.
✅ **OpenAPI single source of truth** — generation from FastAPI, CI gate freshness.
✅ **Static enforcement** — importlinter для boundary violations, не runtime checks.
✅ **CODE_ERRORS.md project convention** — `ErrorCode` enum maps на `AppError.code`, UPPER_SNAKE_CASE с domain prefix.
✅ **YAGNI** — Phase 4 (SSE migration) deferred до конкретного pain.
✅ **Boy scout rule** — каждый touch'нутый файл (`code_syncer`, `file_uploader`, `log_manager`) либо удаляется либо переписывается чисто.

⚠️ **Anti-pattern мы НЕ делаем**:
- ❌ Не пытаемся версионировать API (`/api/v1` остаётся; `/api/v2` появится только если будет breaking change). YAGNI.
- ❌ Не делаем GraphQL endpoint вместо REST. Излишне для MLOps control plane.
- ❌ Не делаем gRPC streaming. Stack — Python+httpx+FastAPI, gRPC overkill.

---

## 17. Next step

**Не начинаю реализацию** до получения ответов на §13 Open Questions от пользователя.

Phase 0 уже committed в feature branch (`claude/dazzling-rosalind-b482fa`), не merged в RESEACRH. Если делать этот план — после approval начинаем с **Phase 1 PR-1** (ProblemDetails foundation) на той же feature branch, накатываем Phase 2 + 3 atomic landings, мерджим всё в RESEACRH одним merge train.
