# План v2: Mac↔pod transport unification — HTTP-only runtime, SSH bootstrap-only (post-packagization)

> **Status:** DRAFT — pending user approval. После approval — реализация на feature branch с atomic merge train в RESEACRH.
> **Дата:** 2026-05-04
> **Author role:** senior Python/DevOps/MLOps инженер + архитектурный проектировщик
> **Триггер:** запрос пользователя «сделать всё взаимодействие с провайдером через uvicorn». Это рефреш плана 2026-05-03-transport-unification-http-only-runtime.md, написанного **до** Phase B packagization (`src/` → 5 uv-workspace пакетов).
>
> **Исторические корни:**
> - [2026-05-03-transport-unification-http-only-runtime.md](2026-05-03-transport-unification-http-only-runtime.md) — v1 плана (paths под старую `src/` топологию, scope тот же).
> - [2026-05-03-monorepo-uv-workspace-packagization.md](2026-05-03-monorepo-uv-workspace-packagization.md) — пакетизация (Phase A+B landed, основа для этого рефреша).
> - [2026-05-02-monitor-cleanup-and-control-plane-redesign.md](2026-05-02-monitor-cleanup-and-control-plane-redesign.md) — Phase 0 (`[OUT]/[ERR]` стрипы, status line) — **landed in RESEACRH**.
> - [2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md](2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md) — pull-only ground-truth — **IMPLEMENTED**.
> - [concurrent-gathering-hippo.md](concurrent-gathering-hippo.md) — provider abstraction PR-1+PR-2 — **landed in RESEACRH**.
>
> **Что изменилось между v1 и v2 (важное):**
> 1. `src/` исчез: 5 пакетов под `packages/{shared,community,pod,providers,control}` с importlinter-enforced графом зависимостей.
> 2. RunPod runner уже живёт в `packages/pod/src/ryotenkai_pod/runner/...` — Phase B перенёс его без изменений API.
> 3. JobClient + SSHTunnelManager уже в `packages/shared/src/ryotenkai_shared/utils/clients/`.
> 4. Provider abstraction — PR-1+PR-2 завершены: manifest-driven registry, capability advertisement, типизированные provider config.
> 5. Phase 0 monitor-cleanup landed: `[OUT]/[ERR]` префиксы скрыты на Mac-стороне, дубль push/pull убран.

---

## 1. Контекст

### 1.1. Где мы сейчас (после Phase B packagization)

**Pod-side (uvicorn runner) уже работает HTTP:**
- `POST /api/v1/jobs` (multipart submit), `GET /api/v1/jobs/{id}`, `POST /api/v1/jobs/{id}/stop`
- `WS /api/v1/jobs/{id}/events` (replay + live)
- `POST /api/v1/control/heartbeat`
- `POST /api/v1/internal/events` (trainer loopback, 127.0.0.1)
- `/healthz`, `/readyz`, `/version`

**Mac-side ходит на pod через 4 канала** (как и в v1 плане):

| Канал | Что несёт | Чем реализовано (post-packagization) |
|---|---|---|
| **HTTP (uvicorn)** | submit/status/stop | [`packages/shared/src/ryotenkai_shared/utils/clients/job_client.py`](packages/shared/src/ryotenkai_shared/utils/clients/job_client.py) (httpx) |
| **WebSocket (uvicorn)** | lifecycle/telemetry events | то же — `JobClient.subscribe_events` (websockets) |
| **SSH `exec_command`** | dmesg, nvidia-smi, tail/head логов, stat sizes, runtime_check | разбросано по `packages/control/src/ryotenkai_control/pipeline/stages/...` |
| **SSH `rsync`/`tar pipe`/`SCP`** | code sync, config + dataset upload, model adapter download | те же deployment helpers |

Каждый канал — свой error contract, свои тесты, свой mental model. **Цель плана** — сократить до **2 каналов** (HTTP+WS runtime + SSH bootstrap-only).

### 1.2. Целевая архитектура (Variant C — без изменений с v1)

**Runtime канал — единственный**: HTTP+WS на uvicorn в pod, через SSH local-port-forward tunnel (один ControlMaster для всего).

**Bootstrap surface — ровно 3 SSH вызова + опциональный 4-й** (см. Open Questions §13.Q-NEW-1):
1. `ssh mkdir -p /workspace/runs/<run_id>/`
2. `rsync packages/ → pod` (incremental, 3.4 MB → 0 ms на repeat)
3. `ssh nohup python -m uvicorn ryotenkai_pod.runner.main:app --host 127.0.0.1 --port 8080 ...`
4. *(возможно)* `ssh uv pip install <plugin-deps>` — см. Q-NEW-1

После того как `/healthz` отвечает 200 — **никаких SSH `exec_command` от Mac на pod**. Ловится AST sentinel test'ом (Phase 3).

### 1.3. Real numbers (verified 2026-05-04)

| Артефакт | Размер | Канал сейчас | Канал предлагается |
|---|---|---|---|
| `packages/` tree (5 пакетов) | ~5 MB | SSH rsync (incremental) | **SSH rsync (bootstrap-only)** ✅ keep |
| `pipeline_config.yaml` | ~10 KB | tar-pipe-SCP | **HTTP POST /files/upload** |
| Dataset (HelixQL) | ~2.5 MB | tar-pipe-SCP | **HTTP POST /files/upload (streaming)** |
| `trainer.stdio.log` | до 100 MB | SCP scheduled pull (5s) | **HTTP GET /logs/{name}?offset=N** (chunked) |
| `runner.log` | до 5 MB | SCP scheduled pull (5s) | **HTTP GET /logs/{name}** (chunked) |
| `dmesg` output | <50 KB | SSH exec | **HTTP GET /diagnostics?include=dmesg** |
| `nvidia-smi` snapshot | <1 KB | SSH exec | **HTTP GET /resources** |
| `runtime_check.py --check-source` | ~stdout | SSH exec | **HTTP POST /runtime/import-check** |
| Model adapter (HF upload) | до 500 MB | SCP stream | **SSH SCP** ✅ keep (out of scope, см. §10) |

**Итого SSH `exec_command` вызовов от Mac на pod**: было ~9 категорий, станет **0 (post-bootstrap)**, всё bootstrap = 3-4 вызова.

### 1.4. Что НЕ задевается этим планом

- **RunPod SDK граница** (Mac → cloud RunPod GraphQL API): create_pod, start_pod, terminate_pod. Это provider boundary, не наш runner. Документируется в `SSH_SURFACE.md`.
- **Pod → Mac data pull** (HFUploader адаптер download через SCP): out of scope. Данные в одну сторону достаточно.
- **Web frontend WS** (`/runs/{id}/.../logs/stream`): отдельный WS на Mac API server (НЕ runner), читает локальные файлы. Не пересекается.
- **Trainer ↔ runner loopback** (`POST /api/v1/internal/events`): уже HTTP на 127.0.0.1.
- **MLflow tracking** (Mac → MLflow server): отдельный HTTP канал.
- **Provider HTTP-ификация (per-provider Mac↔cloud-API через uvicorn)**: out of scope. Plan «всё через uvicorn» = всё **runtime** Mac↔pod через uvicorn. Cloud-level provider operations остаются на собственных SDK (RunPod GraphQL, future AWS EC2 API, etc.).

---

## 2. Целевая архитектура (диаграмма с обновлёнными путями)

```
┌──────────────────────── Mac orchestrator (packages/control) ──────────────────────┐
│                                                                                    │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐                              │
│  │TrainingMonitor│  │GPUDeployer │  │ ResumeService │                              │
│  └──────┬───────┘  └──────┬─────┘  └──────┬───────┘                              │
│         │                 │               │                                        │
│         │                 │ bootstrap     │ cloud SDK (RunPod)                     │
│         │                 │ (3-4 SSH)     │ — ITerminalActionProvider              │
│         │                 │               │                                        │
│         ▼                 ▼               ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │ JobClient (packages/shared, httpx + websockets, problem+json typed)      │    │
│  │   ── lifecycle ────────────                                              │    │
│  │     submit_job        POST /api/v1/jobs                                  │    │
│  │     get_status        GET  /api/v1/jobs/{id}                             │    │
│  │     request_stop      POST /api/v1/jobs/{id}/stop                        │    │
│  │     subscribe_events  WS   /api/v1/jobs/{id}/events                      │    │
│  │   ── operations ───────────                                              │    │
│  │     send_heartbeat    POST /api/v1/control/heartbeat                     │    │
│  │     health_check      GET  /healthz                                      │    │
│  │   ── data (NEW Phase 2) ─                                                │    │
│  │     upload_file       POST /api/v1/files/upload   (chunked multipart)    │    │
│  │     read_log          GET  /api/v1/logs/{name}?offset=                   │    │
│  │     get_log_size      GET  /api/v1/logs/{name}/size                      │    │
│  │   ── diagnostics (NEW Phase 2) ──                                        │    │
│  │     get_diagnostics   GET  /api/v1/diagnostics?include=                  │    │
│  │     get_resources     GET  /api/v1/resources                             │    │
│  │   ── runtime (NEW Phase 2) ──                                            │    │
│  │     check_imports     POST /api/v1/runtime/import-check                  │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────│───────────────────────────────────────────────────┘
                                 │ ssh -L 127.0.0.1:18080 → pod:8080
                                 │ (один SSH ControlMaster для всего)
                                 ▼
┌──────────────────────────── Pod (packages/pod, uvicorn runner) ────────────────────┐
│                                                                                     │
│  packages/pod/src/ryotenkai_pod/runner/api/                                         │
│  ├─ jobs.py          (existing — submit/status/stop)                                │
│  ├─ events.py        (existing — WS lifecycle/telemetry)                            │
│  ├─ control.py       (existing — heartbeat)                                         │
│  ├─ internal.py      (existing — trainer loopback)                                  │
│  │                                                                                  │
│  ├─ errors.py        ◄── PR-1.2 NEW (RFC 9457 problem+json handlers)                │
│  ├─ diagnostics.py   ◄── PR-2.1 NEW                                                 │
│  ├─ resources.py     ◄── PR-2.2 NEW                                                 │
│  ├─ logs.py          ◄── PR-2.3 NEW                                                 │
│  ├─ files.py         ◄── PR-2.4 NEW                                                 │
│  └─ runtime.py       ◄── PR-2.5 NEW                                                 │
│                                                                                     │
│  packages/pod/src/ryotenkai_pod/runner/diagnostics/  ◄── NEW (collectors layer)     │
│                                                                                     │
│  /healthz, /readyz, /version (existing, unchanged)                                  │
│                                                                                     │
│  DTOs импортируются из packages/shared/src/ryotenkai_shared/contracts/runner_api/   │
│  (Phase 0 consolidation — single source of truth для Mac+pod)                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1. Bootstrap surface (что остаётся через SSH)

После Phase 3 от Mac на pod **ровно 3-4 категории SSH-вызовов**, всё bootstrap:

| # | Когда | Команда | Файл |
|---|---|---|---|
| 1 | Stage 1 deployer init | `mkdir -p /workspace/runs/<run_id>` | [runner_launcher.py:?](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/runner_launcher.py) |
| 2 | Stage 1 code shipping | `rsync packages/ → /workspace/` | [code_syncer.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py) (subprocess rsync, не exec_command) |
| 3 | Stage 1 launch | `nohup uvicorn ryotenkai_pod.runner.main:app ...` | [runner_launcher.py:254](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/runner_launcher.py:254) |
| 4 | (под open question Q-NEW-1) | `uv pip install <plugin-deps>` | [dependency_installer.py:102,149](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/dependency_installer.py) |

После того как `/healthz` отвечает 200 → **никаких SSH `exec_command` от Mac**. Любой такой вызов — bug, ловится:
- **Phase 3 PR-3.2:** importlinter contract (forbid `SSHClient` import в non-bootstrap modules).
- **Phase 3 PR-3.1:** AST sentinel test (forbid `.exec_command(` call в non-bootstrap modules) — потому что importlinter ловит только imports, не runtime calls.

### 2.2. RunPod SDK граница (отдельная)

```
Mac packages/providers/.../runpod ──HTTPS──► RunPod GraphQL API
   create_pod(image, gpu, ...)
   start_pod(pod_id)
   terminate_pod(pod_id)
   get_pod(pod_id)
```

Это HTTP, но не наш runner. Документируется в `docs/architecture/SSH_SURFACE.md` (Phase 3.1) как **provider boundary**.

---

## 3. Inventory: что переезжает / что остаётся (post-packagization)

### 3.1. SSH `exec_command` callers (категоризированный список)

#### (a) Bootstrap-legitimate — **остаются** (allowlist)

| File | Function | Line | Purpose |
|---|---|---|---|
| [runner_launcher.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/runner_launcher.py) | `_launch_uvicorn` | 254 | nohup uvicorn (стартует runner) |
| [code_syncer.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py) | sync (rsync via subprocess) | 171, 263 | rsync packages/ — не exec_command, но через ssh wrapper |
| [dependency_installer.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/dependency_installer.py) | run_pip_install / run_uv_install | 102, 149 | `uv pip install` для plugin deps (см. Q-NEW-1) |
| [file_uploader.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/file_uploader.py) | tar-pipe-SCP guards | 293, 328, 357 | mkdir/chown/checksum для bulk upload (см. Q-NEW-2) |

#### (b) Runtime SSH calls — **мигрируем в Phase 2**

| File | Function | Line | Что делает | Куда мигрирует |
|---|---|---|---|---|
| [training_monitor.py](packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py) | `_postmortem_diagnostics` | 818-832 | dmesg, nvidia-smi для post-failure | `GET /api/v1/diagnostics?include=dmesg,gpu,kernel` (PR-2.1) |
| [log_manager.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/log_manager.py) | `_get_file_size` | 222 | `stat -L -c %s` | `GET /api/v1/logs/{name}/size` (PR-2.3) |
| [log_manager.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/log_manager.py) | `download` | 140, 153, 262 | `tail -c`, `head -c` инкрементальный | `GET /api/v1/logs/{name}?offset=N` (PR-2.3) |
| [code_syncer.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py) | `_verify_importability` | 292, 407 | `runtime_check.py --check-source` | `POST /api/v1/runtime/import-check` (PR-2.5) |

#### (c) File transfer — **остаются как SSH** (не exec_command, а subprocess SCP/rsync)

| File | Type | Why keep |
|---|---|---|
| [file_uploader.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/file_uploader.py) | tar-pipe + SCP fallback | **Мигрирует в Phase 2 PR-2.4** на `POST /files/upload` для config + dataset (≤10 MB). Большие файлы остаются в плане как future work. |
| [code_syncer.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py) | rsync packages/ | bootstrap, остаётся |
| [hf_uploader.py](packages/control/src/ryotenkai_control/pipeline/stages/model_retriever/hf_uploader.py) | SCP stream (model adapter download) | **Out of scope** — pod→Mac data pull, не Mac→pod runtime |
| [metrics_buffer_retriever.py:254](packages/control/src/ryotenkai_control/pipeline/stages/model_retriever/metrics_buffer_retriever.py:254) | `cat <buffer>` exec | **Мигрирует в Phase 2 PR-2.3** (logs API generalized) или PR-2.4-extension |

### 3.2. Existing infrastructure status

| Component | Existing? | Where |
|---|---|---|
| **JobClient** (httpx + websockets) | ✅ exists | [packages/shared/.../utils/clients/job_client.py](packages/shared/src/ryotenkai_shared/utils/clients/job_client.py) |
| **SSHTunnelManager** | ✅ exists | [packages/shared/.../utils/clients/ssh_tunnel.py](packages/shared/src/ryotenkai_shared/utils/clients/ssh_tunnel.py) |
| **ControlPlaneHeartbeat** | ✅ exists | [packages/control/.../pipeline/heartbeat/heartbeat.py](packages/control/src/ryotenkai_control/pipeline/heartbeat/heartbeat.py) |
| **MacHeartbeat** (pod-side) | ✅ exists | [packages/pod/.../runner/heartbeat.py](packages/pod/src/ryotenkai_pod/runner/heartbeat.py) |
| **Runner DTOs** (JobSpec etc.) | ✅ but in pod-only | [packages/pod/.../runner/api/schemas.py](packages/pod/src/ryotenkai_pod/runner/api/schemas.py) — **переедет в shared (Phase 0)** |
| **ProblemDetails / RFC 9457** | ❌ NOT implemented | — **создаём в Phase 1** |
| **APIError / ErrorCode** | ❌ NOT implemented | сейчас `HTTPException(detail={"code": "..."})` — **мигрирует в Phase 3** |
| **/diagnostics, /resources, /logs, /files, /runtime** | ❌ NOT implemented | — **создаём в Phase 2** |
| **LogManager** (Mac SSH-based) | ✅ exists, **удаляется** | [packages/control/.../stages/managers/log_manager.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/log_manager.py) → заменяется LogFetcher (HTTP) в PR-2.3 |
| **FileUploader** (tar-pipe-SCP) | ✅ exists, **переписывается** | [packages/control/.../deployment/file_uploader.py](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/file_uploader.py) → HTTP multipart в PR-2.4 |
| **CodeSyncer._verify_importability** | ✅ exists, **переписывается** | [packages/control/.../deployment/code_syncer.py:292](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py:292) → HTTP в PR-2.5 |
| **runtime_check.py** | ✅ exists в pod entrypoint | docker/training/runtime_check.py — **сохраняется как библиотечный модуль**, вызывается через PR-2.5 endpoint |
| **importlinter contracts** | ✅ 7 KEPT, 1 BROKEN (control→pod, pre-existing) | [pyproject.toml `[tool.importlinter]`](pyproject.toml) — **+1 contract в Phase 3.2** |

### 3.3. Текущее состояние error contract

**Сейчас:** `HTTPException(status_code=409, detail={"code": "JOB_STATE_INVALID", ...})` — bare dicts, без `application/problem+json` Content-Type, без RFC 9457 поля `type`/`title`/`instance`.

Найдено в:
- [jobs.py](packages/pod/src/ryotenkai_pod/runner/api/jobs.py): 6 raise sites
- [internal.py](packages/pod/src/ryotenkai_pod/runner/api/internal.py): 3 raise sites
- [control.py](packages/pod/src/ryotenkai_pod/runner/api/control.py): clean (всегда возвращает `ControlHeartbeatResponse`)

Migration policy (Phase 1): новые endpoints (Phase 2) сразу используют `APIError(ErrorCode.X, ...)`. Existing endpoints (jobs, internal) мигрируют в Phase 3.3.

---

## 4. Фазированный план

### Phase 0: DTO consolidation — **NEW в v2** (1 PR, ~250 LOC, ~30 import sites)

**Зачем NEW:** в v1 плана это было PR-3.3 в самом конце, но реальность packagization показывает что лучше сделать первым шагом — иначе все 5 новых endpoints в Phase 2 будут плодить DTO в pod-only месте, потом всё придётся двигать.

#### PR-0: Move runner DTOs into shared contracts

**Adds:**
- `packages/shared/src/ryotenkai_shared/contracts/__init__.py` (NEW)
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/__init__.py` (NEW)
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/jobs.py` (NEW): `JobSpec`, `JobSnapshotResponse`, `JobSubmittedResponse`, `JobStopAcceptedResponse`, `JobState` enum
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/events.py` (NEW): `EventResponse`, custom WS close codes
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/internal.py` (NEW): `InternalEventRequest`
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/control.py` (NEW): `ControlHeartbeatRequest`, `ControlHeartbeatResponse`

**Removes:**
- `packages/pod/src/ryotenkai_pod/runner/api/schemas.py` (содержимое перетекает в shared.contracts.runner_api).

**Modifies:**
- `packages/pod/src/ryotenkai_pod/runner/api/jobs.py`, `events.py`, `control.py`, `internal.py` — `from ... .schemas import` → `from ryotenkai_shared.contracts.runner_api...`
- `packages/shared/src/ryotenkai_shared/utils/clients/job_client.py` — добавить typed deserialization (`JobSnapshotResponse.model_validate(json)` вместо bare dicts)

**Sentinel test:**
- `packages/shared/tests/sentinel/test_runner_api_dto_location.py` — AST-walker assertion: `pod.runner.api.*` НЕ содержит классов с именами `Job*`, `Event*`, `*Response`, `*Request`, `*Spec`. Источник истины — только `shared.contracts.runner_api`.

**Why this PR first:** даёт чистую базу для Phase 1 (ProblemDetails ляжет рядом в `shared.contracts.problem_details`) + Phase 2 (новые endpoints добавляют DTO сразу в правильное место).

**Risks:** `import-linter` enforces shared = leaf. DTOs in shared → both pod (allowed: `pod → shared`) and control via JobClient (allowed: `control → shared`). ✓

---

### Phase 1: Error contract foundation (1 PR, ~350 LOC)

**Цель:** RFC 9457 `application/problem+json` как единый transport для ошибок **до** того как добавлять 5 новых endpoints. Без него каждый новый endpoint унаследует ad-hoc 409/422 detail-shape.

#### PR-1: ProblemDetails + ErrorCode + FastAPI exception handler

**Decision (см. §13.Q-NEW-3):** roll our own лёгкий слой (~150 LOC) поверх `BaseModel`. Не используем `fastapi-problem-details` lib, потому что:
- (+) Контроль над расширениями (`code`, `trace_id`, `request_id`).
- (+) Один способ сериализации Pydantic-модели — конформит styleguide проекта.
- (+) Нет внешнего dep, который добавляет 3 transitive packages.
- (–) Маленькая работа — мы её повторяем.

**Adds:**
- `packages/shared/src/ryotenkai_shared/contracts/problem_details.py` (NEW, ~120 LOC):
  ```python
  class ProblemDetails(BaseModel):
      """RFC 9457 § 3 base + project extensions."""
      type: str = "about:blank"           # URI или "about:blank"
      title: str                          # human-readable summary
      status: int                         # HTTP status
      detail: str | None = None           # human-readable explanation
      instance: str | None = None         # request path
      # Project extensions:
      code: str                           # ErrorCode value (machine-readable)
      trace_id: str | None = None         # for log correlation
      request_id: str | None = None       # FastAPI middleware-set
      errors: list[FieldError] | None = None  # для Pydantic validation

  class ErrorCode(StrEnum):
      """All error codes — domain prefix UPPER_SNAKE_CASE."""
      # jobs domain
      JOB_NOT_FOUND = "JOB_NOT_FOUND"
      JOB_STATE_INVALID = "JOB_STATE_INVALID"
      ...
  ```
- `packages/pod/src/ryotenkai_pod/runner/api/errors.py` (NEW, ~100 LOC):
  ```python
  class APIError(Exception):
      def __init__(self, code: ErrorCode, status: int, detail: str | None = None,
                   title: str | None = None, **extras): ...

  async def api_error_handler(request, exc: APIError) -> JSONResponse: ...
  async def http_exception_handler(request, exc: HTTPException): ...      # adapts existing → problem+json
  async def validation_exception_handler(request, exc: RequestValidationError): ...
  async def generic_exception_handler(request, exc: Exception): ...        # 500 INTERNAL_ERROR + log traceback at ERROR
  ```
- `packages/shared/src/ryotenkai_shared/utils/clients/problem_details.py` (NEW, ~80 LOC):
  ```python
  class APIException(Exception):
      """Mac-side typed exception parsed from server's problem+json."""
      code: ErrorCode
      status: int
      detail: str | None
      ...

  class TransportError(APIException):
      """Tunnel down / network — code=TRANSPORT_UNREACHABLE."""

  def parse_problem_details(response: httpx.Response) -> APIException: ...
  ```

**Modifies:**
- `packages/pod/src/ryotenkai_pod/runner/main.py` — register exception handlers via lifespan startup (~5 LOC).
- `packages/shared/src/ryotenkai_shared/utils/clients/job_client.py` — wrap httpx errors через `parse_problem_details`.

**Tests** (per Phase 1 scope, 7 categorial):
- positive: 1 raise APIError → 1 problem+json response with all fields.
- negative: malformed body / non-200 без problem+json → fallback parse to `TransportError`.
- boundary: empty `errors` list, 600+ status, missing extension fields.
- invariant: каждое поле модели сериализуется null-stripped (RFC 9457 §3.1 рекомендация).
- dependency-error: handler не имеет access to logger → fallback не падает.
- regression: existing `HTTPException(detail={...})` raises пока что → `http_exception_handler` адаптирует к problem+json (preserves `detail` field как `code` ESI).
- logic-specific: `code=ErrorCode.JOB_NOT_FOUND` идёт через wire без mutation.
- combinatorial: 5 ErrorCode × 3 status × 2 with/without trace_id matrix.

**Migration policy:** Phase 1 НЕ переписывает existing endpoints — `http_exception_handler` адаптирует bare-dict стиль к problem+json для backward compat внутри одного релиза. Existing endpoints мигрируют в Phase 3.3.

---

### Phase 2: HTTP runtime endpoints (5 PR, ~1100 LOC)

Каждый PR — атомарный, landится с зелёными тестами + Mac-side caller migration. **Не batch'им** (RP7 — training_monitor.py 99.9-percentile churn hotspot, риск регрессии).

#### PR-2.1: `GET /api/v1/diagnostics` (~250 LOC)

**Pod-side (NEW):**
- `packages/pod/src/ryotenkai_pod/runner/diagnostics/__init__.py`
- `packages/pod/src/ryotenkai_pod/runner/diagnostics/collectors.py` (~150 LOC):
  - `collect_dmesg(filter: KernelFilter | None) -> DmesgReport` — `subprocess.run(["dmesg"], timeout=10)`
  - `collect_nvidia_smi() -> GpuReport` — `subprocess.run(["nvidia-smi", "--query-gpu=...", "--format=csv,noheader"], timeout=5)`
  - `collect_kernel_signals() -> KernelSignalsReport` — grep для OOM/XID/NVRM
- `packages/pod/src/ryotenkai_pod/runner/api/diagnostics.py` (~80 LOC): router `GET /diagnostics?include=dmesg,gpu,kernel`
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/diagnostics.py` (NEW): `DmesgReport`, `GpuReport`, `KernelSignalsReport`, `DiagnosticsResponse`, `DiagnosticsInclude(StrEnum)`

**Mac-side:**
- `packages/shared/src/ryotenkai_shared/utils/clients/job_client.py` — `JobClient.get_diagnostics(include: list[DiagnosticsInclude]) -> DiagnosticsResponse`
- `packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py` lines 818-832 — `_postmortem_diagnostics` (SSH probes) **DELETED**, заменён HTTP вызовом.

**Errors** (новые ErrorCode):
- `DIAGNOSTIC_FAILED` (502 — collector ловит subprocess.SubprocessError)
- `DIAGNOSTIC_TIMEOUT` (504 — subprocess.TimeoutExpired)
- `DIAGNOSTIC_INVALID_INCLUDE` (422 — unknown enum в `?include=`)
- `DIAGNOSTIC_PERMISSION_DENIED` (502 — `dmesg` требует CAP_SYSLOG; collector ловит PermissionError, возвращает блок `{"error": "permission_denied"}` вместо 500). См. RP2.

**Tests (7 категорий):**
- positive: all-blocks include → DiagnosticsResponse заполнен.
- negative: invalid include enum → 422 problem+json `code=DIAGNOSTIC_INVALID_INCLUDE`.
- boundary: huge dmesg output (>10 MB) → truncated, флаг `truncated=true`.
- invariant: один collector failure не валит остальные (graceful per-block error reporting).
- dependency-error: `nvidia-smi` отсутствует → block `error=tool_missing`.
- regression: postmortem assert kernel signals по-прежнему доступны.
- logic-specific: `kernel_signals` filter (OOM grep) работает идемпотентно.
- combinatorial: 3 include flags × 4 subprocess outcomes (ok/timeout/notfound/permission) × 2 runner states (FSM running/idle) = 24 cases.

#### PR-2.2: `GET /api/v1/resources` (~150 LOC)

**Pod-side (NEW):**
- `packages/pod/src/ryotenkai_pod/runner/diagnostics/resources.py` (~80 LOC): `collect_resource_snapshot()` — full GPU/VRAM/CPU/RAM/Temp.
- `packages/pod/src/ryotenkai_pod/runner/api/resources.py` (~40 LOC): router.
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/resources.py` (NEW): `ResourceSnapshot`.

**Mac-side:**
- `JobClient.get_resources() -> ResourceSnapshot`
- `training_monitor._maybe_log_status` — вместо ожидания WS `health_snapshot` poll'ит endpoint каждые 15s. Status line (`running |`) **гарантированно** появляется через 15s даже при early crash (was: только если trainer успел дожить до первого health_snapshot).

**Errors:**
- `RESOURCES_UNAVAILABLE` (502 — nvidia-smi missing / permission denied)
- `RUNNER_NOT_READY` (503 — supervisor не initialized)

**Tests:** 7 категорий аналогично PR-2.1. Особый кейс — *combinatorial*: GPU count × utilization range × VRAM saturation × CPU thermal throttling.

#### PR-2.3: `GET /api/v1/logs/{name}` + `GET /api/v1/logs/{name}/size` (~300 LOC)

**Pod-side (NEW):**
- `packages/pod/src/ryotenkai_pod/runner/api/logs.py` (~120 LOC):
  - `GET /logs/{name}?offset=&limit_bytes=` → `LogChunkResponse(content, total_size, next_offset, truncated)`. **Range pattern** (НЕ HTTP `Range:` header — простой query offset, легче на multipart-mocks).
  - `GET /logs/{name}/size` → `LogSizeResponse(size_bytes)` — lightweight для poll.
  - `LogName(StrEnum)` whitelist: `TRAINER_STDIO`, `RUNNER`. Anti-path-traversal pattern (Kubernetes `kubectl logs --container` style).
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/logs.py` (NEW): response models.

**Mac-side (REPLACES SSH-based LogManager):**
- `packages/control/src/ryotenkai_control/pipeline/stages/managers/log_fetcher.py` (NEW, ~150 LOC) — replaces `LogManager`. Uses `JobClient.read_log` instead of `ssh.exec_command`. Парсит `[OUT]/[ERR]` префикс при чтении (Phase 0 monitor cleanup logic moves here), Mac logger получает чистые строки.
- `packages/control/src/ryotenkai_control/pipeline/stages/managers/log_manager.py` — **DELETED**.
- `packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py` — `LogManager → LogFetcher`.

**Errors:**
- `LOG_NAME_INVALID` (422 — unknown enum value)
- `LOG_NOT_AVAILABLE` (404 — whitelisted name но файла нет)
- `LOG_OFFSET_OUT_OF_RANGE` (416 — offset > total_size)

**Streaming consideration:** `limit_bytes` hard cap 10 MB; для tail-режима Mac читает range chunks 8KB. Используется simple offset model, **не** SSE/WS — потому что polling уже работает (5s) и SSE откладывается на Phase 4.

**Tests:** 7 категорий. *Boundary*: zero-byte файл, файл точно равен limit_bytes, offset = total_size (legal — returns empty). *Combinatorial*: 2 LogName × 4 offset states (zero, mid, exact, OOR) × 3 file states (empty, partial, complete).

#### PR-2.4: `POST /api/v1/files/upload` (~250 LOC)

**Pod-side (NEW):**
- `packages/pod/src/ryotenkai_pod/runner/api/files.py` (~150 LOC): `POST /files/upload` принимает `UploadFile` через FastAPI streaming. Pattern из 2025 best-practice research:
  ```python
  async with aiofiles.open(target_path, 'wb') as f:
      hasher = hashlib.sha256()
      total = 0
      while chunk := await file.read(CHUNK_SIZE):  # CHUNK_SIZE=1MB
          total += len(chunk)
          if total > MAX_FILE_SIZE:
              path.unlink(missing_ok=True)
              raise APIError(ErrorCode.FILE_TOO_LARGE, status=413)
          await f.write(chunk)
          hasher.update(chunk)
      return FileUploadResponse(bytes=total, sha256=hasher.hexdigest())
  ```
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/files.py` (NEW): `FileUploadTarget(StrEnum)` (whitelist: `config`, `dataset`, `community-plugins-zip`), `FileUploadResponse`.
- **Atomic write pattern** (production best practice 2025): write to `<target>.partial`, validate size + hash, then `os.rename()`. Crash mid-upload → no corrupt files.
- **Path whitelist** anti-path-traversal: target_path резолвится **только** через enum mapping, нет user-controlled path strings.

**Mac-side (REPLACES tar-pipe-SCP):**
- `packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/file_uploader.py` — переписывается. Tar-pipe и SCP пути **DELETED**, заменены HTTP multipart через `JobClient.upload_file(target, local_path)`.
- `JobClient.upload_file(target: FileUploadTarget, local_path: Path) -> FileUploadResponse`.

**Errors:**
- `FILE_TARGET_INVALID` (422 — unknown enum)
- `FILE_TOO_LARGE` (413 — > MAX_FILE_SIZE config)
- `FILE_WRITE_FAILED` (502 — disk full / permission denied)
- `FILE_HASH_MISMATCH` (422 — optional client-supplied checksum verify failed)

**Streaming size:** verified pattern для multipart over SSH tunnel. Datasets ≤10 MB сейчас; при росте до 100 MB+ — switch на `streaming_form_data` lib (deferred). MAX_FILE_SIZE: дефолт 100 MB, override через config.

**Phase issue — chicken-and-egg для file upload (RP5):**
Bootstrap order меняется vs текущий:
```
1. SSH: mkdir, rsync packages/        ← code shipped first
2. SSH: nohup uvicorn ...             ← runner up
3. HTTP: wait for /healthz            ← runner ready
4. HTTP: POST /runtime/import-check   ← validate code (PR-2.5)
5. HTTP: POST /files/upload (config)  ← deploy config
6. HTTP: POST /files/upload (dataset)
7. HTTP: POST /files/upload (plugins.zip)
8. HTTP: POST /jobs (submit)
```
Это требует синхронизации `GPUDeployer.execute` order. Тестируется via integration `test_bootstrap_sequence_partial_failure.py`.

**Tests:** 7 категорий. *Boundary*: 0-byte файл, exactly MAX_FILE_SIZE, MAX+1 byte (rejected). *Logic-specific*: SHA-256 round-trip, atomic rename verified (no .partial file post-upload). *Combinatorial*: 3 targets × {0/small/large} × {with/without checksum} = 18.

#### PR-2.5: `POST /api/v1/runtime/import-check` (~150 LOC)

**Pod-side (NEW):**
- `packages/pod/src/ryotenkai_pod/runner/api/runtime.py` (~80 LOC): `POST /runtime/import-check` body=`ImportCheckRequest(modules: list[str])` → `ImportCheckReport(per_module: list[ImportResult])`.
- Subprocess isolation: `subprocess.run([sys.executable, "-c", "import X"], timeout=30)` per module — НЕ загружает torch в runner process (anti-leak).
- `packages/shared/src/ryotenkai_shared/contracts/runner_api/runtime.py` (NEW): `ImportCheckRequest`, `ImportCheckReport`, `ImportResult`.

**Mac-side:**
- `packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py:_verify_importability` — переписывается. SSH exec → `JobClient.check_imports(modules)`.

**Errors:**
- `IMPORT_CHECK_TIMEOUT` (504 — subprocess timeout)
- `IMPORT_CHECK_TOO_MANY_MODULES` (422 — > 50 modules)
- `IMPORT_CHECK_INVALID_MODULE_NAME` (422 — module name doesn't match `[a-z_.][a-z_0-9.]*`)

**Phase ordering:** import-check happens AFTER uvicorn started (chicken-and-egg). Если import-check fails → pipeline halts at Stage 1 deployment с named module (как сейчас), просто через HTTP вместо SSH.

**Tests:** 7 категорий. *Negative*: malicious module name (`os.system('...')`), > 50 modules, invalid python syntax. *Combinatorial*: {good/missing/syntax-broken} × {fast/slow-import} × {1/10/50 modules}.

---

### Phase 3: Cleanup + enforcement (4 PR, ~700 LOC)

#### PR-3.1: SSH surface contract docs + AST sentinel test

**Adds:**
- `docs/architecture/SSH_SURFACE.md` (NEW): allowlist разрешённых SSH вызовов:
  - `ssh exec_command`: только `mkdir -p`, `nohup uvicorn`, (см. Q-NEW-1) `uv pip install`.
  - `subprocess.run(["rsync", ...])`: только bootstrap + bulk file transfer.
  - `subprocess.run(["scp", ...])`: только pod→Mac model adapter download.
- `packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py` (NEW, **критически важный**) — AST-walker assertion: `.exec_command(` calls **НЕ должно быть** в любых modules **кроме** allowlist:
  ```python
  BOOTSTRAP_ALLOWLIST = frozenset({
      "ryotenkai_control.pipeline.stages.managers.deployment.runner_launcher",
      "ryotenkai_control.pipeline.stages.managers.deployment.code_syncer",
      "ryotenkai_control.pipeline.stages.managers.deployment.dependency_installer",
      "ryotenkai_control.pipeline.stages.managers.deployment.file_uploader",
  })

  def test_no_runtime_ssh_exec_command():
      """RP-NEW: importlinter не ловит .exec_command() runtime calls.
      Этот тест walk'ает AST всех packages/control модулей и
      падает если находит .exec_command( вне allowlist'a.
      """
  ```
- Аналогичный sentinel в `packages/providers/tests/sentinel/` для providers (если они когда-нибудь будут SSH-вызывать в runtime).

**Why AST sentinel а не importlinter?** importlinter — static analysis на import statements, не на method calls. Найти `client.exec_command(...)` вне allowlist можно только через AST walker. Sentinel test покрывает provernement gap.

#### PR-3.2: importlinter rule для SSHClient + boundary

**Adds в `pyproject.toml [tool.importlinter]`:**
```toml
[[tool.importlinter.contracts]]
name = "SSHClient может импортироваться только в bootstrap modules"
type = "forbidden"
source_modules = [
    "ryotenkai_control.pipeline.heartbeat",     # heartbeat = HTTP
    "ryotenkai_control.pipeline.stages.training_monitor",
    "ryotenkai_control.pipeline.stages.model_evaluator",
    "ryotenkai_control.pipeline.stages.inference_deployer",
    # ...любой runtime module
]
forbidden_modules = ["ryotenkai_shared.utils.ssh_client"]
ignore_imports = []  # никаких grandfathered violations
```
Bootstrap modules (runner_launcher, code_syncer, dependency_installer, file_uploader) могут импортировать `SSHClient`. Runtime modules — нет.

**Modifies:**
- Refactor известных runtime SSH callsites чтобы они получали SSH через DI (если он реально нужен) ИЛИ удалить SSH wholesale (если перешли на HTTP в Phase 2).

**CI gate:** уже есть `lint-imports` в CI; новый contract автоматически entered. Sentinel test (PR-3.1) — отдельно.

#### PR-3.3: Migrate existing endpoints to APIError pattern

**Modifies:**
- `packages/pod/src/ryotenkai_pod/runner/api/jobs.py` — все `HTTPException(detail={...})` → `raise APIError(ErrorCode.X, ...)`. ~6 raise sites.
- `packages/pod/src/ryotenkai_pod/runner/api/internal.py` — то же. ~3 sites.
- `packages/pod/src/ryotenkai_pod/runner/api/control.py` — ничего, уже clean.
- `packages/shared/src/ryotenkai_shared/utils/clients/job_client.py` — все методы используют `parse_problem_details` + raise typed exceptions. ~5 places.

**Web frontend impact** (separate PR в frontend track, см. Q2):
- `web/src/api/openapi.json` — regenerated. `web/src/api/schema.d.ts` — regenerated.
- `web/src/api/client.ts` — new error parser для problem+json shape.
- Coordinated merge (frontend PR ready ДО backend PR-3.3 merge).

#### PR-3.4: OpenAPI regen + DTO round-trip contract test

**Adds:**
- `scripts/sync_openapi.py` (NEW): regenerates `web/src/api/openapi.json` из FastAPI app (использует `app.openapi()`). Bootstraps runner без RunPod GraphQL вкл. через mock (см. RP9 mitigation).
- `packages/shared/tests/contract/test_openapi_freshness.py` (NEW): CI gate — diff'ит generated OpenAPI с committed; падает если расходятся.
- `packages/shared/tests/contract/test_dto_round_trip.py` (NEW): для каждого endpoint sentence Mac DTO → wire → runner DTO даёт identical object.

**Modifies:**
- `Makefile`: `make sync-openapi`, `make check-openapi`.

---

### Phase 4 (deferred — после Phase 1-3 stabilize): WS → SSE migration

Не начинаем. Решение принимается **после** того как Phase 1-3 поработают в проде. Если появится конкретный pain — открываем follow-up plan. См. v1 §13.Q1 (user choice: defer).

---

## 5. Frontend impact

### 5.1. Что НЕ задевается

- **Web `useLogStream.ts` (`/runs/{id}/.../logs/stream`)**: отдельный WebSocket на Mac API server (НЕ runner). Читает уже-локальные файлы. Refactor Mac↔pod transport не задевает.
- **React Query polling** (3-15s): тривиальный HTTP fetch, формат ответов JSON остаётся.
- **ConfigBuilder, Datasets, ActivityFeed**: REST CRUD на Mac API, нет touch'а.

### 5.2. Что задевается (отдельным PR)

- **Error parser** при PR-3.3: web fetch'и получат новый body shape. `web/src/api/client.ts` нужен update.
- **OpenAPI regeneration** при PR-3.4: `web/src/api/openapi.json` обновляется, TypeScript types регенерируются.

### 5.3. Что МОЖНО добавить в web (out of scope этого плана, бэклог)

- **Diagnostics panel**: используя новый `GET /api/v1/diagnostics` показать live dmesg/gpu в attempt detail UI.
- **Resources widget**: `GET /api/v1/resources` для real-time GPU usage в TopBar.
- **Live log tail через range polling**: web уже tails локальные файлы; можно перейти на pod-direct read через `GET /logs/{name}` если будет доступ к runner endpoint напрямую.

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
| `DIAGNOSTIC_PERMISSION_DENIED` | 502 | diagnostics | Per-block (CAP_SYSLOG missing) |
| `RESOURCES_UNAVAILABLE` | 502 | resources | nvidia-smi missing / permission denied |
| `LOG_NAME_INVALID` | 422 | logs | Unknown name |
| `LOG_NOT_AVAILABLE` | 404 | logs | Whitelisted name но нет файла |
| `LOG_OFFSET_OUT_OF_RANGE` | 416 | logs | Offset > total_size |
| `FILE_TARGET_INVALID` | 422 | files | Unknown target enum |
| `FILE_TOO_LARGE` | 413 | files | Размер > config max |
| `FILE_WRITE_FAILED` | 502 | files | Disk full / permission denied |
| `FILE_HASH_MISMATCH` | 422 | files | Optional checksum verify failed |
| `IMPORT_CHECK_TIMEOUT` | 504 | runtime | Subprocess timeout |
| `IMPORT_CHECK_TOO_MANY_MODULES` | 422 | runtime | > 50 modules |
| `IMPORT_CHECK_INVALID_MODULE_NAME` | 422 | runtime | Pattern mismatch |
| `WS_REPLAY_TRUNCATED` | (4410 ws) | events | Existing |
| `WS_INVALID_PARAMS` | (4422 ws) | events | Existing |
| `TRANSPORT_UNREACHABLE` | (client-side) | transport | Tunnel down / network |
| `INTERNAL_ERROR` | 500 | system | Catch-all, traceback в logs не в body |

---

## 7. DTO inventory (target state, post-Phase 0)

### Phase 0 (consolidation)
**В `packages/shared/src/ryotenkai_shared/contracts/runner_api/`:**
- `jobs.py`: `JobSpec`, `JobSnapshotResponse`, `JobSubmittedResponse`, `JobStopAcceptedResponse`, `JobState` enum
- `events.py`: `EventResponse` + WS close codes
- `internal.py`: `InternalEventRequest`
- `control.py`: `ControlHeartbeatRequest`, `ControlHeartbeatResponse`

### Phase 1 NEW
**В `packages/shared/src/ryotenkai_shared/contracts/`:**
- `problem_details.py`: `ProblemDetails` (RFC 9457), `ErrorCode` enum, `FieldError`, `APIException` (Mac-side)

### Phase 2 NEW (по PR в `runner_api/`)
- **PR-2.1**: `diagnostics.py` — `DmesgReport`, `GpuReport`, `KernelSignalsReport`, `DiagnosticsResponse`, `DiagnosticsInclude`
- **PR-2.2**: `resources.py` — `ResourceSnapshot`
- **PR-2.3**: `logs.py` — `LogName`, `LogChunkResponse`, `LogSizeResponse`
- **PR-2.4**: `files.py` — `FileUploadTarget`, `FileUploadResponse`
- **PR-2.5**: `runtime.py` — `ImportCheckRequest`, `ImportCheckReport`, `ImportResult`

---

## 8. Risk register (15 рисков, 3 итерации audit + новые из packagization)

### Iteration 1 — surface (preserved from v1, paths refreshed)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| RP1 | Web frontend (`web/src/api/openapi.json`) ломается на смене detail:dict → problem+json | High | Medium | PR-3.4 регенерирует openapi; web update — separate PR в том же merge train |
| RP2 | `dmesg` требует CAP_SYSLOG, в RunPod может быть закрыт | Medium | Medium | Collector ловит `PermissionError` → блок `{error: "permission_denied"}`, **не** 500 |
| RP3 | HTTP file upload через SSH tunnel: SSH frame size limits (BufferOverflow) | Low | High | FastAPI streaming через `UploadFile` chunked; tested pattern для multipart over tunnel; verify в PR-2.4 integration test |
| RP4 | Multipart upload memory blow-up на 5GB+ файле | Low (current datasets <10MB) | High | `aiofiles` async chunked write (1MB chunks), `MAX_FILE_SIZE` config cap, atomic write через temp + rename |
| RP5 | Bootstrap chicken-and-egg для file upload (нужно сначала uvicorn, потом upload) | Low | Low | Bootstrap order documented (§4 PR-2.4); integration `test_bootstrap_sequence_partial_failure.py` |
| RP6 | Замена SCP-pull на HTTP range увеличивает latency для log tail (5s polling) | Low | Low | HTTP roundtrip ~50ms, SCP ~80ms — нейтрально/быстрее. Range read tail-only — meaningfully faster |
| RP7 | `training_monitor.py` 99.9% bug-prone hotspot — каждый PR Phase 2 риск регрессии | High | Medium | Каждый PR landит атомарно с зелёными тестами; не batch'им. Полная регрессия после каждого |

### Iteration 2 — deeper (NEW в v2 после packagization)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| RP8 | importlinter lint-imports rule (PR-3.2) ломает CI на existing violations не-трогаемых модулей | Medium | Low | First pass: только new contracts; refactor known violations в том же PR. Pre-existing (control→pod) — отдельно (out of scope) |
| RP9 | DTO move в shared.contracts (Phase 0) ломает 30+ import sites в pod runner + tests | Medium | Medium | Two-step approach: сначала добавляем в shared.contracts с alias re-exports в `pod.runner.api.schemas` (1 PR); потом обновляем imports массово (1 PR). Sentinel test в шаге 2 запрещает старые импорты |
| RP10 | RunPod SDK errors не унифицированы с problem+json | Medium | Low | Out of scope (Mac↔cloud, не наш runner). Document в SSH_SURFACE.md как provider boundary |
| RP11 | `JobClient` transport errors (ssh tunnel down, network) не имеют `code` | High | Medium | Новый `TransportError(code=TRANSPORT_UNREACHABLE)` subclass — отличный от runner-issued errors. Mac code switch'ит на `isinstance(exc, TransportError)` |
| RP12 | Phase 1 catch-all `Exception` handler маскирует debug errors | Low | Medium | Traceback в logs (level=ERROR), response body 500 INTERNAL_ERROR без traceback (security). Grep test что caught traceback printed |
| RP13 (NEW) | importlinter не ловит `.exec_command()` runtime calls (только imports) | High | High | **Phase 3 PR-3.1** AST sentinel test — обязателен для enforcement. Без него после migration кто-то снова добавит SSH вызов и не заметит |
| RP14 (NEW) | runtime_check.py живёт в pod entrypoint (`docker/training/runtime_check.py`) — может быть недоступен после rsync переехал на packages/ | Medium | High | Phase 0 сценарий: runtime_check.py становится либо CLI скриптом в pod package (`packages/pod/.../scripts/runtime_check.py`), либо HTTP-callable библиотекой. Решено: extraction в библиотеку + endpoint Phase 2 PR-2.5 |
| RP15 (NEW) | Cross-package mypy/pyright integration — типы из `shared.contracts` могут не подхватиться runner до полного re-install | Low | Medium | uv-workspace handles dev-installs автоматически; CI gate `make test` reinstalls. Integration test `test_dto_import_from_pod.py` |

### Iteration 3 — production failure modes (NEW + preserved)

| # | Risk | Mitigation |
|---|---|---|
| RP-PROD-1 (NEW) | После rsync packages/, uvicorn стартует со stale `__pycache__` в pod, импортируя устаревшие модули | rsync flag `--delete-excluded` + `find ... -name __pycache__ -exec rm -rf` step в bootstrap script. Verified in PR-2.5 import-check (catches stale state). |
| RP-PROD-2 (NEW) | DTO Pydantic v2 model_dump() between control (новые) и pod (старые после restart) → wire format mismatch | Контракт-test `test_dto_round_trip.py` (PR-3.4) проверяет version compatibility. Schema versioning — out of scope (см. v1 §1.4: API не версионируется до breaking change) |
| RP-PROD-3 (NEW) | Heartbeat HTTP может конкурировать с большим file upload за SSH tunnel bandwidth | Heartbeat priority through separate httpx client with 5s timeout; file upload использует chunked write (event loop yields) |
| RP-PROD-4 | Identified RP3 (SSH tunnel limits) — нужен integration test multipart over tunnel | PR-2.4 test_files_upload_over_ssh_tunnel.py — реальный rsync через mocked tunnel + verify |
| RP-PROD-5 (NEW) | Phase 0 DTO move ломает release pinning: pod image pre-Phase-0 имеет DTOs в `pod.runner.api.schemas`, post-Phase-0 — в `shared.contracts.runner_api`. Mac-side rebuild должен соответствовать pod image | Coordinated release: PR-0 включает version bump в `pod-image-tag` env var; Mac client требует pod image >= `2026.05.04`. Old image fails to import → falls back to "old client + old image" combo |
| RP-PROD-6 | RP12 (catch-all Exception маскирует) | Explicit log at ERROR level + Sentry-style traceback hash в response (для devops correlation) — preserved from v1 |

### Iteration 4 — additional findings (после deep-think audit прохода)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| RP16 (NEW) | Multi-worktree DTO drift: одновременно работают 2+ feature branches которые правят `shared.contracts.runner_api` → merge-conflicts на model fields, потенциально несовместимые wire formats при rsync на pod | Medium | Medium | Feature branches always rebase на RESEACRH перед merge; AST sentinel test phase 0 catches phantom imports if branch-A merges first. Worktrees are local-only, не deployment-active |
| RP17 (NEW) | Mid-migration mixed-state runtime: между Phase 1 и Phase 2 PR-2.1 могут существовать endpoints с problem+json (новые) и bare `HTTPException` (legacy) одновременно — Mac client должен handle оба shapes | Medium | Medium | **Phase 1 ВКЛЮЧАЕТ** универсальный `http_exception_handler` который адаптирует ALL `HTTPException` в problem+json **с дня 1**. Wire-shape unified от Phase 1 onwards. Mac client всегда видит одну форму. Existing endpoints (jobs.py, internal.py) НЕ требуют immediate code change — handler адаптирует |
| RP18 (NEW) | SSH tunnel bandwidth contention: file upload (PR-2.4) + heartbeat (existing) + WS event stream (existing) over одного SSH tunnel; heartbeat может starve если file upload забивает pipe | Medium | Low | OS schedule'ит fairly на TCP layer; httpx async event loop yields на await; heartbeat имеет short timeout (5s default) → если timeout — skip ping (heartbeat tolerant). Test через mock tunnel + concurrent upload+heartbeat |
| RP19 (NEW) | `/diagnostics` DOS amplification: клиент спамит endpoint → dmesg subprocess churn, kernel ring buffer rotation, real warning lost; `nvidia-smi` blocking call competes с trainer GPU | Medium | Medium | **Pod-side rate limit** per endpoint: `/diagnostics` 1 req/s, `/resources` 1 req/s, остальные default. Use simple in-memory token bucket (no slowapi dep needed для loopback-only API). `nvidia-smi` is read-only, не блокирует CUDA |
| RP20 (NEW) | Uvicorn lifespan ordering: exception handlers ДОЛЖНЫ register'ить ДО первого request; если регистрируются в lifespan startup async — race condition → первые 500 ответы могут уйти без problem+json | Low | Medium | **Critical pattern**: handlers register'ятся синхронно в `app = FastAPI(exception_handlers={...})` constructor (не в lifespan). Test: launch uvicorn + immediately POST malformed → assert problem+json в response (не raw HTTPException) |
| RP21 (NEW) | `/files/upload` mid-stream client cancel: клиент (Mac) рвёт httpx stream → pod имеет `<target>.partial` файл → disk fill over time | Low | Medium | **try/finally** + `path.unlink(missing_ok=True)` на любую exception включая `ClientDisconnected`. Test через `httpx.AsyncClient` abort mid-upload. Plus periodic cleanup task `cleanup_partial_uploads_older_than_1h` (Phase 2 PR-2.4 включает) |
| RP22 (NEW) | AST sentinel allowlist может стать stale: новый bootstrap module добавляется, но не registered в `BOOTSTRAP_ALLOWLIST` → sentinel test fails, разраб обходит через add'ение в allowlist без обоснования | Medium | Low | Allowlist живёт в отдельном файле `packages/control/tests/sentinel/bootstrap_allowlist.py` с docstring + комментариями `# why X is bootstrap`. Любой PR расширяющий allowlist требует review с justification. Альтернативно: allowlist auto-derives from модулей содержащих `# bootstrap-module` magic comment (lighter, less rigorous) |

**Итого**: 27 рисков, 24 closed by design, 3 retained as policy (RP10, RP11, RP12). Все documented с concrete mitigation.

---

## 9. Best practices alignment (refresh)

### RFC 9457 (Problem Details for HTTP APIs)
`ProblemDetails`: `type`, `title`, `status`, `detail`, `instance` + extensions `code`, `trace_id`, `request_id`, `errors`. Соответствует §3 спецификации. Sources: [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457.html), [Swagger Problem Details deep dive](https://swagger.io/blog/problem-details-rfc9457-api-error-handling/), [fastapi-problem-details lib reference](https://github.com/g0di/fastapi-problem-details).

**Decision (Q-NEW-3):** roll our own (~150 LOC) вместо `fastapi-problem-details` или `fastapi-problem`. Reasoning: project styleguide enforces explicit Pydantic models, lib add transitive deps, RFC 9457 простой.

### FastAPI streaming uploads (2025 best practices)
PR-2.4 использует `UploadFile` + `aiofiles.open(path, 'wb')` async chunked write — verified production pattern для memory-bounded uploads ([Streaming File Uploads with FastAPI](https://python.plainenglish.io/streaming-file-uploads-and-downloads-with-fastapi-a-practical-guide-ee5be38fdd66), [OneUptime 2026 guide](https://oneuptime.com/blog/post/2026-02-02-fastapi-file-uploads/view), [Async File Uploads Gigabyte-Scale](https://medium.com/@connect.hashblock/async-file-uploads-in-fastapi-handling-gigabyte-scale-data-smoothly-aec421335680)).
- 1 MB chunk size (verified benchmark: 500MB at 1MB chunks = 21s vs 64KB chunks = 48s).
- Track `total_size` per chunk + early reject `> MAX_FILE_SIZE`.
- SHA-256 incrementally в том же loop.
- Atomic temp-file-then-rename (no corrupt files on crash).
- Path whitelist enum (anti-traversal).

### HTTP log streaming (2025 best practices)
PR-2.3 использует **simple offset-based query** (НЕ HTTP `Range:` header) — easier к mock, легче клиенту, range header overkill для tail-style polling. Pattern из [Streaming APIs for Beginners](https://python.plainenglish.io/streaming-apis-for-beginners-python-fastapi-and-async-generators-848b73a8fc06): offset → seek → chunked yield. Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no` (anti-proxy-buffering).

### Kubernetes API conventions
- Subresource pattern: `GET /resources`, `GET /diagnostics?include=...`, `GET /logs/{name}?offset=`.
- LogName enum whitelist vs raw paths — `kubectl logs --container` style.
- All states lower-case (matches `JobState.value`).

### Twelve-factor — logs as event streams
Pod-side `trainer.stdio.log` остаётся ground truth (Phase 0 monitor cleanup `[OUT]/[ERR]` префиксы preserved on disk). Mac читает через HTTP range. Persistent storage = source of truth.

### CODE_ERRORS.md project convention
`ErrorCode` ↔ `AppError.code` (one-to-one). UPPER_SNAKE_CASE с domain prefix. `ProblemDetails.code` несёт значение `ErrorCode`. Mac client raises `APIException(code: ErrorCode)`.

### Single transport principle (Kubernetes pattern)
After Phase 2/3:
- Bootstrap (immutable, документирован): SSH 3-4 calls.
- Runtime (everything observable/operational): HTTP + WS only.
- Provider lifecycle (Mac ↔ cloud): provider-specific SDK, документировано как separate boundary.

This **точно mirror Kubernetes**: kubelet boot via static manifest (immutable), runtime через kube-apiserver HTTP, cloud-controller-manager отдельно.

### Static enforcement (importlinter + AST sentinel)
- importlinter lint-imports — **import statements only** (RFC: [import-linter contract types](https://import-linter.readthedocs.io/en/latest/contract_types.html)).
- AST sentinel — runtime call enforcement (`.exec_command(`). **Critical addition в v2**, потому что v1 не учитывал что importlinter не ловит method calls.

### Frontend best practices
- React Query polling (already used) — **не меняется**.
- Web frontend WS — independent track.
- TypeScript types regenerated from OpenAPI on PR-3.4 → type safety preserved.

---

## 10. Что мы явно НЕ делаем (rejected alternatives)

1. **Bake runner в Docker image** — медленный dev cycle. Уже отвергнуто пользователем в прошлом плане (thin image migration сделана не зря).
2. **Удалять rsync `packages/`** — bootstrap, incremental sync 5 MB → 0 ms на repeat.
3. **WS → SSE migration в этом плане** — deferred Phase 4.
4. **Bake `packages/` в Docker volume вместо rsync** — то же что #1.
5. **Двухступенчатый bootstrap (SSH for runner + HTTP for everything else)** — два canonical paths для shipping кода. Хуже чем один.
6. **HTTP migrate RunPod SDK calls** — это external cloud API, не наш runner. Out of scope.
7. **Web frontend WS migration** — independent track.
8. **Backward compat для existing endpoints HTTPException → APIError** — Phase 0 принципы: «обратную совместимость не пилим», но `http_exception_handler` (PR-1) адаптирует bare-dict → problem+json для in-flight запросов. Existing endpoints мигрируют в PR-3.3.
9. **Pod → Mac data pull migration to HTTP** (HFUploader SCP stream) — out of scope, отдельный план если будет нужно.
10. **HTTP migrate dependency_installer** (`uv pip install`) — Q-NEW-1 решает; default — keep as SSH (bootstrap-extended).
11. **Schema versioning (`/api/v1` → `/api/v2`)** — YAGNI, появится только при breaking change.

---

## 11. Critical files (по фазам, **новые пути**)

### Phase 0
```
packages/shared/src/ryotenkai_shared/contracts/__init__.py                 # NEW
packages/shared/src/ryotenkai_shared/contracts/runner_api/__init__.py      # NEW
packages/shared/src/ryotenkai_shared/contracts/runner_api/jobs.py          # NEW (move from pod)
packages/shared/src/ryotenkai_shared/contracts/runner_api/events.py        # NEW
packages/shared/src/ryotenkai_shared/contracts/runner_api/internal.py      # NEW
packages/shared/src/ryotenkai_shared/contracts/runner_api/control.py       # NEW
packages/pod/src/ryotenkai_pod/runner/api/schemas.py                       # DELETED
packages/pod/src/ryotenkai_pod/runner/api/jobs.py                          # MODIFIED (imports)
packages/pod/src/ryotenkai_pod/runner/api/events.py                        # MODIFIED
packages/pod/src/ryotenkai_pod/runner/api/control.py                       # MODIFIED
packages/pod/src/ryotenkai_pod/runner/api/internal.py                      # MODIFIED
packages/shared/src/ryotenkai_shared/utils/clients/job_client.py           # MODIFIED (typed parsing)
packages/shared/tests/sentinel/test_runner_api_dto_location.py             # NEW
```

### Phase 1
```
packages/shared/src/ryotenkai_shared/contracts/problem_details.py          # NEW (~150 LOC)
packages/pod/src/ryotenkai_pod/runner/api/errors.py                        # NEW (~120 LOC)
packages/shared/src/ryotenkai_shared/utils/clients/problem_details.py      # NEW (~80 LOC)
packages/pod/src/ryotenkai_pod/runner/main.py                              # MODIFIED (~5 LOC, register handlers)
packages/pod/tests/unit/runner/api/test_errors.py                          # NEW
packages/pod/tests/unit/runner/api/test_exception_handlers.py              # NEW
packages/shared/tests/unit/utils/clients/test_problem_details.py           # NEW
```

### Phase 2 PR-2.1 (diagnostics)
```
packages/pod/src/ryotenkai_pod/runner/diagnostics/__init__.py              # NEW
packages/pod/src/ryotenkai_pod/runner/diagnostics/collectors.py            # NEW (~150 LOC)
packages/pod/src/ryotenkai_pod/runner/api/diagnostics.py                   # NEW (~80 LOC)
packages/shared/src/ryotenkai_shared/contracts/runner_api/diagnostics.py   # NEW (~60 LOC)
packages/shared/src/ryotenkai_shared/utils/clients/job_client.py           # MODIFIED (+get_diagnostics)
packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py # MODIFIED (-SSH probes lines 818-832)
packages/pod/src/ryotenkai_pod/runner/main.py                              # MODIFIED (mount router)
packages/pod/tests/unit/runner/diagnostics/                                # NEW dir
```

### Phase 2 PR-2.2 (resources)
```
packages/pod/src/ryotenkai_pod/runner/diagnostics/resources.py             # NEW (~80 LOC)
packages/pod/src/ryotenkai_pod/runner/api/resources.py                     # NEW (~40 LOC)
packages/shared/src/ryotenkai_shared/contracts/runner_api/resources.py     # NEW (~30 LOC)
packages/shared/src/ryotenkai_shared/utils/clients/job_client.py           # MODIFIED (+get_resources)
packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py # MODIFIED (poll vs WS)
```

### Phase 2 PR-2.3 (logs)
```
packages/pod/src/ryotenkai_pod/runner/api/logs.py                          # NEW (~120 LOC)
packages/shared/src/ryotenkai_shared/contracts/runner_api/logs.py          # NEW (~50 LOC)
packages/control/src/ryotenkai_control/pipeline/stages/managers/log_fetcher.py  # NEW (~150 LOC)
packages/control/src/ryotenkai_control/pipeline/stages/managers/log_manager.py  # DELETED
packages/shared/src/ryotenkai_shared/utils/clients/job_client.py           # MODIFIED (+read_log/get_log_size)
packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py # MODIFIED (LogManager → LogFetcher)
```

### Phase 2 PR-2.4 (files)
```
packages/pod/src/ryotenkai_pod/runner/api/files.py                         # NEW (~150 LOC)
packages/shared/src/ryotenkai_shared/contracts/runner_api/files.py         # NEW (~50 LOC)
packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/file_uploader.py  # REWRITTEN (~150 LOC)
packages/shared/src/ryotenkai_shared/utils/clients/job_client.py           # MODIFIED (+upload_file)
```

### Phase 2 PR-2.5 (runtime check)
```
packages/pod/src/ryotenkai_pod/runner/api/runtime.py                       # NEW (~80 LOC)
packages/shared/src/ryotenkai_shared/contracts/runner_api/runtime.py       # NEW (~40 LOC)
packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/code_syncer.py  # MODIFIED (SSH→HTTP for verify)
packages/shared/src/ryotenkai_shared/utils/clients/job_client.py           # MODIFIED (+check_imports)
docker/training/runtime_check.py                                            # MODIFIED (HTTP-callable extraction) OR migrate to packages/pod/.../scripts/
```

### Phase 3
```
docs/architecture/SSH_SURFACE.md                                            # NEW
pyproject.toml [tool.importlinter]                                          # MODIFIED (+1 contract)
packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py         # NEW (AST walker)
scripts/sync_openapi.py                                                     # NEW
packages/shared/tests/contract/test_openapi_freshness.py                    # NEW
packages/shared/tests/contract/test_dto_round_trip.py                       # NEW
packages/pod/src/ryotenkai_pod/runner/api/jobs.py                           # MODIFIED (HTTPException → APIError)
packages/pod/src/ryotenkai_pod/runner/api/internal.py                       # MODIFIED
packages/shared/src/ryotenkai_shared/utils/clients/job_client.py            # MODIFIED (typed exceptions)
web/src/api/client.ts                                                       # MODIFIED (problem+json parser, separate frontend PR)
web/src/api/openapi.json                                                    # REGENERATED
web/src/api/schema.d.ts                                                     # REGENERATED
```

---

## 12. Verification (по фазам, **обновлённые команды**)

### Phase 0
```bash
uv sync && uv run pytest packages/shared/tests/sentinel/test_runner_api_dto_location.py
uv run pytest packages/pod/tests/unit/runner/api/  # all green с новыми imports
uv run lint-imports                                # 7 contracts kept
uv run mypy packages/pod packages/shared           # types resolve through contracts
```

### Phase 1
```bash
uv run pytest packages/pod/tests/unit/runner/api/test_errors.py \
              packages/pod/tests/unit/runner/api/test_exception_handlers.py \
              packages/shared/tests/unit/utils/clients/test_problem_details.py -v
uv run ruff check packages/pod/src/ryotenkai_pod/runner/api/ \
                  packages/shared/src/ryotenkai_shared/utils/clients/ \
                  packages/shared/src/ryotenkai_shared/contracts/
uv run mypy packages/pod packages/shared
```

### Phase 2
```bash
# Per-PR unit
uv run pytest packages/pod/tests/unit/runner/diagnostics/ -v       # PR-2.1
uv run pytest packages/pod/tests/unit/runner/api/test_resources.py # PR-2.2
uv run pytest packages/pod/tests/unit/runner/api/test_logs.py      # PR-2.3
uv run pytest packages/pod/tests/unit/runner/api/test_files.py     # PR-2.4
uv run pytest packages/pod/tests/unit/runner/api/test_runtime.py   # PR-2.5

# Mac-side migrations
uv run pytest packages/control/tests/unit/pipeline/stages/managers/ -v

# Integration
uv run pytest packages/control/tests/integration/test_http_runtime_e2e.py -v
```

### Phase 3
```bash
uv run pytest packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py
uv run pytest packages/shared/tests/contract/test_openapi_freshness.py
uv run pytest packages/shared/tests/contract/test_dto_round_trip.py
uv run lint-imports  # 8 contracts kept (новый SSHClient boundary)
```

### Manual (after all phases)
1. Запустить pipeline на RunPod: `.venv/bin/ryotenkai run start -c <config>`.
2. **Acceptance:**
   - В логе deployer ровно 3 SSH `exec_command` (mkdir, uvicorn-launch, и rsync wrapper). Дополнительный `ssh` exec — bug, AST sentinel test должен был поймать в CI.
   - Postmortem dmesg/gpu приходит через HTTP (виден в `pipeline.log` как `[HTTP] GET /diagnostics`).
   - Status line `running |` появляется через 15s **ВСЕГДА** (даже при early crash), потому что poll-based, не event-based.
   - File upload (config + dataset) через HTTP — заметна в swagger UI на `http://localhost:18080/docs`.

---

## 13. Open questions (новые после packagization)

### Q-NEW-1 — `dependency_installer.py` (`uv pip install` через SSH) мигрировать или оставить?

**Сейчас:** [dependency_installer.py:102,149](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/dependency_installer.py) делает `ssh.exec_command('uv pip install ...')` для plugin deps после rsync.

**Варианты:**
- **(A)** Оставить как SSH bootstrap-extended. Документировать в SSH_SURFACE.md. Один SSH вызов, простая семантика. Но: stdout streaming для long-running pip install сложен через exec_command, текущий код просто wait'ит до конца.
- **(B)** Мигрировать в Phase 2 PR-2.6 `POST /api/v1/runtime/install-deps` со streaming response (SSE-like). Pro: единый transport, structured errors, progress streaming. Con: добавляет 6-й PR в Phase 2 (~250 LOC), увеличивает scope, нужен subprocess streaming pattern в pod.

**Рекомендация:** **(A)** — keep as SSH bootstrap-extended. Reasoning: pip install — bootstrap-adjacent (происходит ровно один раз после rsync, до старта uvicorn в случае plugin deps). Migration в HTTP даёт мало (один вызов на job, не runtime hot path). YAGNI.

**Status:** **AWAITING USER DECISION**.

### Q-NEW-2 — `file_uploader.py` SSH guards (mkdir, chown, hash check)

**Сейчас:** [file_uploader.py:293,328,357](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/file_uploader.py) делает SSH guards вокруг tar-pipe-SCP: mkdir target dir, chown, hash verify.

**После PR-2.4:** tar-pipe-SCP **DELETED**, заменён HTTP `POST /files/upload`. Что делать с guards?

**Варианты:**
- **(A)** Удалить — endpoint сам делает `mkdir -p` (Pod-side `pathlib.Path.mkdir(parents=True)`).
- **(B)** Оставить SSH guards для корнер-кейсов (нестандартные target paths).

**Рекомендация:** **(A)** — endpoint самодостаточен, Pod-side не нужны guards.

**Status:** RESOLVED **(A)**.

### Q-NEW-3 — RFC 9457 lib vs custom

**Варианты:**
- **(A)** Custom (~150 LOC, full control).
- **(B)** [`fastapi-problem-details`](https://github.com/g0di/fastapi-problem-details) (battle-tested, auto-converts HTTPException, validation errors).
- **(C)** [`fastapi-problem`](https://nrwldev.github.io/fastapi-problem/) (slim, exception-classes-based).

**Рекомендация:** **(A)** Custom — потому что project styleguide enforces explicit Pydantic models, нужен tight контроль над extensions (trace_id, request_id, errors), нет smell of "magic auto-conversion".

**Status:** RESOLVED **(A)** — но обновляемся, если на implementation выйдут конкретные boilerplate pains.

### Q-NEW-4 — runtime_check.py: где живёт после plan?

**Сейчас:** `docker/training/runtime_check.py` — standalone script, копируется в pod образ.

**После PR-2.5:** endpoint `POST /api/v1/runtime/import-check`. Где живёт логика?

**Варианты:**
- **(A)** Migrate `runtime_check.py` → `packages/pod/src/ryotenkai_pod/runner/runtime_check.py` (proper package member). Endpoint импортирует `from .runtime_check import check_imports`. Standalone script теряется.
- **(B)** Keep standalone CLI script, endpoint subprocess'ит его (как было).
- **(C)** Migrate в pod package + retain CLI entrypoint via `console_scripts` (`ryotenkai-runtime-check`).

**Рекомендация:** **(C)** — migrate в pod package с CLI entrypoint. Pro: единый source of truth, тесты автоматом, dev может локально вызвать `ryotenkai-runtime-check --check-source`. Con: console_scripts entry в `packages/pod/pyproject.toml`.

**Status:** RESOLVED **(C)**.

### Q-NEW-5 — DTO move strategy: big-bang или alias re-exports?

**Сейчас:** все DTO в `packages/pod/src/ryotenkai_pod/runner/api/schemas.py`.

**Варианты:**
- **(A) Big-bang в Phase 0:** один PR — move + update all imports атомарно.
- **(B) Alias re-exports:** PR-0a добавляет в `shared.contracts.runner_api` + alias в `pod.runner.api.schemas`; PR-0b массово обновляет imports в одном merge train.

**Рекомендация:** **(B)** — RP9 mitigation. Лучше split на 2 sub-PR'а в Phase 0. PR-0a: добавить + alias (no breakage). PR-0b: update imports + delete alias + sentinel test. Reviewable изменения, легко rollback PR-0b если что-то сломается.

**Status:** RESOLVED **(B)**.

### Q-NEW-6 — Heartbeat lifecycle changes?

**Сейчас:** ControlPlaneHeartbeat стартует **после** `JobClient.submit_job` (см. [training_launcher.py:331-333](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/training_launcher.py:331)).

**Должен ли план менять lifecycle?** В Phase 2 добавляются другие HTTP endpoints (diagnostics, logs, files) до submit. Heartbeat стартует **только после submit** — это значит до submit'а если tunnel ляжет — никто не узнает. Пограничный случай: post-`/healthz`, pre-`POST /jobs`.

**Варианты:**
- **(A)** Ничего не трогаем: heartbeat стартует после submit, до этого есть `/healthz` polling в pre-flight.
- **(B)** Heartbeat стартует после `/healthz` 200 (раньше). До submit heartbeat уже идёт.

**Рекомендация:** **(A)** — heartbeat существует для tracking жизни Mac (дать pod знать что Mac alive для термotermination policy). До submit pod не имеет active job, нет смысла ждать heartbeat. `/healthz` polling в pre-flight достаточен.

**Status:** RESOLVED **(A)**.

### Q-NEW-7 — Какой версионинг для pod image после Phase 0?

**Контекст:** RP-PROD-5 (DTO move ломает release pinning).

**Варианты:**
- **(A)** Pod image tag bump после Phase 0 (`ryotenkai-runtime:2026.05.04`), Mac client требует matching tag.
- **(B)** Schema version field в WS handshake (`/api/v1/jobs/{id}/events?since=N&schema_version=2`) — auto-fallback на старый формат.
- **(C)** Atomic deploy: Phase 0 PR landит одновременно (1) shared contracts (2) pod image (3) Mac control. Никакая из combinations Mac-старый × pod-новый не может в природе случиться.

**Рекомендация:** **(C)** — atomic deploy. Reasoning: project не имеет mid-flight rolling deploy semantics (один Mac, один pod, один pipeline). YAGNI для (B).

**Status:** RESOLVED **(C)**.

---

## 14. Total scope (refresh, после Iteration 4 audit)

| Phase | PRs | LOC | Risk | Closes |
|---|---|---|---|---|
| Phase 0 | 2 (PR-0a + PR-0b) | ~250 | Low | DTO consolidation in shared.contracts.runner_api |
| Phase 1 | 1 | ~400 | Low | RFC 9457 problem+json foundation + universal HTTPException adapter (RP17) + sync handler registration (RP20) |
| Phase 2 | 5 | ~1200 | Medium | runtime через HTTP (diagnostics, resources, logs, files, import-check) + rate limit (RP19) + .partial cleanup (RP21) |
| Phase 3 | 4 | ~750 | Medium | enforcement (linter + AST sentinel + bootstrap_allowlist file (RP22) + SSH docs + DTO round-trip + existing endpoints migrate) |
| **Total** | **12** | **~2600** | **Low-Medium** | full HTTP runtime + bootstrap-only SSH + clean DTO surface + AST-enforced boundary |

**Ожидаемый результат:**
- SSH `exec_command` вызовов от Mac на pod: было ~9 категорий, станет **3 bootstrap-only (или 4 с opt Q-NEW-1)**.
- HTTP endpoints: было 6, станет **11** (+5).
- Error contract: ad-hoc → RFC 9457 problem+json.
- DTO source of truth: pod-only → `packages/shared/src/ryotenkai_shared/contracts/runner_api/`.
- CI gates: openapi freshness + importlinter rules + AST sentinel + dto round-trip.

---

## 15. Test strategy (по plan policy — все 7 категорий per PR)

Каждый PR в Phase 1+2+3 включает **7 категорий тестов**:

1. **Positive + negative** — happy path + obvious failure modes.
2. **Boundary** — все возможные edge cases (zero-byte, exact limit, off-by-one).
3. **Invariant** — properties that hold across all inputs (e.g. "ProblemDetails.code is always non-empty").
4. **Dependency-error** — что если subprocess недоступен / nvidia-smi missing / disk full.
5. **Regression** — каждый удалённый SSH-вызов имеет regression test что замена работает идентично.
6. **Logic-specific** — для конкретной логики (SHA-256 round-trip, AST sentinel matches, atomic rename).
7. **Combinatorial** — parametrized matrix (e.g. для PR-2.4: 3 targets × 3 sizes × 2 with/without checksum = 18).

Покрытие per PR documented в §4.

---

## 16. Best-practices conformance check (post-write)

После написания плана проверяю соответствие community best practices:

✅ **Kubernetes API conventions** — subresource patterns, enum whitelist для path params, level-аналог `kubectl logs --since-time`.
✅ **RFC 9457 problem+json** — single error contract, `type`/`title`/`status`/`detail`/`instance` + extensions.
✅ **Twelve-factor logs** — pod-side ground truth + Mac читает чанки через range, не push'ит.
✅ **FastAPI multipart streaming (2025)** — `UploadFile` + `aiofiles` chunked, memory-bounded, atomic write, path whitelist.
✅ **OpenAPI single source of truth** — generation from FastAPI, CI gate freshness.
✅ **Static enforcement (importlinter)** — boundary violations через imports.
✅ **AST sentinel test (NEW в v2)** — runtime call enforcement, gap'ит importlinter.
✅ **CODE_ERRORS.md project convention** — `ErrorCode` enum maps на `AppError.code`, UPPER_SNAKE_CASE.
✅ **YAGNI** — Phase 4 (SSE migration) deferred, schema versioning deferred, dependency_installer migration optional.
✅ **Boy scout rule** — каждый touch'нутый файл (`code_syncer`, `file_uploader`, `log_manager`) либо удаляется либо переписывается чисто.
✅ **uv-workspace import contracts** — DTO ляжет в shared (leaf), respect direction control←shared, pod←shared.

⚠️ **Anti-patterns мы НЕ делаем:**
- ❌ Не пытаемся версионировать API (`/api/v1` остаётся; `/api/v2` появится только при breaking change). YAGNI.
- ❌ Не делаем GraphQL endpoint вместо REST. Излишне для MLOps control plane.
- ❌ Не делаем gRPC streaming. Stack — Python+httpx+FastAPI, gRPC overkill.
- ❌ Не используем `fastapi-problem-details` lib без обоснования — roll our own легче и контролируемее (Q-NEW-3).
- ❌ Не используем HTTP `Range:` header для logs — простой `?offset=` query достаточен и легче на mocks (PR-2.3).

---

## 17. Audit trail (3 итерации, обновлённый)

**Iteration 1** (architecture decomposition):
- ✅ Identified: пользователь сказал «всё через uvicorn после bootstrap». Variant C (rsync packages/ + uvicorn = SSH bootstrap; всё runtime — HTTP).
- ✅ Identified: RunPod SDK — отдельная boundary (Mac↔cloud), не часть Mac↔pod.
- ✅ Identified: Web frontend WS — отдельный канал, не задевается.
- ✅ Identified (NEW в v2): Phase B packagization меняет paths — все file references обновлены.

**Iteration 2** (scope and risk balancing):
- ✅ File upload через HTTP — viable для текущих <10MB datasets. FastAPI streaming pattern verified в 2025 web research.
- ✅ Logs streaming через HTTP — natural fit, replaces SCP scheduled pull.
- ✅ Phase 4 (WS→SSE) — defer.
- ✅ NEW в v2: DTO consolidation moved в Phase 0 (был PR-3.3 в v1), atomic split на 0a/0b (RP9 mitigation).
- ✅ NEW в v2: dependency_installer migration — open question Q-NEW-1 (recommendation: keep SSH).

**Iteration 3** (production readiness):
- ✅ Identified RP3 (SSH tunnel limits) — нужен integration test multipart over tunnel.
- ✅ Identified RP9 (DTO move ripples) — split в two PR.
- ✅ Identified RP12 (catch-all Exception маскирует) — explicit logging at ERROR level.
- ✅ NEW в v2: RP13 — importlinter не ловит `.exec_command()` runtime calls, добавлен AST sentinel test.
- ✅ NEW в v2: RP14 — runtime_check.py location resolved Q-NEW-4 (migrate в pod package с CLI entrypoint).
- ✅ NEW в v2: RP-PROD-5 — atomic Phase 0 deploy (Q-NEW-7).
- ✅ NEW в v2: RP-PROD-1 — `__pycache__` stale state на pod после rsync (mitigation: rsync flags + import-check catches).

**Iteration 4** (additional findings via deep-think audit):
- ✅ NEW: RP16 — multi-worktree DTO drift (mitigation: rebase discipline + AST sentinel).
- ✅ NEW: RP17 — mid-migration mixed-state — **критическое design decision**: Phase 1 включает универсальный `http_exception_handler` адаптирующий ALL `HTTPException` → problem+json от дня 1 (вместо постепенной миграции, которая дала бы wire shape inconsistency).
- ✅ NEW: RP18 — SSH tunnel bandwidth contention (mitigation: heartbeat tolerant timeouts, file upload chunked async yields).
- ✅ NEW: RP19 — `/diagnostics` DOS amplification (mitigation: pod-side simple in-memory rate limit).
- ✅ NEW: RP20 — uvicorn lifespan ordering для exception handlers (handlers register синхронно в `FastAPI(exception_handlers={...})`, не в async lifespan startup — критический pattern).
- ✅ NEW: RP21 — `/files/upload` mid-stream cancel cleanup (try/finally + .partial unlink + periodic cleanup task).
- ✅ NEW: RP22 — AST sentinel allowlist staleness (mitigation: separate file с docstring justifications, PR review требует обоснование на расширение).

---

## 18. Next step

**Не начинаю реализацию** до получения user approval по:
1. Q-NEW-1 (dependency_installer migration: keep SSH или мигрировать в HTTP).
2. Любые правки к scope / phasing.

После approval — начинаем с **Phase 0 PR-0a** (alias re-exports) на свежей feature branch (предлагаю кодовое имя `claude/transport-v2-<hash>`).
