# План: Monitor cleanup + Control-plane HTTP redesign

> Status: **APPROVED scope = full roadmap Phase 0-3** (user decision 2026-05-02). Q1+Q2+Q4 resolved (см. §13 updated answers). Готов к реализации Phase 0.
> Author: daniil + claude (deep-think пасс, 2 параллельных Plan agent'а, 3-iteration risk audit)
> Worktree: `dazzling-rosalind-b482fa`
> Trigger:
> - 3 visible bugs в `run_20260502_113553_r8rul/attempts/attempt_1/logs/training_monitor.log`
> - стратегический запрос пользователя на унификацию Mac↔pod через uvicorn HTTP API + единый error contract
> Связанные планы:
> - [2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md](2026-05-01-22-20-02-stage-execution-loop-polymorphic-token.md) — Phase 1 (PodLayout + pull-only ground truth) IMPLEMENTED
> - [2026-05-02-fail-fast-prevention-and-log-visibility.md](2026-05-02-fail-fast-prevention-and-log-visibility.md) — PR-A/B/C (import gate + push tail + diagnostic grace) IMPLEMENTED, merged

---

## 1. Контекст

### 1.1. Что user видит сейчас в `training_monitor.log`

Из real-run `run_20260502_113553_r8rul`:

```
17:39:35 [TRAINER:STDERR] 2026-05-02 11:39:31 loader:112 DEBUG - [SECRETS] Loading from environment only (no secrets.env found)
17:39:35 [TRAINER:STDERR] [TRAINER:M2] Loading config from config/pipeline_config.yaml
17:39:35 [TRAINER:STDERR] 2026-05-02 11:39:31 run_training:160 INFO - Starting LLM Training
... (30 строк push-tail с префиксом [TRAINER:STDERR])
17:39:35 [MONITOR:POSTMORTEM] non-zero exit detected — collecting diagnostics
17:39:35 [MONITOR:POSTMORTEM:trainer] [ERR] 2026-05-02 11:39:25 factory:141 DEBUG - [SF:STRATEGY_REGISTERED]...
... (30 СНОВА строк pull-postmortem с префиксом [ERR])
```

**Явные баги** (operator-visible):

1. **Двойной dump tail'а**: `_log_trainer_exited_tail` (PR-B push) + `_dump_local_log_tail` (PR-A pull) показывают пересекающиеся ~30 строк stderr trainer'а.
2. **`[OUT]/[ERR]` префикс просачивается в operator log**: pod-side internal serialization (Supervisor merge stdout+stderr в один файл) леcится в Mac logger через `_dump_local_log_tail` raw `readlines()`.
3. **Status line отсутствует в этом run'е**: trainer упал за 11s до первого `health_snapshot` event'а — `_maybe_log_status` ни разу не выстрелил. Существующий `health_snapshot` based status line работает только для long-running runs.

### 1.2. Стратегические запросы пользователя (за рамками 3 багов)

- Унифицировать всё Mac↔pod через uvicorn HTTP API runner'а (вход с понятными контрактами).
- Унифицировать ошибки по образцу `src/docs/codestyle/CODE_ERRORS.md` + использовать HTTP коды.
- DTO для interaction'а Mac ↔ runner.
- "Не плодить хаос — определять границы".

### 1.3. Inventory: что сейчас Mac↔pod (фактическая карта)

**Через HTTP/WS** (`src/runner/api/*.py`):
- `POST /api/v1/jobs` (multipart submit с plugins zip)
- `GET /api/v1/jobs/{id}` (snapshot)
- `POST /api/v1/jobs/{id}/stop`
- `POST /api/v1/control/heartbeat`
- `POST /api/v1/internal/events` (loopback only — trainer→runner)
- `WS /api/v1/jobs/{id}/events` (event stream)

**Через прямой SSH (`ssh_client.exec_command`) от Mac**:
- `dmesg` / grep oom / `nvidia-smi` — `training_monitor.py:_collect_death_diagnostics` (827-840)
- `stat -c%s` / `wc -c` / `tail` — `log_manager.py` (delta-pull для trainer.stdio.log + runner.log)
- `rsync` / tar pipe — `code_syncer.py` (sync src/ + community/)
- `python /opt/helix/runtime_check.py [--check-source]` — `dependency_installer.py` + PR-A `code_syncer.py`
- `pip install` — `dependency_installer.py`
- `python -m uvicorn src.runner.main:app` — `runner_launcher.py` (bootstrap)
- file uploads (config, dataset) — `file_uploader.py` (tar pipe scp)

**Через RunPod SDK от Mac (внешний cloud API, не runner)**:
- `sdk.get_pod` / `sdk.start_pod` — `training_monitor._recover_pod_if_needed` (pod recovery когда runner недоступен)

**Error contract сейчас**:
- runner возвращает `HTTPException` с ad-hoc `detail: dict` shape (например `{"code": "job_in_progress", "current_state": "...", "message": "..."}` в `jobs.py:182-193`)
- Mac client (`src/api/clients/job_client.py`) raises `JobClientError`, `JobNotFoundError`, `ReplayTruncatedError`. Парсит ad-hoc.
- Нет общей `ProblemDetails` / `ErrorCode` enum
- `Result[T, AppError]` действует на Python-side но не пересекает HTTP границу.

### 1.4. Что показал репозиторный анализ

- `training_monitor.py` — 99.9% bug-prone hotspot (Repowise), 4 dependents, sole owner, "increasing" trend → **большие рефакторинги имеют высокий риск регрессии**.
- `runner/main.py`, `supervisor.py`, `jobs.py` — все 90%+ churn-heavy → активная разработка, новые фичи безопаснее.
- `events.py` — 64% stable → можно эволюционировать без больших опасений.

---

## 2. Принципы решения

1. **YAGNI применять с умом**: пользователь явно просит unification; но это не "сделать прям щас" — это roadmap. Разделить immediate fix и strategic foundation.
2. **Каждая фаза landится атомарно с зелёными тестами** — не batch'ить множество PR.
3. **CODE_ERRORS.md is law**: `Result[T, AppError]` Python-side; HTTP-граница maps на RFC 9457 problem+json.
4. **Без обратной совместимости** (явно от пользователя): старый код удаляется чисто, новый занимает его место.
5. **Boy scout rule на hotspots**: в Phase 0/1 не трогаем то что не сломано.
6. **Bootstrap SSH остаётся** — HTTP не может поднять сам себя. Граница чётко документируется.

---

## 3. Целевая архитектура (target state, end of Phase 2)

```
┌──────────────────────────── Mac orchestrator ─────────────────────────────┐
│                                                                            │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │TrainingStage │  │PodDeployer     │  │ PodRecovery  │  │ModelRetrvr  │ │
│  └──────┬───────┘  └────────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                   │                 │                 │         │
│         │                   │ bootstrap       │ cloud SDK       │         │
│         │                   │ (SSH-rsync,     │ (RunPod SDK     │         │
│         │                   │  uvicorn        │  для recovery)  │         │
│         │                   │  launcher)      │                 │         │
│         │                   ▼                 ▼                 │         │
│         │              ┌─────────────────────────────────┐      │         │
│         │              │  SSH surface — DOCUMENTED       │      │         │
│         │              │  bootstrap-only allowlist       │      │         │
│         │              │  (rsync, scp tar, uvicorn exec) │      │         │
│         │              └─────────────────────────────────┘      │         │
│         │                                                       │         │
│         ▼                                                       │         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ JobClient (HTTP+WS, problem+json parser, typed exceptions)      │    │
│  │   • submit_job  • get_status  • request_stop                    │    │
│  │   • subscribe_events  • send_heartbeat                          │    │
│  │   • get_diagnostics  • get_resources  • read_log (range)        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────│─────────────────────────────────────────┘
                                   │ ssh -L 127.0.0.1:8080  (одна ControlMaster
                                   │                          для всего)
                                   ▼
┌────────────────────────── Pod (uvicorn runner) ────────────────────────────┐
│  POST /api/v1/jobs               (existing)                                │
│  GET  /api/v1/jobs/{id}          (existing)                                │
│  POST /api/v1/jobs/{id}/stop     (existing)                                │
│  WS   /api/v1/jobs/{id}/events   (existing)                                │
│  POST /api/v1/control/heartbeat  (existing)                                │
│  POST /api/v1/internal/events    (existing — loopback)                     │
│                                                                            │
│  GET  /api/v1/healthz            (existing)                                │
│  GET  /api/v1/diagnostics?include=dmesg,gpu,kernel   ◄── PR-2.1 (NEW)     │
│  GET  /api/v1/resources                              ◄── PR-2.2 (NEW)     │
│  GET  /api/v1/logs/{name}?offset=&limit_bytes=       ◄── PR-2.3 (NEW)     │
│  POST /api/v1/runtime/import-check                   ◄── PR-3.2 (NEW)     │
│                                                                            │
│  ProblemDetailsExceptionHandler ─→ application/problem+json  (PR-1)        │
│  Pydantic schemas (single source of truth, OpenAPI-exported)              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Фазированный план (4 фазы, 7 PR)

### Phase 0 — Stop the bleeding (immediate, 3 PR, 1 файл, ~80 LOC)

Цель: устранить **3 visible bugs** в `training_monitor.log` минимальной кровью. Низкий риск, быстрый pain relief.

#### PR-0.1: Strip `[OUT]/[ERR]` prefix при чтении на Mac

**Что**: `_dump_local_log_tail` парсит `[OUT] `/`[ERR] ` префикс, эмитит чистую строку через logger.

**Почему**: pod-side это internal serialization (Supervisor merge stdout+stderr в один файл с префиксом для разделения), на Mac оно protect leak'ается через raw `readlines()`. Pod-side формат **остаётся** (нужен для disk forensics + правильного push-tail формирования в supervisor).

**Файлы**:
- `src/pipeline/stages/training_monitor.py` — `_dump_local_log_tail` (~5 LOC).

**Решение по logger level** (см. §13 Open Question 2): по умолчанию **strip полностью, всё через `logger.info`** (ниже severity → не triggerит alerts). Если operator хочет видеть `[ERR]` подсветку — отдельный CLI flag/env.

#### PR-0.2: De-dup push-tail vs pull-postmortem

**Что**: `_collect_death_diagnostics` принимает `skip_trainer_log_dump: bool` параметр. `_handle_trainer_exited` определяет `embedded_tail_emitted = bool(stderr_tail or stdout_tail)` после `_log_trainer_exited_tail` и передаёт в diagnostics.

**Почему**: оба механизма НУЖНЫ для разных failure modes:
- **Platform eviction** (pod исчез до SCP): push-tail доехал по WS, pull-postmortem видит `<<MISSING>>`. Operator всё равно видит причину смерти.
- **Normal failed exit** (pod жив, SCP работает): оба показывают одно и то же → **скрываем pull-копию когда push сработал**.

`runner.log` — НЕ затрагивается push-tail (тот carries только trainer stdout/stderr); pull-postmortem всегда дампит `runner.log` — он не дублируется.

**Файлы**:
- `src/pipeline/stages/training_monitor.py` — `_handle_trainer_exited` + `_collect_death_diagnostics` (~30 LOC).

#### PR-0.3: Status line robustness

**Что**:
- Renames `ALIVE` → `running` (соответствует develop).
- Дополняет format пытаться брать `gpu_temp_c`, `vram_used_gb`/`vram_total_gb` если приходят из `health_snapshot`; иначе показывает `—`.
- НЕ меняет runner-side `health_snapshot` payload (это deferred — Phase 2.2 endpoint `GET /resources` даст мгновенный snapshot).

**Почему**: текущий status line работает только когда trainer **успел дожить до health_snapshot** (~30s). При early crash пользователь видит **только terminal event**, и кажется что monitor ничего не делает. Это исправит Phase 2.2 (PR-2.2 `GET /resources` для immediate snapshot), а в Phase 0 — лишь косметика того что есть.

**Файлы**:
- `src/pipeline/stages/training_monitor.py` — `_maybe_log_status` (~10 LOC).

**Тесты Phase 0** (для всех 3 PR):
- positive: prefix stripped, dedup работает, status line формат matches.
- negative: empty tails → guard НЕ срабатывает → postmortem дампит trainer.log.
- boundary: `[OUT] [OUT]` (вложенный, защита от bug в supervisor) — strip только outer.
- invariants: pod-side `trainer.stdio.log` файл сохраняет original `[OUT]/[ERR]` префикс (ground truth).
- regression: `test_training_monitor_v2_payload.py` тесты остаются зелёными.

---

### Phase 1 — Error contract foundation (1 PR, ~300 LOC)

Цель: заложить единый error transport (RFC 9457 problem+json + ErrorCode registry) **без переписывания существующих endpoints**. Фундамент для Phase 2 endpoint'ов.

#### PR-1: ProblemDetails + ErrorCode + FastAPI exception_handler

**Что**:

1. **`src/runner/api/errors.py` (NEW)**:
   - `class ProblemDetails(BaseModel)` — RFC 9457 поля (`type`, `title`, `status`, `detail`, `instance`) + наши extensions (`code`, `details`).
   - `class ErrorCode(StrEnum)` — реестр UPPER_SNAKE_CASE кодов (см. §5).
   - `class APIError(Exception)` — runner-side typed exception, `to_problem_details() -> ProblemDetails`.

2. **`src/runner/api/exception_handlers.py` (NEW)**:
   - `register(app)` — добавляет handlers для:
     - `APIError` → 4xx/5xx с body=ProblemDetails, content-type=`application/problem+json`.
     - `RequestValidationError` → 422 ProblemDetails(`code=JOB_SPEC_INVALID`).
     - `Exception` (catch-all) → 500 ProblemDetails(`code=INTERNAL_ERROR`), traceback в logger не в response.

3. **`src/api/clients/problem_details.py` (NEW)** — Mac mirror DTO + parser:
   - `parse_problem_details(response: httpx.Response) -> ProblemDetails | None` (None если content-type != problem+json).
   - `class APIException(Exception)` с полем `code: ErrorCode | str` и `problem: ProblemDetails`.
   - `raise_for_problem(response)` helper.

4. **`src/runner/main.py`** — вызвать `register_exception_handlers(app)` после mount routers.

5. **Existing endpoints НЕ переписываются** в этом PR. Они продолжают raise `HTTPException(detail={...})`. Новый exception handler добавляется как **available but not enforced**. Постепенная миграция в Phase 2.

**Почему такой подход**:
- Опасный путь — переписать сразу все ошибки в одном PR: высокий риск, web frontend ломается, hard to review.
- Безопасный путь — добавить инфраструктуру + миграция per-endpoint в каждом следующем PR. Каждый PR в Phase 2 переписывает свои новые endpoints на `APIError(ErrorCode.X)` сразу.

**Тесты PR-1** (7 категорий):
- positive: `APIError(ErrorCode.JOB_NOT_FOUND, detail="...")` → 404 problem+json.
- negative: existing `HTTPException(404)` continue работает (legacy путь не сломан).
- boundary: `details: dict` с unicode + nested.
- invariants: `status` в body == HTTP status; `type` URI начинается с `https://ryotenkai.dev/errors/`.
- dependency-error: `RuntimeError` в handler → 500 INTERNAL_ERROR (traceback в логи, не в body).
- regression: existing `test_jobs_api.py` тесты зелёные.
- combinatorial: matrix `(ErrorCode × http_status)` — каждый код имеет ровно один правильный status.

**Файлы**:
- NEW: `src/runner/api/errors.py`, `src/runner/api/exception_handlers.py`, `src/api/clients/problem_details.py`
- NEW: `src/tests/unit/runner/api/test_errors.py`, `src/tests/unit/runner/api/test_exception_handlers.py`, `src/tests/unit/api/clients/test_problem_details.py`
- MODIFIED: `src/runner/main.py` (~5 LOC).

---

### Phase 2 — HTTP unification monitor probes (2-3 PR, ~600 LOC)

Цель: убрать SSH `dmesg`/`nvidia-smi` probes из training_monitor; заменить мгновенным `GET /resources` (для status line при early crash); опционально перевести log streaming на HTTP.

**Эта фаза опциональна**: пользователь подтвердит scope в §13.

#### PR-2.1: Diagnostic endpoint (`GET /api/v1/diagnostics`)

**Что**:
- `src/runner/diagnostics/collectors.py` (NEW): pure functions `collect_dmesg(grep, tail)`, `collect_nvidia_smi()`, `collect_kernel_signals()`. Возвращают typed reports.
- `src/runner/api/diagnostics.py` (NEW): router `GET /diagnostics?include=dmesg,gpu,kernel` → `DiagnosticsResponse`.
- `src/runner/api/schemas_diagnostics.py` (NEW): `DmesgReport`, `GpuReport`, `KernelSignalsReport`, `DiagnosticsResponse`, `DiagnosticsInclude(StrEnum)`.

- `src/api/clients/job_client.py`: `get_diagnostics(include) -> DiagnosticsResponse`.
- `src/pipeline/stages/training_monitor.py:_collect_death_diagnostics`: SSH probes block (827-858) **удалён**, заменён вызовом `client.get_diagnostics(...)`. Если runner мёртв → graceful warning в logger, не падать.

**Использует `APIError` + `ErrorCode` от Phase 1** для всех ошибок endpoint'а:
- `DIAGNOSTIC_FAILED` (502) — collector raises (permission, subprocess error)
- `DIAGNOSTIC_TIMEOUT` (504) — `subprocess.TimeoutExpired`
- `DIAGNOSTIC_INVALID_INCLUDE` (422) — unknown key

**Тесты**:
- positive: include=all → все три блока с данными.
- negative: invalid include → 422 problem+json.
- boundary: dmesg permission denied → блок с `error: "permission_denied"`, status code 200 (другие блоки success).
- invariants: collector failures изолированы (один не валит другие).
- dependency-error: subprocess timeout → 504.
- regression: training_monitor postmortem всё ещё содержит kernel/GPU info.
- combinatorial: matrix `(include_flags × subprocess_outcome × runner_state)`.

#### PR-2.2: `GET /api/v1/resources` for instant status snapshot

**Что**:
- `src/runner/diagnostics/resources.py` (NEW): `collect_resource_snapshot() -> ResourceSnapshot` (gpu_util, vram_used_gb, vram_total_gb, vram_pct, gpu_temp_c, ram_used_gb, ram_total_gb, cpu_pct, timestamp).
- `src/runner/api/resources.py` (NEW): router `GET /resources` → `ResourceSnapshot`.
- `src/runner/api/schemas_resources.py` (NEW): `ResourceSnapshot`.
- `src/api/clients/job_client.py`: `get_resources() -> ResourceSnapshot`.
- `src/pipeline/stages/training_monitor.py:_maybe_log_status`:
  - **Изменение**: вместо ожидания `health_snapshot` event — poll `client.get_resources()` каждые `STATUS_LINE_INTERVAL_SEC` (15s).
  - Закрывает gap "trainer упал за 11s — status line не появилась" (мы успеем ОДИН раз poll'нуть на T+15s или сразу после spawn).
  - Status line теперь **гарантированно появляется** даже если training fail'ится за 5 секунд.

**Тесты**:
- positive: poll возвращает все поля.
- negative: pod нет GPU → `gpu_util=None`, рендерится как `—`.
- boundary: nvidia-smi падает → `error` field, snapshot без GPU.
- regression: status line всё ещё пишется (не сломали formatting).

#### PR-2.3 (optional, может быть отдельной фазой): `GET /api/v1/logs/{name}` для streaming

**Что**:
- `src/runner/api/logs.py` (NEW): `GET /logs/{name}?offset=&limit_bytes=` → `LogChunkResponse`.
- `LogName(StrEnum)` whitelist (`TRAINER_STDIO`, `RUNNER`) — anti-path-traversal.
- `src/pipeline/stages/managers/log_fetcher.py` (NEW): replaces `LogManager`. Парсит `[OUT]/[ERR]` префикс при чтении (PR-0.1 теперь redundant — заменяется этим), эмитит через logger без префикса.

**Замечание**: Это самый дорогой PR (rewrite LogManager + удаление old SSH-tail logic). Если scope «как хочется быстро» — можно отложить в Phase 3.

---

### Phase 3 — Strategic, deferred (3-4 PR, ~700 LOC)

Цель: завершить unification, document SSH surface, DTO consistency.

#### PR-3.1: SSH surface contract docs + soft deprecation

- `docs/architecture/SSH_SURFACE.md` — allowlist разрешённых SSH вызовов:
  - `rsync` для `code_syncer`
  - `scp tar` для `file_uploader` (config + dataset)
  - `python -m uvicorn` для `runner_launcher`
  - `pip install` для `dependency_installer` (pre-runner bootstrap)
- `src/utils/ssh_client.py` — добавить deprecation warning logger в `exec_command` если caller не в allowlist (path-based check, soft enforcement first).

#### PR-3.2: Migrate `runtime_check.py --check-source` SSH → HTTP

- `src/runner/api/runtime.py` (NEW): `POST /runtime/import-check` body=`ImportCheckRequest` → `ImportCheckReport`.
- Issue: import-check случается **до** uvicorn launch (он часть deployment). Это значит runtime endpoint доступен только **после** второго рестарта. Решение: import-check переходит в "post-launch validation" — выполняется первым после `runner_launcher` health check проходит. Если fail — abort deployment.
- `code_syncer.py:_verify_importability` — больше не SSH-exec, а HTTP call.

#### PR-3.3: DTO mirror через shared package

- `src/contracts/runner_api/` (NEW): move all schemas из `src/runner/api/schemas*.py`. Both `src/runner/api/*` и `src/api/clients/*` импортируют отсюда.
- `scripts/sync_openapi.py` (NEW): generate `web/src/api/openapi.json` from FastAPI app. CI gate `test_openapi_freshness`.

#### PR-3.4: Migrate existing endpoints to APIError pattern

- jobs.py, control.py, events.py, internal.py — заменить все `HTTPException(detail={...})` на `raise APIError(ErrorCode.X, ...)`.
- Mac client — все methods используют `parse_problem_details` + raise typed exceptions.

---

## 5. Error contract — ErrorCode registry

| ErrorCode | HTTP | Domain | Semantic |
|---|---|---|---|
| `JOB_NOT_FOUND` | 404 | jobs | `job_id` не существует или не активен |
| `JOB_STATE_INVALID` | 409 | jobs | FSM transition не разрешён в текущем state |
| `JOB_SPEC_INVALID` | 422 | jobs | Pydantic validation, JSON parse, `extra="forbid"` |
| `JOB_IN_PROGRESS` | 409 | jobs | Active non-terminal job blocks submit |
| `PLUGIN_UNPACK_FAILED` | 422 | jobs | ZIP corrupt / community/ extraction error |
| `SPAWN_FAILED` | 422 | jobs | exec() failure: FileNotFoundError, OSError |
| `RUNNER_NOT_READY` | 503 | system | Startup not complete |
| `RUNNER_BUSY` | 409 | system | Supervisor занят |
| `DIAGNOSTIC_FAILED` | 502 | diagnostics | dmesg/nvidia-smi unavailable, permission denied |
| `DIAGNOSTIC_TIMEOUT` | 504 | diagnostics | subprocess.TimeoutExpired |
| `DIAGNOSTIC_INVALID_INCLUDE` | 422 | diagnostics | Unknown key в `?include=` |
| `LOG_NOT_AVAILABLE` | 404 | logs | Whitelisted log name, нет файла |
| `LOG_OFFSET_OUT_OF_RANGE` | 416 | logs | Offset > total_size |
| `LOG_NAME_INVALID` | 422 | logs | Unknown `name` (не в LogName enum) |
| `IMPORT_CHECK_TOO_MANY_MODULES` | 422 | runtime | > 50 modules |
| `IMPORT_CHECK_TIMEOUT` | 504 | runtime | Subprocess timeout |
| `WS_REPLAY_TRUNCATED` | (4410 ws) | events | Existing — не меняем |
| `WS_INVALID_PARAMS` | (4422 ws) | events | Existing |
| `INTERNAL_ERROR` | 500 | system | Catch-all, traceback в logs не в body |

**Mapping на Python-side `AppError`**: каждый `ErrorCode` имеет соответствующий `AppError` subclass (`ProviderError`, `TrainingError`, etc.) с тем же `code` field. RFC 9457 = HTTP transport; `AppError` = Result transport. Один объект `APIError` рендерится в оба формата.

---

## 6. DTO inventory

### Существующие (не меняются в Phase 0/1)

`JobSpec`, `JobSubmittedResponse`, `JobSnapshotResponse`, `JobStopAcceptedResponse`, `EventResponse`, `InternalEventRequest`, `ControlHeartbeatRequest`, `ControlHeartbeatResponse`.

### Phase 1 (NEW)

- `ProblemDetails(BaseModel)` — RFC 9457
- `ErrorCode(StrEnum)` — реестр кодов
- `APIError(Exception)` — runner-side
- `APIException(Exception)` — Mac client side

### Phase 2 (NEW)

- `DmesgReport`, `GpuReport`, `KernelSignalsReport`, `DiagnosticsResponse`, `DiagnosticsInclude(StrEnum)`
- `ResourceSnapshot`
- `LogName(StrEnum)`, `LogChunkResponse`, `LogSizeResponse`

### Phase 3 (NEW + REORGANIZED)

- `ImportCheckRequest`, `ImportCheckReport`
- All schemas → moved to `src/contracts/runner_api/`

---

## 7. Risk register (12 рисков, 3 итерации audit)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| RP1 | Phase 0 PR-0.2 `embedded_tail_emitted` flag врёт когда один из tail'ов пустой | Medium | Low | Проверять `bool(stderr) or bool(stdout)`, не AND. Test для both cases. |
| RP2 | Phase 0 PR-0.1 `[OUT]/[ERR]` strip ломает existing `test_dump_local_log_tail` тесты | Medium | Low | Обновить тесты в том же PR; assertion на чистую строку без префикса. |
| RP3 | Phase 1 web frontend (`web/src/api/openapi.json`) ломается на смене shape `detail:dict` → `problem+json` | Low | Medium | В Phase 1 НЕ переписываем existing endpoints — только инфраструктура. Web не задет. Migration в Phase 3.4 с openapi sync. |
| RP4 | Phase 2 `dmesg` требует `CAP_SYSLOG`; в RunPod может быть закрыт | Medium | Medium | Collector ловит `PermissionError` → возвращает `error=permission_denied` блок (не 500). |
| RP5 | Phase 2.2 `get_resources` polled каждые 15s = 4 SSH calls/min × 60min = 240/час | Medium | Low | Acceptable. Один HTTP roundtrip ~50-100ms. Если станет issue — adaptive backoff. |
| RP6 | Phase 2 health_snapshot WS события становятся redundant (теперь pollем resources) | Low | Low | Оставить health_snapshot — он несёт metrics для MLflow. Status line использует HTTP poll, MLflow — WS events. Два независимых channel. |
| RP7 | Phase 3.2 import-check после runner launch добавляет latency к deploy | Medium | Low | Acceptable — это всё ещё < 5s overhead, vs сегодняшние 30s на retry crashes. |
| RP8 | Phase 3.3 DTO move из `src/runner/api/schemas` в `src/contracts/runner_api/` ломает 50+ import sites | High | Medium | Разбить на 2 PR: первый только move + alias re-exports; второй обновляет imports. |
| RP9 | training_monitor.py 99.9% bug-prone — каждый PR в Phase 0/2 риск регрессии | High | Medium | Каждый PR landит атомарно с зелёными тестами; не batch'ить. Полная регрессия `test_training_monitor_v2*.py` после каждого. |
| RP10 | Phase 1 `APIError` exception handler не работает с middleware order (CORS, logging) | Low | Medium | Тест: integration test со всеми middlewares + `raise APIError(...)` → правильный response. |
| RP11 | RunPod SDK errors (pod recovery в monitor) не унифицированы с problem+json | Medium | Low | Out of scope: RunPod SDK = Mac↔cloud, не Mac↔pod. Документировать в SSH_SURFACE.md как known limitation. |
| RP12 | Phase 0 PR-0.3 status line пытается читать `gpu_temp_c` из `health_snapshot` payload, но runner его не шлёт | Medium | Low | Graceful — `payload.get("gpu_temp_c")` → None → render `—`. Phase 2.2 даст полный snapshot. |

### Audit iteration 1: ordering

- ✅ Изначально предлагалось PR-1 (error contract) первым. Pragmatic agent показал что 3 visible bugs можно починить за 80 LOC в одном файле — pain relief раньше. Reordered: Phase 0 first, Phase 1 second.

### Audit iteration 2: scope creep

- ✅ Architectural agent предлагал `GET /logs/{name}` с заменой LogManager в Phase 2. Это самый рискованный PR (LogManager имеет много callers). Решение: сделать его optional PR-2.3 ИЛИ перенести в Phase 3. Зависит от user'a (см. §13 Q3).
- ✅ DTO mirror через shared package (Phase 3.3) — не критично для решения текущих болей. Закладываем как roadmap.

### Audit iteration 3: production failure modes

- ✅ Phase 2.2 `GET /resources` polled каждые 15s — что если runner мёртв? Status line должен gracefully handle: `try: r = await client.get_resources() except APIException: r = None` → render `[MONITOR] (runner unreachable) | ...`.
- ✅ Phase 1 catch-all `Exception` handler — может maskировать debug errors. Mitigation: traceback всё равно в logs, response 500 INTERNAL_ERROR без traceback.
- ✅ Phase 0 PR-0.2 dedup может ошибочно скрыть tail если payload schema_version=1 (legacy). Решение: condition `schema_version >= 2 AND embedded_tail_emitted`. v1 payloads без push-tail → postmortem дампит как раньше.

---

## 8. Best-practices alignment

### RFC 9457 (Problem Details for HTTP APIs)

`ProblemDetails` имеет `type` (URI), `title` (short summary), `status`, `detail` (human-readable), `instance` (per-occurrence), плюс наш extension `code` (machine-readable enum), `details` (structured). Соответствует §3 спецификации.

Sources: [Swagger Problem Details RFC9457](https://swagger.io/blog/problem-details-rfc9457-api-error-handling/), [Codecentric Problem Details deep dive](https://www.codecentric.de/en/knowledge-hub/blog/charge-your-apis-volume-19-understanding-problem-details-for-http-apis-a-deep-dive-into-rfc-7807-and-rfc-9457).

### Kubernetes API conventions

- Endpoint naming: `GET /resources` (subresource pattern), `GET /diagnostics?include=...` (field selector), `GET /logs/{name}?offset=` (`kubectl logs --since-time` style range).
- `LogName` whitelist enum vs raw paths — anti-path-traversal как в k8s `ContainerLogOptions`.
- `JobState` lower-case (`pending`, `running`, `completed`, `failed`, `cancelled`) — matches Kubernetes conventions.

### OpenAPI 3.x

- `extra="forbid"` в Pydantic = `additionalProperties: false`.
- `ErrorCode(StrEnum)` = `enum` field в spec.
- All examples в `description=` rendered в Swagger UI.

### Twelve-factor logs

- Pod-side `trainer.stdio.log` — ground truth с `[OUT]/[ERR]` префиксом для post-hoc forensics.
- Mac-side читает structured (Phase 0 PR-0.1 strip prefix; Phase 2.3 endpoint).
- Persistent storage = source of truth, transport — pure data.

### CODE_ERRORS.md project convention

- `ErrorCode` ↔ `AppError.code` (one-to-one). UPPER_SNAKE_CASE с domain prefix (`JOB_*`, `DIAGNOSTIC_*`, `LOG_*`).
- `ProblemDetails.code` несёт значение `ErrorCode` (StrEnum value).
- Mac client raises `APIException(code: ErrorCode)` — те же типы.

---

## 9. Critical files (по фазам)

### Phase 0
```
src/pipeline/stages/training_monitor.py                    # MODIFIED, ~45 LOC
src/tests/unit/pipeline/test_training_monitor_v2.py        # MODIFIED (assertions)
src/tests/unit/pipeline/test_training_monitor_v2_payload.py # MODIFIED
```

### Phase 1
```
src/runner/api/errors.py                                    # NEW (~120 LOC)
src/runner/api/exception_handlers.py                        # NEW (~80 LOC)
src/api/clients/problem_details.py                          # NEW (~80 LOC)
src/runner/main.py                                          # MODIFIED (~5 LOC)
src/tests/unit/runner/api/test_errors.py                    # NEW
src/tests/unit/runner/api/test_exception_handlers.py        # NEW
src/tests/unit/api/clients/test_problem_details.py          # NEW
```

### Phase 2.1 (diagnostics)
```
src/runner/diagnostics/__init__.py                          # NEW
src/runner/diagnostics/collectors.py                        # NEW (~150 LOC)
src/runner/api/diagnostics.py                               # NEW (~80 LOC)
src/runner/api/schemas_diagnostics.py                       # NEW (~60 LOC)
src/api/clients/job_client.py                               # MODIFIED (+get_diagnostics)
src/pipeline/stages/training_monitor.py                     # MODIFIED (-SSH probes block, +HTTP call)
src/runner/main.py                                          # MODIFIED (mount router)
src/tests/unit/runner/diagnostics/                          # NEW dir
```

### Phase 2.2 (resources)
```
src/runner/diagnostics/resources.py                         # NEW (~80 LOC)
src/runner/api/resources.py                                 # NEW (~40 LOC)
src/runner/api/schemas_resources.py                         # NEW (~30 LOC)
src/api/clients/job_client.py                               # MODIFIED (+get_resources)
src/pipeline/stages/training_monitor.py                     # MODIFIED (_maybe_log_status pollуem)
```

### Phase 2.3 (logs) + Phase 3 — пока в roadmap, конкретные файлы определяются после approval scope

---

## 10. Verification

### After Phase 0

Manual reproduction текущего бага:
1. Намеренно ломаем pipeline_config (e.g. provider=invalid).
2. Запускаем pipeline.
3. **Acceptance**:
   - В `training_monitor.log` traceback виден **один раз**, не два.
   - Нет `[ERR]` / `[OUT]` префиксов в Mac log lines.
   - `[MONITOR] running | ...` строка появляется (если successful trainer init), либо явно "monitor active, waiting for first health snapshot".

### After Phase 1

```bash
pytest src/tests/unit/runner/api/test_errors.py src/tests/unit/runner/api/test_exception_handlers.py src/tests/unit/api/clients/test_problem_details.py -v
ruff check src/runner/api/ src/api/clients/
mypy src/runner/api/ src/api/clients/
```

### After Phase 2

End-to-end live run:
1. Стандартный run — status line каждые 15s **с самого start**, не с T+30s.
2. `_collect_death_diagnostics` показывает kernel/GPU данные через HTTP, не SSH probes.
3. Если runner мёртв (форсированно убить uvicorn): postmortem warning "diagnostics unavailable" вместо exception.

---

## 11. Что мы явно НЕ делаем

1. **Не переписываем existing endpoints на APIError в Phase 1**. Это deferred в Phase 3.4. Phase 1 — только infrastructure + new endpoints используют сразу.

2. **Не убираем RunPod SDK calls** (pod recovery). Это Mac↔cloud, не Mac↔pod. Внешний API провайдера, не наш runner.

3. **Не убираем bootstrap SSH** (`rsync`, `runner_launcher` `uvicorn`, `dependency_installer`, `file_uploader` для config/dataset). HTTP не может поднять сам себя. Документируется в `SSH_SURFACE.md` (Phase 3.1).

4. **Не делаем Web frontend migration**. `web/src/api/openapi.json` обновляется в Phase 3.3 как часть DTO sync. Frontend ownership separate из этого плана.

5. **Не строим CLI/web client error parsing** в Phase 1. `JobClient` — единственный consumer; CLI и web используют его. Migration не нужна.

6. **Не вводим `application/problem+json` для существующих 200 success responses**. RFC 9457 — только для ошибок 4xx/5xx.

---

## 12. Phasing recommendation matrix

| Scope | PRs | LOC | Risk | Closes user complaints | Strategic foundation? |
|---|---|---|---|---|---|
| **Phase 0 only** | 3 | ~80 | Low | ✅ all 3 visible bugs | ❌ no |
| **Phase 0 + 1** | 4 | ~380 | Low-Medium | ✅ all 3 + roadmap-ready | ✅ error contract foundation, no behavior change |
| **Phase 0 + 1 + 2** | 6-7 | ~1000 | Medium | ✅ all 3 + status line works on early crash + SSH probes removed from monitor | ✅✅ HTTP unification of monitor concerns |
| **Phase 0 + 1 + 2 + 3** | 10-11 | ~1700 | Medium-High | ✅ all + full HTTP transport + DTO consistency + SSH surface documented | ✅✅✅ full architectural cleanup |

**Моё экспертное рекомендация (не yes-man)**:

Phase 0 — **обязательно сейчас**. Это 80 LOC, очевидные баги, низкий риск. Без него monitor.log остаётся уродливым.

Phase 1 — **рекомендую сейчас**. ~300 LOC, без behavior change, закладывает правильный фундамент. Phase 2/3 endpoints сразу используют это.

Phase 2 — **рекомендую как следующий итерационный шаг**, не одним батчем. Полезно потому что:
- `GET /resources` решает реальную проблему status line при early crash.
- `GET /diagnostics` убирает SSH `dmesg`/`nvidia-smi` из monitor → monitor становится "pure observer" вместо смеси concerns.

Phase 3 — **рекомендую как роадмап на 2-3 недели**. Делать по мере того как появляются конкретные триггеры (например: web frontend нуждается в обновлённой openapi.json → runs PR-3.3). Не делать всё сразу.

---

## 13. User decisions (resolved 2026-05-02)

### Q1 RESOLVED — push-tail убираем

**User decision**: «убери блядский стриминг логов через WebSocket» → **Вариант B** (удалить push-tail).

**Что удаляется в Phase 0**:
- `Supervisor._read_stdio_tail` (supervisor.py).
- `Supervisor._reap` обратно к v1 schema payload `{exit_code, signal, cancellation_requested}` (без `stderr_tail`/`stdout_tail`/`stdio_log_path`/`schema_version`).
- `TRAINER_EXITED_SCHEMA_VERSION` constant (supervisor.py).
- `STDIO_TAIL_MAX_BYTES`, `STDIO_TAIL_MAX_LINES` constants.
- `redact_secrets` integration в supervisor (но сам helper остаётся в `src/utils/secret_redaction.py` — он universal-purpose).
- `_log_trainer_exited_tail` (training_monitor.py).
- `_handle_trainer_exited` упрощается: больше нет schema_version branching.

**Тесты которые удаляются**:
- `src/tests/unit/runner/test_supervisor_stdio_tail.py` (целиком — 13 тестов покрывают только push path).
- `src/tests/unit/pipeline/test_training_monitor_v2_payload.py::TestTailLogging` + `TestSchemaVersionGate` тесты (не работают без push).

**Что остаётся**:
- Pull-postmortem (`_collect_death_diagnostics` + `_dump_local_log_tail`) — единственный источник tail'а на Mac.
- PR-C `TERMINATED_AFTER_DIAGNOSTIC_GRACE` (30s grace) покрывает race с pod cleanup.
- При platform eviction <30s видим `<<MISSING>>` — accepted edge case (rare RunPod aggressive eviction).

### Q2 RESOLVED — strip prefix полностью

**User decision**: **Вариант A** — `[OUT]`/`[ERR]` strip, всё через `logger.info`.

Pod-side формат файла не меняется (Supervisor продолжает писать с префиксом для disk forensics). Mac-side `_dump_local_log_tail` парсит и эмитит чистую строку.

### Q3 RESOLVED — full roadmap Phase 0-3

**User decision**: **Вариант D** — полный roadmap, 10-11 PR, ~1700 LOC. Атомарные landings, не batch.

### Q4 RESOLVED — ship всё через `src/` rsync, плюс defer architectural refactor в Phase 3

**User insight**: «пакет пайплайна это уже отдельная логика, не правильно спроектировано, как лучше доставлять код на провайдер? По идее это же как агент идет? Просто тупо весь код доставлять на провайдер каждый раз».

**Industry research**:
- **Bake-in image** (k8s Jobs, Run:ai) — immutable, slow iteration.
- **Sync всего workspace** (mutagen/rsync, RunPod community pattern) — простота, нет drift, real-time.
- **Selective whitelist** (Bazel/Pants) — чистая архитектура, **drift hell**.

Текущий проект использует selective whitelist БЕЗ enforcement → drift сегодня (`src.providers.runpod.training.provider` импортирует `src.pipeline` который НЕ в списке).

**Decision**: разделить shipping policy и enforcement.

**Phase 0 (immediate, ~5 LOC)**:
- `CodeSyncer.REQUIRED_MODULES` сужается до **`["src"]` directory** (с уже существующими `EXCLUDE_PATTERNS`: `tests`, `__pycache__`, `*.pyc`, `.pytest_cache`, `*.md`).
- Removed: explicit list of submodule paths.
- `_REQUIRED_SRC_MODULES` в `runtime_check.py --check-source` остаётся для **import smoke test** (entry points only) — это validation, не shipping.

**Real numbers** (verified 2026-05-02):
- Текущий ship: ~5.2 MB / ~80k LOC
- Full src/ ship: ~3.4 MB / ~93k LOC (минус tests/cache даёт меньше чем selective потому что selective уже включал часть тяжёлого)
- Дельта на pod: +1.4 MB / +13k LOC (`src/pipeline` 19k, `src/api` 8k, `src/cli` 5k, `src/reports` 5k, `src/evaluation` 1.4k, `src/cli_state` 200)
- rsync first-run overhead: ~150 ms (на 25 MB/s сети)
- rsync repeat overhead если ничего не менялось: ~50 ms (metadata roundtrip)
- В контексте deploy ~3-4 мин cold-start: **шум**, не заметно
- Mac-only код **физически на pod, но не загружается в RAM** (trainer не импортирует)
- Никаких secrets в этом коде — только source

**User decision (final)**: ship всё `src/`. EXCLUDE_PATTERNS оставляет на disk только то, что не нужно (tests, cache, .md). Architectural enforcement (что pod-side код НЕ импортирует Mac-only код) идёт в Phase 3 через static linter — это **отдельная плоскость** (enforcement vs shipping).

**Phase 3 (architectural cleanup)**:
- Static linter rule (`importlinter`/`grimp` или ad-hoc AST scan): запретить `src.providers.* → src.pipeline.*`, `src.providers.* → src.api.*`, `src.providers.* → src.cli.*`. CI fail если нарушено.
- Refactor существующего violation: `src.providers.runpod.training.provider` import из `src.pipeline` → переместить общий код в `src.utils` или `src.contracts`.
- Drift-guard test остаётся (entry points validation), но не drift-of-shipping (всё доезжает по умолчанию).

### Q5 (NEW) — pre-existing bug: import gate validation на runtime

**Замеченный side observation** в текущем run'е: gate проверял **только верхушку** (`src.providers`), не sub-modules. После Phase 0 Q4 fix (ship src/ полностью) — sub-modules доезжают **на pod**, но gate всё равно может НЕ ловить runtime ImportError в трансзитивных импортах.

**Phase 0 PR-0.4** (~10 LOC): расширить `runtime_check.py --check-source` чтобы импортировал не только entry points, а каждый из них **транзитивно через `import src.training.run_training`** (уже работает — импорт run_training запускает все top-level imports). Drift-test проверяет что entry points covered.

Текущий drift test уже это делает. Рискзначит — конкретный sub-module failure ловится только если impacted entry point его импортирует. Если `src.providers.runpod.training.provider` импортируется только trainer-side _провайдером во время run-init, не на cold-start — gate его может пропустить. Это нужно verify в реализации Phase 0.

---

## 14. Audit trail

**Iteration 1** (initial decomposition):
- ✅ Identified: пользователь жалуется на 3 visible bugs + 3 strategic asks. Не путать.
- ✅ Identified: training_monitor.py — 99.9% bug-prone hotspot, sole owner. Большие рефакторинги имеют high risk.
- ✅ Decision: разделить на immediate (Phase 0) + foundation (Phase 1) + roadmap (Phase 2/3).

**Iteration 2** (Plan agent synthesis):
- ✅ Architectural agent предложил 5 PR full unification — over-engineering для текущей боли.
- ✅ Pragmatic agent предложил 3 PR — minimal scope.
- ✅ Synthesis: Pragmatic для Phase 0, Architectural's PR-1 для Phase 1, остальное — Phase 2/3 как roadmap (опционально).

**Iteration 3** (production failure modes):
- ✅ Identified RP9: каждый PR в monitor.py — bug-prone hotspot. Mitigation: атомарные landings, не batch.
- ✅ Identified RP12: `gpu_temp_c` отсутствует в `health_snapshot` payload — Phase 0 PR-0.3 graceful fallback на `—`, Phase 2.2 закроет полностью.
- ✅ Identified RP3: web frontend ломается на problem+json — Phase 1 НЕ переписывает existing endpoints, Phase 3.3 решает с openapi sync.

---

## 15. Next step

**Не начинаю реализацию** до получения ответов на §13 Open Questions от пользователя.
