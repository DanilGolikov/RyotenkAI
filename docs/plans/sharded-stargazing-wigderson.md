# Unified Error Model — Core → CLI → API (→ Frontend later)

## Context

В RyotenkAI сейчас сосуществует **три параллельных мира ошибок**:

1. **Wire-level (golden):** `packages/pod/src/ryotenkai_pod/runner/api/errors.py` уже эмитит RFC 9457 `application/problem+json` через `APIError` + 4 FastAPI handlers, опираясь на `ProblemDetails` + `ErrorCode` enum (41 значение, never-delete) в `packages/shared/src/ryotenkai_shared/contracts/problem_details.py`. Mac-сторона парсит это в `APIException`/`TransportError`.
2. **Control API (ad-hoc):** `packages/control/src/ryotenkai_control/api/exceptions.py::install_exception_handlers` возвращает голый `{detail, code: lowercase}` — НЕ RFC 9457, несовместимо с pod wire-form.
3. **Internal domain (parallel universe):** `packages/shared/.../utils/result.py` — `AppError` dataclass + `Result[T, E]` Success/Failure. `AppError.code` — это plain `str` (`"CONFIG_ERROR"`), не пересекается с `ErrorCode` enum. Используется в **~20 pipeline stages**. Плюс ~40 unrelated exception классов (`SupervisorBusy`, `PipelineStateLockError`, `RunPodAPIError`, `SSHConnectionError`, `HFAuthError`, `ProjectNotFoundError` …), наследуемых от `Exception`/`RuntimeError`/`LookupError` без общего корня.

CLI (`packages/control/.../cli/errors.py::die`) умеет рендерить локально создаваемые `typer.Exit`, но **не ловит** `APIException` от runner и не имеет top-level adapter для `RyotenkAIError`. Trainer subprocess сообщает о падении только через exit-code + EventBus event без структурированного payload. Provider boundary (RunPod/SSH/HF SDK) — отдельные `RuntimeError`-rooted hierarchies, не маппятся в `ErrorCode`.

**Цель**: одна типизированная internal иерархия (`RyotenkAIError`), один boundary protocol (RFC 9457 problem+json), один CLI renderer, end-to-end correlation через `request_id` + `trace_id`. Без backwards-compat shims (политика проекта). Pipeline conditions[] (k8s/OpenShift operator pattern) включаются как **Phase G** — отдельная единица проектирования поверх готового error-фундамента.

## Design summary

- **Один корень — `RyotenkAIError(Exception)`** в `packages/shared/src/ryotenkai_shared/errors/`. Поля: `code: ErrorCode`, `status: int`, `title: str`, `detail: str|None`, `context: dict|None`, `cause: Exception|None`. Подклассы — два абстрактных маркера: `DomainError` (4xx, юзерские ошибки, нужно показывать) и `InfrastructureError` (5xx, внешние/transient/баги). Третий `InternalError` — catch-all 500. Concrete subclasses pin `code`/`status` как `ClassVar`.
- **RFC 9457 — единственный wire-protocol для HTTP.** Pod handlers поднимаются из `packages/pod/.../runner/api/errors.py` в `packages/shared/src/ryotenkai_shared/api/error_handlers.py` и переиспользуются control API. Mac-side parser (`parse_problem_details`) остаётся как есть.
- **`AppError` целиком удалён. `Result[T, E]` — выкорчеван** (по решению пользователя). 20 pipeline stages переписаны на raise-based: stage возвращает T при успехе, raises `RyotenkAIError` при ошибке. Stage execution loop ловит и переводит attempt в failed-state.
- **Один тип имени для ошибки: `RyotenkAIError` везде.** Wire-parsed exceptions (Mac client) и in-process exceptions — один тип, плюс factory `RyotenkAIError.from_problem(problem)`. `TransportError(RyotenkAIError)` — подкласс с `code=TRANSPORT_UNREACHABLE`. `APIError` и `APIException` удаляются (становятся семантическими алиасами в финальной фазе, удаляются в Phase F).
- **Request-ID middleware** в обоих FastAPI приложениях через shared `RequestIDMiddleware` + `contextvars.ContextVar`. `_build_response()` читает из контекста и стампит в `ProblemDetails.request_id`. `trace_id` (8-hex per-error) остаётся отдельно — grep-anchor для конкретного error event, тогда как `request_id` (16-hex per-request) — анкер для всех логов запроса. Совместимо с будущим OpenTelemetry traceparent.
- **CLI — kubectl/Terraform-style рендеринг.** Top-level `wrap_command(fn)` decorator на каждой Typer-команде ловит `RyotenkAIError`/`ValidationError`/`FileNotFoundError`/`yaml.YAMLError` и рендерит `error: <title>` + `hint:` + `code:` + `trace:` + `request:`. Exit codes: 4xx → 2 (user-fixable), 5xx → 1 (system error).
- **Phase G — Operator-style conditions[] на pipeline stages.** Отдельная фаза поверх готового error-фундамента: `PipelineStage.conditions: list[Condition]` с полями `type/status/reason/message/lastTransitionTime`. `reason` reuse `ErrorCode` enum (k8s metav1.Status precedent). Storage в pipeline state JSON. Поверхность для warnings/progress hints (не только финальная ошибка).

## Layer 1: Internal exception hierarchy

**Новый модуль**: `packages/shared/src/ryotenkai_shared/errors/`

```
errors/
  __init__.py         # public re-exports
  base.py             # RyotenkAIError + DomainError/InfrastructureError/InternalError
  domain.py           # 4xx concrete subclasses
  infra.py            # 5xx concrete subclasses
  _render.py          # _DEFAULT_TITLES, _new_trace_id (lifted from pod)
  _factory.py         # RyotenkAIError.from_problem(), from_exception()
```

**`base.py` shape** (~15 lines):

```python
class RyotenkAIError(Exception):
    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR
    status: ClassVar[int] = 500
    title_default: ClassVar[str | None] = None

    def __init__(self, detail: str | None = None, *,
                 context: dict[str, Any] | None = None,
                 cause: Exception | None = None) -> None:
        super().__init__(f"{self.code.value}: {detail or self.title_default or ''}")
        self.detail = detail
        self.context = context or {}
        if cause is not None:
            self.__cause__ = cause

    @property
    def title(self) -> str:
        return self.title_default or _DEFAULT_TITLES[self.code]

    def as_problem(self, *, instance: str | None = None,
                   trace_id: str | None = None,
                   request_id: str | None = None) -> ProblemDetails: ...

    @classmethod
    def from_problem(cls, problem: ProblemDetails) -> "RyotenkAIError": ...
```

**Concrete seed (~15 classes; больше добавляется по мере миграции существующих exception):**

- `domain.py`: `ConfigInvalidError` (CONFIG_INVALID/400), `ConfigDriftError` (CONFIG_DRIFT/409), `ConfigFileNotFoundError` (CONFIG_FILE_NOT_FOUND/404), `JobSpecInvalidError` (JOB_SPEC_INVALID/422), `JobStateInvalidError` (JOB_STATE_INVALID/409), `ProjectNotFoundError` (PROJECT_NOT_FOUND/404), `StateLoadFailedError` (STATE_LOAD_FAILED/404), `StateLockedError` (STATE_LOCKED/409), `LaunchInProgressError` (LAUNCH_IN_PROGRESS/409), `HFAuthFailedError` (HF_AUTH_FAILED/401), `HFNotFoundError` (HF_NOT_FOUND/404).
- `infra.py`: `ProviderUnavailableError` (PROVIDER_UNAVAILABLE/503), `ProviderRateLimitedError` (PROVIDER_RATE_LIMITED/429), `SSHConnectionFailedError` (SSH_CONNECTION_FAILED/502), `SSHExecFailedError` (SSH_EXEC_FAILED/502), `TrainingFailedError` (TRAINING_FAILED/500), `TrainingOOMError` (TRAINING_OOM/500), `LaunchPreparationError` (LAUNCH_PREPARATION_FAILED/500), `WorkspaceStoreFailedError` (WORKSPACE_STORE_FAILED/500), `InferenceUnavailableError` (INFERENCE_UNAVAILABLE/503), `TransportError` (TRANSPORT_UNREACHABLE/599 — Mac-side synthesised).

**`AppError` fate**: удалён полностью. Все 8 dataclass-подклассов (`ConfigError`, `TrainingError`, `DatasetError`, `ModelError`, `StrategyError`, `OOMError`, `DataLoaderError`, `ProviderError`, `InferenceError`) заменяются typed exception subclasses из `domain.py`/`infra.py`. **Risk-flag**: 20+ pipeline stage файлов используют `Result[T, AppError]` — миграция нетривиальна (см. R-MIGRATION-SCOPE ниже).

**`Result[T, E]` fate**: удалён вместе с `Success`/`Failure`/`ResultHelpers`/`ok`/`err`/`Ok`/`Err`. Pipeline stages переписаны на raise-based: stage method возвращает T (или None) при успехе, raises `RyotenkAIError` при failure. Stage execution loop (`packages/control/.../pipeline/execution/stage_execution_loop.py`) ловит `RyotenkAIError` → переводит attempt в failed state, логирует `error.as_problem().model_dump()` для structured logging.

## Layer 2: `ErrorCode` catalog extension

Add to `ErrorCode` StrEnum в `packages/shared/.../contracts/problem_details.py`. Только additive (never-delete policy). Семантика: domain prefix + UPPER_SNAKE_CASE.

| ErrorCode | HTTP | Flavour | Domain |
|---|---|---|---|
| `CONFIG_INVALID` | 400 | Domain | Config |
| `CONFIG_DRIFT` | 409 | Domain | Config |
| `CONFIG_FILE_NOT_FOUND` | 404 | Domain | Config |
| `PROJECT_NOT_FOUND` | 404 | Domain | Workspace |
| `PROJECT_ALREADY_EXISTS` | 409 | Domain | Workspace |
| `PROVIDER_NOT_FOUND` | 404 | Domain | Workspace |
| `INTEGRATION_NOT_FOUND` | 404 | Domain | Workspace |
| `WORKSPACE_STORE_FAILED` | 500 | Infra | Workspace |
| `STATE_LOAD_FAILED` | 404 | Domain | Pipeline state |
| `STATE_LOCKED` | 409 | Domain | Pipeline state |
| `LAUNCH_IN_PROGRESS` | 409 | Domain | Pipeline |
| `LAUNCH_PREPARATION_FAILED` | 500 | Infra | Pipeline |
| `PIPELINE_STAGE_FAILED` | 500 | Infra | Pipeline |
| `RUN_IS_ACTIVE` | 409 | Domain | Pipeline |
| `TRAINING_FAILED` | 500 | Infra | Trainer |
| `TRAINING_OOM` | 500 | Infra | Trainer |
| `DATASET_LOAD_FAILED` | 422 | Domain | Dataset |
| `DATASET_VALIDATION_FAILED` | 422 | Domain | Dataset |
| `MODEL_LOAD_FAILED` | 500 | Infra | Model |
| `INFERENCE_UNAVAILABLE` | 503 | Infra | Inference |
| `PROVIDER_UNAVAILABLE` | 503 | Infra | Providers |
| `PROVIDER_RATE_LIMITED` | 429 | Infra | Providers |
| `PROVIDER_AUTH_FAILED` | 401 | Domain | Providers |
| `SSH_CONNECTION_FAILED` | 502 | Infra | SSH |
| `SSH_EXEC_FAILED` | 502 | Infra | SSH |
| `SSH_TRANSFER_FAILED` | 502 | Infra | SSH |
| `HF_AUTH_FAILED` | 401 | Domain | HF Hub |
| `HF_NOT_FOUND` | 404 | Domain | HF Hub |
| `ENGINE_NOT_REGISTERED` | 404 | Domain | Engines |
| `ENGINE_CONFIG_INVALID` | 422 | Domain | Engines |

`_DEFAULT_TITLES` map в `errors/_render.py` обновляется в lockstep — sentinel `test_error_code_pinned.py` блокирует PR если хоть одна запись отсутствует.

## Layer 3: Promoted shared API error handlers

**Move** (не copy — single-PR cut) из `packages/pod/.../runner/api/errors.py` в `packages/shared/src/ryotenkai_shared/api/error_handlers.py`:

- `APIError` класс → удалён, его роль играет `RyotenkAIError`.
- 4 handler-функции (`api_error_handler`, `http_exception_handler`, `validation_exception_handler`, `generic_exception_handler`) — bodies сохраняются, dispatch на `RyotenkAIError` вместо `APIError`.
- `_DEFAULT_TITLES`, `_build_response`, `_new_trace_id`, `_LEGACY_CODE_ALIASES` → `errors/_render.py` (private).
- `EXCEPTION_HANDLERS: dict[Any, Any]` + helper `install_exception_handlers(app: FastAPI)` — для использования через `FastAPI(exception_handlers=EXCEPTION_HANDLERS)` (синхронная регистрация до первого request, mitigation RP20).

Pod runner переписывает `packages/pod/.../runner/api/errors.py` на тонкий re-export shim — потом удаляется в Phase F.

**ВАЖНЫЙ РИСК (R-FASTAPI-DEP)**: `packages/shared/pyproject.toml` сейчас НЕ содержит `fastapi` в dependencies (есть только pydantic, httpx, websockets, mlflow, …). Plan agent ошибся, утверждая что FastAPI — transitive dep через Pydantic (на самом деле наоборот: FastAPI depends on Pydantic). **Решение**: добавить `fastapi>=0.115.0,<1.0.0` в `packages/shared/pyproject.toml`. Альтернатива (новый пакет `packages/api_common/`) отклонена — overkill для 1 модуля + 1 middleware. Shared уже не "thin leaf" — он тащит mlflow, websockets, httpx; добавление FastAPI семантически на уровне с этими.

**Сентинель-валидация Phase B**: `tests/_lint/test_no_fastapi_outside_shared_api.py` — `from fastapi import` запрещён в `packages/shared/` вне subdirectory `api/`, чтобы случайно не утечь FastAPI imports в leaf-модули (`contracts/`, `utils/`, `infrastructure/`).

## Layer 4: Control API integration

**Delete**: `packages/control/src/ryotenkai_control/api/exceptions.py` целиком. 4 ad-hoc handler-функции там заменяются shared `EXCEPTION_HANDLERS`, mounted в `packages/control/src/ryotenkai_control/api/main.py:96` через `FastAPI(exception_handlers=EXCEPTION_HANDLERS, …)`.

**Migrate control exceptions** в 3 группы:

1. **Subclass new domain hierarchy**:
   - `PipelineStateLoadError` → `StateLoadFailedError(DomainError, code=STATE_LOAD_FAILED, status=404)`
   - `PipelineStateLockError` → `StateLockedError(DomainError, code=STATE_LOCKED, status=409)`
   - `ProjectNotFoundError` (currently `LookupError`) → subclass `RyotenkAIError(DomainError, code=PROJECT_NOT_FOUND, status=404)`
   - `LaunchAlreadyRunningError` → `LaunchInProgressError(DomainError, code=LAUNCH_IN_PROGRESS, status=409)`
   - `RunIsActiveError` → `RunIsActiveError(DomainError, code=RUN_IS_ACTIVE, status=409)`

2. **Subclass infrastructure hierarchy**:
   - `WorkspaceStoreError`, `IntegrationStoreError`, `ProjectStoreError`, `ProviderStoreError` → `WorkspaceStoreFailedError(InfrastructureError, code=WORKSPACE_STORE_FAILED, status=500)`
   - `LaunchPreparationError` → `LaunchPreparationError(InfrastructureError, code=LAUNCH_PREPARATION_FAILED, status=500)`
   - `MLflowManagerNotInitializedError` → `InternalError(code=INTERNAL_ERROR, status=500)`
   - `EngineRegistryError`/`EngineNotRegistered`/`EngineConfigError` → `EngineNotRegisteredError`/`EngineConfigInvalidError`
   - `JobSubmissionLoadError` → `InternalError` (служебный path)

3. **Delete inline**: `_FileUploadFailed`, `_ImportCheckFailed` (private helpers, заменяемые typed raise в call site); `PluginPackError` → `TrainerSpawnFailedError`; `IntegrationServiceError`/`ProjectServiceError`/`ProviderServiceError`/`LaunchAlreadyRunningError` — переименовать в domain-typed.

Control-side `ValueError` handler в exceptions.py исчезает — `RequestValidationError` теперь handled uniformly shared validation_exception_handler. Bare `raise ValueError(...)` в API services → `raise JobSpecInvalidError(...)` в call site.

## Layer 5: Provider boundary translation

**Pattern**: каждый provider adapter получает приватный модуль `_translate.py` с функцией маппинга `vendor_exception → InfrastructureError subclass`. Vendor exception классы (RunPod*, SSH*, HF*) **никогда** не пересекают boundary `packages/providers/` или `packages/shared/.../infrastructure/`.

Пример (RunPod):

```python
# packages/providers/src/.../runpod/_translate.py
from ryotenkai_shared.errors import (
    ProviderRateLimitedError, ProviderUnavailableError,
)
from ryotenkai_shared.infrastructure.runpod_api.protocol import (
    RunPodRateLimitedError, RunPodTransientError, RunPodAPIError,
)

def to_ryotenkai(exc: RunPodAPIError) -> InfrastructureError:
    if isinstance(exc, RunPodRateLimitedError):
        return ProviderRateLimitedError(
            detail=str(exc),
            context={"retry_after_seconds": getattr(exc, "retry_after", None)},
            cause=exc,
        )
    if isinstance(exc, RunPodTransientError):
        return ProviderUnavailableError(detail=str(exc), cause=exc)
    return ProviderUnavailableError(detail=f"runpod: {exc}", cause=exc)
```

**Translation table (Phase E full scope, 10 seeds):**

| Vendor exception | New typed exception | ErrorCode |
|---|---|---|
| `RunPodRateLimitedError` | `ProviderRateLimitedError` | `PROVIDER_RATE_LIMITED` |
| `RunPodTransientError` | `ProviderUnavailableError` | `PROVIDER_UNAVAILABLE` |
| `RunPodPartialResponseError` | `ProviderUnavailableError` | `PROVIDER_UNAVAILABLE` |
| `SSHConnectionError` | `SSHConnectionFailedError` | `SSH_CONNECTION_FAILED` |
| `SSHExecError` | `SSHExecFailedError` | `SSH_EXEC_FAILED` |
| `SSHTransferError` | `SSHTransferFailedError` | `SSH_TRANSFER_FAILED` |
| `HFAuthError` | `HFAuthFailedError` | `HF_AUTH_FAILED` |
| `HFNotFoundError` | `HFNotFoundError` (renamed) | `HF_NOT_FOUND` |
| `HFRateLimitedError` | `ProviderRateLimitedError` | `PROVIDER_RATE_LIMITED` |
| `JobClientRateLimitedError` | `ProviderRateLimitedError` | `PROVIDER_RATE_LIMITED` |

**Importlinter contract** (новый): `ryotenkai_providers.*._translate` — единственный модуль, которому разрешено импортировать vendor exception types. Sentinel-test добавляется в Phase E.

## Layer 6: Trainer subprocess exit payload

**Новый contract**: `packages/shared/src/ryotenkai_shared/contracts/trainer_exit.py`

```python
class TrainerExitPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: ErrorCode
    message: str
    traceback_summary: str | None = None  # last 30 lines, paths stripped
    exit_code: int
    wall_seconds: float
    schema_version: int = 1
```

**Lifecycle**:

1. **Trainer** при старте регистрирует `atexit` handler в `packages/pod/src/ryotenkai_pod/trainer/` entrypoint. При любой `RyotenkAIError` (или generic Exception) — handler ловит, пишет JSON в `<artifact_root>/trainer-exit.json` с правильным `code` (TRAINING_FAILED / TRAINING_OOM / INTERNAL_ERROR). Traceback **обрезается до 30 строк** + paths sanitized regex `r'/Users/.+?/site-packages/' → '<sp>/'` (security: no path leak).
2. **Supervisor** в `packages/pod/.../runner/supervisor.py` при reap:
   - Если файл существует + проходит Pydantic-валидацию → `EventBus.publish("trainer_exited", payload)` с структурированным payload.
   - Если файл отсутствует И `exit_code != 0` → синтезируется payload `code=INTERNAL_ERROR, message="trainer exited without payload"`. **No guessing** (strict policy).
3. **EventBus** `trainer_exited` event schema bumps `schema_version=2`. Consumer в `packages/control/.../pipeline/training_monitor.py` декодирует через типизированную модель, raises `TrainingFailedError`/`TrainingOOMError`/`InternalError` accordingly.

**RISK R-SIGKILL**: при SIGKILL (cgroup OOM-killer, exit_code=137) atexit handler НЕ run → файл отсутствует → fallback INTERNAL_ERROR. Это теряет информацию о реальной причине. **Mitigation**: supervisor проверяет exit_code и если == 137 (или -9) — синтезирует payload `code=TRAINING_OOM, message="trainer killed by signal SIGKILL — likely OOM"`. Это эвристика, но безопасная: false-positive TRAINING_OOM лучше чем noisy INTERNAL_ERROR для частого OOM-сценария. Документируется в `_render.py` docstring.

## Layer 7: Request-ID middleware

**Новый модуль**: `packages/shared/src/ryotenkai_shared/api/request_id.py`

```python
REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("X-Request-ID") or secrets.token_hex(8)
        token = REQUEST_ID.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            REQUEST_ID.reset(token)
```

`_build_response()` в `error_handlers.py` читает `REQUEST_ID.get()` и стампит в `ProblemDetails.request_id`.

**Distinct semantics**:
- `request_id` (16 hex chars, set by middleware): один на весь запрос, в response headers + во всех логах запроса.
- `trace_id` (8 hex chars, set by handler): один на каждый emit error (один request может produce несколько лог-линий через chained errors).

Совместимо с будущим OpenTelemetry: `request_id` → `traceparent` header, `trace_id` → span_id. См. Future work.

**Mount points** (order-sensitive — middleware **первой** в stack):
- `packages/pod/src/ryotenkai_pod/runner/api/app.py` (runner FastAPI)
- `packages/control/src/ryotenkai_control/api/main.py:96` (control FastAPI)

**Logger integration**: `loguru` configuration (где-то в `shared/utils/logging`) добавляет filter, который читает `REQUEST_ID.get()` и stamps `request_id` в каждый log record. Без этого — request_id есть в response, но нет в server log, корреляция сломана. Sentinel `tests/_lint/test_logger_carries_request_id.py` валидирует.

## Layer 8: CLI rendering

**Extend** `packages/control/src/ryotenkai_control/cli/errors.py`:

```python
def die_from_ryotenkai(exc: RyotenkAIError, *, request_id: str | None = None) -> typer.Exit:
    err_console.print(f"[{COLOR_ERR}]error:[/{COLOR_ERR}] {exc.title}")
    if exc.detail:
        err_console.print(f"  [{COLOR_DIM}]hint:[/{COLOR_DIM}] {exc.detail}")
    parts = [exc.code.value]
    if isinstance(exc, TransportError):
        parts.append(f"status={exc.status}")
    if hasattr(exc, "trace_id") and exc.trace_id:
        parts.append(f"trace={exc.trace_id}")
    if request_id:
        parts.append(f"request={request_id}")
    err_console.print(f"  [{COLOR_DIM}]code:[/{COLOR_DIM}] {'  '.join(parts)}")
    raise typer.Exit(code=2 if 400 <= exc.status < 500 else 1)


def wrap_command(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RyotenkAIError as exc:
            raise die_from_ryotenkai(exc, request_id=_current_request_id())
        except ValidationError as exc:
            raise die(f"invalid input:\n{format_validation_errors(exc)}")
        except FileNotFoundError as exc:
            raise die(f"file not found: {exc}")
        except yaml.YAMLError as exc:
            raise die(f"invalid YAML: {exc}")
    return wrapper
```

**Exact rendering** (kubectl/Terraform-style):

```
error: Job not found
  hint: job_id="abc123" is not active
  code: JOB_NOT_FOUND  trace=a3b1c2d4  request=8e7f6c5b4a3d2e1f
```

**Exit codes**:
- 4xx HTTP-mapped → exit 2 (user error — fixable by changing inputs)
- 5xx HTTP-mapped → exit 1 (system error — retry or report)
- `InternalError` → exit 1
- `TransportError` → exit 1 (system unreachable)

`--verbose`/`-v` (уже на root Typer app) добавляет 4-ю строку: `context: <pretty JSON of exc.context>`.

CLI use: каждая Typer-команда в `packages/control/.../cli/commands/*.py` оборачивается `@wrap_command` (или применяется через `app.command(decorator=wrap_command)`). Существующий `die()`/`die_from_ryotenkai()` остаются для inline-использования (config loader, validation).

## Layer 9: Sentinel tests (PR-blocking)

Все под `tests/_lint/` (mirror existing `test_no_protocol_mocking.py` pattern):

1. **`test_exception_root.py`** — AST walk: каждый `class .*Error.*(...)` под `packages/*/src/` должен иметь `RyotenkAIError` в MRO ИЛИ быть в allow-yaml (`tests/_lint/exception_root_allowlist.yaml`: vendor SDK exception types вне `_translate.py`, Pydantic models суффиксом `Error`).

2. **`test_error_code_pinned.py`** — для каждого concrete subclass `RyotenkAIError` (не abstract трёх) class-level `code: ClassVar[ErrorCode]` и `status: ClassVar[int]` обязаны быть pinned. `_DEFAULT_TITLES` в `errors/_render.py` содержит запись для каждого `ErrorCode` member.

3. **`test_no_naked_httpexception.py`** — AST walk: `raise HTTPException(...)` запрещён вне allowlist (только `packages/shared/.../api/error_handlers.py::http_exception_handler` остаётся как FastAPI-internal adapter).

4. **`test_no_traceback_in_context.py`** — `traceback.format_exc()` никогда не должен попадать в `RyotenkAIError.context`. AST-pattern: `XError(context={..., "traceback": traceback.format_exc()})` — bad. То же для `TrainerExitPayload.traceback_summary` — должна быть санитизирована (regex strip paths).

5. **`test_no_fastapi_outside_shared_api.py`** — `from fastapi import` запрещён в `packages/shared/` вне `api/` subdirectory (защита от утечки FastAPI в leaf-модули `contracts/`/`utils/`/`infrastructure/`).

6. **`test_logger_carries_request_id.py`** — синхронный smoke: запросить FastAPI app с `X-Request-ID: testid`, проверить что в logs-buffer (loguru capture) есть строка с этим ID.

7. **`test_no_apperror_or_result.py`** — добавляется в Phase A2: AST walk: `from ryotenkai_shared.utils.result import` запрещён ВЕЗДЕ кроме самого `result.py` (который удаляется в финале Phase A2). После завершения A2 sentinel становится permanent.

8. **`test_no_protocol_mocking.py`** (update existing) — allow-list канонических fakes `FakeRyotenkAIError`, `FakeTransportError`, `FakeRyotenkAIErrorAPIBoundary` в `tests/_fakes/`.

9. **`test_provider_translate_isolated.py`** (Phase E) — importlinter contract: vendor exception types (`RunPodAPIError`, `SSHError`, `HFHubError` и потомки) могут импортироваться ТОЛЬКО внутри `_translate.py` модулей или в `packages/shared/.../infrastructure/.../protocol.py`.

## Layer 10: Phase G — Operator-style conditions[] на pipeline stages

**Цель**: rich observability surface для long-running pipeline stages, поверх готового error-фундамента. Не заменяет FSM (FSM остаётся для жизненного цикла state), а augments его warnings/progress.

**Contract**: `packages/shared/src/ryotenkai_shared/contracts/pipeline_conditions.py`

```python
class ConditionStatus(StrEnum):
    TRUE = "True"
    FALSE = "False"
    UNKNOWN = "Unknown"

class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str            # CamelCase, e.g. "Available", "Progressing", "Degraded"
    status: ConditionStatus
    reason: ErrorCode    # reuse error code enum (k8s metav1.Status precedent)
    message: str | None = None
    last_transition_time: datetime
```

**Storage**: добавляется в `packages/control/.../pipeline/state/models.py::PipelineStageState`:

```python
class PipelineStageState(BaseModel):
    stage_id: str
    status: StageStatus
    # NEW Phase G:
    conditions: list[Condition] = Field(default_factory=list)
```

**Update semantics** (k8s convention): `last_transition_time` обновляется только когда `status` меняется. Если повторно публикуется тот же condition с тем же `status` — `last_transition_time` сохраняется (полезно для "Degraded for 5min"). Helper `update_condition(state, type, status, reason, message)` инкапсулирует это правило.

**API surface**: control API возвращает `conditions[]` в `GET /pipeline/state/{run_id}` (новое поле в response model). CLI command `ryotenkai status <run_id>` рендерит kubectl-style table:

```
STAGE          AVAILABLE  PROGRESSING  DEGRADED  AGE
deployment     True       False        False     2m
training       Unknown    True         False     1m
```

**Phase G scope**:
- Contract + storage migration (Pipeline state JSON schema bump)
- Стандартные condition types: `Available` (готов к работе), `Progressing` (идёт операция), `Degraded` (что-то не так, но операция продолжается), `OOMRisk` (warning), `RateLimited` (warning).
- Helper `update_condition()` + tests.
- CLI rendering.
- API endpoint extension.
- **НЕ в scope Phase G**: webhook subscriptions, condition history (только last_transition_time), correlation с MLflow metrics.

**Тесты**: 7 классов для `update_condition()` (особенно TestInvariants: idempotency на same-status update, TestRegressions: last_transition_time не reset'ится при no-op update).

## Migration phases (A → G)

| Phase | Scope | Files touched | Risk |
|---|---|---|---|
| **A1** | `errors/` module (RyotenkAIError + DomainError/InfrastructureError/InternalError + 15 concrete subclasses + factories). Extend ErrorCode enum (~30 codes). Sentinels 1, 2, 5. No behaviour change. | `packages/shared/src/ryotenkai_shared/errors/*` (new), `contracts/problem_details.py` (enum), `tests/_lint/*` | low |
| **A2** | **DESTRUCTIVE**: удалить `packages/shared/.../utils/result.py` (AppError + Result + Success/Failure + ResultHelpers). Переписать ~20 pipeline stages: `Result[T, AppError]` → `T` returns + raises `RyotenkAIError`. Stage execution loop ловит и transitions attempt to failed. Sentinel 7. | 20 файлов в `packages/control/.../pipeline/stages/`, `pipeline/orchestrator.py`, `pipeline/execution/stage_execution_loop.py`, `pipeline/launch/launch_preparator.py`, `pipeline/mlflow_attempt/manager.py` | **HIGH** — big refactor, см. R-MIGRATION-SCOPE |
| **B** | Promote pod handlers → `packages/shared/src/ryotenkai_shared/api/error_handlers.py`. Add FastAPI to shared deps. Rewire pod runner imports. Add sentinel 5 enforcement. | `packages/shared/src/ryotenkai_shared/api/error_handlers.py` (new), `packages/shared/pyproject.toml`, `packages/pod/.../runner/api/errors.py` (becomes shim), runner `FastAPI()` site | low |
| **C** | Replace control `api/exceptions.py` → shared `EXCEPTION_HANDLERS`. Mount `RequestIDMiddleware` + logger integration in both apps. Migrate control exceptions to typed hierarchy. Sentinels 3, 6. **Breaking wire-shape для текущего frontend — мерджим сразу** (frontend rewrite подхватит). | `packages/control/.../api/exceptions.py` (delete), `api/main.py:96`, `pipeline/state/*.py`, `workspace/*/store.py`, `api/services/*.py`, `shared/api/request_id.py` (new) | medium |
| **D** | `TrainerExitPayload` contract. Trainer `atexit` handler. Supervisor reader + SIGKILL/OOM heuristic. EventBus schema_version=2. | `shared/contracts/trainer_exit.py` (new), `pod/.../trainer/__main__.py` (or entrypoint), `pod/.../runner/supervisor.py`, `pod/.../runner/event_bus.py`, `control/.../pipeline/training_monitor.py` | medium |
| **E** | Provider boundary translation modules. По одному provider per PR (RunPod → SSH → HF → engines). Importlinter contract + sentinel 9. | `packages/providers/src/.../*/_translate.py` (new × 4), vendor-call sites | low (additive) |
| **F** | CLI `wrap_command` на каждой Typer-команде. `die_from_ryotenkai` + integration. Delete legacy `RuntimeError`-rooted exception классы после миграции callers. Удалить `APIError`/`APIException` aliases. | `packages/control/src/ryotenkai_control/cli/*.py`, ~12 callsite files | medium |
| **G** | Pipeline conditions[] (operator pattern). Contract + storage migration + `update_condition` helper + CLI table render + API endpoint extension. | `shared/contracts/pipeline_conditions.py` (new), `control/.../pipeline/state/models.py`, `control/.../pipeline/state/transitioner.py`, `control/.../cli/commands/status.py`, `control/.../api/routers/pipeline.py` | medium |

**Phase ordering rationale**: A1 → A2 → B → C → D → E → F → G. A1 безопасный additive. A2 — big bang refactor pipeline (нельзя half-migrate Result), требует своего PR. B/C wire-level cuts, безопасные после A2. D — cross-process protocol, нужен после B (нужны typed errors). E — additive per provider. F — после E (translate уже есть, можно catch typed). G — отдельная фича, после всего фундамента.

## Critical files to modify

| Path | What changes |
|---|---|
| [packages/shared/src/ryotenkai_shared/contracts/problem_details.py](packages/shared/src/ryotenkai_shared/contracts/problem_details.py) | Extend `ErrorCode` enum (+30 codes) |
| [packages/shared/src/ryotenkai_shared/errors/](packages/shared/src/ryotenkai_shared/errors/) **(new)** | Full typed exception hierarchy |
| [packages/shared/src/ryotenkai_shared/api/error_handlers.py](packages/shared/src/ryotenkai_shared/api/error_handlers.py) **(new)** | Promoted handlers from pod |
| [packages/shared/src/ryotenkai_shared/api/request_id.py](packages/shared/src/ryotenkai_shared/api/request_id.py) **(new)** | RequestIDMiddleware + ContextVar |
| [packages/shared/src/ryotenkai_shared/contracts/trainer_exit.py](packages/shared/src/ryotenkai_shared/contracts/trainer_exit.py) **(new)** | TrainerExitPayload model |
| [packages/shared/src/ryotenkai_shared/contracts/pipeline_conditions.py](packages/shared/src/ryotenkai_shared/contracts/pipeline_conditions.py) **(new, Phase G)** | Condition model |
| [packages/shared/pyproject.toml](packages/shared/pyproject.toml) | Add `fastapi>=0.115.0,<1.0.0` |
| [packages/shared/src/ryotenkai_shared/utils/result.py](packages/shared/src/ryotenkai_shared/utils/result.py) | **DELETE** in Phase A2 |
| [packages/pod/src/ryotenkai_pod/runner/api/errors.py](packages/pod/src/ryotenkai_pod/runner/api/errors.py) | Phase B: thin re-export; Phase F: delete |
| [packages/pod/src/ryotenkai_pod/runner/supervisor.py](packages/pod/src/ryotenkai_pod/runner/supervisor.py) | Phase D: read trainer-exit.json |
| [packages/control/src/ryotenkai_control/api/exceptions.py](packages/control/src/ryotenkai_control/api/exceptions.py) | Phase C: **DELETE** |
| [packages/control/src/ryotenkai_control/api/main.py](packages/control/src/ryotenkai_control/api/main.py) | Phase C: mount EXCEPTION_HANDLERS + RequestIDMiddleware |
| [packages/control/src/ryotenkai_control/pipeline/state/models.py](packages/control/src/ryotenkai_control/pipeline/state/models.py) | Phase G: conditions[] field |
| [packages/control/src/ryotenkai_control/cli/errors.py](packages/control/src/ryotenkai_control/cli/errors.py) | Phase F: add `die_from_ryotenkai`, `wrap_command` |
| [packages/control/src/ryotenkai_control/pipeline/execution/stage_execution_loop.py](packages/control/src/ryotenkai_control/pipeline/execution/stage_execution_loop.py) | Phase A2: catch RyotenkAIError instead of unwrap Failure |
| ~20 файлов под `packages/control/.../pipeline/stages/` | Phase A2: переписать Result returns на raise |
| `packages/providers/src/.../*/_translate.py` × 4 **(new)** | Phase E: vendor translation |
| `tests/_lint/test_*.py` × 8 **(new + 1 update)** | Sentinel suite |

## Test strategy (7 классов per CLAUDE.md)

Для `RyotenkAIError` base в [tests/unit/shared/errors/test_base.py](tests/unit/shared/errors/test_base.py):

1. **TestPositive** — construct каждый concrete subclass с `detail=`/`context=`/`cause=`; `code`/`status`/`title` match class defaults; `as_problem()` returns valid `ProblemDetails`; `from_problem(p)` round-trips.
2. **TestNegative** — `RyotenkAIError()` без subclass → `code=INTERNAL_ERROR, status=500`; instantiate abstract `DomainError()` напрямую — allowed (markers, не abstract в Python sense, но style-checked).
3. **TestBoundary** — `detail=""` vs `detail=None` produce different `as_problem().model_dump(exclude_none=True)` (empty string preserved, None stripped per RFC 9457 §3.1); `context={}` vs `context=None`; max context size (если ввести limit).
4. **TestInvariants** — `code`/`status` immutable после construct; `MRO ∋ RyotenkAIError` для каждого concrete; `_DEFAULT_TITLES[cls.code]` exists; constants pinning: `_new_trace_id()` length=8, hex; PROBLEM_JSON_MEDIA_TYPE = "application/problem+json".
5. **TestDependencyErrors** — `from_exception(orig)` preserves `__cause__`; `str(exc)` включает vendor message; `context` НЕ contains traceback; logger integration: `logger.error(exc)` produces log with `request_id` from ContextVar.
6. **TestRegressions** — `traceback.format_exc()` never escapes to `as_problem().context` (frozen regression test, ref: AppError leak); migration debt: каждая удалённая AppError subclass имеет replacement test.
7. **TestLogicSpecific** — `status → exit_code` mapping table (400→2, 422→2, 429→2, 500→1, 502→1, 503→1, 599→1, INTERNAL→1); `request_id` propagation через 3 уровня call stack; `trace_id` uniqueness across 1000 generations; HTTP status-class → flavour mapping (4xx→DomainError, 5xx→InfrastructureError).

Плюс per-concrete-class TestPositive smoke в `tests/unit/shared/errors/test_domain.py` и `test_infra.py` — exhaustive class-by-class — overkill, base покрывает invariants.

Для `RequestIDMiddleware`, `TrainerExitPayload`, `Condition` — отдельные test_modules с теми же 7 классами.

## Verification (end-to-end smoke)

После полного merge phases A-F (Phase G отдельно):

1. **CLI end-to-end** (manual, до автомата):
   ```bash
   ryotenkai status nonexistent-run-id
   # Expected:
   # error: Pipeline state not found
   #   hint: run_id="nonexistent-run-id" has no state file
   #   code: STATE_LOAD_FAILED  trace=<8hex>  request=<16hex>
   # exit code: 2
   ```

2. **Control API contract**:
   ```bash
   curl -i http://localhost:8000/api/v1/pipeline/state/nonexistent
   # Expected:
   # HTTP/1.1 404 Not Found
   # Content-Type: application/problem+json
   # X-Request-ID: <16hex>
   # {"type":"about:blank","title":"...","status":404,"detail":"...","instance":"/api/v1/...","code":"STATE_LOAD_FAILED","trace_id":"<8hex>","request_id":"<16hex>"}
   ```

3. **Trainer crash propagation** (force a TRAINING_FAILED in test mode):
   - Kill trainer pid with non-zero exit OR force RaiseError in test mode.
   - Verify `trainer-exit.json` written.
   - Verify Supervisor reads it, publishes EventBus event with `schema_version=2`.
   - Verify control's training_monitor raises `TrainingFailedError`, attempt transitions to failed.
   - CLI `ryotenkai run logs <id>` shows `error: Training failed` with sanitized traceback (no paths).

4. **Sentinel suite green**:
   ```bash
   .venv/bin/python -m pytest tests/_lint -q
   ```

5. **Unit test mutation gate** (per agent test policy):
   ```bash
   bash scripts/mutation/validate_agent_output.sh
   ```

6. **Importlinter contracts**:
   ```bash
   uv run lint-imports
   ```
   Verifies: shared no upward deps; providers `_translate` isolation; pod ↔ trainer no cross-imports.

7. **Mock policy / no protocol mocking**:
   ```bash
   .venv/bin/python -m pytest tests/_lint/test_no_protocol_mocking.py -q
   ```

## Identified risks & open questions (fixed in plan)

**R-FASTAPI-DEP** — `packages/shared/pyproject.toml` НЕ содержит fastapi. Plan agent ошибочно сказал, что это transitive dep. **Resolution**: добавить `fastapi>=0.115.0,<1.0.0` в shared deps в Phase B. Альтернатива (новый api_common package) отклонена. Sentinel test_no_fastapi_outside_shared_api.py изолирует FastAPI в shared/api/ subdirectory.

**R-MIGRATION-SCOPE** — AppError + Result используется в ~20 pipeline stage files (orchestrator.py, gpu_deployer.py, training_launcher.py, dataset_validator/*, model_retriever/*, etc.). "Удалить полностью" — это значительный refactor, не "additive". **Resolution**: split в Phase A1 (additive errors/ module) + A2 (destructive Result removal). A2 — отдельный PR, high-risk, требует:
- Stage execution loop переписан на try/except вместо unwrap.
- Каждый stage method: `def execute(...) -> T` raises, не `Result[T, AppError]`.
- Полный test coverage по 7-классовому policy + mutation gate gate-pass per file.
- xfail debt: некоторые existing stage tests могут временно сломаться — нужны xfail-debt:phase-a2-stage-XYZ tokens в `docs/migration/xfail_debt.md`.

**R-SIGKILL-FALLBACK** — trainer atexit handler НЕ run при SIGKILL (cgroup OOM killer, exit_code=137). Strict fallback INTERNAL_ERROR теряет реальную причину для частого OOM-сценария. **Resolution**: supervisor проверяет `exit_code in (137, -9)` и синтезирует payload `code=TRAINING_OOM, message="trainer killed by signal SIGKILL — likely OOM"`. Эвристика, но false-positive TRAINING_OOM лучше чем noisy INTERNAL_ERROR.

**R-CONTROL-API-BREAKING** — Phase C меняет wire-shape control API с `{detail, code: lowercase}` на RFC 9457 problem+json. Существующий frontend (`web/src/api/client.ts`) НЕМЕДЛЕННО сломается. **User decision (зафиксировано)**: мерджим сразу, frontend rewrite подхватит. Compat shim не делаем (политика "обратной совместимости не пилим").

**R-LOGGER-REQUEST-ID-INTEGRATION** — `request_id` без интеграции с loguru даёт ID только в response headers, но не в server logs → корреляция сломана. **Resolution**: добавить loguru filter в `shared/utils/logging/setup.py` (или эквивалент), читающий `REQUEST_ID.get()`. Sentinel `test_logger_carries_request_id.py` blocks PR без интеграции.

**R-HTTPException-RESIDUAL** — после Phase F sentinel `test_no_naked_httpexception.py` запрещает `raise HTTPException()` вне `http_exception_handler`. Но FastAPI internally raises HTTPException (e.g., 422 на validation, 405 на method not allowed) — handler ловит. Sentinel должен whitelist только наш `http_exception_handler`, не FastAPI internals.

**R-VENDOR-TRANSLATE-ISOLATION** — vendor exception types (RunPodAPIError, SSHError, HFHubError) ДОЛЖНЫ быть catch'ed только в `_translate.py` модулях. Без importlinter contract — кто-нибудь catches их в pipeline stage и завязывает архитектуру на vendor SDK. **Resolution**: Phase E добавляет importlinter contract + sentinel `test_provider_translate_isolated.py`.

**R-XFAIL-DEBT-TRACKER** — Phase A2 + C + F могут потребовать временные xfail для тестов, которые ассертят старые wire shapes / Result patterns. **Resolution**: каждый xfail с `xfail-debt:errors-unification-<phase>-<id>` token + row в `docs/migration/xfail_debt.md`. Resolve в течение 30 дней после merge соответствующей фазы.

**R-RESULT-IS-NOT-DEAD-IN-COMMUNITY** — `packages/community/.../loader.py` использует `LoadFailure` (свой dataclass, не AppError-rooted). НЕ затронуто Phase A2 — это independent abstraction. **Confirmed via grep** (community LoadFailure ≠ shared AppError).

**R-PHASE-G-STORAGE-MIGRATION** — Phase G добавляет `conditions[]` в pipeline state JSON. Existing state files без поля → Pydantic default `[]` (Phase G design выше). Но если schema bump (например, добавление required field позже) — нужна migration story. **Resolution**: Phase G документирует schema_version для PipelineStageState; future required-field additions идут через migration scripts (вне scope).

## Future work (out of scope of this plan)

- **i18n** через `Accept-Language` header → `detail` translation table; `title` stays English per RFC 9457 §3 stability requirement.
- **OpenTelemetry adoption** — `traceparent` header read/written; `request_id` augmented (не replaced) до full 128-bit OTEL trace ID; `trace_id` (8 hex) остаётся как human-grep anchor.
- **Retry hints на `InfrastructureError`** — `retry_after_seconds: int | None`, `retryable: bool` fields в `ProblemDetails.context["retry"]` для client backoff logic.
- **gRPC-style `details[]`** — `ProblemDetails.details: list[Any]` для stacking нескольких typed sub-errors (Google AIP-193 precedent).
- **Frontend ProblemDetails consumer library** — после frontend rewrite: общий TypeScript client + UI components для рендеринга problem+json (toast, modal, inline form errors).
- **Phase G extension: condition history** — append-only log конкретно для long-running stages (за пределами `last_transition_time`).
