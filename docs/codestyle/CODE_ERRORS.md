# Error handling — unified model (RFC 9457 + typed exceptions)

> **Status**: post Phase A1-G + post-audit fixes + Phase H (2026-05-17).
> Single source of truth. Older versions of this file described a
> `Result[T, AppError]` system that has been **deleted**. CI sentinels
> block its reintroduction.

---

## TL;DR for LLM agents

**Quick reference:**

| ❌ Do NOT write | ✅ Write instead | Blocked by |
|---|---|---|
| `return Err(ProviderError(...))` | `raise ProviderUnavailableError(detail="...", context={...})` | `tests/_lint/test_no_apperror_or_result.py` |
| `from ryotenkai_shared.utils.result import ...` | `from ryotenkai_shared.errors import RyotenkAIError, ...` | same sentinel |
| `raise HTTPException(404, "...")` | `raise StateLoadFailedError(detail="...")` | `tests/_lint/test_no_naked_httpexception.py` |
| `raise APIError(ErrorCode.X, status=404, ...)` | `raise <ConcreteSubclass>(detail="...")` | (APIError class deleted in Phase F) |
| `raise APIException(...)` | `raise RyotenkAIError(...)` or use `from_problem()` factory | (APIException class deleted in Phase F) |
| `raise RuntimeError("...")` in domain code | `raise PipelineStageFailedError(detail="...")` | `tests/_lint/test_exception_root.py` |
| `Err(X)`, `Ok(Y)`, `Failure`, `Success`, `Result[T, E]` | typed `raise`, plain `return` | sentinel + file deletion |
| `result.is_failure() / unwrap_err()` | `try: ... except RyotenkAIError as exc: ...` | sentinel + file deletion |
| `raise ConfigInvalidError(detail=f"user {email} bad")` | `raise ConfigInvalidError(detail="bad config", context={"email_hash": ...})` | `tests/_lint/test_no_pii_in_raise_sites.py` |
| Deleting an `ErrorCode` enum member | Never. Keep + mark deprecated in docstring | `tests/_lint/test_error_code_never_shrinks.py` |
| `from fastapi import ...` in `packages/shared/` outside `api/` | Move to `packages/shared/.../api/` or import handlers | `tests/_lint/test_no_fastapi_outside_shared_api.py` |
| Catching `RunPodAPIError` / `SSHError` / `HFAuthError` in control/pod | Translate at the boundary adapter; let typed `RyotenkAIError` propagate | importlinter contract `vendor SDK exception types` |

**The only base for raised exceptions is `RyotenkAIError`.** Subclasses live in `packages/shared/src/ryotenkai_shared/errors/{domain,infra}.py`. Wire format is RFC 9457 `application/problem+json`. CLI renders kubectl-style.

---

## Contents

- [Architecture in one diagram](#architecture-in-one-diagram)
- [Choosing the right typed exception](#choosing-the-right-typed-exception)
- [Adding a new error type](#adding-a-new-error-type)
- [Raising errors — patterns](#raising-errors--patterns)
- [Catching errors — patterns](#catching-errors--patterns)
- [Vendor SDK boundaries (RunPod, SSH, HF, Docker, MLflow)](#vendor-sdk-boundaries)
- [HTTP API surface (FastAPI handlers)](#http-api-surface)
- [CLI surface (Typer commands)](#cli-surface)
- [Subprocess boundaries (trainer / worker)](#subprocess-boundaries)
- [Error persistence (where errors land)](#error-persistence)
- [Tests — 7-class policy](#tests--7-class-policy)
- [CI sentinels that block PRs](#ci-sentinels-that-block-prs)
- [Secrets loading (RYOTENKAI_SECRETS_FILE)](#secrets-loading)

---

## Architecture in one diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ Internal raise sites (anywhere in packages/*/src/)               │
│   raise StateLoadFailedError(detail=..., context={...})          │
│                          │                                       │
│                          ▼ RyotenkAIError propagates up          │
├──────────────────────────────────────────────────────────────────┤
│ Boundary handlers (one of):                                      │
│   • FastAPI (pod + control)  → RFC 9457 problem+json on stderr   │
│   • CLI (Typer wrap_command) → kubectl-style on stderr           │
│   • Worker subprocess main() → kubectl-style + pipeline.log H1   │
│   • Trainer atexit hook      → trainer-exit.json                 │
├──────────────────────────────────────────────────────────────────┤
│ Persistence (Phase H):                                           │
│   • pipeline.log H1 summary block at the end of every run        │
│   • pipeline_state.json H2 typed AttemptFailure                  │
│   • <runs_base>/init_error.log H3 fallback for pre-pipeline fail │
└──────────────────────────────────────────────────────────────────┘
```

---

## Choosing the right typed exception

All exports: `packages/shared/src/ryotenkai_shared/errors/__init__.py`.

| Domain | Use exception | Status |
|---|---|---|
| **Provider auth** (missing API key, bad credentials) | `ProviderAuthFailedError` | 401 |
| **Provider unavailable** (SDK call failed, no instances, transient) | `ProviderUnavailableError` | 503 |
| **Provider rate-limited** | `ProviderRateLimitedError` | 429 |
| **SSH connection failure** | `SSHConnectionFailedError` | 502 |
| **SSH exec failure** | `SSHExecFailedError` | 502 |
| **SSH transfer failure** | `SSHTransferFailedError` | 502 |
| **Training failed (generic runtime)** | `TrainingFailedError` | 500 |
| **Training OOM** (cgroup/CUDA) | `TrainingOOMError` | 500 |
| **Training wall-clock timeout** (supervisor watchdog) | `TrainingTimeoutError` | 500 |
| **Strategy chain invalid** | `StrategyChainInvalidError` | 422 |
| **Dataset load failed** | `DatasetLoadFailedError` | 422 |
| **Dataset validation failed** | `DatasetValidationFailedError` | 422 |
| **Model load failed** | `ModelLoadFailedError` | 500 |
| **Inference service unavailable** | `InferenceUnavailableError` | 503 |
| **Config invalid** (schema, validation) | `ConfigInvalidError` | 400 |
| **Config drift** | `ConfigDriftError` | 409 |
| **Config file missing** | `ConfigFileNotFoundError` | 404 |
| **Pipeline state load failed** | `StateLoadFailedError` | 404 |
| **Pipeline state locked** | `StateLockedError` | 409 |
| **Launch in progress** | `LaunchInProgressError` | 409 |
| **Launch preparation failed** | `LaunchPreparationError` | 500 |
| **Pipeline stage failed (catch-all)** | `PipelineStageFailedError` | 500 |
| **HF auth failed** | `HFAuthFailedError` | 401 |
| **HF resource not found** | `HFNotFoundError` | 404 |
| **Workspace store IO failed** | `WorkspaceStoreFailedError` | 500 |
| **Workspace entity not found** | `ProjectNotFoundError` / `ProviderNotFoundError` / etc. | 404 |
| **Job spec invalid** (Pydantic validation) | `JobSpecInvalidError` | 422 |
| **Job not found** | `JobNotFoundError` | 404 |
| **Job state invalid** (FSM rejection) | `JobStateInvalidError` | 409 |
| **Tunnel/transport unreachable** (Mac client) | `TransportError` | 599 |
| **Catch-all server bug** | `InternalError` | 500 |

If none fits — **STOP**, do not invent a new class on the fly. See next section.

---

## Adding a new error type

Two coordinated edits:

1. **`packages/shared/src/ryotenkai_shared/contracts/problem_details.py`** — add `ErrorCode` enum member (UPPER_SNAKE_CASE, domain prefix).

2. **`packages/shared/src/ryotenkai_shared/errors/{domain,infra}.py`** — add concrete subclass:
   ```python
   class TrainingTimeoutError(InfrastructureError):
       """Trainer subprocess exceeded wall-clock timeout (watchdog killed it)."""
       code: ClassVar[ErrorCode] = ErrorCode.TRAINING_TIMEOUT
       status: ClassVar[int] = 500
   ```

3. **`packages/shared/src/ryotenkai_shared/errors/_render.py::_DEFAULT_TITLES`** — register human title (stable across occurrences, per RFC 9457 §3).

4. **`tests/_lint/error_code_history.yaml`** — append the new code value. CI sentinel `test_error_code_never_shrinks.py` enforces.

5. **`packages/shared/src/ryotenkai_shared/errors/__init__.py`** — re-export.

6. **Tests** — add to `tests/unit/shared/errors/test_domain.py` or `test_infra.py` parametrized smoke list.

**Choosing flavour**: `DomainError` (4xx — caller's fault, recoverable) vs `InfrastructureError` (5xx — external/transient/bug). Both are abstract markers; the concrete subclass picks one as base.

---

## Raising errors — patterns

```python
# Idiomatic — typed + context + cause chaining
raise StateLoadFailedError(
    detail=f"run_id={run_id!r} has no state file",
    context={"run_id": run_id, "expected_path": str(path)},
    cause=original_exception,  # optional; sets __cause__
)

# At a vendor SDK boundary — translate, never leak vendor type
try:
    response = runpod_sdk.create_pod(...)
except RunPodRateLimitedError as exc:
    raise ProviderRateLimitedError(
        detail="RunPod throttled the request",
        context={"retry_after_seconds": exc.retry_after},
        cause=exc,
    )

# When you have a Pydantic ValidationError
raise JobSpecInvalidError(
    detail="Request body failed validation",
    errors=[FieldError(loc=e["loc"], type=e["type"], msg=e["msg"], input=e.get("input"))
            for e in validation_error.errors()],
)
```

**Never put PII in `detail`** — sentinel blocks `f"... {email} ..."` patterns. Use `context={"email_hash": hash(email)}` if needed.

**Never put `traceback.format_exc()` in `context`** — sentinel `test_no_traceback_in_context.py` blocks. Tracebacks go to server log only; sanitize via `ryotenkai_shared.contracts.trainer_exit.sanitize_traceback` if you must carry one across a process boundary.

---

## Catching errors — patterns

```python
# Default: don't catch. Let RyotenkAIError propagate to the boundary
# handler (FastAPI handler / CLI wrap_command / worker main()).

# When you must catch (cleanup, retry, fallback):
try:
    deploy_gpu(ctx)
except ProviderRateLimitedError as exc:
    time.sleep(exc.context.get("retry_after_seconds", 5))
    raise  # re-raise, don't swallow
except ProviderUnavailableError as exc:
    logger.warning("Provider down; falling back to single_node: %s", exc.detail)
    fallback()

# Never use bare except:
except Exception:  # ❌ catches KeyboardInterrupt + SystemExit + RyotenkAIError
    ...

# Narrow catches OK:
except (RyotenkAIError, ValidationError) as exc:
    ...
```

`error.code` returns the `ErrorCode` enum member; use `error.code.value` for the string. `error.detail` is the human message. `error.context` is the structured dict.

---

## Vendor SDK boundaries

Files in `packages/shared/src/ryotenkai_shared/infrastructure/*/protocol.py` define vendor exception types (`RunPodAPIError`, `SSHError`, `HFHubError`). These are caught **only** inside their adapter modules and translated to typed `RyotenkAIError` subclasses. Importing them anywhere else trips an importlinter contract.

**Pattern**:

```python
# packages/providers/src/ryotenkai_providers/runpod/sdk_adapter.py — OK to catch
def create_pod(spec):
    try:
        return _runpod_sdk.create_pod(spec)
    except RunPodRateLimitedError as e:
        raise ProviderRateLimitedError(detail=str(e), cause=e)
    except RunPodAPIError as e:
        raise ProviderUnavailableError(detail=str(e), cause=e)

# packages/control/.../some_stage.py — must NOT see vendor types
out = adapter.create_pod(spec)  # raises ProviderRateLimitedError, never RunPodAPIError
```

Allow-list for legitimate translation modules: `pyproject.toml [[tool.importlinter.contracts]]` → "Vendor SDK exception types stay inside infrastructure adapters".

---

## HTTP API surface

Both pod runner FastAPI and control FastAPI register **the same** shared handlers from `packages/shared/src/ryotenkai_shared/api/error_handlers.py`:

```python
from ryotenkai_shared.api.error_handlers import EXCEPTION_HANDLERS
from ryotenkai_shared.api.request_id import RequestIDMiddleware

app = FastAPI(exception_handlers=EXCEPTION_HANDLERS, ...)
app.add_middleware(RequestIDMiddleware)
```

Handler dispatch:
- `RyotenkAIError` → `as_problem()` → RFC 9457 `application/problem+json`
- `RequestValidationError` (Pydantic) → 422 `JobSpecInvalidError` shape
- Catch-all `Exception` → 500 `INTERNAL_ERROR`, traceback only to server log

Never `raise HTTPException(...)` in routes — sentinel `test_no_naked_httpexception.py` blocks. The legacy adapter still translates HTTPException to problem+json for FastAPI-internal raises (405, etc.), but routes themselves use typed errors.

`RequestIDMiddleware` reads `X-Request-ID` header (or generates 16-hex), exposes via `contextvars.ContextVar` `REQUEST_ID`, returns the same value in response headers, and propagates into every log record via the `RequestIDLogFilter` on root logger. Body's `request_id` field is stamped from the contextvar.

---

## CLI surface

Every Typer command in `packages/control/src/ryotenkai_control/cli/commands/*.py` is wrapped:

```python
from ryotenkai_control.cli.errors import wrap_command

@app.command()
@wrap_command
def start(config: Path = ...):
    # raise anything — wrap_command catches and renders
    ...
```

`wrap_command` catches `RyotenkAIError`, `ValidationError`, `FileNotFoundError`, `yaml.YAMLError`, `KeyboardInterrupt`. Renders kubectl-style:

```
error: Provider authentication failed
  hint: RUNPOD_API_KEY is required when using provider 'runpod'. ...
  code: PROVIDER_AUTH_FAILED  trace=a3b1c2d4  request=8e7f6c5b4a3d2e1f
```

Exit codes:
- 4xx → exit 2 (user error, fixable)
- 5xx + `InternalError` + `TransportError` → exit 1
- `KeyboardInterrupt` → exit 130 (POSIX SIGINT)
- `--verbose` adds a 4th `context:` line with JSON dump

`wrap_command` mints a fresh `request_id` per command invocation if one isn't already in the contextvar — every CLI run has correlation.

---

## Subprocess boundaries

### Worker (`pipeline/worker.py`)

Spawned by `ryotenkai run start`. Its `main()` has the same top-level handler shape as `wrap_command` but uses `print_ryotenkai_error()` (non-raising twin of `die_from_ryotenkai`):

```python
def main(argv):
    ...
    try:
        return _run_pipeline(args)
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr); return 130
    except RyotenkAIError as exc:
        return print_ryotenkai_error(exc, request_id=request_id)
    except Exception as exc:
        logger.error("Unhandled", exc_info=exc)
        return print_ryotenkai_error(InternalError(detail="..."), ...)
```

### Trainer (`pod/trainer/`)

Trainer is a child process of the pod runner. On unhandled exception or non-zero exit, it writes `<artifact_root>/trainer-exit.json` (Pydantic model `TrainerExitPayload` in `shared/contracts/trainer_exit.py`):

```json
{
  "code": "TRAINING_OOM",
  "message": "GPU 0 ran out of memory at step 1500",
  "traceback_summary": "...30 lines, paths sanitized...",
  "exit_code": 137,
  "wall_seconds": 1234.5,
  "schema_version": 1
}
```

Supervisor reads on reap, publishes typed `trainer_exited` EventBus event. If the file is missing AND `exit_code in (137, -9)` (SIGKILL → cgroup OOM heuristic), supervisor synthesizes `TRAINING_OOM`. If exit_code != 0 and missing file with no heuristic match → `INTERNAL_ERROR` synthesized.

**Watchdog**: if trainer doesn't exit within `RYOTENKAI_TRAINER_TIMEOUT` (default 6h), supervisor SIGTERM → 30s grace → SIGKILL → synthesizes `TrainerExitPayload(code=TRAINING_TIMEOUT, payload_source="watchdog_timeout")`.

---

## Error persistence

Three places an error lands. Phase H, 2026-05-17.

### 1. `pipeline.log` final summary block (Н1)

Worker writes the **last** entry of every run (success or failure):

```
================================================================================
Pipeline FAILED at attempt 1
================================================================================
  error:    Provider unavailable
  code:     PROVIDER_UNAVAILABLE
  stage:    GPU Deployer (Stage 2/6)
  request:  df3a29d5887c25aa
  detail:   runpod SDK call failed in create_pod: ...
  context:  {...}
  attempts: 1
  outcome:  FAILED
================================================================================
```

`tail pipeline.log` → outcome immediately. Helper: `worker._log_pipeline_outcome(...)`.

### 2. `pipeline_state.json::PipelineAttemptState.failure` (Н2)

Typed `AttemptFailure` dataclass in `pipeline/state/models.py`. Schema bumped 1→2; legacy `error: str` field auto-migrates to `AttemptFailure(code="LEGACY_ERROR", ...)` on load. Web UI / API / automation reads `state.attempts[-1].failure.code` — no separate `outcome.json` file (state IS the source of truth).

Writer: `AttemptController.record_failure(failure: AttemptFailure)`.

### 3. `<runs_base>/init_error.log` (Н3)

**Only** when failure happened BEFORE `pipeline.log` was attached (startup-time: missing secrets, bad config path, bootstrap errors, etc.). Same kubectl-style format. Detection: `was_pipeline_log_ever_opened()` accessor in `shared/utils/logger.py` reads a sticky module-level flag.

If you change the `_attach_pipeline_file_handler` flow, **preserve** the `_pipeline_file_was_ever_attached = True` assignment — H3 detection depends on it.

---

## Tests — 7-class policy

Every new error-related method gets 7 test classes per `.claude/CLAUDE.md`:

1. **TestPositive** — happy path
2. **TestNegative** — error paths
3. **TestBoundary** — empty/whitespace/zero/max
4. **TestInvariants** — pin code/status/title defaults
5. **TestDependencyErrors** — external dep fails (mock-via-fake)
6. **TestRegressions** — bug guards with explicit reason
7. **TestLogicSpecific** — parametrized truth tables (status→exit-code, etc.)

Use **canonical fakes** from `tests/_fakes/` — `MagicMock(spec=Protocol)` is blocked by `test_no_protocol_mocking.py`.

Reference example: `tests/unit/shared/errors/test_base.py` — 80+ tests, 100% coverage, 97% mutation kill rate.

---

## CI sentinels that block PRs

All under `tests/_lint/`:

| Sentinel | Blocks |
|---|---|
| `test_no_apperror_or_result.py` | `from ryotenkai_shared.utils.result import ...` anywhere |
| `test_no_naked_httpexception.py` | `raise HTTPException(...)` outside legacy adapter |
| `test_exception_root.py` | `class *Error(Exception)` must inherit `RyotenkAIError` |
| `test_error_code_pinned.py` | concrete `RyotenkAIError` subclass must pin `code` + `status` |
| `test_error_code_never_shrinks.py` | enum members must never be removed (history snapshot) |
| `test_no_traceback_in_context.py` | `traceback.format_exc()` must not land in `context=` |
| `test_no_pii_in_raise_sites.py` | f-strings interpolating PII variables in `detail=` |
| `test_no_fastapi_outside_shared_api.py` | `from fastapi import` in `packages/shared/` outside `api/` |
| `test_logger_carries_request_id.py` | logger filter must inject `request_id` from contextvar |
| `test_condition_reason_format.py` | `update_condition(..., reason=...)` literal must be CamelCase |
| `test_no_protocol_mocking.py` | `MagicMock(spec=Protocol)` — use canonical fakes |
| importlinter contract (vendor SDK isolation) | catching `RunPod*`/`SSH*`/`HF*` exception types outside `infrastructure/` |

Run locally: `.venv/bin/python -m pytest tests/_lint -q`.

---

## Secrets loading

The loader (`packages/shared/src/ryotenkai_shared/config/secrets/loader.py`) finds `secrets.env` through three paths, in priority order:

1. `env_file=` keyword arg (programmatic).
2. `RYOTENKAI_SECRETS_FILE` env var — set once in `~/.zshrc` for non-standard layouts (containers, tarballs, mounted volumes).
3. Filesystem walk-up — collects **every** uv-workspace root (directory containing `pyproject.toml` + `packages/`) from innermost to outermost; tries `<root>/secrets.env` and `<root>/config/secrets.env` at each level.

Loader has **zero knowledge of git, jj, hg, `.git` files, or any tool-specific markers**. Pure filesystem traversal. Do not reintroduce VCS coupling — was rolled back as architectural mistake in commit `aa27e80`.

Operators with non-workspace layouts (Docker without uv setup, downloaded tarball) → set `RYOTENKAI_SECRETS_FILE` explicitly.

---

## Pipeline conditions (Phase G — observability)

`PipelineStageState.conditions: list[Condition]` provides k8s/OpenShift-style live status surface in addition to FSM. Helper `update_condition()` is idempotent — `last_transition_time` only changes on actual status flip.

Standard types: `Available`, `Progressing`, `Degraded`, `OOMRisk`, `RateLimited`.

`Condition.reason` is **CamelCase free-form string** (not `ErrorCode` enum) — k8s convention uses positive reasons like `"AsExpected"` that don't fit a failure-only enum. Sentinel `test_condition_reason_format.py` enforces format at literal call sites.

Emitters live in `pipeline/state/attempt_controller.py::record_condition` — extend there, not from random code.

---

## When this doc disagrees with code

The code wins. CI sentinels listed above are the authoritative contract; this doc is best-effort prose around them. If you find a discrepancy:

1. Read the sentinel test source — it's the actual rule.
2. Check the latest commit log under `git log --grep "errors" --oneline -20` for recent refactors.
3. Update this doc in the same PR as the rule change.

Reference plan with full history: `docs/plans/sharded-stargazing-wigderson.md`.
