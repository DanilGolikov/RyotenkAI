# Error handling in this project

Guidelines for working with errors for consistency and extensibility.

---

## Contents

- [Instructions for LLMs](#instructions-for-llms)
- [Principles](#principles)
- [Error hierarchy](#error-hierarchy)
- [Result pattern](#result-pattern)
- [Formatting rules](#formatting-rules)
- [Antipatterns](#antipatterns)
- [Exceptions and except](#exceptions-and-except)
- [Tests](#tests)

---

## Instructions for LLMs

> This document is for developers and LLM agents. When generating or refactoring code in this project — read and apply the rules below.

**Quick reference:** `Err("x")` → `Err(XxxError(message="x", code="XXX_CODE"))` | `Result[T, str]` → `Result[T, AppError]` | In tests: `str(res.unwrap_err())` to check message text.

**When editing code in this project, follow these rules.**

### Mandatory rules (MUST)

1. **Never return `Err("string")`.** Always use `Err(SomeError(message="...", code="CODE"))`.
2. **Never declare `Result[T, str]`.** The error type must be only `AppError` or a subclass.
3. **When asserting errors in tests** — `unwrap_err()` returns an object. Use `str(res.unwrap_err())` for substring checks or `res.unwrap_err().code` for the code.
4. **When mocking a function that returns an error** — the mock must return `Err(ProviderError(...))` or `Err(AppError(...))`, not `Err("string")`.

### Choosing an error type

| Context | Type |
|---------|------|
| SSH, Docker, RunPod, pod lifecycle | `ProviderError` |
| vLLM, deploy, health check, endpoints | `InferenceError` |
| Strategies (SFT, DPO, ORPO), datasets, checkpoints | `TrainingError`, `StrategyError`, `DataLoaderError` |
| Configuration, secrets | `ConfigError` |

### Error return template

```python
return Err(ProviderError(message="problem description", code="MODULE_ERROR_MEANING"))
```

Code: UPPER_SNAKE_CASE, domain prefix (`SSH_`, `POD_`, `INFERENCE_`, `SFT_`, etc.).

### Pre-commit checklist

- [ ] No `Err("...")` or `Err(f"...")`
- [ ] No `Result[..., str]` in signatures
- [ ] In tests: `str(res.unwrap_err())` when checking text, not `res.unwrap_err()` directly
- [ ] Mocks return `Err(AppError(...))`, not `Err("...")`

### Common mistakes when generating code

- Generating `return Err("message")` — replace with `Err(XxxError(message="message", code="XXX_CODE"))`.
- Generating `assert "x" in res.unwrap_err()` — replace with `assert "x" in str(res.unwrap_err())`.
- Generating `monkeypatch.setattr(..., lambda: Err("fail"))` — replace with `Err(AppError(message="fail", code="TEST_ERROR"))`.

---

## Principles

1. **Explicitness** — errors are returned via `Result[T, E]`, not exceptions (for recoverable errors).
2. **Typing** — always `Result[T, AppError]` or a subclass (`ProviderError`, `ConfigError`, etc.), **never** `Result[T, str]`.
3. **Codes** — every error has a `code` for logging, monitoring, and branching.
4. **Context** — `message` plus optional `details` for diagnostics.

---

## Error hierarchy

All errors inherit from `AppError` (`src/utils/result.py`):

```
AppError (base)
├── ConfigError          # configuration, validation
├── ProviderError        # SSH, Docker, RunPod, pod lifecycle
├── InferenceError       # vLLM, health checks, endpoints
└── TrainingError
    ├── DatasetError
    ├── ModelError
    ├── StrategyError
    ├── DataLoaderError
    └── OOMError
```

**Choosing a type:**
- Providers (SSH, RunPod, single_node) → `ProviderError`
- Inference (deploy, health, activate_for_eval) → `InferenceError`
- Training (strategies, datasets, checkpoints) → `TrainingError` and subclasses
- Config, secrets → `ConfigError`

---

## Result pattern

### Returning an error

```python
from src.utils.result import Err, Ok, ProviderError, Result

def connect(host: str) -> Result[SSHInfo, ProviderError]:
    if not host:
        return Err(ProviderError(
            message="Host is required",
            code="SSH_HOST_MISSING",
        ))
    # ...
    return Ok(ssh_info)
```

### Handling the result

```python
res = connect(host)
if res.is_failure():
    err = res.unwrap_err()
    logger.error("Connect failed: %s", err)
    return Err(err)  # re-raise
value = res.unwrap()
```

### Re-raise with wrapping

```python
inner = some_operation()
if inner.is_failure():
    return Err(InferenceError(
        message=str(inner.unwrap_err()),
        code="RUNPOD_EVAL_ACTIVATE_FAILED",
    ))
```

### Assertions in tests

```python
assert res.is_failure()
err = res.unwrap_err()
assert err.code == "SSH_HOST_MISSING"
assert "Host is required" in str(err)
# or
assert "Host is required" in str(res.unwrap_err())
```

---

## Formatting rules

### 1. Signatures

```python
# Correct
def load_dataset(path: Path) -> Result[Dataset, DataLoaderError]:
def deploy() -> Result[EndpointInfo, AppError]:  # if mixed types

# Wrong
def load_dataset(path: Path) -> Result[Dataset, str]:
def deploy() -> Result[EndpointInfo, str]:
```

### 2. Creating errors

```python
# Correct — always with code
return Err(ProviderError(message="Connection refused", code="SSH_CONNECT_FAILED"))
return Err(ConfigError(message="Invalid path", code="CONFIG_PATH_INVALID", details={"path": str(p)}))

# Wrong
return Err("Connection refused")
return Err(f"Failed: {e}")
```

### 3. Error codes

- Uppercase snake_case: `POD_SSH_READY_TIMEOUT`, `DATA_LOADER_LOAD_FAILED`
- Domain prefix: `POD_*`, `SSH_*`, `INFERENCE_*`, `SFT_*`, `ORPO_*`
- Unique within the module

### 4. Result variance

`Result[X, SubError]` is not a subtype of `Result[X, AppError]`. Public methods should return `Result[..., AppError]` or wrap:

```python
# Internal method returns ProviderError
def _sync_files() -> Result[None, ProviderError]: ...

# Public — AppError
def deploy_files(self) -> Result[None, AppError]:
    res = self._sync_files()
    if res.is_failure():
        return Err(res.unwrap_err())  # ProviderError is compatible with AppError
    return Ok(None)
```

---

## Antipatterns

| Do not | Instead |
|--------|---------|
| `Err("string")` | `Err(ProviderError(message="...", code="CODE"))` |
| `Result[T, str]` | `Result[T, ProviderError]` or another `AppError` |
| `except Exception: pass` | Catch specific types or log |
| `str(err)` in tests without `str()` | `str(res.unwrap_err())` — `unwrap_err()` returns an object |
| Duplicated string codes | Extract to a constant when used more than 3 times (WPS226) |

---

## Exceptions and except

- **Recoverable** (network, validation, config) → `Result`, not raise.
- **Unrecoverable** (assert, critical bugs) → may raise; top level will catch.
- **except Exception** — only where a defensive catch is needed (cleanup, best-effort logging). Narrow to specific types where possible:

```python
# Narrow catch
except (OSError, TimeoutError, urllib.error.URLError) as e:
    return Err(ProviderError(message=str(e), code="NETWORK_ERROR"))

# Defensive (cleanup, non-critical)
except Exception as e:
    logger.debug("Non-fatal: %s", e)
```

---

## Tests

### Mocks

```python
# Mock returns AppError
monkeypatch.setattr("module.func", lambda: Err(ProviderError(message="fail", code="TEST_ERROR")))

# Outdated mock
monkeypatch.setattr("module.func", lambda: Err("fail"))
```

### Assertions

```python
# Error is an object, not a string
assert res.is_failure()
assert "expected text" in str(res.unwrap_err())
assert res.unwrap_err().code == "EXPECTED_CODE"
```

---

## Extending

When adding a new domain:

1. Add an `AppError` subclass in `src/utils/result.py` if a separate type is needed.
2. Export it in `__all__`.
3. Use it everywhere in the new module instead of `str` or bare `AppError`.
4. Update this document when the hierarchy changes.
