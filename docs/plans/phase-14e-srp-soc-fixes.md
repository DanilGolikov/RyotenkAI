# Phase 14.E — Core SRP/SoC Fixes

> Status: **DRAFT — pending user approval**
> Author: daniil + agent
> Date: 2026-04-28
> Worktree: `nice-jepsen-07d789`
> Predecessors: Phase 14.A (commit `1819cc4`), 14.B (uncommitted but green), 14.C (planned), 14.D+F (planned)
> Successor: Phase 14 complete after this lands
> Migration policy: NO BACKWARDS COMPATIBILITY — single-PR full cutover
> Effort: ~2 engineer-days (audit reduced from 3 — KISS won't-fix on 2 of 8 violations)
> Risk: Low — focused architectural cleanup; tests cover all touched paths
> Out of: Phase 14 roadmap in [`phase-14a-provider-abstraction.md`](./phase-14a-provider-abstraction.md)

## 14.E Context — what we're fixing AND what we're NOT

Phase 14.A's audit listed 8 SRP/SoC violations. After detailed audit, the scope has narrowed:

**Must fix (3 violations, high-value architectural cleanup):**
* **V1** — `src/runner/main.py:148-189` — circular-binding closure between `EventJournal.on_rotate` and `bus.publish` via dict-as-mutable-cell hack. Temporal inversion.
* **V3** — `src/runner/api/events.py:149-150` + `src/runner/api/jobs.py:231-233` — heartbeat `mark_active()` called inline in API handlers. Layer violation.
* **V5** — `src/training/mlflow/resilient_transport.py:97-123` — module imports `requests` and `urllib3` at module-load time to enumerate retryable exception types. Library coupling.

**Nice to have (3 violations, low-priority hygiene; included if time allows):**
* **V4** — `src/training/mlflow/metrics_buffer.py::MetricsDecimator.__init__` — conflates timing anchor with policy capture.
* **V6** — `src/runner/event_journal.py::EventJournal.__init__` — validation + initialization mixed.
* **V8** — `src/runner/api/control.py:105-106` — endpoint imports `MacHeartbeat` class to read a constant.

**Won't fix (KISS — audit-rejected):**
* **V2** — `src/runner/pod_terminator.py::decide_and_act` "mixing concerns". Method is 80 LOC, well-commented, decision logic already extracted (`decide_terminal_outcome`), heartbeat already extracted (`_check_heartbeat_with_retries`). Splitting into `HeartbeatGate + LifecycleDispatcher + telemetry decorator` adds 3 classes for marginal clarity gain. Test surface is solid (50+ tests via Phase 14.B). **Decision: keep as-is.**
* **V7** — `src/runner/main.py::_periodic_journal_health_check` "hardcoded thresholds". Function ALREADY accepts both as parameters with module-level defaults — idiomatic Python. No change needed unless a runner-config layer lands elsewhere. **Decision: keep as-is.**

This narrowing is deliberate: Phase 14.A audit identified concerns; Phase 14.E audit (this plan) verified each against KISS. Two violations didn't survive the verification — kept.

## 14.E Architectural decisions (locked, audit-validated)

### 14.E.1 V1: Circular-binding closure — **deferred binding via `EventBus.attach_journal()`**

**Current state** (`main.py:148-189`):

```python
rotate_publisher = {"bus": None}     # mutable cell

def _on_rotate(*, from_seq, to_seq, file_size_bytes, oldest_remaining_seq):
    target = rotate_publisher.get("bus")
    if target is None:
        return    # silent no-op if bus not yet bound
    target.publish(EVENTS_ROTATED, {...})

journal = EventJournal(root_dir=..., on_rotate=_on_rotate)
bus = EventBus(journal=journal)
rotate_publisher["bus"] = bus   # late-bind

```

**Problem**: temporal inversion. The `on_rotate` callback holds a closure over a future `bus.publish` reference; if anything fires between `EventJournal()` construction and the late-bind, the rotation event is silently dropped.

**Fix**: invert control. Make the bus register itself as the journal's rotation observer AFTER both are constructed:

```python
journal = EventJournal(root_dir=workspace / EVENTS_DIR_REL)
bus = EventBus(journal=journal)
bus.attach_journal_rotation_listener()   # NEW method on EventBus

# Internally EventBus.attach_journal_rotation_listener does:
#   self._journal.set_rotation_callback(self._publish_rotation_event)
```

**`EventJournal` changes**:
* `on_rotate` parameter on `__init__` becomes optional (default `None`).
* New method `set_rotation_callback(cb)` that wires the observer post-construction.
* Existing constructor keeps backward-compat for unit tests that pass `on_rotate=...` directly.

**Why this approach**:
* No new dictionary cells.
* Type-safe — `set_rotation_callback` has a typed signature, no `Optional` runtime cell.
* Test isolation preserved — tests that construct `EventJournal` standalone don't change.
* `EventBus` becomes the canonical owner of rotation telemetry, matching its existing role as the bus that emits `EVENTS_ROTATED`.

**Alternatives considered + rejected**:
* (B) Build bus first then journal — wrong order (bus needs the journal at construction).
* (C) Collapse Bus + Journal into one class — couples persistence and pub-sub; violates current SRP.

### 14.E.2 V3: WS heartbeat leak — **`@track_activity` decorator on send paths**

**Current state**:

```python
# src/runner/api/events.py:149-150 — inside WebSocket yield loop
await websocket.send_json(event.to_dict())
if heartbeat is not None:
    heartbeat.mark_active()

# src/runner/api/jobs.py:231-233 — inside REST GET handler
heartbeat = getattr(request.app.state, "heartbeat", None)
if heartbeat is not None:
    heartbeat.mark_active()
```

**Problem**: API handlers know about the heartbeat — cross-cutting concern leaks into endpoint logic. Phase 11.E intent was middleware, audit confirmed.

**Fix — single helper function, two call patterns**:

```python
# src/runner/api/_activity.py — NEW module

def mark_heartbeat_if_present(app_state: Any) -> None:
    """Pull heartbeat from app.state and call mark_active() if it exists.

    Phase 14.E: centralizes the heartbeat-marking pattern that was
    previously inlined in API handlers. Treats heartbeat as truly
    optional — no app.state attribute, no heartbeat, no exception.
    """
    heartbeat = getattr(app_state, "heartbeat", None)
    if heartbeat is not None:
        heartbeat.mark_active()
```

**For REST**:
```python
# src/runner/api/jobs.py — replace inline check
mark_heartbeat_if_present(request.app.state)
```

**For WS**:
```python
# src/runner/api/events.py — wrap the send loop
async def _send_with_activity(ws: WebSocket, payload: dict, app_state: Any) -> None:
    await ws.send_json(payload)
    mark_heartbeat_if_present(app_state)

# In the WS handler:
await _send_with_activity(websocket, event.to_dict(), websocket.app.state)
```

**Why this approach**:
* No FastAPI middleware (FastAPI's `@app.middleware("http")` doesn't see WebSocket frames; would only fix REST).
* No subclassing — minimal new abstractions.
* Test seam preserved: `mark_heartbeat_if_present` is pure-function with `app_state` injected → easy to unit test.
* Co-locates the "is heartbeat tracking" decision in one module; future heartbeat changes update one call site, not N.

**Alternatives considered + rejected**:
* (A) FastAPI HTTP middleware — doesn't cover WS path; need a second mechanism anyway.
* (B) `@activity_tracked` async decorator on endpoints — Pythonic but FastAPI's dependency injection is the standard pattern; decorator stacking is non-standard.
* (C) `WSActivityTracker` class wrapping WebSocket — over-engineering for one mark_active call.

### 14.E.3 V5: ResilientMLflowTransport library coupling — **`ExceptionClassifier` protocol**

**Current state** (`resilient_transport.py:97-123`):

```python
def _optional_exception_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [...]
    try:
        from requests import exceptions as requests_exceptions
        types.extend([
            requests_exceptions.RequestException,
            requests_exceptions.ConnectionError,
            requests_exceptions.Timeout,
        ])
    except Exception:
        pass
    try:
        import urllib3.exceptions as urllib3_exceptions
        types.extend([...])
    except Exception:
        pass
    return tuple(types)

_TRANSPORT_EXCEPTION_TYPES = _optional_exception_types()
```

The retry loop catches `_TRANSPORT_EXCEPTION_TYPES`, which depends on whether `requests` and `urllib3` are importable.

**Problem**: module-level imports of optional libraries; module knows about library exception hierarchies. If MLflow ever swaps to httpx, the transport module needs editing.

**Fix — inject classifier**:

```python
# src/training/mlflow/resilient_transport.py — refactor

class ExceptionClassifier(Protocol):
    """Decides whether an exception is a transient transport failure
    that warrants retry. Phase 14.E: replaces module-level imports
    of requests/urllib3 with caller-injected classification."""

    def is_retryable(self, exc: BaseException) -> bool: ...


class _DefaultClassifier:
    """Conservative default — retries on built-in transport-shaped
    exceptions only. Caller wraps with library-specific classifier
    at bootstrap time."""

    _BUILTIN_TRANSPORT_TYPES = (
        ConnectionError, TimeoutError, OSError, IOError,
    )

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, self._BUILTIN_TRANSPORT_TYPES)


class ResilientMLflowTransport:
    def __init__(
        self, ...,
        classifier: ExceptionClassifier | None = None,
    ) -> None:
        self._classifier = classifier or _DefaultClassifier()
```

**Bootstrap layer adds the requests/urllib3 knowledge**:

```python
# src/training/mlflow/_classifier_bootstrap.py — NEW

def make_default_classifier_for_mlflow() -> ExceptionClassifier:
    """Build a classifier that knows about MLflow's transport stack.

    Optional imports — falls back to built-in types if libraries
    absent. The caller (e.g. trainer init) decides when to upgrade
    from the default classifier.
    """
    extra_types: list[type[BaseException]] = []
    try:
        from requests import exceptions as requests_exceptions
        extra_types.extend([
            requests_exceptions.RequestException,
            requests_exceptions.ConnectionError,
            requests_exceptions.Timeout,
        ])
    except Exception:
        pass
    try:
        import urllib3.exceptions as urllib3_exceptions
        extra_types.extend([
            urllib3_exceptions.ProtocolError,
            urllib3_exceptions.MaxRetryError,
        ])
    except Exception:
        pass
    return _ExtendedClassifier(extra_types)
```

**Wiring**: `ResilientMLflowTransport` constructed during trainer init passes `classifier=make_default_classifier_for_mlflow()`. The transport module itself imports nothing optional.

**Why this approach**:
* Transport module is testable without library imports.
* Bootstrap layer is one extra file (~30 LOC) with explicit "this is where library knowledge lives".
* Existing tests inject mock classifier → easy.
* Future library swap = update one bootstrap file.

**Alternative considered + rejected**: keep the `_optional_exception_types` pattern as-is. Audit notes: "low risk of breakage" — true, but the goal of Phase 14 is provider/library decoupling. Keeping it preserves the leak; fixing now is cheaper than fixing later.

### 14.E.4 V4 (NICE-TO-HAVE): MetricsDecimator policy split

Audit found `MetricsDecimator.__init__` reads timing anchor + extracts policy from config. Fix:

```python
# src/training/mlflow/metrics_buffer.py — refactor

@dataclass(frozen=True)
class DecimationPolicy:
    """Pure config — decimation rules without runtime state."""
    keep_all: bool
    window_first_seconds: float
    window_first_keep_every: int
    window_mid_boundary_seconds: float
    window_mid_keep_every: int
    window_late_keep_every: int

    @classmethod
    def from_config(cls, config: Any) -> "DecimationPolicy": ...


class MetricsDecimator:
    """Runtime state — applies a DecimationPolicy.
    Anchor (start_time) is constructor-time runtime concern.
    """
    def __init__(
        self, *,
        policy: DecimationPolicy,
        training_start_time: float | None = None,
    ) -> None: ...
```

**Test impact**: ~15 tests via `test_metrics_buffer.py` and `test_metrics_buffer_config.py`. Migration: tests construct `DecimationPolicy(...)` inline OR call `DecimationPolicy.from_config(mock_cfg)`.

**Decision**: include if scope allows. Audit priority: Low. Skip if total effort exceeds 2 days.

### 14.E.5 V6 (NICE-TO-HAVE): EventJournal init validation extract

Audit found `EventJournal.__init__` validates parameters AND initializes filesystem state. Fix: extract `validate_journal_config(...)` pure function called at the top of `__init__`.

```python
def validate_journal_config(
    *,
    file_size_cap: int,
    max_files: int,
    fsync_batch: int,
    fsync_interval_ms: int,
) -> None:
    """Phase 14.E: pure validation. No filesystem touch. Use at
    bootstrap time for fail-fast on bad config."""
    if file_size_cap <= 0:
        raise ValueError("file_size_cap must be positive")
    # ... etc
```

`__init__` calls `validate_journal_config(**locals_subset)` first, then proceeds with state init.

**Decision**: include — small (~30 LOC), zero risk.

### 14.E.6 V8 (NICE-TO-HAVE): Control endpoint constant relocation

Audit found `src/runner/api/control.py:105-106` imports `MacHeartbeat` class to read `EXPLICIT_HEARTBEAT_TTL_SECONDS`.

**Fix**: leave the constant on `MacHeartbeat` (its semantic home) but expose it via a module-level alias re-export so the API endpoint imports `from src.runner.heartbeat import EXPLICIT_HEARTBEAT_TTL_SECONDS` (already module-importable) instead of re-importing the class.

```python
# src/runner/heartbeat.py — at module top, add
EXPLICIT_HEARTBEAT_TTL_SECONDS = MacHeartbeat.EXPLICIT_HEARTBEAT_TTL_SECONDS
```

Then control.py:
```python
from src.runner.heartbeat import EXPLICIT_HEARTBEAT_TTL_SECONDS
explicit_default = EXPLICIT_HEARTBEAT_TTL_SECONDS
```

**Decision**: include — 5-minute fix.

## 14.E Scope

### IN-scope (must fix)

| Item | Description |
|---|---|
| `src/runner/main.py` lifespan | Replace circular-binding closure with `EventBus.attach_journal_rotation_listener()` |
| `src/runner/event_bus.py` | New method `attach_journal_rotation_listener()` |
| `src/runner/event_journal.py` | New method `set_rotation_callback(cb)`; `on_rotate` constructor param becomes optional |
| `src/runner/api/_activity.py` (NEW) | `mark_heartbeat_if_present(app_state)` helper |
| `src/runner/api/events.py` | Use `_activity.mark_heartbeat_if_present` instead of inline `if heartbeat is not None` |
| `src/runner/api/jobs.py` | Same |
| `src/training/mlflow/resilient_transport.py` | `ExceptionClassifier` protocol + `_DefaultClassifier`; constructor accepts injected classifier |
| `src/training/mlflow/_classifier_bootstrap.py` (NEW) | `make_default_classifier_for_mlflow()` builds extended classifier with optional `requests`/`urllib3` imports |
| Trainer init code (find via grep) | Pass `classifier=make_default_classifier_for_mlflow()` when constructing `ResilientMLflowTransport` |

### IN-scope (nice-to-have, included)

| Item | Description |
|---|---|
| `src/training/mlflow/metrics_buffer.py` | Extract `DecimationPolicy` dataclass; `MetricsDecimator.__init__` takes policy + start_time |
| `src/runner/event_journal.py` | Extract `validate_journal_config(...)` pure function called from `__init__` |
| `src/runner/heartbeat.py` | Re-export `EXPLICIT_HEARTBEAT_TTL_SECONDS` at module level |
| `src/runner/api/control.py` | Use module-level import |

### OUT-of-scope (won't fix per KISS)

| Item | Reason |
|---|---|
| V2: `decide_and_act` split | 80 LOC, decision/heartbeat already extracted; splitting adds 3 classes for no clarity gain |
| V7: `_periodic_journal_health_check` thresholds | Function already takes parameters with sensible defaults — idiomatic Python |
| Async-ifying anything | YAGNI |
| Adding a runner-config layer (config dataclasses for thresholds) | YAGNI |
| Centralized telemetry decorator (gather all bus.publish into one place) | Too speculative without a concrete user |

## 14.E New abstractions (signatures only)

```python
# src/runner/event_bus.py — adds:

class EventBus:
    # ... existing ...

    def attach_journal_rotation_listener(self) -> None:
        """Phase 14.E: register self as the journal's rotation observer.

        Called by the lifespan AFTER both bus and journal exist.
        Replaces the circular-binding-closure hack in main.py.
        """
        if self._journal is None:
            return    # journal init failed; nothing to attach
        self._journal.set_rotation_callback(self._publish_rotation_event)

    def _publish_rotation_event(
        self, *, from_seq: int, to_seq: int,
        file_size_bytes: int, oldest_remaining_seq: int | None,
    ) -> None:
        """Internal — called by journal on rotation."""
        self.publish(EVENTS_ROTATED, {
            "from_seq": from_seq, "to_seq": to_seq,
            "file_size_bytes": file_size_bytes,
            "oldest_remaining_seq": oldest_remaining_seq,
        })
```

```python
# src/runner/event_journal.py — adds:

class EventJournal:
    def __init__(
        self, *,
        root_dir: Path,
        on_rotate: RotationCallback | None = None,    # now Optional
        # ... other params unchanged ...
    ) -> None:
        validate_journal_config(...)    # extracted (V6)
        # ... state init ...
        self._on_rotate = on_rotate

    def set_rotation_callback(self, cb: RotationCallback) -> None:
        """Phase 14.E: post-construction binding. Replaces circular
        closure in lifespan."""
        self._on_rotate = cb


def validate_journal_config(
    *, file_size_cap: int, max_files: int,
    fsync_batch: int, fsync_interval_ms: int,
) -> None:
    """Phase 14.E (V6): pure parameter validation."""
    if file_size_cap <= 0:
        raise ValueError(...)
    # ... etc
```

```python
# src/runner/api/_activity.py — NEW

def mark_heartbeat_if_present(app_state: Any) -> None:
    """Phase 14.E (V3): pull heartbeat from app.state and call
    mark_active() if it exists. Centralizes the optional-heartbeat
    pattern that was inlined in API handlers."""
    heartbeat = getattr(app_state, "heartbeat", None)
    if heartbeat is not None:
        heartbeat.mark_active()


async def send_ws_with_activity(
    ws: "WebSocket", payload: dict, app_state: Any,
) -> None:
    """Phase 14.E (V3): wrap WS send so heartbeat marking happens
    after every successful frame."""
    await ws.send_json(payload)
    mark_heartbeat_if_present(app_state)
```

```python
# src/training/mlflow/resilient_transport.py — refactor

class ExceptionClassifier(Protocol):
    def is_retryable(self, exc: BaseException) -> bool: ...


class _DefaultClassifier:
    """Built-in transport-shaped exceptions only. Conservative."""
    _BUILTIN_TRANSPORT_TYPES = (
        ConnectionError, TimeoutError, OSError,
    )

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, self._BUILTIN_TRANSPORT_TYPES)


class ResilientMLflowTransport:
    def __init__(
        self, *, ...,
        classifier: ExceptionClassifier | None = None,
    ) -> None:
        self._classifier = classifier or _DefaultClassifier()

    # Retry loop replaces:
    #   except _TRANSPORT_EXCEPTION_TYPES as exc:
    # with:
    #   except BaseException as exc:
    #       if not self._classifier.is_retryable(exc):
    #           raise
```

```python
# src/training/mlflow/_classifier_bootstrap.py — NEW

def make_default_classifier_for_mlflow() -> ExceptionClassifier:
    """Build classifier that knows MLflow's transport stack
    (requests + urllib3 if importable). Phase 14.E: explicit place
    for library knowledge that the transport module no longer holds.
    """
    types: list[type[BaseException]] = [
        ConnectionError, TimeoutError, OSError,
    ]
    try:
        from requests import exceptions as r
        types.extend([r.RequestException, r.ConnectionError, r.Timeout])
    except Exception:
        pass
    try:
        import urllib3.exceptions as u
        types.extend([u.ProtocolError, u.MaxRetryError])
    except Exception:
        pass
    return _ExtendedClassifier(tuple(types))


class _ExtendedClassifier:
    def __init__(self, extra_types: tuple[type[BaseException], ...]) -> None:
        self._types = extra_types

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, self._types)
```

```python
# src/training/mlflow/metrics_buffer.py — refactor (V4 nice-to-have)

@dataclass(frozen=True)
class DecimationPolicy:
    """Phase 14.E (V4): pure config, no runtime state."""
    keep_all: bool
    window_first_seconds: float
    window_first_keep_every: int
    window_mid_boundary_seconds: float
    window_mid_keep_every: int
    window_late_keep_every: int

    @classmethod
    def from_config(cls, config: Any) -> "DecimationPolicy":
        # Existing _extract_decimation logic moves here.
        ...


class MetricsDecimator:
    def __init__(
        self, *,
        policy: DecimationPolicy,
        training_start_time: float | None = None,
    ) -> None: ...
```

## 14.E Migration order — single PR, atomic commits

| # | Commit | Scope | Why this order |
|---|---|---|---|
| 1 | `feat(runner/event_journal): extract validate_journal_config (V6)` | Pure addition | Smallest, lowest-risk — lands first |
| 2 | `feat(runner/event_journal): add set_rotation_callback method (V1 prep)` | Pure addition | No consumers yet |
| 3 | `feat(runner/event_bus): add attach_journal_rotation_listener method (V1 prep)` | Pure addition | Pairs with (2) |
| 4 | `refactor(runner/main): use deferred binding for journal rotation (V1)` | Lifespan rewrite | Cuts over to (2)+(3); deletes circular-binding cell |
| 5 | `feat(runner/api/_activity): add mark_heartbeat_if_present helper (V3 prep)` | Pure addition | New file |
| 6 | `refactor(runner/api/events): use _activity helper for heartbeat marking (V3)` | One-line change | Drops inline check |
| 7 | `refactor(runner/api/jobs): use _activity helper for heartbeat marking (V3)` | One-line change | Drops inline check |
| 8 | `feat(runner/heartbeat): re-export EXPLICIT_HEARTBEAT_TTL_SECONDS (V8 prep)` | Pure addition | Module-level alias |
| 9 | `refactor(runner/api/control): import constant directly (V8)` | One-line change | Cuts over |
| 10 | `feat(training/mlflow): add ExceptionClassifier Protocol + _DefaultClassifier (V5 prep)` | Pure addition | New abstraction |
| 11 | `feat(training/mlflow): add _classifier_bootstrap module (V5 prep)` | Pure addition | Library knowledge moves here |
| 12 | `refactor(training/mlflow/resilient_transport): use injected classifier (V5)` | Module rewrite | Drops module-level optional imports |
| 13 | `refactor(training): pass classifier to ResilientMLflowTransport at construction (V5 wiring)` | Trainer init update | Wires bootstrap into use-site |
| 14 | `feat(training/mlflow): extract DecimationPolicy dataclass (V4)` | Pure addition | Adds dataclass |
| 15 | `refactor(training/mlflow/metrics_buffer): MetricsDecimator takes policy (V4)` | Constructor change | Cutover |
| 16 | `test: extend coverage for V1, V3, V5, V6 fixes` | Test additions | After all impl lands |
| 17 | `docs(plans): mark Phase 14.E DONE` | Bookkeeping | — |

PR is **NOT** squash-merged.

## 14.E Critical files to modify

**NEW:**
- `src/runner/api/_activity.py` — `mark_heartbeat_if_present`, `send_ws_with_activity`
- `src/training/mlflow/_classifier_bootstrap.py` — `make_default_classifier_for_mlflow`
- `src/tests/unit/runner/api/test_activity.py` — coverage for the helper
- `src/tests/unit/runner/test_lifespan_rotation_binding.py` — coverage for V1 fix
- `src/tests/unit/training/mlflow/test_classifier.py` — coverage for V5

**MODIFIED:**
- `src/runner/main.py::_lifespan` — drop circular-binding cell; use `bus.attach_journal_rotation_listener()`
- `src/runner/event_bus.py` — add `attach_journal_rotation_listener` + `_publish_rotation_event`
- `src/runner/event_journal.py` — add `set_rotation_callback`; extract `validate_journal_config`; `on_rotate` becomes Optional
- `src/runner/api/events.py` — use `_activity.mark_heartbeat_if_present`
- `src/runner/api/jobs.py` — same
- `src/runner/api/control.py` — use module-level constant import
- `src/runner/heartbeat.py` — re-export `EXPLICIT_HEARTBEAT_TTL_SECONDS`
- `src/training/mlflow/resilient_transport.py` — accept `classifier`; drop module-level optional imports
- `src/training/mlflow/metrics_buffer.py` — extract `DecimationPolicy`; `MetricsDecimator` takes policy
- Trainer init code (find via grep `ResilientMLflowTransport(` callsites) — pass classifier
- Existing tests touching the refactored shapes

**REUSE (unchanged):**
- `src/runner/pod_terminator.py` — V2 won't-fix
- `src/runner/main.py::_periodic_journal_health_check` — V7 won't-fix

## 14.E Risks (3 deepsink iterations)

### Iteration 1 — initial sweep

| ID | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R-1 | **V1 lifespan refactor breaks rotation event delivery** — if `attach_journal_rotation_listener` isn't called, no rotations are observed | M | L | Test (`test_lifespan_rotation_binding.py`) asserts post-lifespan-construction that calling `journal.rotate()` triggers `EVENTS_ROTATED` on the bus. Lifespan ordering is single-line — easy to review. |
| R-2 | **V3 helper breaks WebSocket heartbeat tracking** — `mark_active` not called per-frame | M | L | Test the helper with a stub websocket; assert `mark_active` called exactly once per `send_ws_with_activity`. Existing `test_api_events.py` heartbeat assertions (none today, but Phase 11.E intent was middleware) get NEW coverage in this PR. |
| R-3 | **V5 classifier injection wires wrong classifier** — trainer init builds `ResilientMLflowTransport()` without classifier kwarg → falls back to `_DefaultClassifier` → tests for `requests.RequestException` retry FAIL | H | M | Find all call sites via grep (`ResilientMLflowTransport(`); audit each. Test that `make_default_classifier_for_mlflow()` returns a classifier that recognizes `requests.RequestException`. |
| R-4 | **V4 (DecimationPolicy split) breaks metrics_buffer config tests** — config-side tests construct `MetricsDecimator(config=mock)` directly | M | M | Tests migrate to `MetricsDecimator(policy=DecimationPolicy.from_config(mock_cfg))`. Mass-update via search-replace + visual review. |

### Iteration 2 — deepsink R-3 (V5 wiring)

`ResilientMLflowTransport` is constructed in trainer init. Current state (verify in implementation):

```python
# Hypothetical current location: src/training/runtime/mlflow_setup.py or similar
transport = ResilientMLflowTransport(...)
```

After 14.E:

```python
from src.training.mlflow._classifier_bootstrap import make_default_classifier_for_mlflow
transport = ResilientMLflowTransport(..., classifier=make_default_classifier_for_mlflow())
```

**Concern**: missing the kwarg silently downgrades retry coverage (default classifier doesn't catch `requests.HTTPError`). The default classifier protects builtin transport types only.

**Mitigation**: in the same PR commit (commit 13 in migration order) update ALL callsites. CI catches via dedicated test that asserts:
```python
def test_resilient_mlflow_transport_default_classifier_does_NOT_catch_requests_error(self) -> None:
    """Pin: by default, ResilientMLflowTransport does NOT retry
    requests-library errors. Caller must wire bootstrap classifier."""
    import requests
    transport = ResilientMLflowTransport()    # default classifier
    classifier = transport._classifier
    fake_exc = requests.exceptions.ConnectionError("test")
    assert not classifier.is_retryable(fake_exc), (
        "Default classifier should NOT retry requests-library errors. "
        "Production code must pass make_default_classifier_for_mlflow()."
    )

def test_make_default_classifier_for_mlflow_DOES_catch_requests_error(self) -> None:
    """Pin: bootstrap classifier extends default with requests + urllib3."""
    import requests
    classifier = make_default_classifier_for_mlflow()
    assert classifier.is_retryable(requests.exceptions.ConnectionError("test"))
```

This makes the contract explicit and fails CI if someone removes the wiring.

### Iteration 3 — deepsink R-1 (V1 lifespan ordering)

Current lifespan order (post-V1):
```python
journal = EventJournal(root_dir=..., on_rotate=None)   # no callback yet
bus = EventBus(journal=journal)
bus.attach_journal_rotation_listener()                  # binds AFTER both exist
```

**Concern**: between `EventBus(...)` returning and `attach_journal_rotation_listener()` being called, can `journal.rotate()` fire?

**Analysis**: `EventBus.__init__` stores the journal reference but doesn't trigger any rotations (rotations are append-driven; bus init doesn't append). Between bus construction and the explicit `attach` call, no events flow. Safe.

**Concern 2**: what if a subsequent lifespan refactor reorders? E.g. someone moves journal-only-init code BETWEEN journal and bus construction.

**Mitigation**: dedicated test:
```python
def test_attach_journal_rotation_listener_is_idempotent(self) -> None:
    """Pin: calling attach twice doesn't double-register; calling
    before journal is set is a no-op."""
    bus = EventBus(journal=None)
    bus.attach_journal_rotation_listener()    # no-op, journal=None
    journal = EventJournal(...)
    bus._journal = journal                    # simulate late binding
    bus.attach_journal_rotation_listener()    # binds
    bus.attach_journal_rotation_listener()    # idempotent — second call is harmless
```

If `set_rotation_callback` is implemented to overwrite (not append), idempotent. Test pins the behaviour.

### Iteration 4 — cross-cutting concerns

| ID | Concern | Resolution |
|---|---|---|
| R-5 | **`_DefaultClassifier` wraps `BaseException`** — too broad, might catch `KeyboardInterrupt` | Use `Exception` not `BaseException` in the catch. Pin via test: `KeyboardInterrupt` is NOT classified as retryable. |
| R-6 | **V5 introduces an extra import + bootstrap step** for every test that constructs `ResilientMLflowTransport` | Test default constructor (no classifier kwarg) → `_DefaultClassifier` is used → minimal retries. Tests that exercise retry paths inject their own classifier. |
| R-7 | **V8 module-level alias might cause circular import** between `heartbeat.py` and `control.py` | `heartbeat.py` doesn't import `control.py` (one-way dep). The alias is a forward declaration — safe. |
| R-8 | **EventJournal `on_rotate` becoming Optional changes external test contract** | Existing tests pass `on_rotate=...` explicitly — still works (Optional with default None). Tests that rely on the constructor-bound callback continue without change. New test for `set_rotation_callback` is additive. |
| R-9 | **Risk of "v1 lands but v3 doesn't" mid-PR**: each commit is atomic; CI runs per-commit; reviewer sees the granularity | Commit-by-commit migration order (§ above) ensures each commit is green. PR overall stays reviewable. |
| R-10 | **`mark_heartbeat_if_present` is called from inside async websocket loop — ensure no implicit blocking** | `mark_heartbeat_if_present` is sync (just calls `heartbeat.mark_active()` which is sync). No I/O. Safe. |

## 14.E Open questions (resolved)

| OQ | Question | Resolution |
|---|---|---|
| OQ-1 | Should V1 use `set_rotation_callback` (post-init) or `with_rotation_callback(cb)` returning a new instance? | Setter — simpler, matches existing mutable-attribute style of `EventJournal._on_rotate`. Returning new instance breaks all existing tests that construct journal once. |
| OQ-2 | V3: should the heartbeat helper handle `app_state` being `None`? | YES — defensive: `getattr(app_state, "heartbeat", None)` returns `None` if `app_state` is None too (TypeError caught? no — getattr on None raises). Add `if app_state is None: return`. |
| OQ-3 | V5: classifier should also classify `socket.error` / `socket.timeout`? | `socket.error` is alias for `OSError` (already covered). `socket.timeout` is alias for `TimeoutError` (already covered). No additions needed. |
| OQ-4 | V5: should `_DefaultClassifier` be public? | Public via `from src.training.mlflow.resilient_transport import _DefaultClassifier as DefaultClassifier` — keep the underscore convention but offer alias for external use. Most callers use `make_default_classifier_for_mlflow()` anyway. |
| OQ-5 | V4: `DecimationPolicy.from_config` location — classmethod on `DecimationPolicy` or module-level function? | Classmethod — discoverable via the dataclass, single source of truth. |
| OQ-6 | V6: `validate_journal_config` raises ValueError or a custom exception? | `ValueError` — matches pre-14.E behaviour. No new exception type needed. |
| OQ-7 | Phase 14.E doesn't fix `decide_and_act` (V2). Will future work need it? | Won't-fix per § 14.E.0. If a future requirement creates pressure (e.g. need to swap dispatcher per provider), revisit then. YAGNI today. |

## 14.E Test plan (7-category coverage)

### V1: rotation binding (~5 tests)

`src/tests/unit/runner/test_lifespan_rotation_binding.py` (NEW):

1. Positive — lifespan boots, journal rotates, bus emits `EVENTS_ROTATED`.
2. Negative — lifespan boots with journal=None, no `EVENTS_ROTATED` emitted on subsequent appends.
3. Boundary — `attach_journal_rotation_listener()` called twice → idempotent (one rotation = one event).
4. Invariants — `EventBus` constructed without journal can still call `attach_journal_rotation_listener()` (no-op).
5. Regressions — pre-14.E behaviour preserved: rotation events have same payload shape (from_seq, to_seq, file_size_bytes, oldest_remaining_seq).

### V3: heartbeat helper (~6 tests)

`src/tests/unit/runner/api/test_activity.py` (NEW):

1. Positive — `mark_heartbeat_if_present(app_state_with_heartbeat)` calls `mark_active()` exactly once.
2. Negative — `mark_heartbeat_if_present(app_state_without_heartbeat)` is a no-op.
3. Negative — `mark_heartbeat_if_present(None)` is a no-op (no AttributeError).
4. Positive — `send_ws_with_activity(ws, payload, app_state)` calls `ws.send_json(payload)` AND `mark_active()` in that order.
5. Boundary — heartbeat present but `mark_active` raises → exception propagates (don't swallow).
6. Regressions — pre-14.E test count for `test_api_events.py` + `test_api_jobs.py` preserved (existing tests not regressed by the helper introduction).

### V5: ExceptionClassifier (~10 tests)

`src/tests/unit/training/mlflow/test_classifier.py` (NEW):

1. Positive — `_DefaultClassifier().is_retryable(ConnectionError())` is True.
2. Positive — `make_default_classifier_for_mlflow().is_retryable(requests.exceptions.ConnectionError())` is True.
3. Negative — `_DefaultClassifier().is_retryable(KeyboardInterrupt())` is False.
4. Negative — `_DefaultClassifier().is_retryable(requests.exceptions.ConnectionError())` is False — explicit pin.
5. Boundary — empty `_ExtendedClassifier(types=())` falls back to default.
6. Invariants — `ExceptionClassifier` Protocol accepts any class with `is_retryable(exc) -> bool`.
7. Dependency errors — if `requests` not importable, `make_default_classifier_for_mlflow()` returns classifier with builtin types only (no exception).
8. Same for `urllib3`.
9. Regressions — `ResilientMLflowTransport(...)` without classifier kwarg uses `_DefaultClassifier`.
10. Logic-specific — `ResilientMLflowTransport(classifier=fake_classifier)` retries based on fake's verdict.

### V4 (NICE): DecimationPolicy (~5 tests)

`src/tests/unit/training/mlflow/test_decimation_policy.py` (NEW or extend existing):

1. Positive — `DecimationPolicy(...)` constructs; frozen.
2. Positive — `DecimationPolicy.from_config(mock)` parses correctly.
3. Boundary — `from_config` with missing fields → uses defaults.
4. Invariants — `MetricsDecimator(policy=p)` does not mutate `p`.
5. Regressions — pre-14.E `MetricsDecimator(config=...)` migration: tests pass after `MetricsDecimator(policy=DecimationPolicy.from_config(...))` substitution.

### V6 (NICE): validate_journal_config (~4 tests)

Extend `src/tests/unit/runner/test_event_journal.py`:

1. Positive — valid params pass.
2. Negative — `file_size_cap=0` raises ValueError.
3. Negative — `max_files=0` raises ValueError.
4. Boundary — minimum valid params (`file_size_cap=1`, `max_files=1`).

### V8 (NICE): module-level constant (~1 test)

Extend `src/tests/unit/runner/test_heartbeat.py`:

1. Logic-specific — `from src.runner.heartbeat import EXPLICIT_HEARTBEAT_TTL_SECONDS` works; value matches `MacHeartbeat.EXPLICIT_HEARTBEAT_TTL_SECONDS`.

## 14.E Verification

### Unit tests

```bash
# Phase 14.E new + extended
pytest src/tests/unit/runner/test_lifespan_rotation_binding.py -v
pytest src/tests/unit/runner/api/test_activity.py -v
pytest src/tests/unit/training/mlflow/test_classifier.py -v
pytest src/tests/unit/training/mlflow/test_decimation_policy.py -v   # if V4 included

# Existing tests still green
pytest src/tests/unit/runner/ -v
pytest src/tests/unit/training/mlflow/ -v
pytest src/tests/unit/api/ -v
```

### Regression

```bash
pytest src/tests/unit/ -q --tb=line
# Pre-existing slim-venv failures (Phase 6.6) unchanged.
```

### Cleanup grep checks

```bash
# V1: no more circular cell
! grep -n 'rotate_publisher = {"bus"' src/runner/main.py
! grep -n '"bus": None' src/runner/main.py

# V3: no inline heartbeat checks in API handlers
! grep -n 'heartbeat.mark_active()' src/runner/api/events.py
! grep -n 'heartbeat.mark_active()' src/runner/api/jobs.py
grep -n 'mark_heartbeat_if_present' src/runner/api/

# V5: no module-level optional imports in transport
! grep -n 'from requests' src/training/mlflow/resilient_transport.py
! grep -n 'import urllib3' src/training/mlflow/resilient_transport.py
grep -n 'from requests' src/training/mlflow/_classifier_bootstrap.py

# V8: control.py uses module-level constant
! grep -n 'MacHeartbeat as _Heartbeat' src/runner/api/control.py
grep -n 'EXPLICIT_HEARTBEAT_TTL_SECONDS' src/runner/api/control.py
```

### Manual smoke

1. Runner boots → trigger journal rotation (write enough events) → confirm `EVENTS_ROTATED` event published on bus.
2. WebSocket connection → send/receive frames → check `app.state.heartbeat.is_alive()` returns True after each frame.
3. Trainer with MLflow → cause requests.HTTPError → confirm transport retries (would NOT retry pre-14.E if classifier wiring missed).

## 14.E Effort + rollout

| Step | Effort | Notes |
|---|---|---|
| V6 `validate_journal_config` extraction + tests | 1h | Lowest risk, lands first |
| V1 deferred binding + tests | 2h | EventJournal.set_rotation_callback + EventBus.attach_journal_rotation_listener + lifespan rewrite |
| V3 `mark_heartbeat_if_present` + WS wrapper + tests | 2h | Helper + 2 callsite swaps + new test file |
| V8 constant relocation | 30min | Trivial |
| V5 ExceptionClassifier protocol + bootstrap + transport refactor + tests | 4h | Most consequential — touches retry semantics |
| V5 trainer-init wiring (find + update callsites) | 1.5h | Ensure no missed sites |
| V4 DecimationPolicy split + tests (if scope permits) | 2h | Nice-to-have |
| Documentation + commit messages | 1h | |
| Code review buffer | 2.5h | |
| **Total (high-priority only: V1, V3, V5, V6, V8)** | **~13h (~1.6 engineer-days)** |
| **Total (with nice-to-haves V4 included)** | **~15h (~2 engineer-days)** |

Single PR with 17 commits per migration order. Land after 14.D+F (which provides the cleanup foundation) and Phase 14 closes.

## 14.E Migration & rollback

**Migration:** No operator-facing changes. The fixes are internal architectural cleanup — runner boots identically, trainer logs identically, WebSocket frames identical.

**Rollback:** Revert the 17 commits in reverse order. The architecture supports stepwise rollback because each commit is atomic and additive-then-cutover (commits 1-3, 5, 8, 10-11 are pure additions; cutovers in 4, 6-7, 9, 12-13, 15 each touch one consumer).

**Operator failure mode if rolled out incorrectly:**

* V1 lifespan refactor: missed call to `attach_journal_rotation_listener` → no rotation events on bus → operator dashboards stop seeing rotations. Caught by smoke test (manual smoke step 1).
* V3 helper bug: heartbeat not marked → pod terminator sees stale heartbeat → wrong terminal decision. Caught by V3 helper test + integration smoke.
* V5 missing classifier wiring: trainer's MLflow retries silently degrade. Caught by retry-coverage test (commit 16).

**Won't-fix items (V2, V7) carry no rollback risk** — they remain unchanged.

No DB migrations, no config-schema changes, no backwards-compat shims.
