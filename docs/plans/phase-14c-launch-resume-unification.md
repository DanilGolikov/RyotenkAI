# Phase 14.C — LaunchResumeService Unification

> Status: **DRAFT — pending user approval**
> Author: daniil + agent
> Date: 2026-04-28
> Worktree: `nice-jepsen-07d789`
> Predecessors: Phase 14.A (commit `1819cc4`), Phase 14.B (uncommitted but green in worktree)
> Successor: Phase 14.D + 14.F (planned to merge into single "Provider-leak elimination" plan), Phase 14.E
> Migration policy: NO BACKWARDS COMPATIBILITY — single-PR full cutover
> Effort: ~2.5 engineer-days
> Risk: Medium-Low — refactor of two consumer call-sites; backed by 45+ existing tests of the duplicated logic
> Out of: Phase 14 roadmap in [`phase-14a-provider-abstraction.md`](./phase-14a-provider-abstraction.md)

## 14.C Context — what we're fixing

Phase 11.C-2 wired the in-pod resume flow ("Mac wakes up, pod is sleeping, probe + capacity-aware retry then continue") into two surfaces:

* **CLI**: `src/cli/commands/run.py::_resume_pod_if_needed(run_dir)` — lines 189-337, **108 LOC**. Streams progress via `typer.echo`; raises `die(...)` on failure.
* **REST**: `src/api/services/launch_service.py::resume_pod_for_run(run_dir)` — lines 143-286, **99 LOC**. Returns a `ResumePodResponse` dataclass; silent (no progress emitted).

**The resume body is byte-identical** between the two — same `PodAvailabilityProbe` instantiation, same `RunPodAPIClient` client, same `_query_pod` closure, same `_resume_call` closure, same `resume_pod_with_retry(metadata.pod_id, resume_call=..., is_capacity_error=...)` invocation, same outcome handling. The only differences are:

1. **Output shape** — CLI echoes progress lines + raises `die()`; REST returns a `ResumePodResponse` dataclass.
2. **`provider != "runpod"` gate** — both paths early-return with a "doesn't have an in-pod resume mechanism" message but the gating is duplicated.
3. **API key handling** — both check `os.environ["RUNPOD_API_KEY"]`; CLI prints "skipping" and returns silently, REST returns `ok=False`.

A single new module — `src/pipeline/launch/resume_service.py::LaunchResumeService` — eliminates the duplicate. Both surfaces become thin adapters: CLI translates a progress callback into `typer.echo`, REST converts the final `ResumeOutcome` into `ResumePodResponse`.

**Secondary cleanup**: `src/pipeline/launch/pod_availability.py::_RUNPOD_STATUS_MAP` (lines 176-182) is RunPod GraphQL `desiredStatus` vocabulary leaking into shared code. Phase 14.A's `RunPodProvider.probe_availability` already imports it (with a Phase 14.C TODO marker at provider.py:605-606). This phase relocates the map into RunPod's runtime package, drops the import path from shared code, and tightens `PodAvailabilityProbe` to a transport-agnostic shape that delegates the mapping to the provider.

## 14.C Architectural decisions (locked, deepsink-validated)

### 14.C.1 Where the new service lives — **`src/pipeline/launch/resume_service.py`**

Three options considered:

1. **`src/pipeline/launch/resume_service.py`** — sibling to `pod_availability.py`, `restart_options.py`. ✅ **Picked.**
2. `src/api/services/resume_service.py` — keeps the API service layer co-located.
3. `src/launch/` — new top-level.

Rationale for (1):
* `pod_availability.py` (the probe), `restart_options.py` (the resume validator), `pod_metadata` infra all live in `src/pipeline/launch/` — the new service belongs in the same package.
* `src/api/services/` is FastAPI-specific; the service is consumed by both API and CLI, so neutral location wins.
* New top-level (`src/launch/`) requires more import-chain churn for no benefit.

### 14.C.2 Sync vs async — **sync facade with internal `asyncio.run`**

Both consumers (CLI and `launch_service.py:resume_pod_for_run`) are sync. The existing implementation already runs `asyncio.run(resume_pod_with_retry(...))` internally — the new service preserves this shape. CLI runs from a sync Typer command; REST runs through FastAPI's `run_in_threadpool` wrapper (already in place).

Rationale: matches the rest of the launch infrastructure (`launch_service.launch()` is sync, `RunContextStore` is sync), avoids forcing async on Mac-side launch flow that has no other async pressure. The async island stays inside `resume_pod_with_retry`.

### 14.C.3 Progress reporting — **callback-based, optional**

The CLI shows ~5 progress lines ("Probing pod...", "Pod status: ...", "Resuming pod...", "Pod resumed in..."). The REST surface stays silent. Three approaches considered:

1. **Optional `on_progress: Callable[[ResumeProgress], None]` callback.** ✅ **Picked.**
2. Async generator yielding events.
3. CLI does its own progress reporting (probe + resume separately, bypassing the service for the streaming part).

Rationale for (1):
* Sync — matches the service shape.
* Optional — REST passes `None`, CLI passes `lambda evt: typer.echo(evt.message)`.
* Typed — `ResumeProgress` is a small frozen dataclass with `kind`, `message`, `detail`.
* No async complexity, no generator state machine.
* Tests can pass a list-appending callback for assertion.

### 14.C.4 Failure surface — **`ResumeOutcome` dataclass, NOT `Result[T, E]`**

Phase 14.B's `LifecycleActionResult` set the precedent: when the failure modes are part of the success vocabulary (already-gone, capacity-exhausted, probe-failed), a typed dataclass with explicit fields beats `Result[T, E]` because consumers grep for outcome strings.

```python
@dataclass(frozen=True)
class ResumeOutcome:
    """Final state of a resume attempt — covers all 5 PodAvailability
    branches plus the legacy / no-metadata case."""
    availability: str          # PodAvailability.value or "skipped"
    ok: bool
    message: str
    elapsed_seconds: float | None = None  # None unless we actually called resume
    attempts_made: int = 0                # 0 unless we actually called resume
    capacity_exhausted: bool = False      # surfaced for CLI's "die with capacity hint"
```

CLI's `die()` adapter inspects `ok=False` + `availability ∈ {GONE, SLEEPING_RESUME_FAILED, PROBE_FAILED}` to pick the right hint. REST's `ResumePodResponse` adapter reads `availability`, `ok`, `message` 1:1.

### 14.C.5 Provider gating — **`isinstance(provider, ITerminalActionProvider)` not `provider_name == "runpod"`**

Phase 14.A's `ITerminalActionProvider` Protocol exists exactly for this reason. Both `launch_service.py:190` and `run.py:230` currently check `metadata.provider != "runpod"` — string comparison. The new service:

1. Resolves the provider class from `metadata.provider` via a registry (see § 14.C.6 below).
2. Checks `isinstance(provider, ITerminalActionProvider)` — type-system gate.
3. If False → return `ResumeOutcome(availability="running", ok=True, message="provider has no in-pod resume mechanism")`.

Single-node already implements `IGPUProvider.probe_availability` returning `state="running"` (Phase 14.A). It does NOT implement `ITerminalActionProvider` (Phase 14.A § OQ-2). The gate cleanly skips it.

When a future third provider lands with lifecycle support, it just inherits `ITerminalActionProvider` — no string-check site needs editing.

### 14.C.6 Provider resolution — **factory function, lazy import**

`PodMetadata.provider` is a string. The service needs to turn it into a live `ITerminalActionProvider | None` instance. Approach:

```python
def _resolve_lifecycle_provider(provider_name: str) -> ITerminalActionProvider | None:
    """Build a lifecycle-capable provider instance for a given name.

    Returns None when:
      * Provider name unknown.
      * Provider doesn't conform to ITerminalActionProvider.
      * Required env (e.g. RUNPOD_API_KEY) absent — service surfaces
        "skipped: missing creds" outcome.

    Lazy imports per provider to keep CLI startup fast for non-resume
    paths.
    """
```

Closed registry of two entries (RunPod + future). For RunPod, `_resolve_lifecycle_provider("runpod")` returns a minimal `RunPodProvider` instance built from env (NOT through the heavy Pydantic config validator — bypass via `object.__new__` + attribute injection, same pattern as Phase 14.A test fixtures). For single-node, returns `None` (no lifecycle).

This **does** add a tiny duplication with `src/runner/runtime/provider_registry.py` (Phase 14.B) — both have closed dicts mapping provider names to clients. Acceptable: 14.B's registry is runner-side (for in-pod hooks); 14.C's resolver is Mac-side (for control-plane lifecycle calls). They serve different consumers and have different bootstrap shapes (14.B requires `RUNPOD_POD_ID`; 14.C reads `pod_id` from `PodMetadata`). Phase 14.E or 14.F can extract a shared `Mac<->Runner provider locator` if the duplication grows.

### 14.C.7 `_RUNPOD_STATUS_MAP` relocation — **moved into RunPodProvider, deleted from shared code**

Current state (Phase 14.A):
* `src/pipeline/launch/pod_availability.py:176-182` defines `_RUNPOD_STATUS_MAP`.
* `src/providers/runpod/training/provider.py:605-612` imports + uses it (with TODO comment).

Phase 14.C action:
1. Move `_RUNPOD_STATUS_MAP` (and `_GONE_ERROR_MARKERS` already there as static method) into `src/providers/runpod/training/provider.py` as a module-level Final dict.
2. Delete the import path from `src/pipeline/launch/pod_availability.py`.
3. Remove the TODO comment.
4. **`PodAvailabilityProbe` class itself stays** in `pod_availability.py` for now — it has a transport-agnostic `query_pod` callable injected, and its job is "translate provider response into `PodAvailability` enum". Phase 14.D may collapse it further when string-checks are eliminated; for 14.C we keep it because the new service still uses it as the probe transport.
5. The probe's `_RUNPOD_STATUS_MAP` reference becomes an injected `status_mapper: Callable[[str], PodAvailability]` — RunPod-specific mapping passes a callable that consults the (now provider-local) map.

**Alternative considered + rejected**: collapse `PodAvailabilityProbe` entirely, have the service call `provider.probe_availability()` directly and translate the `AvailabilityVerdict.state` literals to `PodAvailability` enum values. Rejected for this phase because:
* `PodAvailability` enum (in `pod_availability.py`) and `AvailabilityVerdict.state` Literal type carry the same vocabulary but are NOT the same type. Cross-walk requires a translation layer.
* `PodAvailabilityProbe` has its own retry budget logic (5-min capacity-aware). Collapsing into `provider.probe_availability` would force the budget into the provider too — wrong layer.
* Phase 14.D will revisit this collapsing-question when it eliminates the env hardcodes.

For now: provider owns the **mapping** (string → enum); probe owns the **transport orchestration** (call provider, time it, return verdict).

### 14.C.8 Retry policy ownership — **service owns budget, provider owns single-shot**

Phase 11.C's `resume_pod_with_retry` lives in `src/providers/runpod/training/lifecycle.py` (or similar — Phase 11.C planning section). It takes:
* `pod_id` — string
* `resume_call: AsyncCallable` — closure over the SDK
* `is_capacity_error: Callable[[str], bool]` — RunPod-specific marker check

After Phase 14.C the **service** invokes `resume_pod_with_retry` directly with the provider-local `is_capacity_error`. We do NOT push the retry into `ITerminalActionProvider.resume()` (Phase 14.A) because:

1. `ITerminalActionProvider.resume()` is sync (Phase 14.A § OQ-1) — adding capacity retry would force async.
2. Capacity-aware retry is a **service-level concern** (5-min budget, exponential backoff). Provider exposes single-shot semantics; service composes the retry on top.
3. Phase 14.B's `IPodLifecycleClient.resume()` (runner-side) has its own retry knobs. Mac-side and runner-side don't need to share retry policy.

Trade-off: service still imports `is_capacity_error_message` from `src/providers/runpod/sdk_adapter.py` — a tiny RunPod-specific leak. Mitigated by:
* Importing it lazily inside the service's RunPod branch (same as the current code).
* Extracting `provider.is_capacity_error(message: str) -> bool` is a candidate for Phase 14.D (unify with the rest of the env-hardcode cleanup).

### 14.C.9 `ResumePodResponse` REST contract — **kept verbatim**

The REST endpoint that consumes `resume_pod_for_run` returns `ResumePodResponse(availability, ok, message)` — three fields, JSON-serializable. Web UI parses this shape. Changing it = breaking change for the frontend.

Phase 14.C keeps the dataclass byte-identical. The REST adapter (`launch_service.resume_pod_for_run`) becomes a 5-line wrapper over `LaunchResumeService.resume()`:

```python
def resume_pod_for_run(run_dir: Path) -> ResumePodResponse:
    outcome = _service.resume(run_dir)
    return ResumePodResponse(
        availability=outcome.availability,
        ok=outcome.ok,
        message=outcome.message,
    )
```

CLI gets a thicker adapter (~30 LoC) that translates `on_progress` events into `typer.echo` and `ResumeOutcome` into `die()` / silent-return.

## 14.C Scope

### IN-scope

| Item | Description |
|---|---|
| `src/pipeline/launch/resume_service.py` (NEW) | `LaunchResumeService`, `ResumeOutcome`, `ResumeProgress`, `_resolve_lifecycle_provider` |
| `src/api/services/launch_service.py::resume_pod_for_run` REWRITE | Becomes 5-line wrapper |
| `src/cli/commands/run.py::_resume_pod_if_needed` REWRITE | Becomes ~30-line adapter (echo + die translations) |
| Move `_RUNPOD_STATUS_MAP` from shared `pod_availability.py` to `src/providers/runpod/training/provider.py` | Drop the layer leak |
| `PodAvailabilityProbe` constructor extension | Accept `status_mapper: Callable[[str], PodAvailability]` (default = RunPod's mapper for backwards compat with existing tests, removed in 14.D) |
| Migrate ~45 tests | Existing `test_launch_service_resume_pod.py` + CLI tests + `test_pod_availability.py` |
| Add new tests for `LaunchResumeService` | 7-cat coverage; pin progress callback contract |
| Drop `provider != "runpod"` string checks | Replaced by `isinstance(provider, ITerminalActionProvider)` |

### OUT-of-scope

| Item | Phase |
|---|---|
| Collapse `PodAvailabilityProbe` into `provider.probe_availability()` | 14.D — when probe's `status_mapper` parameter goes away |
| Extract `provider.is_capacity_error(message)` method | 14.D |
| Eliminate the duplication between Mac-side `_resolve_lifecycle_provider` and runner-side `provider_registry` | 14.E or 14.F (pending real need) |
| Generalize resume to non-RunPod providers | YAGNI until a third provider with lifecycle |
| Async-ify launch_service / CLI | YAGNI |
| Web UI changes for `ResumePodResponse` | None — wire-shape preserved |

## 14.C New abstractions (signatures only)

```python
# src/pipeline/launch/resume_service.py

@dataclass(frozen=True)
class ResumeProgress:
    """Single progress event emitted by LaunchResumeService.resume().

    ``kind`` discriminator (string for JSON-friendly payloads):
      * "probing"  — about to call probe
      * "verdict"  — probe returned, message describes status
      * "resuming" — about to call resume (capacity-aware retry)
      * "resumed"  — pod is now reachable, message has timing
      * "skipped"  — provider doesn't support resume, or no metadata
    ``detail`` — kind-specific structured payload for tests / future
    consumers. CLI ignores ``detail``, only displays ``message``.
    """
    kind: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumeOutcome:
    """Terminal state of a resume attempt.

    ``availability`` is a :class:`PodAvailability` value or the string
    ``"skipped"`` (legacy run, missing creds, non-lifecycle provider).
    ``ok=True`` means the pipeline can continue with the run. ``ok=False``
    means the caller MUST surface the message to the user (CLI dies,
    REST returns non-ok response).
    """
    availability: str
    ok: bool
    message: str
    elapsed_seconds: float | None = None
    attempts_made: int = 0
    capacity_exhausted: bool = False


ProgressCallback = Callable[[ResumeProgress], None]


class LaunchResumeService:
    """Provider-agnostic resume orchestrator. Phase 14.C single source
    of truth for the wake-pod flow previously duplicated across CLI
    + REST.

    Test seam: ``provider_resolver`` lets tests inject a fake.
    """

    def __init__(
        self,
        *,
        provider_resolver: Callable[[str], ITerminalActionProvider | None] | None = None,
    ) -> None: ...

    def resume(
        self,
        run_dir: Path,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ResumeOutcome: ...
```

```python
# src/api/services/launch_service.py — resume_pod_for_run becomes:

def resume_pod_for_run(run_dir: Path) -> ResumePodResponse:
    """Phase 14.C — thin REST adapter over LaunchResumeService."""
    from src.pipeline.launch.resume_service import LaunchResumeService
    outcome = LaunchResumeService().resume(run_dir)
    return ResumePodResponse(
        availability=outcome.availability,
        ok=outcome.ok,
        message=outcome.message,
    )
```

```python
# src/cli/commands/run.py — _resume_pod_if_needed becomes:

def _resume_pod_if_needed(run_dir: Path) -> None:
    """Phase 14.C — thin CLI adapter over LaunchResumeService."""
    from src.pipeline.launch.resume_service import (
        LaunchResumeService, ResumeProgress,
    )

    def _echo(evt: ResumeProgress) -> None:
        typer.echo(f"  {evt.message}")

    outcome = LaunchResumeService().resume(run_dir, on_progress=_echo)
    if outcome.ok:
        return

    if outcome.availability == PodAvailability.GONE.value:
        raise die("Pod has been terminated; cannot resume in-place.", hint=...)
    if outcome.capacity_exhausted:
        raise die(f"Pod resume capacity unavailable: {outcome.message}", hint=...)
    if outcome.availability == PodAvailability.PROBE_FAILED.value:
        raise die(f"Pod probe failed: {outcome.message}", hint=...)
    raise die(f"Pod resume failed: {outcome.message}", hint=...)
```

```python
# src/providers/runpod/training/provider.py — relocated map

# Phase 14.C: moved here from src/pipeline/launch/pod_availability.py.
# Lives in the provider module that owns the RunPod GraphQL vocabulary.
_RUNPOD_STATUS_MAP: Final[dict[str, "PodAvailability"]] = {
    "RUNNING": PodAvailability.RUNNING,
    "EXITED": PodAvailability.SLEEPING_RESUMABLE,
    "STOPPED": PodAvailability.SLEEPING_RESUMABLE,
    "PAUSED": PodAvailability.SLEEPING_RESUMABLE,
    "TERMINATED": PodAvailability.GONE,
    "DEAD": PodAvailability.GONE,
}
```

```python
# src/pipeline/launch/pod_availability.py — PodAvailabilityProbe gains
# an injected status_mapper to drop the RunPod hardcode.

class PodAvailabilityProbe:
    def __init__(
        self, *,
        query_pod: Callable[[str], dict],
        status_mapper: Callable[[str], "PodAvailability"] | None = None,
    ) -> None:
        # Default mapper: RunPod's, lazily imported. Phase 14.D will
        # remove the default — every caller will pass their own.
        if status_mapper is None:
            from src.providers.runpod.training.provider import (
                map_runpod_desired_status_to_availability,
            )
            status_mapper = map_runpod_desired_status_to_availability
        self._status_mapper = status_mapper
        ...
```

## 14.C Migration order — single PR, atomic commits

| # | Commit | Why this order |
|---|---|---|
| 1 | `feat(pipeline/launch): add LaunchResumeService skeleton + ResumeOutcome/ResumeProgress` | Lands the abstraction first; no consumer changes yet. |
| 2 | `test(pipeline/launch): unit tests for LaunchResumeService` (7-cat) | Service is testable in isolation with fake provider_resolver. |
| 3 | `refactor(providers/runpod): move _RUNPOD_STATUS_MAP into RunPodProvider module` | Provider-local relocation; pod_availability.py probe gets `status_mapper` injection point with backward-compat default. |
| 4 | `refactor(api/services): switch resume_pod_for_run to LaunchResumeService` | One consumer cuts over; REST tests adapt. |
| 5 | `refactor(cli/commands): switch _resume_pod_if_needed to LaunchResumeService` | Second consumer cuts over; CLI tests adapt. |
| 6 | `refactor(pipeline/launch): drop status_mapper backward-compat default` | After (3) all callers pass their own mapper; remove the lazy import-back. |
| 7 | `test: migrate test_launch_service_resume_pod.py + test_pod_availability.py + CLI tests to new shape` | Bulk test migration; ~45 tests adjusted. |
| 8 | `docs(plans): mark Phase 14.C DONE; update successor link in 14.B` | Bookkeeping. |

PR is **NOT** squash-merged — Phase 14 convention.

## 14.C Critical files to modify

**NEW:**
- `src/pipeline/launch/resume_service.py` — `LaunchResumeService`, `ResumeOutcome`, `ResumeProgress`, `_resolve_lifecycle_provider`
- `src/tests/unit/pipeline/launch/test_resume_service.py` — 7-cat coverage

**MODIFIED:**
- `src/api/services/launch_service.py::resume_pod_for_run` — collapse to 5-line wrapper
- `src/cli/commands/run.py::_resume_pod_if_needed` — collapse to ~30-line adapter
- `src/pipeline/launch/pod_availability.py::PodAvailabilityProbe.__init__` — add `status_mapper` parameter
- `src/providers/runpod/training/provider.py` — receive `_RUNPOD_STATUS_MAP` from pod_availability.py + expose `map_runpod_desired_status_to_availability` helper
- `src/tests/unit/pipeline/launch/test_pod_availability.py` — update tests to pass mapper explicitly
- `src/tests/unit/api/services/test_launch_service_resume_pod.py` — adapt to mock service + fake provider_resolver
- CLI tests (e.g. `src/tests/unit/cli/commands/test_run_resume.py` if exists) — same pattern

**REUSE (do NOT modify):**
- `src/pipeline/launch/pod_availability.py::PodAvailability`, `load_pod_metadata_for_run`, `resume_pod_with_retry` — Phase 11.C primitives, untouched
- `src/pipeline/state/models.py::PodMetadata` — Phase 11.C-1 schema, untouched
- `src/providers/runpod/training/api_client.py::RunPodAPIClient` — transport, untouched
- `src/providers/runpod/sdk_adapter.py::is_capacity_error_message` — used by service via lazy import (same as today)
- `src/providers/training/interfaces.py::ITerminalActionProvider` — Phase 14.A Protocol, untouched
- `src/api/services/launch_service.py::ResumePodResponse` — wire shape preserved

**DELETED:**
- `_RUNPOD_STATUS_MAP` constant in `src/pipeline/launch/pod_availability.py`
- The `if metadata.provider != "runpod"` string check in both consumer sites
- The `if not api_key:` early returns in both consumer sites (moved into the service's resolver)

## 14.C Risks (3 deepsink iterations)

### Iteration 1 — initial sweep

| ID | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R-1 | **`ResumePodResponse` REST contract drift** — accidentally change the field shape and break Web UI | H | L | Pin via existing test (`test_launch_service_resume_pod.py`); REST adapter is 5 lines, easy to review. Frozen dataclass invariant. |
| R-2 | **CLI progress output changes user-visible UX** — different formatting, different ordering, missing lines | M | M | Test the CLI adapter with a list-collecting `_echo` callback; assert progression matches pre-14.C output line-by-line. Snapshot-test where reasonable. |
| R-3 | **`provider_resolver` factory hides bugs** — if it returns a stale or wrong provider, the service silently falls into "skipped" branch | M | L | Two tests pin: (a) RunPod metadata + valid env → returns ITerminalActionProvider instance; (b) RunPod metadata + missing API_KEY → returns None and service emits `skipped` outcome with explicit `message` mentioning the missing var. |
| R-4 | **`PodAvailabilityProbe.status_mapper` default introduces a circular import** — pod_availability imports provider, provider already imports from pod_availability | M | M | Lazy import inside the default factory (`if status_mapper is None: from src.providers... import ...`). Test that pure-import of `pod_availability` does not pull provider module. |

### Iteration 2 — deepsink R-2 (CLI UX preservation)

The CLI today emits this exact progression on a sleeping pod:

```
Probing pod {pod_id} ({provider})...
  Pod status: sleeping_resumable
  {verdict.message}
  Resuming pod (capacity-aware retry, ≤5min budget)...
  Pod resumed in 3.2s (1 attempt(s))
```

After 14.C the service emits 4-5 `ResumeProgress` events:

1. `kind="probing"`, message=`"Probing pod pod-abc (runpod)..."`
2. `kind="verdict"`, message=`"Pod status: sleeping_resumable"` + verdict.message
3. `kind="resuming"`, message=`"Resuming pod (capacity-aware retry, ≤5min budget)..."`
4. `kind="resumed"`, message=`"Pod resumed in 3.2s (1 attempt(s))"`

CLI's `_echo` callback prepends `"  "` (2 spaces) to each message. The first one (probing) had no leading spaces in the original code — `typer.echo(f"Probing pod {metadata.pod_id} ({metadata.provider})...")` direct.

**Decision**: CLI's echo callback handles indentation per `kind` (probing = 0 spaces, others = 2 spaces). Pin via test that captures the exact list of strings emitted to stdout.

Alternative considered: have the service emit pre-indented strings. Rejected — couples display semantics to the service. Indent stays in the adapter.

### Iteration 3 — deepsink R-3 (provider_resolver)

The resolver's job: turn `metadata.provider` string into either `ITerminalActionProvider | None`. Concretely for RunPod:

```python
def _resolve_lifecycle_provider(provider_name: str) -> ITerminalActionProvider | None:
    if provider_name == "runpod":
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            return None  # Service emits "skipped: missing API key"
        # Build a minimal RunPodProvider via __new__ — bypass the heavy
        # Pydantic config validator. Same pattern as Phase 14.A test
        # fixtures + Phase 14.B provider_registry's RunPod path.
        from src.providers.runpod.training.provider import RunPodProvider
        provider = object.__new__(RunPodProvider)
        provider._api_key = api_key
        # ... minimal attribute injection ...
        return provider
    if provider_name == "single_node":
        return None  # No lifecycle; service emits "skipped: provider has no resume"
    return None  # Unknown provider; same skipped path
```

**Concern raised**: bypassing Pydantic via `__new__` means a future `RunPodProvider.__init__` change could leave the resolver building broken instances.

**Mitigation**: factor out the construction into a classmethod on `RunPodProvider`:
```python
@classmethod
def from_resume_metadata(cls, *, api_key: str) -> "RunPodProvider":
    """Build a minimal provider just for resume-flow API calls.
    Skips the full config validation chain — only attributes that
    `terminate/pause/resume/probe_availability` consume are populated."""
```

This gives the resolver one named entry-point per provider, makes the Pydantic-bypass intent explicit, and gives a single seam to update if `RunPodProvider.__init__` evolves. Phase 14.C adds this method on RunPod; future providers add their equivalent.

**Concern**: `_resolve_lifecycle_provider` duplicates Phase 14.B's `src/runner/runtime/provider_registry.py::resolve_lifecycle_client_from_env` shape.

**Resolution**: acknowledged minor duplication. The two registries differ in:
* 14.B's runner registry produces `IPodLifecycleClient` (async, in-pod transport).
* 14.C's Mac-side resolver produces `ITerminalActionProvider` (sync, Mac-side SDK).

These are two distinct Protocols (Phase 14.B § 1.1) so the registries can't share types. Phase 14.E or 14.F may extract a shared "provider locator" if a third provider lands; for now this is YAGNI.

### Iteration 4 — cross-cutting concerns

| ID | Concern | Resolution |
|---|---|---|
| R-5 | **Phase 11.C `resume_pod_with_retry` 5-min budget is RunPod-specific** — should we generalize? | NO. The budget knobs (10s/30s/60s/120s + 5-min cap) are RunPod's empirical capacity-recovery curve. Different providers will have different curves. Phase 14.D considers `provider.resume_retry_budget_seconds` if a second lifecycle provider lands. |
| R-6 | **`is_capacity_error_message` import is a RunPod leak** in the new service | YES it is. Acceptable for 14.C; flagged for Phase 14.D. The leak is one line (`from src.providers.runpod.sdk_adapter import is_capacity_error_message`) and is gated on `metadata.provider == "runpod"`. |
| R-7 | **Test fixture proliferation** — `_FakeProvider`, `_FakeRunPodAPIClient`, `_FakeProbe`, ... | Reuse Phase 14.A's `_mk_provider` pattern (object.__new__ + attribute injection). New tests use 2 fakes max: `_FakeProgressCollector` (callback) and `_FakeProvider` (ITerminalActionProvider stub). |
| R-8 | **Web UI still calls `POST /api/v1/runs/{run_id}/resume-pod`** which routes to `resume_pod_for_run` | Wire-shape preserved (R-1 mitigation). Web UI tests not affected. Confirmed by reading `web/src/api/runs.ts` — only reads `availability`, `ok`, `message`. |
| R-9 | **Existing `test_pod_availability.py` mocks the map directly** in some tests — relocation breaks them | Test migration commit (commit 7) adapts: tests that constructed `PodAvailabilityProbe` without a mapper get the lazy default; tests that asserted on `_RUNPOD_STATUS_MAP[...]` import from the new location. |
| R-10 | **`asyncio.run()` from inside the service** when called from a test that already has a running loop | Same constraint as Phase 14.B § R-3 — doc this in the service's class docstring. Pytest with `pytest-asyncio` keeps loop scope per-function, so unit tests are safe. The integration test that goes through TestClient runs through `run_in_threadpool` (preserves Phase 11.C's pattern). |
| R-11 | **`PodAvailabilityProbe`'s budget logic conflicts with retry-after-resume timing** | Pre-14.C the probe + resume are two phases with their own budgets. 14.C preserves: probe runs once (no retry), resume has the 5-min budget. The service emits separate `kind="verdict"` and `kind="resuming"` events so callers see the boundary. |
| R-12 | **`die()` import in the CLI adapter pulls Typer at module load time** | `die` already imported at top of `run.py`. No change. |

## 14.C Open questions (resolved)

| OQ | Question | Resolution |
|---|---|---|
| OQ-1 | Should the service own the `RunPodAPIClient` instance lifecycle (one per `service.resume()` call vs cached on the service)? | One per call. The service is constructed fresh each time anyway (no shared state). Caching would save ~50ms in a flow that already takes 5+ seconds. Not worth the complexity. |
| OQ-2 | Should `ResumeOutcome.elapsed_seconds` measure end-to-end (probe + resume) or resume-only? | Resume-only. Pre-14.C the message says "Pod resumed in 3.2s" — that's just the resume-call time. Preserve. End-to-end timing can be added in a future telemetry phase. |
| OQ-3 | Should the service publish events to the EventBus (Phase 12.A)? | NO. EventBus is in-pod; this service is Mac-side. If we want telemetry on resume attempts, that's a Mac-side telemetry phase (out of 14.C scope). |
| OQ-4 | What about CLI's `--skip-pod-probe` flag — does it still work after 14.C? | Yes. The CLI flag is checked BEFORE `_resume_pod_if_needed` is called. The service is unaware of the flag. Verified by reading run.py callsite. |
| OQ-5 | Does Phase 14.C close any of the 17 string-check sites listed in 14.A's audit? | YES — 2 of 17. The `provider != "runpod"` checks in `launch_service.py:190` and `run.py:230` are eliminated. The remaining 15 are 14.D + 14.F scope. |
| OQ-6 | What happens if a future provider adds `ITerminalActionProvider` but no `_resolve_lifecycle_provider` registry entry? | The resolver returns `None` → service emits `skipped` outcome → user sees message and proceeds without resume. NOT a crash. Adding a third provider = registry entry update. Phase 14.A's two-source-of-truth invariant ensures the provider's `caps.supports_lifecycle_actions` flag matches `isinstance` — failing the invariant test catches the omission. |
| OQ-7 | Do we need a feature flag to roll back to the old code path? | NO. Per project policy "no backwards compat". Revert via `git revert` on the 8 commits. |

## 14.C Test plan (7-category coverage)

### 14.C.10.1 — `test_resume_service.py` (NEW, ~25-30 tests)

Tests against `LaunchResumeService` with a fake `provider_resolver`:

1. **Positive — RunPod sleeping pod, capacity available** — outcome.ok=True, attempts_made=1, message contains timing.
2. **Positive — RunPod running pod** — outcome.ok=True, no resume call, on_progress emits `verdict` event with status=running.
3. **Positive — single_node provider** — resolver returns None → outcome.availability="skipped", ok=True.
4. **Positive — legacy run (no PodMetadata)** — outcome.availability="running", ok=True, message mentions "legacy".
5. **Negative — missing API key** — resolver returns None → outcome.availability="skipped", ok=True with "RUNPOD_API_KEY not in env" message.
6. **Negative — pod GONE** — outcome.ok=False, availability="gone".
7. **Negative — capacity exhausted** — outcome.ok=False, capacity_exhausted=True, availability="sleeping_resume_failed".
8. **Negative — probe failed** — outcome.ok=False, availability="probe_failed".
9. **Boundary — pod in PAUSED state** — same flow as EXITED/STOPPED (all map to SLEEPING_RESUMABLE).
10. **Invariants — frozen ResumeOutcome, frozen ResumeProgress** — assignment raises.
11. **Invariants — on_progress called in correct order** — list-collector callback yields [`probing`, `verdict`, `resuming`, `resumed`] for the happy path.
12. **Invariants — on_progress=None never raises** — REST flow path (no callback).
13. **Dependency errors — resolver raises** — service surfaces the exception verbatim (no swallowing).
14. **Regressions — ResumeOutcome.message format unchanged** — pin "Pod resumed in X.Xs (N attempt(s))" exact format.
15. **Logic-specific — `_resolve_lifecycle_provider("runpod")` with valid env returns ITerminalActionProvider conformer**.
16. **Logic-specific — `_resolve_lifecycle_provider("lambda")` (unknown) returns None**.

### 14.C.10.2 — `test_launch_service_resume_pod.py` (UPDATED, ~10 tests)

Existing 8 test classes get adapted: instead of mocking `PodAvailabilityProbe` directly, they mock `LaunchResumeService.resume()` to return a scripted `ResumeOutcome` and assert the REST adapter's translation.

### 14.C.10.3 — `test_run_resume.py` (NEW or extension to existing CLI test, ~6 tests)

CLI adapter tests:

1. **Positive — happy path** — typer.echo gets the right lines in the right order; no `die`.
2. **Negative — GONE** — `die()` raised with hint mentioning `run restart`.
3. **Negative — capacity exhausted** — `die()` with capacity hint.
4. **Negative — probe failed** — `die()` with retry hint.
5. **Boundary — legacy run** — silent return, no echoes.
6. **Boundary — `--skip-pod-probe` set** — service not called.

### 14.C.10.4 — `test_pod_availability.py` (UPDATED, ~5 tests)

Existing tests construct `PodAvailabilityProbe(query_pod=...)` — they get the lazy default mapper (RunPod's). Add 2 new tests:

* **Boundary — explicit status_mapper injected** — pass a fake mapper that returns hardcoded values; assert probe uses it.
* **Invariants — default mapper is lazily imported** — import `pod_availability` does NOT trigger import of `src.providers.runpod`. Test by reading `sys.modules` before/after.

## 14.C Verification

### Unit tests

```bash
# Phase 14.C new + migrated
pytest src/tests/unit/pipeline/launch/test_resume_service.py -v
pytest src/tests/unit/pipeline/launch/test_pod_availability.py -v
pytest src/tests/unit/api/services/test_launch_service_resume_pod.py -v
pytest src/tests/unit/cli/commands/ -v -k "resume"

# Existing decision-matrix tests still green (no provider-detection regression)
pytest src/tests/unit/runner/test_pod_terminator.py -v
pytest src/tests/unit/providers/runpod/runtime/ -v
pytest src/tests/unit/providers/training/ -v
```

Expected: ~30-50 new/migrated tests pass; zero regressions.

### Regression

```bash
pytest src/tests/unit/ -q --tb=line
# Pre-existing slim-venv failures (Phase 6.6 docker_image cleanup) unchanged.
# All other suites green. New count = pre-14.C count + ~20 net new tests.
```

### Cleanup grep checks

```bash
# String-checks eliminated (2 of 17)
! grep -n 'provider != "runpod"' src/api/services/launch_service.py
! grep -n 'provider != "runpod"' src/cli/commands/run.py

# Map relocated
! grep -n "_RUNPOD_STATUS_MAP" src/pipeline/launch/pod_availability.py
grep -n "_RUNPOD_STATUS_MAP" src/providers/runpod/training/provider.py

# Single source of truth for resume orchestration
test -f src/pipeline/launch/resume_service.py
```

### Contract verification (manual / via repl)

```python
from pathlib import Path
from src.pipeline.launch.resume_service import LaunchResumeService, ResumeOutcome

# Invariant 1: service produces ResumeOutcome for a real run dir.
svc = LaunchResumeService()
outcome = svc.resume(Path("/path/to/run"))
assert isinstance(outcome, ResumeOutcome)

# Invariant 2: provider gating uses Protocol, not strings.
from src.providers.training.interfaces import ITerminalActionProvider
from src.providers.single_node.training.provider import SingleNodeProvider
assert not isinstance(SingleNodeProvider(...), ITerminalActionProvider)
# → service treats single_node as skipped without ever string-comparing.
```

### Manual smoke

1. `ryotenkai run resume <run_dir>` on a real RunPod sleeping pod → progress lines match pre-14.C UX byte-for-byte.
2. Web UI "Resume" button → `POST /api/v1/runs/<id>/resume-pod` → JSON response shape unchanged (`{availability, ok, message}`).
3. `ryotenkai run resume <run_dir>` on a single_node run → silent return (Phase 11.C-2 behaviour preserved; no "skipping pod probe" output because we hit the resolver-returns-None branch first now, not the api-key branch).

**Note on (3)**: This is a SUBTLE UX change. Pre-14.C, single_node + RUNPOD_API_KEY missing → CLI prints `(skipping pod probe: RUNPOD_API_KEY not in env)`. Post-14.C, single_node short-circuits before the API-key check → silent return. Documented in plan; not regression-tested because the message is informational. CLI smoke confirms.

## 14.C Effort + rollout

| Step | Effort |
|---|---|
| `LaunchResumeService` + `ResumeOutcome` + `ResumeProgress` skeleton (§ 14.C.10.1 lines 1-9) | 2h |
| `_resolve_lifecycle_provider` factory + `RunPodProvider.from_resume_metadata` classmethod | 2h |
| Service body — probe, resume, outcome translation | 2h |
| `_RUNPOD_STATUS_MAP` move + `PodAvailabilityProbe` `status_mapper` injection | 1h |
| REST adapter rewrite (`launch_service.py::resume_pod_for_run`) | 30min |
| CLI adapter rewrite (`run.py::_resume_pod_if_needed`) | 1h |
| Unit tests for service (~25 cases × 7-cat) | 4h |
| Test migrations (REST tests, CLI tests, probe tests) | 3h |
| Manual smoke + regression | 1h |
| Documentation + commit messages | 1h |
| Code review buffer | 3h |
| **Total** | **~20h (~2.5 engineer-days)** |

Single PR with the 8 commits in § 14.C above. Land before starting the merged 14.D+F plan (status map relocation matters for 14.D's env-hardcode cleanup).

## 14.C Migration & rollback

**Migration:** No operator-facing changes. The service preserves CLI progress output byte-for-byte (modulo the silent-skip on single-node-without-API-key — see Manual Smoke note 3). REST `ResumePodResponse` shape unchanged. No env contract changes.

**Rollback:** Revert the 8 commits in reverse order. Commits 1-3 are purely additive (no behaviour change). Commits 4-5 are the breaking refactor — they're atomic per consumer, so each can be reverted independently if needed. Commits 6-8 are mechanical (cleanup + docs).

**Operator failure mode if rolled out incorrectly:**
* CLI: bug in adapter → typer.echo gets wrong format → operator sees garbled output but the pipeline still works (the service's outcome is what controls the actual pod state). Caught by smoke test.
* REST: bug in adapter → JSON shape mismatch → Web UI shows error but the resume itself happened (or didn't) per the service's logic. Caught by REST contract test.

No DB migrations, no config-schema changes, no backwards-compat shims.
