# Phase 14.B — Pod Terminator Provider Extraction

> Status: **DONE** — implemented in this worktree (single PR, 8 commits worth of atomic edits applied in-place).
> Author: daniil + agent
> Date: 2026-04-27
> Worktree: `nice-jepsen-07d789`
> Predecessors: Phase 14.A (commit `1819cc4`)
> Successor: Phase 14.C (LaunchResumeService unification) — unblocked
> Migration policy: NO BACKWARDS COMPATIBILITY — single-PR full cutover
> Effort: ~3.5 engineer-days estimated; actual ~1 day with agent assistance
> Risk: Medium — touched the code path that runs on EVERY terminal transition; failure mode is "pod doesn't pause/terminate" which is a money/data correctness bug
> Out of: Phase 14 roadmap in [`phase-14a-provider-abstraction.md`](./phase-14a-provider-abstraction.md)
>
> ### What landed
>
> 1. **New `src/runner/runtime/`** package: `lifecycle_client.py`
>    (Protocol + `LifecycleActionResult`, 21 tests) +
>    `provider_registry.py` (env-driven resolver +
>    `BootstrapConfigError` + helpers, 29 tests).
> 2. **New `src/providers/runpod/runtime/lifecycle_client.py`** —
>    RunPod GraphQL transport (verbatim move of `_call_mutation`,
>    `_ALREADY_GONE_RE`, `DEFAULT_RUNPOD_GRAPHQL_URL` from
>    `pod_terminator.py`), 26 tests.
> 3. **New `src/providers/single_node/runtime/lifecycle_client.py`** —
>    `NoOpPodLifecycleClient`, 12 tests.
> 4. **`src/runner/pod_terminator.py` rewrite** — dropped
>    `httpx`/`re`/`os` imports, dropped `DEFAULT_RUNPOD_GRAPHQL_URL`,
>    `_ALREADY_GONE_RE`, `_call_mutation`, `_call_terminate`,
>    `_call_stop`. `PodTerminator.__init__` now takes
>    `client: IPodLifecycleClient`, `resource_id`, `volume_kind`,
>    `keep_on_error`. `decide_and_act` no longer takes `env=...`.
>    Added `provider` + `attempts_made` fields to `pod_stop_attempt`
>    event payload.
> 5. **`src/runner/main.py::_lifespan` rewrite** — single env-reading
>    seam via `resolve_lifecycle_client_from_env(os.environ)` +
>    helper functions for volume kind / keep-on-error / resource id.
>    `BootstrapConfigError` propagates so uvicorn exits non-zero.
> 6. **Test migration** — `test_pod_terminator.py` rewritten to use
>    `_FakeLifecycleClient` (no httpx); pre-14.B
>    "missing creds → SKIPPED" tests removed (gated at lifespan boot
>    now). `test_pod_terminator_retry.py` adapted with same
>    `_FakeClient`. `test_lifespan_journal_wiring.py` now sets the
>    runtime provider env. `runner_client` / `runner_client_real`
>    conftest fixtures default to `single_node` so existing tests
>    keep booting.
> 7. **New `test_main_lifespan_bootstrap.py`** — 15 tests covering
>    happy paths (runpod + single_node), bootstrap failures (4
>    rows), volume_kind/keep_on_error clamping, app.state wiring,
>    Protocol invariants, regression (pod_terminator no longer
>    exports GraphQL constants / imports httpx), and snapshot
>    independence (post-boot env mutation doesn't change
>    terminator state).
>
> Test result: **103 new tests; 0 regressions.** Full
> `src/tests/unit/runner/` suite: 376 passed (up from 367 pre-14.B).
> Provider tests unchanged at 593 passed / 25 pre-existing failures
> (all from Phase 6.6 `docker_image` cleanup, unrelated to 14.B).

## 14.B Context — why THIS sub-phase is next

`src/runner/pod_terminator.py` today (post-Phase-11.E) carries a
**RunPod-shaped GraphQL transport** as a private implementation
detail: `_call_mutation`, `_call_terminate`, `_call_stop`,
`DEFAULT_RUNPOD_GRAPHQL_URL`, `_ALREADY_GONE_RE`. That code lives in
the shared `src/runner/` namespace because — historically — the
runner only ever needed to talk to RunPod. Phase 14.A added the
abstraction surface (`ITerminalActionProvider` + `VolumeKind` +
`RYOTENKAI_RUNTIME_PROVIDER`); Phase 14.B is the first sub-phase
that **uses** that surface to evict the leak.

Concretely, after 14.B:

* All RunPod GraphQL transport lives in
  `src/providers/runpod/runtime/lifecycle_client.py`.
* Single_node has a parity `NoOpPodLifecycleClient` in
  `src/providers/single_node/runtime/lifecycle_client.py`.
* The runner reads `RYOTENKAI_RUNTIME_PROVIDER` **once at lifespan**
  and never again. Provider-specific env (`RUNPOD_API_KEY`,
  `RUNPOD_POD_ID`, `RUNPOD_VOLUME_KIND`, `RUNPOD_KEEP_ON_ERROR`)
  is read in the same place.
* `PodTerminator` becomes a **pure decision-and-dispatch** class
  whose only knowledge of "RunPod vs single_node" is the
  `IPodLifecycleClient` it was constructed with.
* `decide_and_act` no longer takes `env=...`; lifespan-static
  config rides on `PodTerminator.__init__`.
* Bootstrap fails fast on misconfigured env (`BootstrapConfigError`),
  not 4 hours later as a silent `outcome=skipped`.

The hardest decision in this phase is reconciling Phase 14.A's
**sync** `ITerminalActionProvider` (Mac-side, `Result[None, ProviderError]`)
with the runner's **async, retry-aware, idempotent-marker-detecting**
needs. § 14.B.1.1 below documents the locked outcome: the two
Protocols stay separate by design, with a CI-checked invariant
linking them.

## 14.B Architectural decisions (locked)

### 14.B.1.1 Sync vs async — **Async (Option A) — LOCKED**

A **new** runner-side Protocol `IPodLifecycleClient` lives in
`src/runner/runtime/lifecycle_client.py` with **async** methods. It
does **not** subclass, alias, or import `ITerminalActionProvider`.

**Rationale (in order of importance):**

1. **The runner already owns an event loop** (FastAPI/uvicorn).
   Today's `_call_mutation` is async + uses `httpx.AsyncClient`
   because the lifespan + `decide_and_act` chain is async. Calling
   `asyncio.run()` from inside a running loop raises `RuntimeError`.
   Wrapping with `loop.run_in_executor` would re-introduce the
   threadpool/cancellation-leak class of bugs Phase 11.B
   specifically avoided.
2. **Phase 14.A's `ITerminalActionProvider` is Mac-side.** It's
   called from `LaunchResumeService` / Mac-side cleanup paths that
   today are sync. Forcing it async would have rippled into 5+
   Mac-side callers and re-opened the Phase 14.A review.
3. **Audience separation is a feature, not a leak.** Mac-side
   `RunPodProvider.terminate()` calls the SDK and returns
   `Result[None, ProviderError]`; runner-side
   `RunPodPodLifecycleClient.terminate()` calls GraphQL via httpx
   + has retry knobs + emits idempotency markers. Different
   consumers, different cross-cuts.
4. **Two-Protocols-for-one-concept** is mitigated by:
   * Both share method names (`terminate` / `pause` / `resume`).
   * Both are `@runtime_checkable`.
   * Both share `VolumeKind` from `src/providers/training/interfaces.py`.
   * A docstring on each cross-references the other.
   * **CI-checked invariant** (§ 14.B.8.2): for every provider that
     conforms to `ITerminalActionProvider`, there exists a matching
     `IPodLifecycleClient` impl reachable from the registry.

**Why not Option B (make `ITerminalActionProvider` async):** Phase
14.A is shipped sync. Changing it now violates 14.A's "no behaviour
change" guarantee, forces every Mac-side caller to thread-pool, and
re-opens the 14.A review.

**Why not Option C (sync + asyncio.run):** `asyncio.run()` from
inside an active uvicorn loop is a runtime error.
`asyncio.get_event_loop().run_until_complete()` is deprecated.
`nest_asyncio` is banned per repo style.

### 14.B.1.2 Result shape — **`LifecycleActionResult` dataclass (NOT `Result[T, E]`)**

```python
@dataclass(frozen=True)
class LifecycleActionResult:
    outcome: str                            # one of PodTerminalOutcome strings
    attempts_made: int                      # 1..max_attempts (0 for NoOp)
    last_error: str | None = None           # repr of last exception or HTTP body snippet
    raw_response_excerpt: str | None = None # first ~300 chars of response body for forensics
```

**Rationale:**
* `decide_and_act` already returns a flat dict with `decision` +
  `action` strings. Mapping `Result[None, ProviderError]` into
  action-stage `PodTerminalOutcome` would require an extra
  translation layer.
* The action-stage outcome vocabulary
  (`TERMINATED` / `ALREADY_TERMINATED` / `STOPPED` / `ALREADY_STOPPED`
  / `FAILED` / `SKIPPED`) is a **closed enum that already exists**
  on `pod_terminator.PodTerminalOutcome`.
* Idempotency markers ("already gone") are not errors on the runner
  side — they're a **successful** outcome of "intent satisfied".
  `Result` would lose this nuance unless wrapped as
  `Ok(AlreadyGone)`.
* Telemetry consumers (`pod_stop_attempt` event) already grep for
  outcome strings; preserving the vocabulary keeps dashboards
  unchanged.

The Mac-side `ITerminalActionProvider` keeps `Result[None, ProviderError]`
because Mac-side cleanup callers already speak the `Result` dialect.

### 14.B.1.3 NoOp client — `outcome="skipped"` for all actions

Single_node case: `NoOpPodLifecycleClient` returns
`LifecycleActionResult(outcome="skipped", attempts_made=0,
last_error=None)` for `terminate` / `pause` / `resume`.

`PodTerminalOutcome.SKIPPED` already exists for "decision wanted to
act but couldn't". Mapping single_node's "no cloud lifecycle to act
on" into the same bucket means downstream operators see one
explicit branch (`outcome == "skipped"`) regardless of *why* it
skipped (creds missing vs no provider lifecycle). The
`pod_stop_attempt` event payload includes `provider_name` so
dashboards can disambiguate when needed.

### 14.B.1.4 Env reads → lifespan-static, NOT per-call

`volume_kind`, `keep_on_error`, `resource_id` move to
`PodTerminator.__init__` constructor params. `decide_and_act` no
longer takes `env=...`. `terminal_state`, `heartbeat`, `bus_publish`
remain per-call.

**Rationale:**
* Volume kind / keep-on-error / resource id are **static for the
  runner's lifetime** — the pod doesn't change its volume kind
  mid-run.
* Reading env at call time is the **leak** Phase 14 attacks.
  Hoisting to constructor proves the lifespan is the single
  env-reading seam.
* `terminal_state` MUST be per-call (FSM transitions decide it).
* `heartbeat` and `bus_publish` STAY per-call to preserve Phase
  11.E's `run_terminal_hook` shape and to keep the test surface
  stable.

### 14.B.1.5 Registry placement — `src/runner/runtime/`

New package layout:

```
src/runner/runtime/
  __init__.py              (empty, package marker)
  lifecycle_client.py      (Protocol + LifecycleActionResult)
  provider_registry.py     (env-driven resolver + BootstrapConfigError)

src/providers/runpod/runtime/
  __init__.py
  lifecycle_client.py      (RunPodPodLifecycleClient)

src/providers/single_node/runtime/
  __init__.py
  lifecycle_client.py      (NoOpPodLifecycleClient)
```

`runtime/` (vs `runner/`-root) signals "things the runner needs to
bootstrap from env at lifespan, distinct from API and FSM
machinery". Mirrors the Mac-side
`src/providers/<name>/training/provider.py` layout — adding
`runtime/` as a third sibling for the in-pod control plane is
consistent.

### 14.B.1.6 Verb vocabulary — `pause` not `stop` at the Protocol level

Runner-side Protocol method names: `terminate` / `pause` / `resume`
(matches Phase 14.A). `PodTerminator` continues to emit the
operator-facing strings `STOPPED` / `STOPPED_FOR_RESUME` /
`STOPPED_FOR_RESUME_SHORT_GRACE` because those are **outcomes** not
**verbs** — preserves Phase 11.B/E telemetry vocabulary unchanged.
RunPod's GraphQL still calls `podStop`/`podResume` — that's an
implementation detail of `RunPodPodLifecycleClient.pause()`.

### 14.B.1.7 Bootstrap validation — fail-fast in lifespan

`resolve_lifecycle_client_from_env` raises `BootstrapConfigError`
(new exception class in `provider_registry.py`) on:
* `RYOTENKAI_RUNTIME_PROVIDER` unset or empty.
* Value not in registered set.
* Value=`runpod` but `RUNPOD_API_KEY` or `RUNPOD_POD_ID` missing.

The lifespan **catches it, logs structured fatal, and re-raises so
uvicorn exits non-zero.** NO graceful degradation.

**Rationale:**
* `RYOTENKAI_RUNTIME_PROVIDER` is an explicit bootstrap signal.
  If it's absent, the Mac launcher is broken — we want a loud
  failure at boot, not silent SKIPPED outcomes 4 hours later.
* Today's behaviour silently turns into `SKIPPED` at terminal
  hook time, which is a known operator-dashboard pain point.
* Phase 14.B is no-backcompat — we tighten the contract.

### 14.B.1.8 Old `SKIPPED` branch removal

The current `decide_and_act` branch "creds missing → SKIPPED" goes
away. With env-read moved to lifespan, missing creds = boot
failure; if we got past boot, creds are present and the dispatch
always either succeeds, returns `ALREADY_*`, or `FAILED`. The
single_node case still returns `outcome="skipped"` from the
`NoOpPodLifecycleClient` — the SKIPPED string is still emitted, but
as an explicit "no-op-by-design" outcome from the provider, not as
an error path. Operators reading the event stream see the same
vocabulary they're used to.

## 14.B Scope

### IN-scope

| Item | Description |
|---|---|
| Move `_call_mutation`, `_call_terminate`, `_call_stop` out of `pod_terminator.py` | ~100 LOC migration into `RunPodPodLifecycleClient` |
| Move `DEFAULT_RUNPOD_GRAPHQL_URL`, `_ALREADY_GONE_RE` | Constants follow the code that owns them |
| New `IPodLifecycleClient` Protocol (async, runtime_checkable) | Runner-side abstraction |
| New `LifecycleActionResult` dataclass | Per-call result shape |
| New `RunPodPodLifecycleClient` impl | RunPod GraphQL transport, lifted verbatim from `pod_terminator.py` |
| New `NoOpPodLifecycleClient` impl | single_node parity |
| New `provider_registry.py` with `resolve_lifecycle_client_from_env` | env-driven dispatch + `BootstrapConfigError` |
| `_lifespan` rewrite | Build client first, pass into `PodTerminator(client=..., volume_kind=..., keep_on_error=..., resource_id=...)` |
| `PodTerminator.__init__` signature change | Drop graphql/http params, add client + lifespan-static config |
| `PodTerminator.decide_and_act` signature change | Drop `env=...`, drop inline GraphQL |
| Test migration | GraphQL/HTTP tests move to `tests/unit/providers/runpod/runtime/test_lifecycle_client.py` |
| 7-cat coverage for new modules | See § 14.B.8 |

### OUT-of-scope (deferred)

| Item | Phase |
|---|---|
| `LaunchResumeService` extraction (CLI + REST resume duplication) | 14.C |
| `pod_availability.py` provider-agnostic refactor | 14.C |
| Removing `provider == "runpod"` string checks from `src/api/services/launch_service.py` | 14.D |
| Collapsing `prepare_training_script_hooks` into `required_runtime_env_vars` | 14.D |
| Pipeline-side `_build_job_env` cleanup (still hardcodes `RUNPOD_VOLUME_KIND="persistent"` for all) | 14.D |
| Renaming `pod_terminator.py` → `terminal_hook.py` (cosmetic) | 14.E |
| Mac ↔ runner-side Protocol unification (NEVER — explicit non-goal) | n/a |

## 14.B New abstractions (signatures only)

```python
# src/runner/runtime/lifecycle_client.py

@dataclass(frozen=True)
class LifecycleActionResult:
    outcome: str                                 # one of PodTerminalOutcome strings
    attempts_made: int                           # 1..max_attempts (0 for NoOp)
    last_error: str | None = None
    raw_response_excerpt: str | None = None      # ≤ 300 chars for forensics

@runtime_checkable
class IPodLifecycleClient(Protocol):
    """Runner-side equivalent of :class:`ITerminalActionProvider`.

    See Phase 14.B § 1.1 for why these are two distinct Protocols.
    Cross-reference: src.providers.training.interfaces.ITerminalActionProvider.
    """
    @property
    def provider_name(self) -> str: ...

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult: ...
    async def pause(self, *, resource_id: str) -> LifecycleActionResult: ...
    async def resume(self, *, resource_id: str) -> LifecycleActionResult: ...
```

```python
# src/runner/runtime/provider_registry.py

class BootstrapConfigError(RuntimeError):
    """Raised at lifespan when env config can't satisfy the runner bootstrap."""

def resolve_lifecycle_client_from_env(env: Mapping[str, str]) -> IPodLifecycleClient: ...
def resolve_volume_kind_from_env(env: Mapping[str, str]) -> str: ...      # "persistent"|"network"
def resolve_keep_on_error_from_env(env: Mapping[str, str]) -> bool: ...
def resolve_resource_id_from_env(env: Mapping[str, str]) -> str: ...      # RUNPOD_POD_ID or ""
```

```python
# src/providers/runpod/runtime/lifecycle_client.py

DEFAULT_RUNPOD_GRAPHQL_URL: Final = "https://api.runpod.io/graphql"
_ALREADY_GONE_RE: Final = re.compile(...)   # moved verbatim

class RunPodPodLifecycleClient:
    def __init__(
        self, *,
        api_key: str,
        graphql_url: str = DEFAULT_RUNPOD_GRAPHQL_URL,
        request_timeout: float = 30.0,
        max_attempts: int = 3,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None: ...

    @property
    def provider_name(self) -> str: return PROVIDER_RUNPOD

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult: ...
    async def pause(self, *, resource_id: str) -> LifecycleActionResult: ...
    async def resume(self, *, resource_id: str) -> LifecycleActionResult: ...
```

```python
# src/providers/single_node/runtime/lifecycle_client.py

class NoOpPodLifecycleClient:
    @property
    def provider_name(self) -> str: return PROVIDER_SINGLE_NODE

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
        return LifecycleActionResult(outcome=PodTerminalOutcome.SKIPPED, attempts_made=0)

    async def pause(self, *, resource_id: str) -> LifecycleActionResult:    # same shape
    async def resume(self, *, resource_id: str) -> LifecycleActionResult:   # same shape
```

```python
# src/runner/pod_terminator.py — modified PodTerminator constructor

class PodTerminator:
    def __init__(
        self, *,
        client: IPodLifecycleClient,
        resource_id: str,
        volume_kind: str,                       # "persistent" | "network"
        keep_on_error: bool,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        grace_base_seconds: float | None = None,
        # ... grace + heartbeat retry knobs unchanged ...
    ) -> None: ...

    async def decide_and_act(
        self, *,
        terminal_state: str,
        heartbeat: MacHeartbeat,
        bus_publish: Callable[[str, dict[str, Any]], Any],
        # NOTE: env=... PARAMETER REMOVED
    ) -> dict[str, Any]: ...
```

### Edge-case table for `provider_registry`

| Env state | Outcome |
|---|---|
| `RYOTENKAI_RUNTIME_PROVIDER` unset | `BootstrapConfigError` → uvicorn exits non-zero |
| value = `""` | Treated as unset → `BootstrapConfigError` |
| value = `" runpod "` (whitespace) | NOT stripped — explicit `BootstrapConfigError`. Operators must set it cleanly. Test pins this. |
| value = `"runpod"` + `RUNPOD_API_KEY` missing | `BootstrapConfigError("...requires RUNPOD_API_KEY")` |
| value = `"runpod"` + `RUNPOD_POD_ID` missing | `BootstrapConfigError("...requires RUNPOD_POD_ID")` |
| value = `"single_node"` + nothing else | OK — returns `NoOpPodLifecycleClient()` |
| value = `"lambda"` (unregistered) | `BootstrapConfigError("...not registered...")` |
| `RUNPOD_VOLUME_KIND` invalid (e.g. `"foo"`) | Falls back to `"persistent"` (warning log, NOT fatal) — preserves existing 11.B behaviour |
| `RUNPOD_KEEP_ON_ERROR` invalid (e.g. `"yes"`) | Treated as `false` (only literal `"true"` enables it) — preserves existing behaviour |

## 14.B Critical files to modify

**NEW:**
- `src/runner/runtime/__init__.py` — empty package marker
- `src/runner/runtime/lifecycle_client.py` — `IPodLifecycleClient` + `LifecycleActionResult`
- `src/runner/runtime/provider_registry.py` — resolver + `BootstrapConfigError` + env helpers
- `src/providers/runpod/runtime/__init__.py` — empty package marker
- `src/providers/runpod/runtime/lifecycle_client.py` — `RunPodPodLifecycleClient` (~150 LOC)
- `src/providers/single_node/runtime/__init__.py` — empty package marker
- `src/providers/single_node/runtime/lifecycle_client.py` — `NoOpPodLifecycleClient` (~30 LOC)

**MODIFIED:**
- `src/runner/pod_terminator.py` — drop GraphQL/regex/http_client_factory; add `client`, `resource_id`, `volume_kind`, `keep_on_error` constructor params; drop `env=` from `decide_and_act`
- `src/runner/main.py::_lifespan` — call `resolve_lifecycle_client_from_env` first; build `PodTerminator(client=..., volume_kind=..., keep_on_error=..., resource_id=...)`; let `BootstrapConfigError` re-raise

**MODIFIED MINIMALLY (verify only):**
- `src/pipeline/stages/managers/deployment/training_launcher.py::_build_job_env` — confirm Phase 14.A's `provider.required_runtime_env_vars()` already wires `RYOTENKAI_RUNTIME_PROVIDER`. If not, add the one line. **No other 14.D-scope changes here.**

**REUSE (do NOT modify):**
- `src/providers/runpod/training/api_client.py`,
  `src/providers/runpod/training/sdk_adapter.py` — Mac-side
  transport. Untouched. Phase 14.B is runner-side only.
- `src/runner/heartbeat.py`,
  `src/runner/cancellation_telemetry.py`,
  `src/runner/main.py::_periodic_journal_health_check` —
  Phase 14.E touches these, not us.
- `src/constants.py::RUNTIME_PROVIDER_ENV_VAR`,
  `PROVIDER_RUNPOD`, `PROVIDER_SINGLE_NODE` — landed in Phase 14.A.
  Just import.

**TESTS — NEW:**
- `src/tests/unit/providers/runpod/runtime/test_lifecycle_client.py` —
  GraphQL/HTTP-level coverage migrated from `test_pod_terminator.py` (decision matrix tests stay in place)
- `src/tests/unit/providers/single_node/runtime/test_lifecycle_client.py` —
  NoOp returns SKIPPED, all three methods, 7-cat
- `src/tests/unit/runner/runtime/test_provider_registry.py` —
  edge-case table coverage + `BootstrapConfigError` + invariant cross-check
- `src/tests/unit/runner/test_pod_terminator_dispatch.py` (NEW or
  rename existing) — `decide_and_act` dispatching to mock client,
  no GraphQL knowledge
- `src/tests/unit/runner/test_main_lifespan_bootstrap.py` —
  `_lifespan` calls registry, fail-fast on bad env, success path

**TESTS — MIGRATED (chunks moved):**
- `src/tests/unit/runner/test_pod_terminator.py` — keep matrix +
  retry + grace tests; **move** `_call_mutation` / `_call_terminate`
  / `_call_stop` test classes to the new
  `test_lifecycle_client.py`. Net LOC about flat.

## 14.B Migration order — single PR, atomic commits

| # | Commit | Why this order |
|---|---|---|
| 1 | `feat(runner): add IPodLifecycleClient Protocol + LifecycleActionResult` (3.2 only) | Land the abstraction first — provable nothing breaks because nothing imports it. |
| 2 | `feat(providers/runpod): add RunPodPodLifecycleClient with GraphQL transport` | Now the impl exists. Tests for it can pass. Existing `pod_terminator.py` is untouched. |
| 3 | `feat(providers/single_node): add NoOpPodLifecycleClient` | Trivial; symmetry. |
| 4 | `feat(runner): add provider_registry with BootstrapConfigError` | Wires (1)+(2)+(3) together but doesn't change runtime yet — registry is dead code until lifespan calls it. |
| 5 | `refactor(runner): switch PodTerminator to IPodLifecycleClient` — single combined commit covering: `pod_terminator.py` constructor + `decide_and_act` + `_lifespan` + delete inline GraphQL + delete env reads | **The breaking change.** Atomic because `PodTerminator()` no-arg constructor stops working. |
| 6 | `test(runner): migrate GraphQL/HTTP tests to providers/runpod/runtime/` | After (5) lands, the test surface is split. |
| 7 | `test(runner): add provider_registry tests + lifespan bootstrap tests` | New coverage for new code. |
| 8 | `docs(plans): mark Phase 14.B DONE; update predecessor link in 14.C plan stub` | Bookkeeping. |

PR is **NOT** squash-merged — Phase 14 carries this convention from
14.A so the migration history is preserved.

## 14.B Risks (3 deepsink iterations)

### Iteration 1 — initial sweep

| ID | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R-1 | **Lifespan refactor breaks startup** — new constructor signature, new env-validation paths. If anything goes wrong the runner doesn't come up at all. | H | M | Commit-by-commit migration (§ above); commit 5 is reviewable as one atomic refactor. New `test_main_lifespan_bootstrap.py` covers happy path + each `BootstrapConfigError` branch. |
| R-2 | **Two-Protocols-for-one-concept drift** — RunPod author updates `ITerminalActionProvider.terminate` semantics but forgets `IPodLifecycleClient.terminate` (or vice versa). | M | M | CI invariant test (§ 14.B.8.2): for every concrete `IGPUProvider` impl with `caps.supports_lifecycle_actions=True`, there exists a registered `IPodLifecycleClient` reachable from the registry. Failing test = blocker. Docstrings cross-reference. |
| R-3 | **`_lifespan` env-read order matters** — must call registry BEFORE constructing terminator. Reverse order → `PodTerminator()` no-arg = TypeError. | M | L | Test pins the order; `BootstrapConfigError` from registry surfaces in fixture before terminator construction. |
| R-4 | **Dropping `_ALREADY_GONE_RE` from `__all__`** — no consumer should import it from `pod_terminator`, but if any test does, it breaks silently. | L | L | grep `_ALREADY_GONE_RE` repo-wide before merge. Only one importer = the test that's being moved. |

### Iteration 2 — deepsink on Iteration 1

R-1 is the highest-impact concern. Three hardenings considered:

1. **Wrap registry in try/except inside lifespan, swallow + log,
   fall through to a dummy `NoOpPodLifecycleClient`.** Rejected:
   defeats the fail-fast contract; same silent SKIPPED outcomes
   we're trying to remove.
2. **Add a CLI tool that validates env without booting the runner**
   (`python -m src.runner.runtime.provider_registry --check-env`).
   Deferred: nice-to-have, doesn't gate this phase.
3. **Add a smoke test that builds the lifespan with realistic env
   fixtures.** ✅ Picked. `test_main_lifespan_bootstrap.py` covers
   four matrices: (provider=runpod, full env), (provider=single_node,
   minimal env), (provider=runpod, missing API_KEY → fail),
   (provider unset → fail).

R-2 deepsink — three solutions for the drift problem:

1. **Drop `IPodLifecycleClient`**, reuse `ITerminalActionProvider`
   with `asyncio.run` shims. Rejected per § 14.B.1.1.
2. **Subclass `IPodLifecycleClient` from `ITerminalActionProvider`**
   to enforce method-name parity. Rejected: Protocols don't
   inherit cleanly across sync/async; the subclassing buys little
   and locks in coupling we may regret.
3. **Keep both, add CI invariant test.** ✅ Picked. The test
   (§ 14.B.8.2) iterates over the `_REGISTRY` dict in
   `provider_registry.py`, asserts each registered builder produces
   a client whose `provider_name` matches a Mac-side provider whose
   capabilities match.

### Iteration 3 — cross-cutting concerns

| ID | Concern | Resolution |
|---|---|---|
| R-5 | **`PodTerminalOutcome` import location** — runtime impls import the enum from `src.runner.pod_terminator`. Future Phase 14.E may rename `pod_terminator.py` → `terminal_hook.py` and break those imports. | Accepted minor cost. Phase 14.E will do an `__all__`-driven sweep; renaming a single import path across 3 files is a 30-second sed. Not worth a separate move now. |
| R-6 | **`http_client_factory` test injection** — tests today pass `http_client_factory=...` into `PodTerminator(...)` to mock `httpx.AsyncClient`. After 14.B, that injection point lives on `RunPodPodLifecycleClient`, NOT `PodTerminator`. | Test migration: tests that mock HTTP move to `test_lifecycle_client.py` and inject via the new client. Tests that exercise the matrix mock the **client** (a `IPodLifecycleClient` Protocol stub), not HTTP. Cleaner separation. |
| R-7 | **Phase 14.A's `ITerminalActionProvider.resume`** — Mac-side. Phase 14.B's `IPodLifecycleClient.resume` — runner-side, NEVER called from inside the pod (Mac wakes the pod, pod doesn't wake itself). Why implement it? | Symmetry. The Protocol is closed; `resume` exists on `ITerminalActionProvider` so it exists on `IPodLifecycleClient`. RunPod's impl is one more line; keeps future use (e.g. self-resume after capacity reservation) cheap. |
| R-8 | **`outcome="resumed"` / `"already_running"` strings** for `IPodLifecycleClient.resume` — these are NOT in `PodTerminalOutcome` enum today. | Decision: do NOT add to the enum in this phase (would broaden the public vocabulary unnecessarily). `RunPodPodLifecycleClient.resume()` returns these as raw strings; if/when a caller uses resume from inside the pod, Phase 14.E or beyond can add them. |
| R-9 | **Single_node bootstrap with no `RUNPOD_*` env** — registry must NOT raise on `single_node` even when RunPod env is absent. | `_build_single_node_client` doesn't read RunPod env. Test fixture (`test_provider_registry.py`) covers (provider=single_node, no RunPod env) → returns `NoOpPodLifecycleClient`. |
| R-10 | **Idempotent retry semantics** — `_call_mutation` today retries on connection errors and "already gone" markers; if we move it verbatim, the retry loop runs on the new client. Make sure the test that validates "retry exhaustion → FAILED" still passes after the move. | Migration is verbatim — same retry knobs, same telemetry. The test moves with the code. Net behaviour: bit-for-bit identical. |
| R-11 | **HTTP client lifecycle** — `httpx.AsyncClient` was previously short-lived (created per call). After move, should we long-lived a single `AsyncClient` on the lifecycle client for connection reuse? | KEEP short-lived per call (matches existing behaviour exactly). Connection-reuse is an optimization for Phase 14.E or beyond. Phase 14.B = pure refactor, no behaviour change. |
| R-12 | **PodTerminator owns retry knobs (`grace_base_seconds`, heartbeat retry budget) — but RunPodPodLifecycleClient owns transport retry knobs (`max_attempts`).** Two layers of retry. Could compound. | This is **already the case today** — the inline `_call_mutation` has its own attempt loop, and `decide_and_act` has its grace loop. Migration preserves the layering verbatim. Not a new risk. |

## 14.B Open questions (resolved)

| OQ | Question | Resolution |
|---|---|---|
| OQ-1 | Should `IPodLifecycleClient` extend `ITerminalActionProvider`? | **No.** § 14.B.1.1. Sync/async mismatch + audience separation. |
| OQ-2 | Should `LifecycleActionResult.outcome` be a `Literal[...]` of `PodTerminalOutcome` values? | **No.** Keep as `str` for now — the enum is non-exhaustive (resume adds new strings — R-8). Annotating as the enum would force every new resume outcome into the public vocabulary. Phase 14.E may revisit. |
| OQ-3 | Should the registry support runtime registration (plugin-style)? | **No.** Closed registry of two entries. Adding a third provider = code change in `provider_registry.py`. YAGNI. |
| OQ-4 | What about the `single_node` host failing to clean up its docker container at training end — should `NoOpPodLifecycleClient` invoke that? | **No.** SSH-based docker cleanup is Mac-side `provider.cleanup_pod()` (existing path). NoOp's job is "no cloud lifecycle to act on". |
| OQ-5 | Should we expose `IPodLifecycleClient` from a top-level `src/runner/__init__.py` for ergonomic imports? | **No.** Keep imports specific. Tests reference the full path. |
| OQ-6 | If `RUNPOD_VOLUME_KIND` is `"network"`, RunPod GraphQL `podStop` will fail. Should we surface that at boot? | **No.** Decision matrix in `decide_terminal_outcome` already routes `network` → terminate-only. The boot validation is about "can we talk to the API at all"; volume-kind-vs-action validation lives in the matrix. |
| OQ-7 | Provider constructor takes `api_key: str` plainly. Should this be a `SecretStr` from Pydantic? | **No.** Existing `RunPodAPIClient` and `pod_terminator._call_mutation` take it as `str`. Phase 14.B preserves the convention. Introducing `SecretStr` would be a Phase 14.F cosmetic. |

## 14.B Test plan (7-category coverage)

### 14.B.8.1 — `test_lifecycle_client.py` (NEW under `src/tests/unit/runner/runtime/`)

Tests the Protocol shape itself, NOT impls:

1. **Positive** — Protocol is `runtime_checkable`; a class with all three async methods + `provider_name` matches `isinstance(stub, IPodLifecycleClient)`.
2. **Negative** — Class missing `resume()` does NOT match.
3. **Boundary** — `LifecycleActionResult(outcome="x", attempts_made=0)` with default `last_error=None` constructs successfully and is frozen (mutation raises).
4. **Invariants** — `attempts_made >= 0` is enforced via dataclass post-init validation OR documented (decision: documented + linted via test).
5. **Dependency errors** — N/A (pure type module).
6. **Regressions** — `LifecycleActionResult.outcome` accepts every existing `PodTerminalOutcome` string.
7. **Logic-specific** — `raw_response_excerpt` length cap (≤ 300 chars) is enforced at construction (truncate, not raise).

### 14.B.8.2 — `test_provider_registry.py` (NEW)

The full edge-case matrix from § 14.B.3.3 + the **two-Protocol invariant**:

1. **Positive** — provider=`runpod`, full env → returns `RunPodPodLifecycleClient` with correct `api_key`.
2. **Positive** — provider=`single_node`, minimal env → returns `NoOpPodLifecycleClient`.
3. **Negative** — provider unset → `BootstrapConfigError`.
4. **Negative** — provider=`runpod`, no `RUNPOD_API_KEY` → `BootstrapConfigError` mentioning the var name.
5. **Negative** — provider=`runpod`, no `RUNPOD_POD_ID` → `BootstrapConfigError`.
6. **Negative** — provider=`lambda` (unregistered) → `BootstrapConfigError` listing known providers.
7. **Boundary** — provider=`""` → `BootstrapConfigError` (treated as unset).
8. **Boundary** — provider=`" runpod "` (with whitespace) → `BootstrapConfigError`.
9. **Invariants** — `resolve_volume_kind_from_env` clamps invalid values to `"persistent"`.
10. **Invariants** — `resolve_keep_on_error_from_env` returns `False` for everything except literal `"true"`.
11. **Logic-specific (R-2 invariant)** — for every entry in `_REGISTRY`, building the client succeeds and the client's `provider_name` matches the key. Forces parity if a provider author adds a new entry.
12. **Logic-specific (cross-Protocol invariant)** — for every Mac-side `IGPUProvider` impl in the providers package whose `get_capabilities().supports_lifecycle_actions` is `True`, there's a corresponding entry in the runner-side `_REGISTRY`. (Single test that imports both providers and asserts the link.)

### 14.B.8.3 — `test_pod_terminator_dispatch.py` (NEW or rename)

Tests `PodTerminator` with mock `IPodLifecycleClient` (a Protocol stub). NO httpx, NO GraphQL knowledge:

1. **Positive** — `decide_and_act` with terminal_state=`completed`, mac_alive=`False`, volume_kind=`persistent`, keep_on_error=`False` → calls `client.pause(resource_id=...)`. Outcome from client propagates.
2. **Negative** — terminal_state=`failed`, keep_on_error=`True` → no client call (`KEPT_ALIVE_FOR_DEBUG`).
3. **Boundary** — heartbeat retries: mac_alive flips between `True` and `False` across retry attempts → eventual decision matches the last observation.
4. **Invariants** — `decide_and_act` always returns dict with `decision`, `action`, `outcome` keys.
5. **Dependency errors** — `client.terminate(...)` raises generic `Exception` → outcome=`FAILED`, last_error captured in the bus event.
6. **Regressions** — pre-Phase-14.B `decide_and_act(env=...)` parameter is REMOVED — calling it raises `TypeError`. Pin the new signature.
7. **Logic-specific** — `client.pause()` returns `outcome="already_stopped"` → terminator publishes `pod_stop_attempt` with that outcome (idempotency preserved).

### 14.B.8.4 — `test_runpod_lifecycle_client.py` (MIGRATED from `test_pod_terminator.py`)

Receives the GraphQL/HTTP test classes from `test_pod_terminator.py` verbatim. Naming changes (`test_pod_terminator_*` → `test_lifecycle_client_*`) but body is unchanged:

1. **Positive** — `terminate(resource_id="pod-x")` calls `podTerminate` mutation with correct payload.
2. **Positive** — `pause(resource_id="pod-x")` calls `podStop`.
3. **Positive** — `resume(resource_id="pod-x")` calls `podResume`.
4. **Negative** — Network error → retried up to `max_attempts` then `outcome="failed"`.
5. **Boundary** — HTTP 5xx → retry; HTTP 4xx → no retry, `outcome="failed"`.
6. **Invariants** — `_ALREADY_GONE_RE` markers in response body → `outcome=ALREADY_TERMINATED` / `ALREADY_STOPPED`.
7. **Dependency errors** — `httpx.AsyncClient` raise on `__aenter__` → graceful `outcome="failed"`.
8. **Regressions** — pre-move test cases (mutation envelope shape, headers, timeout) all pass unchanged.
9. **Logic-specific** — `attempts_made` count in result reflects actual loop iterations; `raw_response_excerpt` truncated to 300 chars.

### 14.B.8.5 — `test_single_node_lifecycle_client.py` (NEW)

1. **Positive** — `terminate / pause / resume` all return `outcome="skipped"`, `attempts_made=0`.
2. **Negative** — N/A (NoOp can't fail).
3. **Boundary** — empty `resource_id=""` accepted.
4. **Invariants** — `provider_name` returns `"single_node"`.
5. **Dependency errors** — N/A (no transport).
6. **Regressions** — Pin: `NoOpPodLifecycleClient` does NOT import `httpx` or any RunPod module. (Static-analysis style test using `inspect.getsource`.)
7. **Logic-specific** — `last_error` is always `None` — single_node has no transient failure modes.

### 14.B.8.6 — `test_main_lifespan_bootstrap.py` (NEW)

End-to-end wiring tests using FastAPI `TestClient`:

1. **Positive** — env=`{RYOTENKAI_RUNTIME_PROVIDER=runpod, RUNPOD_API_KEY=k, RUNPOD_POD_ID=p}` → app starts; `app.state.terminator._client` is a `RunPodPodLifecycleClient`.
2. **Positive** — env=`{RYOTENKAI_RUNTIME_PROVIDER=single_node}` → app starts; `app.state.terminator._client` is `NoOpPodLifecycleClient`.
3. **Negative** — env empty → lifespan raises `BootstrapConfigError`; uvicorn would exit non-zero.
4. **Negative** — env=`{RYOTENKAI_RUNTIME_PROVIDER=runpod, RUNPOD_POD_ID=p}` (no API_KEY) → `BootstrapConfigError`.
5. **Boundary** — env=`{RYOTENKAI_RUNTIME_PROVIDER=runpod, RUNPOD_API_KEY=k, RUNPOD_POD_ID=p, RUNPOD_VOLUME_KIND=invalid}` → app starts; volume_kind clamped to `"persistent"` (pin via `app.state.terminator._volume_kind`).
6. **Invariants** — After successful startup, `app.state.terminator` and `app.state.heartbeat` are both wired and reachable via the existing health endpoint.
7. **Logic-specific** — `BootstrapConfigError` message includes both the env var name AND the list of known providers (operator-friendly).

### 14.B.8.7 — Migrated tests in `test_pod_terminator.py` (KEPT)

Decision matrix + grace + heartbeat retry tests stay where they
are. Only the GraphQL/HTTP tests move out (§ 14.B.8.4 receives
them).

## 14.B Verification

### Unit tests
```bash
# Phase 14.B new + migrated
pytest src/tests/unit/runner/runtime/test_lifecycle_client.py -v
pytest src/tests/unit/runner/runtime/test_provider_registry.py -v
pytest src/tests/unit/providers/runpod/runtime/test_lifecycle_client.py -v
pytest src/tests/unit/providers/single_node/runtime/test_lifecycle_client.py -v
pytest src/tests/unit/runner/test_pod_terminator_dispatch.py -v
pytest src/tests/unit/runner/test_main_lifespan_bootstrap.py -v

# Existing matrix/grace/retry tests still green
pytest src/tests/unit/runner/test_pod_terminator.py -v
```

Expected: ~50-70 new/migrated tests pass; zero regressions in
existing matrix tests.

### Regression
```bash
pytest src/tests/unit/runner/ src/tests/unit/providers/ -q --tb=short
# Slim-venv stubs runpod SDK (carry-over from 14.A); no new external deps.
```

### Contract verification (manual / via repl)
```python
import os
from src.runner.runtime.provider_registry import resolve_lifecycle_client_from_env
from src.runner.runtime.lifecycle_client import IPodLifecycleClient
from src.providers.training.interfaces import ITerminalActionProvider

# Invariant 1: registry maps RYOTENKAI_RUNTIME_PROVIDER → matching client.
runpod_env = {
    "RYOTENKAI_RUNTIME_PROVIDER": "runpod",
    "RUNPOD_API_KEY": "k", "RUNPOD_POD_ID": "p",
}
client = resolve_lifecycle_client_from_env(runpod_env)
assert isinstance(client, IPodLifecycleClient)
assert client.provider_name == "runpod"

# Invariant 2: single_node bootstrap with NO RunPod env still works.
sn_client = resolve_lifecycle_client_from_env({"RYOTENKAI_RUNTIME_PROVIDER": "single_node"})
assert sn_client.provider_name == "single_node"

# Invariant 3: missing API key on runpod path is loud.
from src.runner.runtime.provider_registry import BootstrapConfigError
try:
    resolve_lifecycle_client_from_env({"RYOTENKAI_RUNTIME_PROVIDER": "runpod"})
except BootstrapConfigError as e:
    assert "RUNPOD_API_KEY" in str(e)

# Invariant 4: PodTerminator constructor takes the new signature.
import inspect
from src.runner.pod_terminator import PodTerminator
sig = inspect.signature(PodTerminator.__init__)
assert "client" in sig.parameters
assert "graphql_url" not in sig.parameters
```

### Manual smoke
1. Build new image (no env-contract changes for operator):
   `./docker/training/build_and_push.sh --bump minor`.
2. Run a real RunPod SAPO config end-to-end. Verify:
   * Runner starts (lifespan logs include
     `pod_lifecycle_client_resolved provider=runpod`).
   * On natural completion, `pod_stop_attempt` event has
     `provider_name=runpod` and `outcome=stopped` (or
     `stopped_for_resume`).
3. Run a single_node SAPO config. Verify:
   * Runner starts with `pod_lifecycle_client_resolved provider=single_node`.
   * On natural completion, `pod_stop_attempt` event has
     `provider_name=single_node` and `outcome=skipped`.
4. Negative smoke: launch with `RYOTENKAI_RUNTIME_PROVIDER` unset
   in the runner image env → uvicorn exits non-zero immediately
   with `BootstrapConfigError` in the journal.

### Cleanup grep checks
```bash
# No more httpx imports in pod_terminator
! grep -n "import httpx" src/runner/pod_terminator.py
! grep -n "DEFAULT_RUNPOD_GRAPHQL_URL" src/runner/pod_terminator.py
! grep -n "_ALREADY_GONE_RE" src/runner/pod_terminator.py

# RunPod GraphQL is now provider-local
grep -n "api.runpod.io/graphql" src/providers/runpod/runtime/lifecycle_client.py

# decide_and_act no longer takes env=
! grep -nE "def decide_and_act\(.*env" src/runner/pod_terminator.py
```

## 14.B Effort + rollout

| Step | Effort |
|---|---|
| `IPodLifecycleClient` + `LifecycleActionResult` (§ 3.2) | 1h |
| `provider_registry` + `BootstrapConfigError` + edge-case helpers | 2h |
| `RunPodPodLifecycleClient` (verbatim move + adapt return type) | 3h |
| `NoOpPodLifecycleClient` | 30min |
| `PodTerminator` constructor + `decide_and_act` rewrite | 3h |
| `_lifespan` rewrite + bootstrap fail-fast | 2h |
| Test migration (move ~150 LOC, adapt assertions to new return type) | 4h |
| New tests (registry edge cases, lifespan bootstrap, dispatch) | 4h |
| Manual smoke (RunPod + single_node) | 2h |
| Documentation + commit messages | 1h |
| Code review buffer | 4h |
| **Total** | **~26h (~3.5 engineer-days)** |

Single PR with the 8 commits in § 14.B above. Land sequentially
before starting 14.C.

## 14.B Migration & rollback

**Migration:** No operator-facing env changes (the env vars
already exist; we just centralise where they're read).
`RYOTENKAI_RUNTIME_PROVIDER` is set by Phase 14.A's
`required_runtime_env_vars` and was already wired by Phase 14.A.
The launcher needed no further changes for 14.B.

**Rollback:** revert the 8 commits in reverse order. Commits 1-4
are purely additive (no behaviour change). Commit 5 is the
breaking refactor — it's atomic, so revert is clean. Tests in
commits 6-7 follow.

**Operator failure mode if rolled out incorrectly:**
* Pod boots, runner exits with `BootstrapConfigError` in the
  journal → operator sees the error in `journal events`. Loud,
  diagnosable.
* Mac launcher still wires `RYOTENKAI_RUNTIME_PROVIDER` from
  `provider.required_runtime_env_vars` (Phase 14.A). The only way
  this breaks is if `provider.provider_name` returns an
  unregistered string — caught by `BootstrapConfigError("...not
  registered...")`.

No DB migrations, no config-schema changes, no backwards-compat
shims.
