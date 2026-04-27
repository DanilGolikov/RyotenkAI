# Phase 14.A — Provider Capability Abstraction (FOUNDATIONAL)

> Status: **DRAFT — pending user approval**
> Author: daniil + agent
> Date: 2026-04-27
> Worktree: `nice-jepsen-07d789`
> Predecessors: Phase 0-11 (Job Server), Phase 12 (Metrics Durability)
> Successor: Phase 14.B (cannot start before 14.A merges)
> Migration policy: NO BACKWARDS COMPATIBILITY (carry-over)
> Effort: ~1.5 engineer-days
> Risk: Low (purely additive — no behavior change)
> Out of: roadmap stub in [`harmonic-rolling-crayon.md`](./harmonic-rolling-crayon.md)

## Phase 14 — overall context (why this multi-phase refactor exists)

Phase 9-12 landed with explicit RunPod focus. Audit surfaced concrete
leaks that violate the "shared code is provider-agnostic" contract:

* **17 string-checks** across `src/pipeline/`, `src/api/`, `src/cli/`
  asking "is this RunPod or single_node" by literal name comparison.
* `src/runner/pod_terminator.py` hard-codes RunPod GraphQL
  (`api.runpod.io/graphql`, `podStop`/`podTerminate` mutations,
  RunPod-specific error regex).
* `src/pipeline/launch/pod_availability.py` hard-codes
  `_RUNPOD_STATUS_MAP` (RunPod's `desiredStatus` vocabulary).
* `src/api/services/launch_service.py:190` and
  `src/cli/commands/run.py:230` duplicate ~140 lines of resume logic,
  both gated by `if metadata.provider != "runpod": return ok no-op`.
* `src/pipeline/stages/managers/deployment/training_launcher.py:464`
  hard-codes `env["RUNPOD_VOLUME_KIND"] = "persistent"` for ALL
  providers.
* Several core modules violate SRP/SoC (lifespan circular-binding
  closure, `decide_and_act` mixing pure decision + impure dispatch +
  telemetry, WS handler mutating heartbeat in serialization loop,
  `MetricsDecimator` conflating timing anchor with policy capture).

**Goal:** make the codebase **provider-agnostic at the shared level**,
with all RunPod specifics living inside `src/providers/runpod/` and
single_node specifics inside `src/providers/single_node/`. Adding a
third provider should be a 1-line registry change, not a 17-place
grep-replace.

## Phase 14 architectural decisions (locked, user-confirmed)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Two-Protocol split**: `IGPUProvider` (mandatory) + `ITerminalActionProvider` (optional, capability-gated). | Type checker enforces "you can't ask single_node to pause". Adding a third provider is a typed conformance question, not a Liskov-violation review. |
| 2 | **`RYOTENKAI_RUNTIME_PROVIDER` env var** — set by Mac launcher, read by runner at lifespan. Single source of truth. | Implicit detection ("if `RUNPOD_API_KEY` present → RunPod") doesn't scale to 3+ providers and is invisible at the bootstrap seam. |
| 3 | **Six separate plan files**, one per sub-phase. | Per user mandate "на каждую фазу отдельный план" — keeps each plan focused, each commit atomic, each review small. |

## Phase 14 sub-phase roadmap (this plan covers ONLY 14.A)

| # | Sub-phase | Effort | Risk | Plan file |
|---|---|---|---|---|
| **14.A** | **Provider Capability Abstraction** (foundational — must land first) | ~1.5 days | Low | ✅ This file (DONE — commit `1819cc4`) |
| 14.B | Extract RunPod GraphQL client (`IPodLifecycleClient` Protocol, RunPod impl, NoOp impl, runner registry) | ~3.5 days est. / ~1 day actual | Medium | ✅ [`phase-14b-pod-terminator-extraction.md`](./phase-14b-pod-terminator-extraction.md) (DONE in worktree, uncommitted) |
| 14.C | Unify resume flow (`LaunchResumeService`, eliminate the 140-line duplicate) | ~2.5 days | Medium-Low | 📋 [`phase-14c-launch-resume-unification.md`](./phase-14c-launch-resume-unification.md) (DRAFT) |
| 14.D + 14.F | **MERGED** into single plan: Provider-leak elimination (env hardcodes + capability flags + 9 leak sites + AUTHORING.md link + test fakes) | ~3.5 days | Low-Medium | 📋 [`phase-14df-provider-leak-elimination.md`](./phase-14df-provider-leak-elimination.md) (DRAFT) |
| 14.E | Core SRP/SoC fixes — focused (V1 lifespan circular binding, V3 WS heartbeat helper, V5 MLflow transport classifier; 2 won't-fix per KISS) | ~2 days | Low | 📋 [`phase-14e-srp-soc-fixes.md`](./phase-14e-srp-soc-fixes.md) (DRAFT) |

**Total estimated effort across all sub-phases: ~12 engineer-days
(down from initial 13.5 estimate — 14.D+F merge saves coordination
overhead; 14.E focused scope drops 2 won't-fix violations).** This
file delivers ONLY 14.A.

## Phase 14 zone-of-responsibility violations — full audit (informational)

Surfaced by the cross-cutting audit. Listed here so the user has the
full picture even though only 14.A is being executed in this plan.

### HIGH severity (blocking further work)

| File:lines | Type | Addressed in |
|---|---|---|
| `src/runner/pod_terminator.py:609-667` | Provider/transport leak — RunPod GraphQL hardcoded | 14.B |
| `src/runner/main.py:148-183` | Mixed abstractions — circular-binding closure to wire EventJournal `on_rotate` to bus.publish | 14.E |
| `src/api/services/launch_service.py:190` | Provider string check — `if metadata.provider != "runpod"` | 14.C |
| `src/cli/commands/run.py:230` | Duplicate of above (140 lines duplicated) | 14.C |
| `src/pipeline/stages/managers/deployment/training_launcher.py:464` | Hardcoded `env["RUNPOD_VOLUME_KIND"] = "persistent"` for all providers | 14.D |
| `src/pipeline/launch/pod_availability.py:176-182` | `_RUNPOD_STATUS_MAP` hard-codes RunPod GraphQL `desiredStatus` vocabulary | 14.C |

### MEDIUM severity (test maintenance pain)

| File:lines | Type | Addressed in |
|---|---|---|
| `src/runner/pod_terminator.py:335-447` | Side-effect smuggling — `decide_and_act` mixes pure decision + impure GraphQL + telemetry | 14.E |
| `src/runner/pod_terminator.py:182-242` | Tight coupling — `decide_terminal_outcome(mac_alive: bool)` should take `heartbeat: MacHeartbeat` | 14.E |
| `src/runner/api/events.py:144` | Layer violation — WS handler mutates `app.state.heartbeat` inside serialization loop | 14.E |
| `src/training/mlflow/resilient_transport.py:38-49` | Mixed abstractions — module knows about `requests`/`urllib3` library internals | 14.E |
| `src/runner/event_journal.py:157-201` | SRP — `__init__` does both initialization AND config validation | 14.E |
| `src/training/mlflow/metrics_buffer.py:56-95` | Mixed abstractions — `MetricsDecimator.__init__` conflates timing anchor with policy extraction | 14.E |
| `src/pipeline/stages/managers/deployment/dependency_installer.py:65` | `is_single_node_provider()` string check | 14.D |
| `src/pipeline/bootstrap/startup_validator.py:92-115` | Hardcoded `PROVIDER_RUNPOD` checks | 14.D |
| `src/api/services/config_service.py:69-91` | UI config validator mixes RunPod-specific logic | 14.F |

### LOW severity (cosmetic / future-proofing)

| File:lines | Type | Addressed in |
|---|---|---|
| `src/runner/main.py:258-307` | Tight coupling — `_periodic_journal_health_check` hardcodes thresholds at module level | 14.E |
| `src/runner/api/control.py:105-106` | Knowledge inversion — endpoint re-imports `MacHeartbeat` for a constant | 14.E |
| `src/pipeline/state/models.py:130` | `PodMetadata.provider: str = "runpod"` default — harmless but RunPod-leaning | 14.F |

---

## 14.A Context — why THIS sub-phase is foundational

All five subsequent phases (14.B → 14.F) depend on the runner being
able to ask "what kind of provider booted me?" and the launcher being
able to ask "what env vars does this provider need at runtime?". Today
neither question has a clean answer:

* The runner detects RunPod implicitly by checking
  `os.environ.get("RUNPOD_API_KEY")`. Single_node has no equivalent
  signal — it just degrades to `SKIPPED` on terminal hooks.
* The launcher hardcodes `env["RUNPOD_VOLUME_KIND"] = "persistent"`
  for all providers because there's no method to ask the provider
  what volume kind it has.
* `IGPUProvider` Protocol covers `connect/disconnect/check_gpu/...`
  but has zero methods for terminal actions or runtime contract.

**Phase 14.A introduces the capability surface** (Protocol extensions,
new dataclasses, the `RYOTENKAI_RUNTIME_PROVIDER` env contract). It
changes **zero** behavior — every new method is callable but only
exercised by tests. That makes 14.A safe to merge alone, validates
the abstraction shape via real provider impls, and lets every
downstream phase plug in incrementally.

## 14.A Scope

### IN-scope

1. **Extend `src/providers/training/interfaces.py`**:
   - Add `VolumeKind` Enum (`PERSISTENT` / `NETWORK` / `LOCAL_DISK`).
   - Add `AvailabilityVerdict` dataclass (frozen).
   - Extend `ProviderCapabilities` with new fields:
     `supports_lifecycle_actions`, `volume_kind`, `has_pause_resume`,
     `runner_workspace_root`.
   - Add `IGPUProvider.required_runtime_env_vars(...)` method.
   - Add `IGPUProvider.probe_availability(...)` method (always
     defined; single_node returns `state="running"`).
   - Add new `ITerminalActionProvider` Protocol (capability-gated;
     only RunPod implements).

2. **Extend `src/providers/runpod/training/provider.py`**:
   - Implement `required_runtime_env_vars` (returns RunPod env vars
     + `RYOTENKAI_RUNTIME_PROVIDER`).
   - Implement `probe_availability` (delegates to existing
     `RunPodAPIClient.query_pod` adapted to return
     `AvailabilityVerdict`).
   - Implement `ITerminalActionProvider` methods (`terminate`,
     `pause`, `resume` — delegate to existing RunPod lifecycle code).
   - Set new `ProviderCapabilities` fields appropriately.

3. **Extend `src/providers/single_node/training/provider.py`**:
   - Implement `required_runtime_env_vars` returning
     `{RYOTENKAI_RUNTIME_PROVIDER: "single_node"}`.
   - Implement `probe_availability` returning `state="running"`
     (single_node host is always reachable; SSH connect step
     surfaces real errors).
   - Set new `ProviderCapabilities` fields (lifecycle_actions=False,
     volume_kind=LOCAL_DISK, has_pause_resume=False).
   - **Does NOT implement** `ITerminalActionProvider` — type checker
     enforces this at every callsite.

4. **Add `RUNTIME_PROVIDER_ENV_VAR = "RYOTENKAI_RUNTIME_PROVIDER"`**
   constant in `src/constants.py`.

5. **Add factory-level integration test** asserting capability ↔
   Protocol conformance invariant:
   `caps.supports_lifecycle_actions == isinstance(provider, ITerminalActionProvider)`.

### OUT-of-scope (deferred to 14.B-F)

- Replacing existing call sites that read `RUNPOD_*` envs directly (14.D)
- Replacing the 17 string-check sites (14.F)
- Refactoring `PodTerminator` to use `IPodLifecycleClient` (14.B)
- Refactoring `LaunchResumeService` (14.C)
- Lifespan circular-binding fix (14.E)
- `decide_and_act` split (14.E)
- WS heartbeat refresh decorator (14.E)
- `MetricsDecimator.DecimationPolicy` extraction (14.E)
- `prepare_training_script_hooks` collapse into
  `required_runtime_env_vars` (14.D)

This phase is **purely additive**: every new method is callable but
only exercised by tests. Production callsites continue to use the
old code paths until 14.B onwards rewires them.

## 14.A New abstractions (signatures only)

```python
# src/providers/training/interfaces.py

class VolumeKind(str, Enum):
    """Storage semantics for the provider's pod/host workspace."""
    PERSISTENT = "persistent"   # Cloud pod with persistent volume — stoppable, /workspace survives
    NETWORK    = "network"      # Cloud pod with network volume — terminate-only (RunPod constraint)
    LOCAL_DISK = "local_disk"   # Local host (single_node) — no cloud volume semantics

@dataclass(frozen=True)
class AvailabilityVerdict:
    """Outcome of probing a provider for pod/host availability."""
    state: Literal[
        "running",              # ready to accept work
        "sleeping_resumable",   # paused but recoverable (RunPod EXITED/STOPPED/PAUSED)
        "gone",                 # terminated; needs fresh-pod resume
        "probe_failed",         # transient probe error; caller decides
        "unknown",              # provider doesn't track availability (single_node)
    ]
    resource_id: str
    raw_status: str | None = None    # provider-native status string for logs
    message: str = ""                # human-readable hint for operator

# Existing ProviderCapabilities EXTENDED (not replaced):
class ProviderCapabilities:
    # ... existing fields kept verbatim ...
    supports_lifecycle_actions: bool = False     # True iff provider implements ITerminalActionProvider
    volume_kind: VolumeKind = VolumeKind.PERSISTENT
    has_pause_resume: bool = False               # subset of lifecycle actions
    runner_workspace_root: str = "/workspace"    # what HELIX_WORKSPACE / PYTHONPATH resolve to inside the runner

# IGPUProvider extended with two new methods:
class IGPUProvider(Protocol):
    # ... existing methods kept verbatim ...

    def required_runtime_env_vars(
        self, *, resource_id: str | None,
    ) -> dict[str, str]:
        """Env vars the in-pod runner needs at boot. Includes
        RYOTENKAI_RUNTIME_PROVIDER plus any provider-specific creds.
        Single_node returns {RYOTENKAI_RUNTIME_PROVIDER: "single_node"}."""
        ...

    def probe_availability(self, resource_id: str) -> AvailabilityVerdict:
        """Always defined. Single_node returns state='running' without
        any network round-trip. RunPod queries GraphQL via its own client."""
        ...

# NEW capability-gated Protocol — RunPod implements, single_node does NOT:
@runtime_checkable
class ITerminalActionProvider(Protocol):
    """Provider can pause / resume / terminate its compute resource.
    Implemented iff ProviderCapabilities.supports_lifecycle_actions is True.

    Type-system enforcement: senior dev sees a method they expect to call
    only when isinstance(provider, ITerminalActionProvider) — no runtime
    branch on provider name needed."""

    def terminate(self, *, resource_id: str, reason: str) -> Result[None, ProviderError]: ...
    def pause(self, *, resource_id: str) -> Result[None, ProviderError]: ...
    def resume(self, *, resource_id: str) -> Result[None, ProviderError]: ...
```

```python
# src/constants.py — single line
RUNTIME_PROVIDER_ENV_VAR = "RYOTENKAI_RUNTIME_PROVIDER"
```

## 14.A Migration order

1. **Add `VolumeKind`, `AvailabilityVerdict`, `ITerminalActionProvider`** to
   `src/providers/training/interfaces.py`. Extend
   `ProviderCapabilities` fields with safe defaults.
2. **Add constant** `RUNTIME_PROVIDER_ENV_VAR` to `src/constants.py`.
3. **Implement on RunPod**: `required_runtime_env_vars`,
   `probe_availability`, `ITerminalActionProvider` methods. Update
   `get_capabilities()` return.
4. **Implement on single_node**: `required_runtime_env_vars`,
   `probe_availability` (always running). Update
   `get_capabilities()` return. Do **NOT** implement
   `ITerminalActionProvider`.
5. **Factory invariant test**: assert
   `caps.supports_lifecycle_actions == isinstance(provider, ITerminalActionProvider)`
   for both providers.
6. **Per-method unit tests** (7-cat coverage) on each provider impl.

Each step is a separate commit, each commit ships green tests.

## 14.A Critical files to modify

**EXTEND:**
- `src/providers/training/interfaces.py` — Protocol + dataclass additions
- `src/providers/runpod/training/provider.py` — implement new methods
- `src/providers/single_node/training/provider.py` — implement new methods
- `src/constants.py` — add `RUNTIME_PROVIDER_ENV_VAR`

**REUSE (do NOT modify):**
- `src/providers/runpod/training/api_client.py` —
  `RunPodAPIClient.query_pod` is the existing transport for
  `probe_availability`. New code wraps it to return
  `AvailabilityVerdict` instead of the raw dict. Keep the existing
  client untouched.
- `src/providers/runpod/training/sdk_adapter.py` — `start_pod`,
  `stop_pod` — these become the impls behind
  `ITerminalActionProvider.resume` and `.pause`. Do not duplicate;
  delegate.
- `src/pipeline/launch/pod_availability.py::_RUNPOD_STATUS_MAP` —
  the RunPod-specific status mapping. Phase 14.A LEAVES this in
  place; Phase 14.C is what moves it into
  `RunPodProvider.probe_availability`. In 14.A we delegate to the
  existing map for parity.

**TESTS (NEW):**
- `src/tests/unit/providers/training/test_interfaces.py` —
  `VolumeKind` round-trips, `AvailabilityVerdict` invariants
- `src/tests/unit/providers/runpod/training/test_provider_capabilities.py` —
  RunPod impl of new methods, 7-cat coverage
- `src/tests/unit/providers/single_node/training/test_provider_capabilities.py` —
  single_node impl, 7-cat coverage
- `src/tests/unit/providers/training/test_factory_capability_invariant.py` —
  cross-provider conformance test

## 14.A Risks (3 deepsink iterations)

### Iteration 1 — initial sweep

| ID | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R-1 | Forgetting to update factory return type — `Factory.create()` returns `IGPUProvider`, but RunPod ALSO conforms to `ITerminalActionProvider`. Type checker won't flag the latter unless callers do `isinstance`. | M | M | Document in factory docstring. Add explicit factory invariant test in `test_factory_capability_invariant.py` asserting both Protocols match `caps`. |
| R-2 | `VolumeKind` enum migration friction — existing `RUNPOD_VOLUME_KIND` env value is the string `"persistent"` / `"network"`. New enum's `.value` matches but reviewers may forget conversion at the env boundary. | L | L | Document the contract in the enum's docstring. 14.D consumes via `VolumeKind(env_str)` not raw string compare (deferred to 14.D). 14.A only adds the enum, doesn't change call sites. |
| R-3 | `probe_availability` for single_node — single_node has no `resource_id` (connect returns empty string). Must handle empty string AND not make network calls. | L | M | Explicit unit test: `SingleNodeProvider.probe_availability("")` returns instantly with `state="running"`. Hard requirement; failing test = blocker. |
| R-4 | Liskov violation — two sources of truth for "supports lifecycle": `ProviderCapabilities.supports_lifecycle_actions` flag AND `isinstance(p, ITerminalActionProvider)`. They can drift. | M | M | Factory-level runtime assertion: `assert caps.supports_lifecycle_actions == isinstance(provider, ITerminalActionProvider)`. Failing assert at boot = blocker. |

### Iteration 2 — deepsink on R-1 / R-4

The two-sources-of-truth concern (R-4) is the riskiest. Three
solutions considered:

1. **Drop the `supports_lifecycle_actions` capability flag.** Use
   only `isinstance(provider, ITerminalActionProvider)`. Trade-off:
   callsites doing data-driven checks (e.g. CLI listing all
   providers' capabilities) would have to compute this themselves.

2. **Drop `ITerminalActionProvider` Protocol.** Keep just the bool
   flag. Trade-off: lose type-system enforcement; back to
   "single_node has `pause()` that always returns Err".

3. **Keep both, add factory-level invariant assertion.** Trade-off:
   two places to update when provider gains lifecycle support, but
   the factory assertion catches drift at boot. ✅ **Picked.**

R-1 (factory return type) is solved by `runtime_checkable` Protocol
+ explicit invariant test. Type checker sees both Protocols; runtime
test catches any provider impl that's missing one.

### Iteration 3 — deepsink on cross-cutting concerns

| ID | Concern | Resolution |
|---|---|---|
| R-5 | **`required_runtime_env_vars` collides with existing `prepare_training_script_hooks(...).env_vars`** — both return env-var dicts. Which wins? | 14.A keeps both; both return the same data on RunPod (slight redundancy). 14.D explicitly collapses `prepare_training_script_hooks` into `required_runtime_env_vars`. Adding a temporary FIXME comment in `provider.py` to flag the redundancy. |
| R-6 | **`probe_availability` impl on RunPod** — should it use the existing `_RUNPOD_STATUS_MAP` from `pod_availability.py`? Or duplicate the mapping? | Use the existing map by import — 14.A is purely additive, doesn't move code. 14.C is what relocates the map into the RunPod provider package. Document this in the `RunPodProvider.probe_availability` docstring as a TODO. |
| R-7 | **Test coverage philosophy** — should the new methods be unit-tested with mocks of the underlying transports, or integration-tested against real GraphQL? | Unit tests for the method shape + value mapping. Existing integration tests for RunPod GraphQL transport stay. 14.A doesn't add new integration tests. |
| R-8 | **Backwards compat on `ProviderCapabilities` constructor** — adding 4 new fields with defaults. Any code constructing `ProviderCapabilities` manually breaks if it uses positional args. | Audit shows no positional-arg construction of `ProviderCapabilities` outside of provider classes themselves (which we control). Safe. Document with `# Phase 14.A: defaults safe — keyword-only construction is the only used pattern`. |

## 14.A Open questions (3 iterations of deepsink)

### Iteration 1

| OQ | Question | Resolution |
|---|---|---|
| OQ-1 | Should `IGPUProvider.probe_availability` be `async`? | **No.** Existing `IGPUProvider` methods are sync (`connect`, `disconnect`, `check_gpu`). For consistency, `probe_availability` is sync. RunPod's GraphQL call wraps an async client via `asyncio.run` (acceptable — see `RunPodAPIClient.query_pod` for precedent). The `ITerminalActionProvider` methods (`terminate`, `pause`, `resume`) ALSO sync for the same reason. |
| OQ-2 | Should `ITerminalActionProvider` methods take `resource_id: str` per call OR rely on the provider holding state? | **Per-call resource_id.** Provider instance is shared across attempts; state-holding would create stale-id bugs. Caller (Phase 14.B `PodTerminator`, 14.C `LaunchResumeService`) holds the resource_id from `PodMetadata`. |

### Iteration 2

| OQ | Question | Resolution |
|---|---|---|
| OQ-3 | Should `runner_workspace_root` be on `ProviderCapabilities` or a top-level method on `IGPUProvider`? | **`ProviderCapabilities`.** It's static-per-provider (RunPod = `/workspace`, single_node = `/workspace`, future provider X may be `/data`). Capabilities is the right home — it's already the static-config snapshot. |
| OQ-4 | Single_node's `volume_kind` — `LOCAL_DISK` or `PERSISTENT`? | **`LOCAL_DISK`.** It's not a cloud-volume concept. The downstream decision matrix in 14.B treats `LOCAL_DISK` as "no cloud terminate/stop semantics; provider's `disconnect()` handles cleanup". `PERSISTENT` would falsely imply "stoppable" which doesn't apply. |

### Iteration 3

| OQ | Question | Resolution |
|---|---|---|
| OQ-5 | What does `RunPodProvider.required_runtime_env_vars(resource_id=None)` return when called BEFORE `connect()` (no resource_id yet)? | **Return what we know.** Without `resource_id`, RunPod returns `{RUNTIME_PROVIDER, RUNPOD_API_KEY, RUNPOD_KEEP_ON_ERROR, RUNPOD_VOLUME_KIND}`. With `resource_id`, ALSO `RUNPOD_POD_ID`. The launcher always calls AFTER connect, so the empty case is just defensive. |
| OQ-6 | `AvailabilityVerdict.state` literal type — should `"unknown"` exist? | **Yes.** Single_node uses `"running"` (always-on host) but a future provider that doesn't track availability could legitimately return `"unknown"`. Forcing them into one of the four runpod-flavored states would be a type-system lie. |
| OQ-7 | Should we add a `ProviderCapabilities.runtime_provider_id: str` field that mirrors `provider_name`? | **No** — `provider_name` already exists. The launcher reads `provider.provider_name` to set `RYOTENKAI_RUNTIME_PROVIDER` env. Don't duplicate. |

## 14.A Verification

### Unit tests
```bash
# Slim-venv compatible — no datasets/pandas needed
pytest src/tests/unit/providers/training/test_interfaces.py -v
pytest src/tests/unit/providers/runpod/training/test_provider_capabilities.py -v
pytest src/tests/unit/providers/single_node/training/test_provider_capabilities.py -v
pytest src/tests/unit/providers/training/test_factory_capability_invariant.py -v
```

Expected: ~30-40 new tests pass; zero regressions in existing suite.

### Regression
```bash
pytest src/tests/unit/runner/ src/tests/unit/pipeline/ src/tests/unit/api/ \
       src/tests/unit/providers/ -q --tb=short
# Pre-existing slim-venv failures unchanged (datasets module missing).
# All other suites green.
```

### Contract verification (manual / via repl)
```python
from src.providers.training.factory import GPUProviderFactory
from src.providers.training.interfaces import ITerminalActionProvider, ProviderCapabilities

# Build both providers from a fixture config.
runpod = GPUProviderFactory.create("runpod", ...)
single_node = GPUProviderFactory.create("single_node", ...)

# Invariant 1: capability flag matches Protocol conformance.
assert runpod.get_capabilities().supports_lifecycle_actions
assert isinstance(runpod, ITerminalActionProvider)

assert not single_node.get_capabilities().supports_lifecycle_actions
assert not isinstance(single_node, ITerminalActionProvider)

# Invariant 2: required_runtime_env_vars always includes RUNTIME_PROVIDER_ENV_VAR.
from src.constants import RUNTIME_PROVIDER_ENV_VAR
assert RUNTIME_PROVIDER_ENV_VAR in runpod.required_runtime_env_vars(resource_id=None)
assert RUNTIME_PROVIDER_ENV_VAR in single_node.required_runtime_env_vars(resource_id=None)

# Invariant 3: probe_availability never raises.
v = single_node.probe_availability("")
assert v.state == "running"
```

### Manual smoke (optional)
N/A — Phase 14.A is purely additive, no runtime behavior change.
The first runtime-affecting change is Phase 14.B.

## 14.A Effort + rollout

| Step | Effort |
|---|---|
| Interfaces extension | 1h |
| RunPod impl | 2h |
| Single_node impl | 1h |
| Factory invariant test | 30min |
| Per-method unit tests (~30 cases × 7-cat) | 4h |
| Slim-venv test fixture updates | 1h |
| Documentation + commit messages | 1h |
| Code review buffer | 2h |
| **Total** | **~12h (~1.5 engineer-days)** |

Single PR, single commit recommended (it's small enough). Land
sequentially before starting 14.B.

## 14.A Migration & rollback

**Migration:** zero. Phase 14.A is purely additive — every new
method/field has a default and is only called from new tests. No
production callsite changes.

**Rollback:** revert the commit. Each change is in `src/providers/`;
zero impact on shared modules.

No DB migrations, no config schema changes, no env-var contract
changes visible to operators.
