# Phase 14.D + 14.F — Provider-Leak Elimination (merged)

> Status: **DRAFT — pending user approval**
> Author: daniil + agent
> Date: 2026-04-28
> Worktree: `nice-jepsen-07d789`
> Predecessors: Phase 14.A (commit `1819cc4`), 14.B (uncommitted but green), 14.C (planned)
> Successor: Phase 14.E
> Migration policy: NO BACKWARDS COMPATIBILITY — single-PR full cutover
> Effort: ~3.5 engineer-days (combined; pre-merge estimate was 14.D ~1.5 + 14.F ~2 = 3.5)
> Risk: Low-Medium — many small mechanical edits backed by capability-flag tests; failure mode is "config validator misses a missing secret" → caught by startup validation
> Out of: Phase 14 roadmap in [`phase-14a-provider-abstraction.md`](./phase-14a-provider-abstraction.md)

## 14.D+F Context — why merge

The original Phase 14 roadmap split env-hardcode cleanup (14.D) from string-check elimination (14.F). After Phase 14.C lands (which itself eliminates 2 of the 17 string-check sites originally counted), the remaining work for 14.D and 14.F has a single shared narrative:

**"Move every piece of provider-specific knowledge out of `src/pipeline/`, `src/api/`, `src/cli/`, `src/runner/` into `IGPUProvider` capability methods or fields."**

The two sub-phases share:
* The same target Protocol (`IGPUProvider` + `ProviderCapabilities`).
* The same test invariant (Phase 14.A factory test enforces capability ↔ implementation parity).
* The same migration risk profile (mechanical refactor, easy to test).
* The same downstream consumer surfaces (`startup_validator`, `_build_job_env`, `gpu_deployer`, `dependency_installer`, `config_service`, validators).

Splitting them into two PRs would force a half-step state where some leaks are gone but capability flags don't yet exist for the others — confusing for review. Merging into one PR yields one coherent "before/after" picture.

Effort: 14.D estimate was 1.5 days; 14.F estimate was 2 days. Combined estimate is **3.5 days** with no overlap savings (the work is genuinely additive). The benefit is reviewability + atomicity.

## 14.D+F Scope of leaks (post-14.C audit)

The audit identified **9 distinct leak sites** still open after 14.C lands (down from the originally-claimed 17 because 14.C closes 2 + audit reclassified 6 as legitimate-or-deferred):

### Phase 14.D — env-vars + secrets + hooks (4 leaks)

| L | File:line | Leak | Replacement |
|---|---|---|---|
| L1 | `training_launcher.py:464` | `env["RUNPOD_VOLUME_KIND"] = "persistent"` set unconditionally | Replace with `env.update(provider.required_runtime_env_vars(resource_id=...))` |
| L2 | `runpod/training/provider.py:527-541` + `:695+` | `prepare_training_script_hooks` returns same env vars as `required_runtime_env_vars` | Hooks delegate to `required_runtime_env_vars`; remove duplication |
| L3 | `dependency_installer.py:65` + `training_launcher.py:411` + `provider_config.py:38` | `is_single_node_provider()` string check | New capability `ProviderCapabilities.is_local: bool` |
| L4 | `startup_validator.py:92-115` | `PROVIDER_RUNPOD` hardcoded secret-presence check | New method `IGPUProvider.required_secrets() -> list[str]` |

### Phase 14.F — capability flags + cleanup (5 leaks)

| L | File:line | Leak | Replacement |
|---|---|---|---|
| L5 | `gpu_deployer.py:351,462` | `if active_provider == PROVIDER_RUNPOD:` for log download | New capability `ProviderCapabilities.supports_log_download: bool` |
| L6 | `validators/cross.py:238,270` | Provider name string-checks in cross-validator | Schema-level provider registry validation |
| L7 | `config_service.py:83` | `"runpod" in (active_provider or "").lower()` | Use `required_secrets()` from L4 |
| L8 | Test fixtures (`_FakeProvider` variants in `test_inference_deployer.py` + `test_early_pod_release.py`) | Duplicated test doubles | Centralized `FakeGPUProvider` in `src/tests/fixtures/providers.py` |
| L9 | `README.md` | No top-level link to provider authoring docs | Link `src/providers/README.md` from main README (the README itself is already complete) |

### Out of scope (deferred to 14.E or later)

| Item | Why deferred |
|---|---|
| `PodMetadata.provider: str = "runpod"` default (`models.py:130`) | Architectural blocker — changing the default requires either making the field non-optional (breaks legacy attempt deserialization) or threading `active_provider` through the deserializer. This is Phase 14.E SRP/SoC scope. |
| `runtime/provider_registry.py:181` (runner-side `== PROVIDER_RUNPOD`) | Legitimate registry-key match — Phase 14.B § 1.5 documents that the runner registry maps env-var values to client builders. Removing the literal would break the bootstrap contract. NOT a leak. |
| `cli/commands/run.py:230` + `api/services/launch_service.py:190` (`!= "runpod"` check) | Phase 14.C closes both via `isinstance(provider, ITerminalActionProvider)` — already covered. |

## 14.D+F Architectural decisions (locked, deepsink-validated)

### 14.D+F.1 Capability surface — **two new fields + one new method**

Phase 14.A added `ProviderCapabilities` with 4 fields (`supports_lifecycle_actions`, `volume_kind`, `has_pause_resume`, `runner_workspace_root`). 14.D+F adds:

```python
# Phase 14.D+F additions to ProviderCapabilities
is_local: bool = False                  # True for single_node host; False for cloud
supports_log_download: bool = False     # True for cloud (SCP-based); False for local
```

And adds one method to `IGPUProvider`:

```python
def required_secrets(self) -> tuple[str, ...]:
    """Names of secret env vars that MUST be present at startup.

    The startup validator (src.pipeline.bootstrap.startup_validator)
    iterates this list and checks os.environ for each name; any
    missing → StartupValidationError.

    Phase 14.D collapse: replaces the hardcoded
    ``PROVIDER_RUNPOD`` branches in startup_validator.py:92-115.
    """
```

**Why a method (not a capability field):** secrets are environment-driven (e.g. `RUNPOD_API_KEY` is one secret today, but a future provider with multiple credentials would return a tuple). A `tuple[str, ...]` field couldn't distinguish "no secrets" from "secrets are configured later" without adding sentinels. Method gives provider room to compute (e.g. read secret list from its config).

**Why not extend the existing `required_runtime_env_vars`:** that method returns env vars to **inject** into the runner subprocess; it MUST always include the values. `required_secrets` returns names of env vars that MUST be **present in the operator's environment** before launch — different layer (validation vs. injection). Conflating them would force the validator to discriminate "is this a value to set" vs. "is this a precondition to check" by string parsing — fragile.

**Tuples not lists:** `tuple[str, ...]` is hashable + immutable — matches how Phase 14.A treats provider-static metadata. List would imply mutable, which it never is.

### 14.D+F.2 Hooks collapse — **`prepare_training_script_hooks` becomes thin wrapper**

Phase 14.A added `IGPUProvider.required_runtime_env_vars(*, resource_id)` returning a dict; the older `prepare_training_script_hooks(ssh_client, context) -> TrainingScriptHooks` returns a dataclass with `env_vars: dict[str, str]` (plus other fields like `pre_python`, `post_python` script snippets that have other purposes).

Today these overlap: RunPod's `prepare_training_script_hooks().env_vars` returns essentially the same dict as `required_runtime_env_vars()`. The FIXME comment at provider.py:527 calls this out.

Phase 14.D+F:
1. `prepare_training_script_hooks().env_vars` is replaced with `provider.required_runtime_env_vars(resource_id=resource_id)`.
2. Other fields of `TrainingScriptHooks` (`pre_python`, `post_python`, etc.) stay — they have non-env purposes.
3. Callers of `prepare_training_script_hooks` (likely `training_launcher.py`) now call BOTH methods: `required_runtime_env_vars` for env, `prepare_training_script_hooks` for the script snippets. Duplication eliminated.

**Alternative considered**: collapse `prepare_training_script_hooks` entirely, exposing pre/post-Python hooks as separate methods. **Rejected** because:
* Three methods would need three callsites in `_build_job_env` and `_build_runner_command` — more breakable than one method returning a dataclass.
* The dataclass shape is a stable seam for future hook-types.

### 14.D+F.3 String-comparison → capability boolean — **mechanical replacement**

Each leak L3, L5, L6, L7 follows the same recipe:

1. Define a new `bool` field on `ProviderCapabilities` with safe default.
2. Set it to the right value on each provider impl.
3. Replace the callsite's `provider_name == "X"` check with `provider.get_capabilities().is_X`.
4. Update Phase 14.A's two-source-of-truth invariant test to cover the new field.

**Special case L6** (`validators/cross.py:238,270`):
* Line 238: `if active_provider not in {PROVIDER_SINGLE_NODE, PROVIDER_RUNPOD}` — schema-level "is this a known provider" check. **NOT replaced** with capability — instead, replaced with `if active_provider not in PROVIDER_REGISTRY_NAMES` using a tuple of registered names from `src/constants.py` (already exists as `PROVIDER_TYPES`). This is a registry membership check, not a capability check.
* Line 270: `provider_type == PROVIDER_RUNPOD` for inference-config validation. Phase 14.D+F adds `ProviderCapabilities.supports_inference_pod: bool` (already exists in some shape via `provider_type == "cloud"` — verify in implementation). Replace with capability.

### 14.D+F.4 Test fakes centralization — **`src/tests/fixtures/providers.py`**

Audit found 6+ ad-hoc `_FakeProvider` classes across test files (`test_inference_deployer.py`, `test_early_pod_release.py`). Each subset implements just enough of `IGPUProvider` for its test and skips the rest.

Phase 14.F adds `src/tests/fixtures/providers.py`:

```python
class FakeGPUProvider:
    """Test double implementing IGPUProvider in full.

    Default behaviour: every method returns a successful Result with
    sensible defaults. Tests subclass to override specific methods
    or pass kwargs to the constructor (e.g.
    ``FakeGPUProvider(connect_result=Err(...))``).
    """
    def __init__(
        self, *,
        provider_name: str = "fake",
        capabilities: ProviderCapabilities | None = None,
        connect_result: Result[..., AppError] | None = None,
        ...
    ) -> None: ...

class FailingGPUProvider(FakeGPUProvider):
    """Variant: every state-changing method returns Err."""
```

Existing tests migrate gradually (commit-by-commit basis: `test_inference_deployer.py` first, then `test_early_pod_release.py`). New tests use the fixture immediately.

**Why not migrate ALL tests in this PR**: combinatorial test surface is large; phased migration keeps PR reviewable. The fixture is added in 14.F; tests migrate as part of 14.F (in-scope) or future cleanup commits (out-of-scope but unlocked).

### 14.D+F.5 Documentation — **link, don't rewrite**

Audit confirmed `src/providers/README.md` is **complete and accurate** (~119 lines). It documents:
* Both `IGPUProvider` and `IInferenceProvider` Protocols.
* Lifecycle.
* Factory pattern.
* Step-by-step "adding a new provider" guide.

Phase 14.F:
1. Adds a one-line link from `README.md` (project root) to `src/providers/README.md` under a "Contributing" section.
2. Updates `src/providers/README.md` to reflect Phase 14.A/B/C/D/F additions:
   * `ITerminalActionProvider` Protocol (14.A)
   * `IPodLifecycleClient` (14.B)
   * `LaunchResumeService` (14.C)
   * `required_secrets()`, `is_local`, `supports_log_download` (14.D+F)
   * The two-source-of-truth invariant test pattern

No new top-level `AUTHORING.md` file is created — the existing README.md is the right place.

### 14.D+F.6 Two-source-of-truth invariant — **extended for new flags**

Phase 14.A's `test_factory_capability_invariant.py` pinned:
```
caps.supports_lifecycle_actions == isinstance(provider, ITerminalActionProvider)
```

Phase 14.D+F extends with parallel invariants:
* `caps.is_local` matches `provider_name == "single_node"` (today's only local provider; future providers that are also local declare `is_local=True`).
* `caps.supports_log_download` matches `provider.has_method("download_logs")` (capability flag matches actual method presence).
* `provider.required_secrets()` returns at least the env vars referenced by the provider's other methods (e.g. RunPod's `required_runtime_env_vars` includes `RUNPOD_API_KEY`, so `required_secrets()` MUST include it).

The last invariant is the strongest: it forces a provider author to keep the secret list in sync with what they actually consume at runtime. Test reads both methods and asserts the secret-name keys appear in the runtime env keys.

## 14.D+F Scope

### IN-scope

| Item | Description |
|---|---|
| `ProviderCapabilities.is_local: bool = False` | Marks single-node host providers |
| `ProviderCapabilities.supports_log_download: bool = False` | Cloud providers with SCP/HTTP log fetch |
| `ProviderCapabilities.supports_inference_pod: bool = False` (TBD — verify against existing flag) | Replaces `validators/cross.py:270` check |
| `IGPUProvider.required_secrets() -> tuple[str, ...]` | New Protocol method; replaces hardcoded secret-presence checks |
| RunPod impl updates | Set new flags; implement `required_secrets() == ("RUNPOD_API_KEY",)` |
| Single-node impl updates | Set `is_local=True`; `required_secrets() == ()` |
| `prepare_training_script_hooks` collapse | `env_vars` field delegates to `required_runtime_env_vars` |
| `_build_job_env` rewrite | Calls `provider.required_runtime_env_vars(resource_id=...)` directly; drops unconditional `env["RUNPOD_VOLUME_KIND"] = ...` |
| `startup_validator.py` rewrite | Iterates `provider.required_secrets()` instead of hardcoded `if PROVIDER_RUNPOD` |
| `dependency_installer.py:65` | `provider.get_capabilities().is_local` |
| `training_launcher.py:411` | `provider.get_capabilities().is_local` |
| `provider_config.py::is_single_node_provider` | Keep as compatibility shim (one-line `provider.get_capabilities().is_local`) OR delete + inline at callers — **decision in implementation** |
| `gpu_deployer.py:351,462` | `provider.get_capabilities().supports_log_download` |
| `validators/cross.py:238` | Replace with `active_provider not in PROVIDER_TYPES` (registry membership) |
| `validators/cross.py:270` | `provider.get_capabilities().supports_inference_pod` (or `.provider_type == "cloud"`) |
| `config_service.py:83` | Iterate `required_secrets()`; check each against env |
| `src/tests/fixtures/providers.py` (NEW) | `FakeGPUProvider`, `FailingGPUProvider` test doubles |
| Migrate `_FakeProvider` in `test_inference_deployer.py` | Use centralized fixture |
| Migrate provider-mock in `test_early_pod_release.py` | Use centralized fixture |
| Extended `test_factory_capability_invariant.py` | Pin new fields and `required_secrets` invariant |
| README.md "Contributing" section | One-line link to `src/providers/README.md` |
| Update `src/providers/README.md` | Document Phase 14.A/B/C/D+F additions |

### OUT-of-scope

| Item | Phase |
|---|---|
| `PodMetadata.provider: str = "runpod"` default | 14.E (architectural blocker — needs deserializer rework) |
| Migrate ALL test files to centralized FakeGPUProvider | Cleanup commits (after 14.F lands) |
| `src/runner/runtime/provider_registry.py` literal `PROVIDER_RUNPOD` matches | Legitimate registry — not a leak |
| Async-ify any of the touched code paths | YAGNI |
| Replace `PROVIDER_TYPES` constant with a runtime registry | YAGNI; current literal-list is fine for 2 providers |

## 14.D+F New abstractions (signatures only)

```python
# src/providers/training/interfaces.py — additions

@dataclass(frozen=True)
class ProviderCapabilities:
    # ... existing fields preserved verbatim ...
    is_local: bool = False
    supports_log_download: bool = False
    # supports_inference_pod TBD pending impl read of cross.py:270


class IGPUProvider(Protocol):
    # ... existing methods preserved ...

    def required_secrets(self) -> tuple[str, ...]:
        """Names of operator-environment env vars that MUST be set
        before launch. RunPod returns ('RUNPOD_API_KEY',); single-node
        returns (). Used by
        :mod:`src.pipeline.bootstrap.startup_validator`.

        Phase 14.D+F: REPLACES the hardcoded ``PROVIDER_RUNPOD``
        secret check at startup_validator.py:92-115.
        """
        ...
```

```python
# src/providers/runpod/training/provider.py — adds:

def required_secrets(self) -> tuple[str, ...]:
    return ("RUNPOD_API_KEY",)

def get_capabilities(self) -> ProviderCapabilities:
    return ProviderCapabilities(
        # ... existing values ...
        is_local=False,
        supports_log_download=True,
    )
```

```python
# src/providers/single_node/training/provider.py — adds:

def required_secrets(self) -> tuple[str, ...]:
    return ()

def get_capabilities(self) -> ProviderCapabilities:
    return ProviderCapabilities(
        # ... existing values ...
        is_local=True,
        supports_log_download=False,
    )
```

```python
# src/pipeline/bootstrap/startup_validator.py — rewrite of lines 92-115

def _validate_provider_secrets(
    *, active_provider: str, secrets: Secrets, provider: IGPUProvider,
) -> None:
    """Phase 14.D+F: provider-driven secret validation."""
    for secret_name in provider.required_secrets():
        if not _has_secret(secrets, secret_name):
            raise StartupValidationError(
                f"{secret_name} is required when using provider "
                f"{active_provider!r}. Set it in your environment "
                f"or .env file before launching the pipeline."
            )
```

```python
# src/tests/fixtures/providers.py — NEW

@dataclass
class FakeGPUProvider:
    """Test double implementing IGPUProvider in full. Default
    behaviour: every method returns a successful Result. Override
    specific methods via constructor kwargs.
    """
    provider_name: str = "fake"
    capabilities: ProviderCapabilities | None = None
    required_secrets_value: tuple[str, ...] = ()
    # ... per-method override kwargs ...

    def required_secrets(self) -> tuple[str, ...]:
        return self.required_secrets_value

    # ... full IGPUProvider impl ...
```

## 14.D+F Migration order — single PR, atomic commits

| # | Commit | Scope | Why this order |
|---|---|---|---|
| 1 | `feat(providers/training/interfaces): add ProviderCapabilities.is_local/supports_log_download + IGPUProvider.required_secrets` | Pure additions | No consumers yet — additive |
| 2 | `feat(providers/runpod): set capability flags + implement required_secrets` | RunPod impl | Provider conformance |
| 3 | `feat(providers/single_node): set capability flags + implement required_secrets` | Single-node impl | Provider conformance |
| 4 | `test(providers/training): extend factory capability invariant with new flags` | Test invariants | Pin two-source-of-truth |
| 5 | `refactor(pipeline/bootstrap/startup_validator): use provider.required_secrets()` | Validator cutover | First consumer cuts to new API |
| 6 | `refactor(pipeline/stages/managers/deployment): drop _build_job_env hardcodes; use provider.required_runtime_env_vars` | _build_job_env rewrite | Second consumer (env injection) |
| 7 | `refactor(providers/runpod): collapse prepare_training_script_hooks env_vars to delegate` | Hooks collapse | Third consumer (script hooks) |
| 8 | `refactor(pipeline/stages/managers/deployment): replace is_single_node_provider with capabilities.is_local` | First string-check elimination | Two callsites in deployment |
| 9 | `refactor(pipeline/stages): replace gpu_deployer log-download check with capabilities.supports_log_download` | Second string-check elimination | Two callsites in gpu_deployer |
| 10 | `refactor(api/services/config_service): use required_secrets() for UI validation` | UI validator | Third string-check elimination |
| 11 | `refactor(config/validators/cross): use registry membership + capability for cross-validation` | Schema validator | Fourth string-check elimination |
| 12 | `feat(tests/fixtures): add centralized FakeGPUProvider + FailingGPUProvider` | Test infrastructure | Unblocks fake migration |
| 13 | `refactor(tests/unit/pipeline/stages): migrate test_inference_deployer + test_early_pod_release fakes` | Fake migration | Two file migrations |
| 14 | `docs: link src/providers/README.md from main README; update README with Phase 14.A/B/C/D+F additions` | Documentation | Final touch-up |
| 15 | `docs(plans): mark Phase 14.D+F DONE; update successor link in 14.A` | Bookkeeping | — |

PR is **NOT** squash-merged — Phase 14 convention.

## 14.D+F Critical files to modify

**EXTEND:**
- `src/providers/training/interfaces.py` — add 2 capability fields + `required_secrets()` Protocol method
- `src/providers/runpod/training/provider.py` — implement new method, set new flags, collapse hooks
- `src/providers/single_node/training/provider.py` — implement new method, set new flags

**REWRITE:**
- `src/pipeline/bootstrap/startup_validator.py:92-115` — use `provider.required_secrets()`
- `src/pipeline/stages/managers/deployment/training_launcher.py::_build_job_env` (lines ~380-474) — use `provider.required_runtime_env_vars(resource_id=...)`; drop hardcoded `RUNPOD_VOLUME_KIND` line at 464; drop `is_single_node_provider` check at 411
- `src/pipeline/stages/managers/deployment/dependency_installer.py:65` — use `capabilities.is_local`
- `src/pipeline/stages/managers/deployment/provider_config.py::is_single_node_provider` — DELETE (or shim to `capabilities.is_local`)
- `src/pipeline/stages/gpu_deployer.py:351,462` — use `capabilities.supports_log_download`
- `src/api/services/config_service.py:83` — iterate `required_secrets()`
- `src/config/validators/cross.py:238,270` — registry-membership + capability check

**NEW:**
- `src/tests/fixtures/providers.py` — `FakeGPUProvider`, `FailingGPUProvider`

**TEST UPDATES:**
- `src/tests/unit/providers/training/test_factory_capability_invariant.py` — extend with new fields
- `src/tests/unit/providers/training/test_interfaces.py` — pin `required_secrets()` Protocol method exists
- `src/tests/unit/providers/runpod/training/test_provider_capabilities.py` — pin RunPod returns `("RUNPOD_API_KEY",)`
- `src/tests/unit/providers/single_node/training/test_provider_capabilities.py` — pin single-node returns `()`
- `src/tests/unit/pipeline/bootstrap/test_startup_validator.py` — replace hardcoded RUNPOD test cases with provider-driven tests
- `src/tests/unit/pipeline/stages/managers/deployment/test_training_launcher.py` — assert `_build_job_env` calls `provider.required_runtime_env_vars`
- `src/tests/unit/api/services/test_config_service.py` — provider-driven UI validator tests
- `src/tests/unit/config/validators/test_cross.py` — registry-membership tests
- `src/tests/unit/pipeline/stages/test_inference_deployer.py` — use `FakeGPUProvider`
- `src/tests/unit/pipeline/stages/test_early_pod_release.py` — use `FakeGPUProvider`

**DOCUMENTATION:**
- `README.md` — add Contributing link
- `src/providers/README.md` — Phase 14 additions

## 14.D+F Risks (3 deepsink iterations)

### Iteration 1 — initial sweep

| ID | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R-1 | **Capability flag drift** — provider author updates impl method but forgets to update flag (or vice versa) | M | M | Phase 14.A's factory invariant test pattern extended to new fields. Failing test at boot = blocker. |
| R-2 | **`_build_job_env` rewrite breaks env injection** — subtle missing key triggers runtime failure on the pod | H | L | Pin via test asserting `_build_job_env` output dict has same keys as pre-14.D for RunPod (regression set); same for single-node. |
| R-3 | **`required_secrets` introduces a new failure mode** — startup validator now runs provider Protocol method that didn't exist before | M | L | Test that startup validator with a Phase 14-A provider (no `required_secrets`) raises clear error mentioning Phase 14.D+F migration. (Won't happen in production — all providers updated atomically — but defensive test prevents accidental partial rollout.) |
| R-4 | **`prepare_training_script_hooks` collapse** changes script-side env injection | M | L | Pin: hooks's `env_vars` field returns the SAME dict as `required_runtime_env_vars` for RunPod. Diff test. |
| R-5 | **`FakeGPUProvider` doesn't fully implement IGPUProvider** — tests using it pass but impl is incomplete | L | M | Use `runtime_checkable` Protocol assertion in fixture's `__post_init__` (or via a fixture-level test). |

### Iteration 2 — deepsink R-2 (env-injection regression)

`_build_job_env` (training_launcher.py:380-474) is the most-touched file. Today's behaviour:

```python
# Pseudocode of current logic:
env = base_env_dict()
env["HELIX_WORKSPACE"] = workspace
env["PYTHONPATH"] = ...
env["HF_TOKEN"] = ...
env["RUNPOD_VOLUME_KIND"] = "persistent"  # L1
extra = provider.prepare_training_script_hooks(...).env_vars
env.update(extra)  # provider-specific keys merged on top
return env
```

Post-14.D+F:

```python
# Pseudocode of new logic:
env = base_env_dict()
env["HELIX_WORKSPACE"] = workspace
env["PYTHONPATH"] = ...
env["HF_TOKEN"] = ...
provider_env = provider.required_runtime_env_vars(resource_id=resource_id)
env.update(provider_env)  # provider's full env contribution
hooks = provider.prepare_training_script_hooks(...)  # script snippets only
return env, hooks  # OR: caller pulls hooks separately
```

**Concern**: today `env_vars` from hooks runs LAST (overriding base_env keys); after refactor `required_runtime_env_vars` runs after `base_env`. Are there keys that base_env sets and the provider intends to override?

**Mitigation**: read both pre-14.D+F branches to enumerate the keys each contributes:

| Key | base_env | required_runtime_env_vars (RunPod) | hooks.env_vars (RunPod) |
|---|---|---|---|
| `HELIX_WORKSPACE` | ✓ | – | – |
| `PYTHONPATH` | ✓ | – | – |
| `HF_TOKEN` | ✓ | – | – |
| `RYOTENKAI_RUNTIME_PROVIDER` | – | ✓ | (was ✓ before 14.A) |
| `RUNPOD_API_KEY` | – | ✓ | ✓ (legacy) |
| `RUNPOD_KEEP_ON_ERROR` | – | ✓ | ✓ (legacy) |
| `RUNPOD_VOLUME_KIND` | ✓ (L1 leak) | ✓ | – |
| `RUNPOD_POD_ID` | – | ✓ (when resource_id passed) | ✓ (legacy) |

**Verification**: keys are disjoint between base_env and `required_runtime_env_vars` after L1 is removed. So `env.update(provider_env)` is safe — no clobbering of Mac-side base. Confirmed by test assertion that resulting dict has same keys as the pre-14.D+F dict (modulo `RUNPOD_VOLUME_KIND` which is now provider-supplied not base-hardcoded).

### Iteration 3 — deepsink R-1 (capability flag drift)

Phase 14.A's invariant test pattern works for boolean capability flags but is weaker for `required_secrets()` (a method, not a field). The risk: provider author adds a new RunPod env var to `required_runtime_env_vars` but forgets to add the secret name to `required_secrets()` → startup validator passes, runtime injection sets a non-existent secret → trainer fails with cryptic error.

**Mitigation**: a stronger invariant test:

```python
def test_required_secrets_subset_of_runtime_env_keys(self) -> None:
    """For every secret in required_secrets, it must be either:
    * Read by required_runtime_env_vars (the provider injects it), OR
    * Documented as 'used at the pre-launch validation layer only'.

    This prevents 'required at startup, never actually used' bugs.
    """
    provider = _mk_provider()
    secrets = provider.required_secrets()
    runtime_env = provider.required_runtime_env_vars(resource_id="test-pod")
    # RunPod's required_runtime_env_vars READS RUNPOD_API_KEY
    # from operator's env (via os.environ) and INJECTS it. So
    # every name in required_secrets must appear as a value in
    # required_runtime_env_vars (when the provider has been
    # constructed with that secret).
    for secret_name in secrets:
        # Provider must have REFERENCED this secret somewhere
        # in its impl. We verify by checking it appears in the
        # injection dict's KEYS (RunPod injects "RUNPOD_API_KEY"
        # → the value is the secret's content).
        assert secret_name in runtime_env, (
            f"{secret_name} listed in required_secrets() but not "
            f"injected by required_runtime_env_vars(). Provider "
            f"is asking operator for a secret it never uses."
        )
```

**Limitation**: this catches the "ask for a secret, never use it" direction but NOT the "use a secret, never ask for it". The reverse direction is harder to verify without scanning Provider source for `os.environ.get(...)` patterns. Phase 14.D+F accepts the asymmetry; Phase 14.E or beyond may add a static-analysis test.

### Iteration 4 — cross-cutting concerns

| ID | Concern | Resolution |
|---|---|---|
| R-6 | **L9 doc work is trivial** but Phase 14 has been disciplined about documentation | Acknowledge — keep doc commit small (commit 14). README.md gets ~5 lines of new content; provider README gets a "Phase 14 changes" appendix block. |
| R-7 | **Test fake migration risk** — existing tests have specific provider behaviours that the centralized fake might not capture | Migrate file-by-file with diff review. Each migration commit verifies the fake's behaviour matches the file-local _FakeProvider (assertion: same Result type, same exceptions raised, etc.). |
| R-8 | **`provider_config.py::is_single_node_provider` shim vs delete** — Phase 14.F prefers deletion but callers might be in test code that we don't migrate | Decision: KEEP as a one-line shim `return provider.get_capabilities().is_local`. Cost: 1 line. Benefit: tests calling `is_single_node_provider(config)` keep working. Phase 14.E or later can delete if/when test migration completes. |
| R-9 | **`PROVIDER_TYPES` constant** in `src/constants.py` — using it as the registry membership source for `validators/cross.py:238` is hardcoded just-as-much as the explicit set | Acceptable: `PROVIDER_TYPES` IS the canonical registry. New providers are added by appending to it (a one-line change in one place). Phase 14.E may add a dynamic registry but YAGNI for two providers. |
| R-10 | **Inference vs training capability split** — `IGPUProvider` is training-side; do we need parallel additions for `IInferenceProvider`? | Yes for `is_local` (single-node inference exists too). Phase 14.D+F adds the same field on both Protocols. `required_secrets` is currently only training-relevant (RUNPOD_API_KEY); inference may need additional secrets when third-party inference providers join — defer to that future provider's PR. |
| R-11 | **Concurrent test execution + `os.environ` mutation** in startup validator tests | Use `monkeypatch.setenv` (per-test isolation) — same pattern as Phase 14.B test fixtures. Don't use `os.environ.update` directly. |
| R-12 | **`PROVIDER_TYPES` in `cross.py:238`** is a tuple of literals; if 14.D+F replaces with a different constant name, any external consumer breaks | Read the file: if `cross.py` is the only consumer (check via grep), rename freely. If others consume → keep name. |

## 14.D+F Open questions (resolved)

| OQ | Question | Resolution |
|---|---|---|
| OQ-1 | Should `required_secrets()` be a method or a `tuple[str, ...]` field on `ProviderCapabilities`? | **Method.** Per § 14.D+F.1 — secrets are environment-driven; method gives provider room to compute / derive. |
| OQ-2 | Do we add `is_remote: bool` as the inverse of `is_local`? | **No.** YAGNI. `not is_local` is one line; redundant flag is drift risk. |
| OQ-3 | Should `provider_type` (existing field on ProviderCapabilities, `"cloud"` / `"local"` string) be deprecated in favour of `is_local`? | **Keep both for now.** `provider_type` is read by other code paths (catalog, telemetry); keeping it preserves backward-compat. `is_local` is the new test-friendly seam. Phase 14.E may rationalize. |
| OQ-4 | Does `prepare_training_script_hooks` collapse break inference path? | No — this method is `IGPUProvider.prepare_training_script_hooks`, training-only. Inference Protocol has its own hooks. |
| OQ-5 | What about `RUNPOD_KEEP_ON_ERROR` — is it a "secret" or a runtime config? | Runtime config — operator-set behaviour toggle. NOT a secret. `required_secrets()` does not return it. Validator does NOT check its presence (absent = `False` is a valid state). |
| OQ-6 | Add `required_optional_env(): tuple[str, ...]` for runtime knobs like `RUNPOD_KEEP_ON_ERROR`? | **No, YAGNI.** Optional means absent is OK; no validation needed. Operators set it explicitly when they want it. |
| OQ-7 | Should `FakeGPUProvider` also implement `ITerminalActionProvider` (the optional Phase 14.A Protocol)? | **No by default.** Subclass `FakeGPUProvider` → `FakeLifecycleProvider` for tests that need lifecycle. Default provider has `supports_lifecycle_actions=False`. |
| OQ-8 | What about `IInferenceProvider` — does it need `required_secrets()` too? | **Same name, same shape, separate method.** Phase 14.D+F adds it on `IInferenceProvider` too. RunPod inference returns `("RUNPOD_API_KEY",)`; single-node inference returns `()`. |
| OQ-9 | `PROVIDER_TYPES` registry — should it become a runtime registry pattern (like Phase 14.B's `_REGISTRY`)? | **No, YAGNI.** Two providers, low growth rate. Tuple of literals is fine. Refactor when 4+ providers exist. |

## 14.D+F Test plan (7-category coverage)

### 14.D+F.10.1 — `test_interfaces.py` extension (~5 tests)

1. Positive — `ProviderCapabilities(is_local=True, supports_log_download=False)` constructs.
2. Positive — `ProviderCapabilities()` defaults: both new fields are `False`.
3. Invariants — fields are frozen (assignment raises).
4. Logic-specific — `IGPUProvider.required_secrets()` is a Protocol method that runtime-checkable detects on impls.
5. Boundary — provider returning `()` from `required_secrets()` is valid (not None).

### 14.D+F.10.2 — `test_factory_capability_invariant.py` extension (~6 tests)

1. RunPod: `caps.is_local is False`, `caps.supports_log_download is True`.
2. Single-node: `caps.is_local is True`, `caps.supports_log_download is False`.
3. RunPod: `required_secrets() == ("RUNPOD_API_KEY",)`.
4. Single-node: `required_secrets() == ()`.
5. Cross-invariant — for each provider, every name in `required_secrets()` appears as a key in `required_runtime_env_vars(resource_id="test")`.
6. Schema-level — `PROVIDER_TYPES` constant lists exactly the registered provider names; adding a third provider triggers a registry membership update.

### 14.D+F.10.3 — Per-consumer test updates (~30-40 tests)

Each refactored consumer gets its tests updated:

* `test_startup_validator.py` — replace 4 hardcoded RUNPOD assertions with parameterized provider-driven tests.
* `test_training_launcher.py` — `_build_job_env` returns dict containing `provider.required_runtime_env_vars()` keys; pin no `RUNPOD_VOLUME_KIND` is set unconditionally for single-node.
* `test_dependency_installer.py` — replace `is_single_node_provider` mock with `provider.get_capabilities().is_local` mock.
* `test_gpu_deployer.py` — log-download path tested against `supports_log_download` capability.
* `test_config_service.py` — UI validator checks every `required_secrets()` name against env.
* `test_cross.py` — schema validator uses registry membership.

### 14.D+F.10.4 — `test_fixtures/providers.py` (NEW, ~12 tests)

7-cat for the centralized fake:

1. Positive — `FakeGPUProvider()` constructs; conforms to `IGPUProvider`.
2. Positive — `FakeGPUProvider(provider_name="x", capabilities=...)` constructs with overrides.
3. Negative — `FailingGPUProvider().connect(...)` returns `Err(...)`.
4. Boundary — empty kwargs produce a minimally-valid provider.
5. Invariants — `runtime_checkable` Protocol assertion: `isinstance(FakeGPUProvider(), IGPUProvider)`.
6. Dependency errors — N/A (test fixture, no transports).
7. Logic-specific — every method returns a `Result` instance (not a bare value).

### 14.D+F.10.5 — Existing fake migration tests

* `test_inference_deployer.py` — re-run after fake migration; all original tests still pass.
* `test_early_pod_release.py` — same.

## 14.D+F Verification

### Unit tests

```bash
# Phase 14.D+F new + extended
pytest src/tests/unit/providers/training/test_interfaces.py -v
pytest src/tests/unit/providers/training/test_factory_capability_invariant.py -v
pytest src/tests/unit/providers/runpod/training/test_provider_capabilities.py -v
pytest src/tests/unit/providers/single_node/training/test_provider_capabilities.py -v
pytest src/tests/unit/pipeline/bootstrap/test_startup_validator.py -v
pytest src/tests/unit/pipeline/stages/managers/deployment/ -v
pytest src/tests/unit/pipeline/stages/test_gpu_deployer.py -v
pytest src/tests/unit/api/services/test_config_service.py -v
pytest src/tests/unit/config/validators/test_cross.py -v
pytest src/tests/fixtures/test_providers.py -v   # if test file added

# Existing tests still green
pytest src/tests/unit/pipeline/stages/test_inference_deployer.py -v
pytest src/tests/unit/pipeline/stages/test_early_pod_release.py -v
pytest src/tests/unit/runner/ -v   # Phase 14.A/B tests preserved
```

### Regression

```bash
pytest src/tests/unit/ -q --tb=line
# Pre-existing slim-venv failures (Phase 6.6) unchanged.
# All other suites green.
```

### Cleanup grep checks

```bash
# String-checks eliminated
! grep -nE 'PROVIDER_RUNPOD\b' src/pipeline/bootstrap/startup_validator.py
! grep -n 'is_single_node_provider' src/pipeline/stages/managers/deployment/dependency_installer.py
! grep -n 'is_single_node_provider' src/pipeline/stages/managers/deployment/training_launcher.py
! grep -n 'PROVIDER_RUNPOD' src/pipeline/stages/gpu_deployer.py
! grep -n '"runpod" in' src/api/services/config_service.py

# Hardcoded RUNPOD_VOLUME_KIND gone from shared code
! grep -n 'RUNPOD_VOLUME_KIND' src/pipeline/stages/managers/deployment/training_launcher.py

# New capability methods exist
grep -n 'is_local' src/providers/training/interfaces.py
grep -n 'supports_log_download' src/providers/training/interfaces.py
grep -n 'def required_secrets' src/providers/training/interfaces.py
grep -n 'def required_secrets' src/providers/runpod/training/provider.py
grep -n 'def required_secrets' src/providers/single_node/training/provider.py

# Centralized fake exists
test -f src/tests/fixtures/providers.py

# README link added
grep -n 'src/providers/README.md' README.md
```

### Manual smoke

1. `RUNPOD_API_KEY` unset + provider=runpod → startup validator raises `StartupValidationError` mentioning the secret name.
2. `provider=single_node` → no secret check; pipeline boots without RUNPOD_API_KEY.
3. RunPod pod boots — `_build_job_env` injects `RUNPOD_VOLUME_KIND=persistent` via `required_runtime_env_vars` (not via the deleted hardcode).
4. Single-node host runs — environment has NO `RUNPOD_VOLUME_KIND` (provider returns it only when applicable).

## 14.D+F Effort + rollout

| Step | Effort |
|---|---|
| `ProviderCapabilities` field additions + `IGPUProvider.required_secrets()` Protocol | 1h |
| RunPod impl (set flags + implement method + collapse hooks) | 2h |
| Single-node impl | 1h |
| `_build_job_env` rewrite + tests | 3h |
| `startup_validator` rewrite + tests | 2h |
| `dependency_installer` + `training_launcher` is_local cutover | 1h |
| `gpu_deployer` supports_log_download cutover | 1h |
| `config_service` UI cutover | 1h |
| `validators/cross` cutover | 1.5h |
| `FakeGPUProvider` fixture | 2h |
| `_FakeProvider` migration in 2 test files | 2h |
| Two-source-of-truth invariant test extensions | 1h |
| Documentation updates | 1h |
| Manual smoke + regression | 2h |
| Code review buffer | 4h |
| **Total** | **~25.5h (~3.5 engineer-days)** |

Single PR with the 15 commits in § Migration order. Land before starting Phase 14.E (which depends on the cleanup landed here for the SRP/SoC fixes).

## 14.D+F Migration & rollback

**Migration:** Operator-facing change — startup validator now provides clearer error messages naming the missing secret. Otherwise no contract change. Existing `.env` files keep working. CLI commands unchanged.

**Rollback:** Revert the 15 commits in reverse order. Commits 1-4 (Protocol + impls + invariants) are purely additive; commits 5-11 are the consumer rewrites — atomic per consumer, so a single regression can revert just one rewrite. Commits 12-15 are infra + docs; trivial revert.

**Operator failure mode if rolled out incorrectly:**

* Forgot to set a flag on a provider → invariant test fails at boot OR at PR review time.
* Capability mismatch (provider author updates impl but not flag) → factory invariant test fails at boot.
* Centralized FakeGPUProvider missing a method → tests that use it fail at runtime with `AttributeError` — caught by CI before merge.

No DB migrations, no config-schema changes, no backwards-compat shims.
