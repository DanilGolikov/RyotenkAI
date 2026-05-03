# Providers (`src/providers`)

The package contains **runtime provider implementations** of two kinds:

- **Training providers** (`IGPUProvider`) — provide SSH access to a GPU machine for training
- **Inference providers** (`IInferenceProvider`) — deploy/provision an inference endpoint and produce artifacts (manifest + scripts)

**Key principle:** config schemas live in `src/config/providers/*`, execution code lives in `src/providers/*`.

`src/providers/` is the **single source of truth** for providers. `src/pipeline/providers/` and `src/pipeline/inference/providers/` are compat shims with re-exports.

## Structure

```
src/providers/
├── training/
│   ├── interfaces.py        # IGPUProvider (Protocol) + ITerminalActionProvider (capability-gated)
│   │                        #   ProviderCapabilities, AvailabilityVerdict, VolumeKind,
│   │                        #   SSHConnectionInfo, GPUInfo, ProviderStatus, TrainingScriptHooks
│   └── factory.py           # GPUProviderFactory (registry) + auto_register_providers()
├── inference/
│   ├── interfaces.py        # IInferenceProvider (Protocol), EndpointInfo, InferenceArtifacts, PipelineReadinessMode
│   └── factory.py           # InferenceProviderFactory (if/elif by config.inference.provider)
├── single_node/
│   ├── training/            # SingleNodeProvider (local SSH)
│   ├── inference/           # SingleNodeInferenceProvider (vLLM via SSH+Docker)
│   └── runtime/             # NoOpPodLifecycleClient (in-pod runner side, Phase 14.B)
├── runpod/
│   ├── _status_mapper.py    # RunPod desiredStatus → PodAvailability (provider-owned vocabulary, Phase 14.C)
│   ├── training/            # RunPodProvider (cloud GPU via GraphQL API)
│   ├── inference/pods/      # RunPodPodInferenceProvider (volume + pod provisioning)
│   └── runtime/             # RunPodPodLifecycleClient (in-pod runner side, Phase 14.B)
└── constants.py
```

## Common conventions

- **Provider config schema**: in YAML each provider under `providers.<provider>` must have four blocks: `connect`, `training`, `inference`, `cleanup`. Even if a block is unused — it must be present per the schema.
- **Errors and returns**: providers use `Result[T, E]` (`Ok(...)` / `Err("message")`). Error text is short and operational, with a config path when possible (example: `providers.runpod.inference.volume.id`).
- **Secrets**: supplied via `Secrets`. Never put secrets in manifest/README/scripts — scripts read env or `secrets.env`. Required secret names are declared by `IGPUProvider.required_secrets()` and validated at startup.
- **Idempotency**: "list/lookup → create" pattern with deterministic resource names. On transient 5xx — verify existence before retrying create.
- **Logging**: training — `logging.getLogger("ryotenkai")`, inference — `from src.utils.logger import logger`.
- **Two-source-of-truth invariants** (Phase 14.A+): when a provider declares a capability flag, it MUST also implement the corresponding behaviour. Factory-level invariant tests in `src/tests/unit/providers/training/test_factory_capability_invariant.py` enforce this — adding a new provider requires adding rows there.

## Training providers (IGPUProvider)

### IGPUProvider — the core Protocol

Every training provider implements `IGPUProvider` (Phase 14.A
extended). The Protocol is `@runtime_checkable` — `isinstance(p, IGPUProvider)`
returns True for any class that satisfies the shape.

```python
class IGPUProvider(Protocol):
    @property
    def provider_name(self) -> str: ...      # "single_node" / "runpod" / future
    @property
    def provider_type(self) -> str: ...      # "local" / "cloud"

    # --- Lifecycle ---
    def connect(self, *, run: RunContext) -> Result[SSHConnectionInfo, ProviderError]: ...
    def disconnect(self) -> Result[None, ProviderError]: ...
    def get_status(self) -> ProviderStatus: ...
    def check_gpu(self) -> Result[GPUInfo, ProviderError]: ...
    def mark_error(self) -> None: ...
    def get_resource_info(self) -> PodResourceInfo | None: ...

    # --- Capabilities (Phase 14.A + 14.D+F) ---
    def get_capabilities(self) -> ProviderCapabilities: ...

    # --- Runtime contract (Phase 14.A) ---
    def required_runtime_env_vars(
        self, *, resource_id: str | None,
    ) -> dict[str, str]: ...
    def probe_availability(self, resource_id: str) -> AvailabilityVerdict: ...

    # --- Secret validation (Phase 14.D+F) ---
    def required_secrets(self) -> tuple[str, ...]: ...

    # --- Legacy training hook (kept for back-compat) ---
    def prepare_training_script_hooks(
        self, ssh_client: SSHClient, context: dict[str, Any],
    ) -> Result[TrainingScriptHooks, ProviderError]: ...
```

Lifecycle: `connect()` → `check_gpu()` (optional) → training runs → `disconnect()`.

### ITerminalActionProvider — capability-gated Protocol

A **separate** Protocol for providers that support cloud lifecycle
actions (pause / resume / terminate). RunPod implements; single_node
does NOT — type checker rejects `provider.pause()` at the callsite
when the provider is single_node.

```python
@runtime_checkable
class ITerminalActionProvider(Protocol):
    def terminate(self, *, resource_id: str, reason: str) -> Result[None, ProviderError]: ...
    def pause(self, *, resource_id: str) -> Result[None, ProviderError]: ...
    def resume(self, *, resource_id: str) -> Result[None, ProviderError]: ...
```

**Two-source-of-truth invariant** (Phase 14.A § R-4):
```
caps.supports_lifecycle_actions == isinstance(provider, ITerminalActionProvider)
```
The factory test pins this; a provider that updates the flag but forgets to inherit the Protocol (or vice versa) fails CI before merge.

### IPodLifecycleClient — runner-side async Protocol

Distinct from `ITerminalActionProvider`: lives in `src/runner/runtime/`,
runs **inside** the GPU pod (in the in-pod FastAPI runner), is
**async** (FastAPI/uvicorn already owns an event loop), and uses
`LifecycleActionResult` instead of `Result[T, E]` to surface retry
attempts + idempotency markers.

```python
@runtime_checkable
class IPodLifecycleClient(Protocol):
    @property
    def provider_name(self) -> str: ...
    async def terminate(self, *, resource_id: str) -> LifecycleActionResult: ...
    async def pause(self, *, resource_id: str) -> LifecycleActionResult: ...
    async def resume(self, *, resource_id: str) -> LifecycleActionResult: ...
```

Mac-side (`ITerminalActionProvider`) is sync + uses the SDK; runner-side (`IPodLifecycleClient`) is async + uses GraphQL via httpx. See [`docs/plans/phase-14b-pod-terminator-extraction.md`](../../docs/plans/phase-14b-pod-terminator-extraction.md) for the rationale on keeping them as two distinct Protocols.

### ProviderCapabilities — declarative capability flags

```python
@dataclass
class ProviderCapabilities:
    # Pre-Phase 14
    provider_type: str                        # "local" | "cloud"
    supports_multi_gpu: bool = False
    supports_spot_instances: bool = False
    max_runtime_hours: int | None = None
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None

    # Phase 14.A — capability surface for multi-provider refactor
    supports_lifecycle_actions: bool = False
    """True iff provider implements ITerminalActionProvider."""

    volume_kind: VolumeKind = VolumeKind.PERSISTENT
    """PERSISTENT (cloud, stop-able), NETWORK (cloud, terminate-only),
    LOCAL_DISK (single_node host)."""

    has_pause_resume: bool = False
    """Subset of supports_lifecycle_actions — full pause↔resume cycle."""

    runner_workspace_root: str = "/workspace"
    """Where HELIX_WORKSPACE / PYTHONPATH resolve in the in-pod runner."""

    # Phase 14.D+F — provider-leak elimination flags
    is_local: bool = False
    """True for always-on local hosts (single_node).
    Replaces provider_name == "single_node" string-checks."""

    supports_log_download: bool = False
    """True for cloud providers with SCP/HTTP log fetch.
    Replaces provider_name == PROVIDER_RUNPOD checks in GPUDeployer."""
```

Pipeline modules gate on these flags instead of comparing
`provider.provider_name`. New flags are added when a third+
provider needs to express a behaviour the existing two don't share.

### required_secrets — startup validation contract

`startup_validator` iterates the tuple at boot and fails fast if any
secret is missing from the operator's environment (or `secrets.env`).
Adding a provider with a new credential = update the provider's impl
+ the resolver in `src/pipeline/bootstrap/startup_validator.py::_resolve_required_secrets_for_provider`.

```python
# RunPod
def required_secrets(self) -> tuple[str, ...]:
    return ("RUNPOD_API_KEY",)

# single_node
def required_secrets(self) -> tuple[str, ...]:
    return ()
```

### Factory + auto-registration

```python
from src.providers.training.factory import GPUProviderFactory

result = GPUProviderFactory.create(
    provider_name="runpod",          # key from providers: in YAML
    provider_config=config.providers["runpod"],
    secrets=secrets,
)
provider = result.unwrap()

# Convenience method from full pipeline config
result = GPUProviderFactory.create_from_config("single_node", providers_config, secrets)
```

Providers register automatically when their package's `__init__.py` is imported via `GPUProviderFactory.register(...)`.

### Existing implementations

**SingleNodeProvider** (`src/providers/single_node/training/`):
- SSH via alias from `~/.ssh/config` or explicit host/port/user/key
- Health checks before training
- Auto GPU detection via `nvidia-smi`
- Capabilities: `is_local=True, volume_kind=LOCAL_DISK, supports_lifecycle_actions=False, supports_log_download=False`
- `required_secrets() == ()`
- Does NOT implement `ITerminalActionProvider`
- Runner-side: `NoOpPodLifecycleClient` (always returns `outcome="skipped"`)

**RunPodProvider** (`src/providers/runpod/training/`):
- Pod creation via GraphQL API
- Wait for readiness + SSH availability
- Automatic cleanup on disconnect (if `cleanup.auto_delete_pod`)
- Spot instances and multiple GPU types
- Capabilities: `is_local=False, volume_kind=PERSISTENT, supports_lifecycle_actions=True, has_pause_resume=True, supports_log_download=True`
- `required_secrets() == ("RUNPOD_API_KEY",)`
- DOES implement `ITerminalActionProvider`
- Runner-side: `RunPodPodLifecycleClient` (GraphQL via httpx, retry-aware)
- `from_resume_metadata(api_key)` classmethod — minimal-construction factory used by `LaunchResumeService` to bypass the heavy Pydantic validator chain when only `terminate`/`pause`/`resume`/`probe_availability` are needed

## Adding a new training provider

A new GPU provider (e.g. `lambda`, `vast_ai`, `aws`) is a multi-step
contract — every step is a separate file in a separate package, and
every step has a corresponding test that pins the contract.

### 1. Config schema

Pydantic models in `src/config/providers/<provider>/`. Four blocks:
`connect`, `training`, `inference`, `cleanup`. Strict
(`extra="forbid"`) so unknown fields surface as clear errors.

### 2. Constants

Add `PROVIDER_<NAME>: Final[Literal["..."]]` to `src/constants.py`. Used by:
- `provider_name` property on the impl
- `RYOTENKAI_RUNTIME_PROVIDER` env value
- Registry keys (Mac + runner side)

### 3. Mac-side IGPUProvider implementation

`src/providers/<provider>/training/provider.py`. Constructor
`(config: dict, secrets: Secrets)`. Implements every method of the
Protocol. Capability flags reflect actual behaviour:

```python
class MyProvider(IGPUProvider):  # add ITerminalActionProvider if applicable
    def __init__(self, config: dict[str, Any], secrets: Secrets) -> None:
        self._cfg = MyProviderConfig.from_dict(config)
        self._api_key = secrets.my_api_key
        ...

    @property
    def provider_name(self) -> str:
        return PROVIDER_MY

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_type="cloud",
            is_local=False,
            volume_kind=VolumeKind.PERSISTENT,
            supports_lifecycle_actions=True,    # set if implementing ITerminalActionProvider
            supports_log_download=True,
            ...
        )

    def required_secrets(self) -> tuple[str, ...]:
        return ("MY_API_KEY",)

    def required_runtime_env_vars(
        self, *, resource_id: str | None,
    ) -> dict[str, str]:
        env = {RUNTIME_PROVIDER_ENV_VAR: PROVIDER_MY, "MY_API_KEY": self._api_key}
        if resource_id:
            env["MY_RESOURCE_ID"] = resource_id
        return env

    def probe_availability(self, resource_id: str) -> AvailabilityVerdict:
        # Query the cloud API; map status to "running" / "sleeping_resumable" / "gone" / "probe_failed"
        ...

    # ... connect/disconnect/check_gpu/get_status/mark_error/get_resource_info ...
    # ... prepare_training_script_hooks (legacy, kept for back-compat) ...
```

If your provider supports cloud lifecycle (terminate/pause/resume),
ALSO inherit `ITerminalActionProvider` and implement its three
methods — the type system will then enforce capability-gated calls
across the codebase.

### 4. Auto-register on import

In `src/providers/<provider>/training/__init__.py`:

```python
from src.constants import PROVIDER_MY
from src.providers.training.factory import GPUProviderFactory
from .provider import MyProvider

if not GPUProviderFactory.is_registered(PROVIDER_MY):
    GPUProviderFactory.register(PROVIDER_MY, MyProvider)

__all__ = ["MyProvider"]
```

Add `importlib.import_module("src.providers.<provider>.training")` to `auto_register_providers()` in `src/providers/training/factory.py`.

### 5. Startup secret validator

In `src/pipeline/bootstrap/startup_validator.py`, add the provider
name to `_resolve_required_secrets_for_provider`:

```python
def _resolve_required_secrets_for_provider(provider_name: str) -> tuple[str, ...]:
    if provider_name == PROVIDER_RUNPOD:
        return ("RUNPOD_API_KEY",)
    if provider_name == PROVIDER_SINGLE_NODE:
        return ()
    if provider_name == PROVIDER_MY:                  # <— new branch
        return ("MY_API_KEY",)
    return ()
```

### 6. Runner-side IPodLifecycleClient (if your provider supports lifecycle)

Only required if your provider's `supports_lifecycle_actions=True`.
The runner uses this from inside the pod for terminal hooks
(automatic pod stop / terminate after training).

`src/providers/<provider>/runtime/lifecycle_client.py`:

```python
class MyPodLifecycleClient(IPodLifecycleClient):
    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return PROVIDER_MY

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
        # Async HTTP call to your cloud API; return LifecycleActionResult
        # with outcome string + attempts_made + last_error.
        ...

    async def pause(self, *, resource_id: str) -> LifecycleActionResult: ...
    async def resume(self, *, resource_id: str) -> LifecycleActionResult: ...
```

If your provider does NOT support lifecycle, **skip this step** — the
runner registry maps your provider name to `NoOpPodLifecycleClient`
(see step 7).

### 7. Runner-side registry entry

In `src/runner/runtime/provider_registry.py`, add a builder for
your provider:

```python
def _build_my_client(env: Mapping[str, str]) -> IPodLifecycleClient:
    api_key = env.get("MY_API_KEY")
    if not api_key:
        raise BootstrapConfigError(
            f"{RUNTIME_PROVIDER_ENV_VAR}=my requires MY_API_KEY"
        )
    from src.providers.my.runtime.lifecycle_client import MyPodLifecycleClient
    return MyPodLifecycleClient(api_key=api_key)


_REGISTRY: Final[dict[str, Callable[[Mapping[str, str]], IPodLifecycleClient]]] = {
    PROVIDER_RUNPOD: _build_runpod_client,
    PROVIDER_SINGLE_NODE: _build_single_node_client,
    PROVIDER_MY: _build_my_client,                # <— new entry
}
```

If your provider has no lifecycle, point it at `_build_single_node_client` (which produces `NoOpPodLifecycleClient`).

### 8. Mac-side resume support (if applicable)

If your provider supports lifecycle, extend
`src/pipeline/launch/resume_service.py::_default_resolve_lifecycle_provider`
with a branch that builds your provider for the resume flow:

```python
if provider_name == PROVIDER_MY:
    api_key = os.environ.get("MY_API_KEY")
    if not api_key:
        return None
    from src.providers.my.training.provider import MyProvider
    return MyProvider.from_resume_metadata(api_key=api_key)
```

This requires implementing a `from_resume_metadata` classmethod on
your provider (minimal-construction factory bypassing the heavy
Pydantic validator chain — see `RunPodProvider.from_resume_metadata`
for the pattern).

### 9. Tests

Each step has a test pinning the contract:

| Test file | What it pins |
|---|---|
| `src/tests/unit/providers/<provider>/training/test_provider_capabilities.py` | Capability flags + `required_runtime_env_vars` / `required_secrets` shape |
| `src/tests/unit/providers/training/test_factory_capability_invariant.py` | Two-source-of-truth: caps ↔ Protocol parity, secrets ⊆ runtime env keys, expected outcomes per provider |
| `src/tests/unit/providers/<provider>/runtime/test_lifecycle_client.py` | Async terminate/pause/resume happy paths, retry, idempotency markers |
| `src/tests/unit/runner/runtime/test_provider_registry.py` | Registry membership + bootstrap failure messages mention your provider |
| `src/tests/unit/runner/test_main_lifespan_bootstrap.py` | Lifespan boots with valid env; `BootstrapConfigError` on missing creds |

**Add rows for your provider** to:
- `test_factory_capability_invariant.py` parameterized matrix (`_mk_my()` factory + expected capability values)
- `test_main_lifespan_bootstrap.py` env permutations

### 10. Use the centralized fake fixture

For unit tests that need a stub provider (not specifically about
your impl), use `src/tests/fixtures/providers.py::FakeGPUProvider`
or `FailingGPUProvider` — don't roll your own.

```python
from src.tests.fixtures.providers import FakeGPUProvider

provider = FakeGPUProvider(
    provider_name_value="my",
    capabilities=ProviderCapabilities(
        provider_type="cloud", is_local=False,
        supports_log_download=True,
    ),
    required_secrets_value=("MY_API_KEY",),
)
```

## Inference providers (IInferenceProvider)

### Interface

```python
class IInferenceProvider(Protocol):
    @property
    def provider_name(self) -> str: ...
    @property
    def provider_type(self) -> str: ...

    def deploy(self, model_source: str, *, run_id: str, base_model_id: str,
               trust_remote_code: bool = False, lora_path: str | None = None,
               quantization: str | None = None, keep_running: bool = False,
               ) -> Result[EndpointInfo, InferenceError]: ...
    def set_event_logger(self, event_logger: InferenceEventLogger | None) -> None: ...
    def get_pipeline_readiness_mode(self) -> PipelineReadinessMode: ...
    def collect_startup_logs(self, *, local_path: Path) -> None: ...
    def build_inference_artifacts(self, *, ctx: InferenceArtifactsContext) -> Result[InferenceArtifacts, InferenceError]: ...
    def undeploy(self) -> Result[None, InferenceError]: ...
    def health_check(self) -> Result[bool, InferenceError]: ...
    def get_capabilities(self) -> InferenceCapabilities: ...
    def get_endpoint_info(self) -> EndpointInfo | None: ...
    def activate_for_eval(self) -> Result[str, InferenceError]: ...
    def deactivate_after_eval(self) -> Result[None, InferenceError]: ...
```

`PipelineReadinessMode`:
- `WAIT_FOR_HEALTHY` — pipeline waits for a healthy endpoint after `deploy()`
- `SKIP` — resource is "parked"; readiness is handled by generated scripts

Lifecycle: `set_event_logger()` → `deploy()` → `health_check()` → `activate_for_eval()` → `deactivate_after_eval()` → `undeploy()`

### Factory

```python
from src.providers.inference.factory import InferenceProviderFactory

result = InferenceProviderFactory.create(config=config, secrets=secrets)
provider = result.unwrap()
```

The factory picks the implementation from `config.inference.provider` (if/elif pattern).

### Existing implementations

**SingleNodeInferenceProvider** (`src/providers/single_node/inference/`):
- Deploy vLLM via SSH + Docker
- Health check via HTTP
- `PipelineReadinessMode.WAIT_FOR_HEALTHY`
- Artifacts: manifest + chat script + stop script + README

**RunPodPodInferenceProvider** (`src/providers/runpod/inference/pods/`):
- Volume provisioning + pod provisioning
- After provisioning the pod **stops** (STOP mode)
- `PipelineReadinessMode.SKIP` — actual startup via generated scripts
- Artifacts: manifest + start script + stop script + README

### Adding a new inference provider

1. **Config** — provider-specific under `providers.<provider>.inference.*`; global engine settings (vLLM, etc.) from `config.inference.*`
2. **Implementation** — `src/providers/<provider>/inference/<variant>/provider.py`, constructor `(*, config: PipelineConfig, secrets: Secrets)`
3. **Registration** — add a branch in `src/providers/inference/factory.py`:
   ```python
   if provider == PROVIDER_MY:
       from src.providers.my.inference.provider import MyInferenceProvider
       return Success(MyInferenceProvider(config=config, secrets=secrets))
   ```
4. **Artifacts** — implement `build_inference_artifacts(*, ctx)`, return `Result[InferenceArtifacts, InferenceError]` with `manifest`, `chat_script`, `readme`

## Imports

All providers live only under `src/providers/`. Direct imports:

```python
# GPU/training
from src.providers.training.factory import GPUProviderFactory
from src.providers.training.interfaces import (
    IGPUProvider, ITerminalActionProvider,
    SSHConnectionInfo, GPUInfo, ProviderCapabilities,
    AvailabilityVerdict, VolumeKind, ProviderStatus,
)
from src.utils.ssh_client import SSHClient

# Runner-side (in-pod control plane)
from src.runner.runtime.lifecycle_client import (
    IPodLifecycleClient, LifecycleActionResult,
)
from src.runner.runtime.provider_registry import (
    resolve_lifecycle_client_from_env, BootstrapConfigError,
)

# Mac-side resume orchestrator
from src.pipeline.launch.resume_service import (
    LaunchResumeService, ResumeOutcome, ResumeProgress,
)

# Inference
from src.providers.inference.factory import InferenceProviderFactory
from src.providers.inference.interfaces import (
    IInferenceProvider, EndpointInfo, PipelineReadinessMode,
)

# Test fixtures
from src.tests.fixtures.providers import FakeGPUProvider, FailingGPUProvider
```

## Phase 14 reference plans

The Phase 14 multi-step refactor that introduced the current
provider abstractions is documented in detail under
[`docs/plans/`](../../docs/plans/):

- `phase-14a-provider-abstraction.md` — IGPUProvider extension, ITerminalActionProvider, RYOTENKAI_RUNTIME_PROVIDER env contract
- `phase-14b-pod-terminator-extraction.md` — IPodLifecycleClient, runner-side registry, BootstrapConfigError
- `phase-14c-launch-resume-unification.md` — LaunchResumeService, ResumeOutcome, ResumeProgress
- `phase-14df-provider-leak-elimination.md` — capability flags (is_local, supports_log_download), required_secrets, FakeGPUProvider fixture
- `phase-14e-srp-soc-fixes.md` — lifespan deferred binding, heartbeat helper, ExceptionClassifier

Read the relevant plan when extending its surface — it carries the
"why" behind each architectural decision and the deepsink risk
analysis.

## Tests

- Training: verify registration/creation via `GPUProviderFactory`, negative cases (missing secrets, unknown provider), the **two-source-of-truth invariants** in `test_factory_capability_invariant.py`
- Runner-side: verify `IPodLifecycleClient` impl, the runner's `_REGISTRY` membership in `test_provider_registry.py`, lifespan bootstrap matrix in `test_main_lifespan_bootstrap.py`
- Inference: verify `InferenceProviderFactory.create(...)`, `build_inference_artifacts()` returns a valid manifest
- Goal: tests should fail **before** the pipeline starts creating real resources
