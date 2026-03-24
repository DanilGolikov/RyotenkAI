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
│   ├── interfaces.py    # IGPUProvider (Protocol), SSHConnectionInfo, GPUInfo, ProviderCapabilities, ProviderStatus
│   └── factory.py       # GPUProviderFactory (registry) + auto_register_providers()
├── inference/
│   ├── interfaces.py    # IInferenceProvider (Protocol), EndpointInfo, InferenceArtifacts, PipelineReadinessMode
│   └── factory.py       # InferenceProviderFactory (if/elif by config.inference.provider)
├── single_node/
│   ├── training/        # SingleNodeProvider (local SSH)
│   │   ├── provider.py
│   │   ├── config.py    # compat shim → src/config/providers/single_node/
│   │   └── health_check.py
│   └── inference/       # SingleNodeInferenceProvider (vLLM via SSH+Docker)
│       ├── provider.py
│       └── artifacts.py
├── runpod/
│   ├── training/        # RunPodProvider (cloud GPU via GraphQL API)
│   │   ├── provider.py
│   │   ├── api_client.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── lifecycle_manager.py
│   │   └── cleanup_manager.py
│   └── inference/
│       └── pods/        # RunPodPodInferenceProvider (volume + pod provisioning)
│           ├── provider.py
│           ├── api_client.py
│           ├── pod_session.py
│           ├── constants.py
│           └── artifacts.py
└── constants.py
```

## Common conventions

- **Provider config schema**: in YAML each provider under `providers.<provider>` must have four blocks: `connect`, `training`, `inference`, `cleanup`. Even if a block is unused — it must be present per the schema.
- **Errors and returns**: providers use `Result[T, E]` (`Ok(...)` / `Err("message")`). Error text is short and operational, with a config path when possible (example: `providers.runpod.inference.volume.id`).
- **Secrets**: supplied via `Secrets`. Never put secrets in manifest/README/scripts — scripts read env or `secrets.env`.
- **Idempotency**: "list/lookup → create" pattern with deterministic resource names. On transient 5xx — verify existence before retrying create.
- **Logging**: training — `logging.getLogger("ryotenkai")`, inference — `from src.utils.logger import logger`.

## Training providers (IGPUProvider)

### Interface

```python
class IGPUProvider(Protocol):
    @property
    def provider_name(self) -> str: ...      # "single_node" / "runpod"
    @property
    def provider_type(self) -> str: ...      # "local" / "cloud"

    def connect(self, *, run: RunContext) -> Result[SSHConnectionInfo, ProviderError]: ...
    def disconnect(self) -> Result[None, ProviderError]: ...
    def get_status(self) -> ProviderStatus: ...
    def check_gpu(self) -> Result[GPUInfo, ProviderError]: ...
    def get_capabilities(self) -> ProviderCapabilities: ...
    def get_resource_info(self) -> dict[str, Any] | None: ...
```

Lifecycle: `connect()` → (optional) `check_gpu()` → ... → `disconnect()`

### Factory

```python
from src.providers.training.factory import GPUProviderFactory

# Create a provider
result = GPUProviderFactory.create(
    provider_name="runpod",          # key from providers: in YAML
    provider_config=config.providers["runpod"],
    secrets=secrets,
)
provider = result.unwrap()

# Convenience method from full pipeline config
result = GPUProviderFactory.create_from_config("single_node", providers_config, secrets)
```

Providers register automatically when their `__init__.py` is imported via `GPUProviderFactory.register(...)`.

### Provider constructor

```python
class MyGPUProvider:
    def __init__(self, config: dict[str, Any], secrets: Secrets):
        # Parse config into a Pydantic model
        self._cfg = MyProviderConfig.from_dict(config)
```

### Existing implementations

**SingleNodeProvider** (`src/providers/single_node/training/`):
- SSH via alias from `~/.ssh/config` or explicit host/port/user/key
- Health checks before training
- Auto GPU detection via `nvidia-smi`
- Workspace management (`cleanup.cleanup_workspace`)

**RunPodProvider** (`src/providers/runpod/training/`):
- Pod creation via GraphQL API
- Wait for readiness + SSH availability
- Automatic cleanup on disconnect (if `cleanup.auto_delete_pod`)
- Spot instances and multiple GPU types

### Adding a new training provider

1. **Config** — Pydantic models in `src/config/providers/<provider>/` (four blocks: connect/training/inference/cleanup)
2. **Implementation** — `src/providers/<provider>/training/provider.py`, constructor `(config: dict, secrets: Secrets)`
3. **Registration** — in the provider's `__init__.py`:
   ```python
   from src.providers.training.factory import GPUProviderFactory
   from src.constants import PROVIDER_MY

   if not GPUProviderFactory.is_registered(PROVIDER_MY):
       GPUProviderFactory.register(PROVIDER_MY, MyProvider)
   ```
4. **Auto-discovery** — add `importlib.import_module("src.providers.<provider>.training")` to the factory's `auto_register_providers()`

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

### Provider constructor

```python
class MyInferenceProvider:
    def __init__(self, *, config: PipelineConfig, secrets: Secrets):
        provider_cfg = config.get_provider_config("my_provider")
        self._cfg = MyInferenceConfig.model_validate(provider_cfg)
```

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
from src.providers.training.interfaces import IGPUProvider, SSHConnectionInfo
from src.utils.ssh_client import SSHClient

# Inference
from src.providers.inference.factory import InferenceProviderFactory
from src.providers.inference.interfaces import IInferenceProvider, EndpointInfo, PipelineReadinessMode
```

## Tests

- Training: verify registration/creation via `GPUProviderFactory`, negative cases (missing secrets, unknown provider)
- Inference: verify `InferenceProviderFactory.create(...)`, `build_inference_artifacts()` returns a valid manifest
- Goal: tests should fail **before** the pipeline starts creating real resources
