# Architectural Stabilization: Discriminated Unions + Auto-IDs

**Date:** 2026-05-06
**Branch baseline:** `RESEACRH @ 8dd8ee9`
**Owner:** daniil
**Status:** PLANNED — awaiting approval before execution.

---

## 1. Context

The codebase has accumulated **four** instances of the same anti-pattern: a `string-typed selector field` plus a `flat namespace of variant configs`, with `if/elif` branching elsewhere to dispatch on the selector. Every time we want to add a new variant (engine / adapter / dataset source / etc.), we edit a dozen files scattered across `shared/`, `providers/`, `community/`, and `control/`. This is the opposite of OCP (Open/Closed Principle).

This plan **unifies** all four instances under a single discriminated-union pattern (`kind` discriminator + Pydantic Tag-based union from day one), **plus** a related UX problem — plugin instance IDs that today must be hand-typed by users and provide no value beyond a string label.

### Concrete pain points being fixed

| # | Concern | Today's shape | Pain |
|---|---|---|---|
| **1. Inference Engine** | `inference.engine: Literal["vllm"]` + `inference.engines.vllm: VLLMConfig` | Adding SGLang = ~12 file edits across packages. Each provider hardcodes `self._engine = VLLMEngine()`. |
| **2. Training Adapter** | `training.type: str` + flat `training.lora`, `training.qlora`, `training.adalora` (only one is non-None) | Adding new adapter type = manual `if/elif` updates in validators + `get_adapter_config()` dispatch. No type-safe binding between selector and config. |
| **3. Dataset Source** | `dataset.source_type: Literal["local"|"huggingface"]` + flat `source_local: ... \| None`, `source_hf: ... \| None` | Same pattern: selector + namespace, with manual cross-validation. |
| **4. Plugin instance IDs** | `EvaluatorPluginConfig.id: str` (required, user-typed) + `DatasetValidationPluginConfig.id: str` (same) | UX friction: user must hand-craft unique id for every plugin invocation, even when they invoke the same plugin with different params. Risk of typos and collisions. |

### Boundary conditions

- **No backwards compatibility** (user explicit choice): «обратную совместимость не пилим, от старого/плохого кода избавляемся». Old shapes are deleted, not deprecated.
- **All four concerns landed together in one cohesive PR-train.** User chose «всё вместе» — discriminated unions are too similar to fragment.
- **Plan A for Inference Engine** (full plugin system with `engine.toml` manifests + `EngineRegistry` + new workspace member `packages/engines/`). User explicit choice.
- **Auto-id strategy: hash-based** — `f"{plugin}_{md5(params)[:8]}"`. Deterministic, survives reordering, MLflow artifact path stable across runs.
- **Tag-based discriminator from day one** — `Annotated[Annotated[T, Tag("…")], Discriminator("kind")]`. Works with single-variant unions (today's vLLM-only case), enforces type-safety from the first commit.

---

## 2. TL;DR

```
DISCRIMINATED UNION REFACTORS (3)

A) inference.engine: EngineConfigUnion         ← Tag-based discriminated
   was: engine: Literal["vllm"] + engines.vllm: VLLMConfig
   plus: NEW packages/engines/ with engine.toml manifests + EngineRegistry

B) training.adapter: AdapterConfigUnion         ← Tag-based discriminated
   was: type: str + lora|qlora|adalora flat fields
   moves config from training/schema.py 17-83 into per-adapter Pydantic classes

C) dataset.source: DatasetSourceUnion           ← Tag-based discriminated
   was: source_type: Literal[…] + source_local|source_hf flat fields
   uses existing DatasetSourceLocal/DatasetSourceHF, adds kind discriminator

NOT MIGRATING:
   • Providers — already type-safe via dict-key + manifest pattern; adding `kind`
     would duplicate provider.toml id. Decision RD-2 in §16.
   • Strategies — current generic `params: dict[str, Any]` is intentional
     (avoids per-strategy boilerplate); migrate later if/when strategies
     diverge in required fields.
   • Integrations — only 2 hardcoded fields (mlflow, huggingface), no selector pattern.

AUTO-ID FOR PLUGIN INSTANCES (1)

D) EvaluatorPluginConfig.id      ← optional, hash-based default
   DatasetValidationPluginConfig.id  ← same

   id: str | None = None
   if id is None: id = f"{plugin}_{md5(params, sort_keys=True)[:8]}"
   collision suffix _2, _3 if duplicate hash within parent list.

NEW WORKSPACE MEMBER     packages/engines/         ← engine plugin discovery
DELETED                  packages/shared/.../inference/__about__.py (image dict)
                         constants: INFERENCE_ENGINE_VLLM, SUPPORTED_INFERENCE_ENGINES
                         InferenceEnginesConfig (flat namespace)
                         training.adapter flat fields (lora/qlora/adalora)
                         dataset source_local/source_hf flat fields

Adding a new engine ⇒ ~140 LOC across one new folder.
Adding a new adapter ⇒ new Pydantic class + 1 line in AdapterConfigUnion.
Adding a new dataset source ⇒ new Pydantic class + 1 line in DatasetSourceUnion.
```

---

## 3. Why three unions plus auto-id, all in one PR-train

The four concerns share **the same shape and the same fix**. Doing them separately means:

- **3× the migration risk** — each refactor breaks tests in the same way; one regression-fix template applies to all.
- **3× the review fatigue** — reviewer reads the same Pydantic Tag-based discriminator pattern three times.
- **3× the YAML migration** — every test fixture and example YAML touches all three at once anyway.

Doing them together:
- Single migration script (`scripts/migrate_config_to_discriminated_unions.py`) handles all old YAML shapes.
- Single regression sweep.
- Single Pydantic Tag-based discriminator pattern documented once, applied three times.
- Single set of importlinter contracts and sentinel tests.

Auto-id for evaluator/validator plugins is grouped because:
- It touches the same config files (evaluation/schema.py, datasets/validation.py).
- Same migration script can rewrite explicit `id:` lines to omit them when they match auto-generated.
- User asked explicitly to bundle.

---

## 4. Industry research findings (2025-2026)

| Source | Pattern | What we adopt |
|---|---|---|
| **NVIDIA Triton Inference Server** | Backend C-API; engines are compiled shared libraries. | NOT our pattern — we run engines as black-box containers. |
| **HuggingFace TGI** | OpenAI-compatible HTTP. **Maintenance mode since Dec 2025.** | Confirms vLLM/SGLang/MAX as the active engine set. |
| **vLLM / SGLang / MAX / Ollama** | OpenAI-compatible HTTP at `/v1/*`. | `api_dialect: "openai_compatible"` is our lingua franca. |
| **KServe ServingRuntime CRD** | Declarative manifest: image + supported_model_formats + capabilities. | **Direct analogue** for `engine.toml`. |
| **BentoML / OpenLLM** | Service classes embedding engines as Python libs. | NOT our pattern — we run engines as containers. |
| **LiteLLM** | Provider proxy with single API surface. | Inspired our `OpenAICompatibleInferenceClient`. |
| **Pydantic v2 discriminated unions** (Rust-backed) | `Annotated[Union[A, B], Field(discriminator="kind")]` | Our config primitive. Tag-based for single-member case. |
| **Terraform / Pulumi** | Auto-generated stable URN from logical name + parents. | Inspires hash-based auto-id (`{plugin}_{md5(params)[:8]}`). |

---

## 5. Architectural decisions (codified upfront)

| ID | Decision | Rationale |
|---|---|---|
| **AD-1** | Tag-based discriminator from day one (`Annotated[T, Tag("…")] + Discriminator("kind")`) | Works with single-variant unions; uniform contract; type-safety at PR-1, not at "second variant added". |
| **AD-2** | New workspace member `packages/engines/` | Symmetry with `packages/community/` and `packages/providers/`. Marketplace-ready. Keeps importlinter contracts clean. |
| **AD-3** | NOT applying `kind` to providers | Provider id is already encoded via `provider.toml` manifest + dict key in YAML. Adding internal `kind` would duplicate. |
| **AD-4** | Auto-id is hash-based, not positional | Deterministic across reorderings → MLflow artifact paths stable across runs. |
| **AD-5** | Config classes for variants live next to their runtime | `VLLMEngineConfig` lives in `packages/engines/.../vllm/config.py`, NOT `packages/shared/.../inference/engines/vllm.py`. Pod imports config classes (cheap — pure Pydantic, no runtime deps). |
| **AD-6** | Discriminator field uniformly named `kind` (not `type`/`engine`/`source_type`/`strategy_type`) | Prior art is divergent. Pick one name, apply everywhere new. Old fields are renamed during migration. |
| **AD-7** | `LaunchSpec` (structured) instead of shell string from engine | Decouples engine from execution mechanism. Forward-compatible to k8s providers. |
| **AD-8** | Remove `quantization` kwarg from `IInferenceProvider.deploy()` API | Engine-specific knob; provider reads from its `engine_cfg` directly. Deploy API stays generic. |
| **AD-9** | Image name follows convention `f"{prefix}/inference-{engine_id}:{engine_version}"` | `[image].default` in `engine.toml` is fully OPTIONAL. Author drops Dockerfile + manifest, image name auto-derives. Override chain (env > provider > manifest > convention) preserves flexibility. |
| **AD-10** | `engine.toml [engine].version` IS the image tag (our integration contract semver, NOT upstream version) | New upstream vLLM 0.7→0.8 = bump our `version`, rebuild image. Separate `upstream_version` field is informational only (logs, MLflow tags, observability). |
| **AD-11** | Providers stay dict-keyed (NOT migrated to discriminated union) | Adding internal `kind` to provider configs would duplicate the dict key + manifest id (three sources of truth). Multiple-instances-per-provider-type is a marginal use case; deferred. |

---

## 6. Concern A: Inference Engine plugin system (Plan A — full)

### 6.1 New workspace member: `packages/engines/`

```
packages/engines/
├── pyproject.toml                          # workspace member
├── README.md                                # engine author guide
├── scripts/
│   └── check_engine_manifests.py           # drift detector — runs in CI
├── src/ryotenkai_engines/
│   ├── __init__.py
│   ├── interfaces.py                       # IInferenceEngine Protocol, EngineCapabilities
│   ├── manifest.py                         # EngineManifest pydantic + invariants
│   ├── registry.py                         # EngineRegistry (filesystem discovery)
│   ├── images.py                           # resolve_image()
│   ├── capabilities.py                     # EngineCapabilities pydantic
│   ├── _config_union.py                    # Tag-based discriminated union builder
│   ├── errors.py                           # EngineRegistryError, EngineConfigError
│   └── vllm/
│       ├── __init__.py
│       ├── engine.toml                     # manifest
│       ├── runtime.py                      # VLLMEngineRuntime(IInferenceEngine)
│       └── config.py                       # VLLMEngineConfig (kind="vllm")
└── tests/
    ├── unit/                               # all 8 categories
    ├── contract/test_engine_protocol_parity.py
    └── sentinel/test_no_provider_imports.py
```

### 6.2 `IInferenceEngine` Protocol

**File:** `packages/engines/src/ryotenkai_engines/interfaces.py` (NEW).

```python
from typing import Protocol, runtime_checkable, Literal, ClassVar
from pydantic import BaseModel, Field
from ryotenkai_shared.utils.result import Result, AppError


class EngineCapabilities(BaseModel):
    """Mirrors engine.toml [capabilities] 1:1. Drift-detected in CI."""
    api_dialect: Literal["openai_compatible", "custom"]
    supports_lora: bool
    supports_quantization: bool
    supports_streaming: bool
    supports_tensor_parallel: bool
    supported_quantizations: tuple[str, ...] = ()
    supported_dtypes: tuple[str, ...]
    default_port: int = Field(ge=1, le=65535)


class BaseEngineConfig(BaseModel):
    """All engine configs derive from this. Subclasses MUST set
    `kind` to a Literal matching their engine.toml id."""
    kind: str
    model_config = {"extra": "forbid"}


class LaunchSpec(BaseModel):
    """Structured engine launch description. Provider decides how to wrap
    (docker, k8s Pod, systemd unit). Forward-compatible."""
    image: str
    container_name: str
    args: tuple[str, ...]            # CLI args for engine binary
    env: dict[str, str]              # extra env vars
    port: int                        # container-side port
    volumes: tuple[tuple[str, str], ...]   # (host, container)


@runtime_checkable
class IInferenceEngine(Protocol):
    """Pure command/spec builder. NO side effects. NO async. NO IO."""
    engine_id: ClassVar[str]
    config_class: ClassVar[type[BaseEngineConfig]]

    def get_capabilities(self) -> EngineCapabilities: ...

    def build_launch_spec(
        self,
        *,
        cfg: BaseEngineConfig,
        image: str,
        container_name: str,
        port: int,
        workspace_host_path: str,
        model_path_in_container: str,
    ) -> LaunchSpec: ...

    def build_healthcheck_command(self, *, host: str, port: int) -> str: ...

    def build_default_endpoint_url(self, *, host: str, port: int) -> str: ...

    def validate_config(self, cfg: BaseEngineConfig) -> Result[None, AppError]: ...
```

### 6.3 Engine manifest (`engine.toml`)

```toml
schema_version = 1

[engine]
id               = "vllm"
name             = "vLLM"
version          = "1.0.0"           # our integration contract semver → image tag
upstream_version = "0.7.0"           # informational: actual upstream vLLM version
description      = "PagedAttention-based OpenAI-compatible LLM server."
stability        = "stable"
homepage         = "https://github.com/vllm-project/vllm"

[capabilities]
api_dialect              = "openai_compatible"
supports_lora            = true
supports_quantization    = true
supports_streaming       = true
supports_tensor_parallel = true
supported_quantizations  = ["awq", "gptq", "fp8", "bitsandbytes"]
supported_dtypes         = ["bfloat16", "float16", "float32"]
default_port             = 8000

# [image] is FULLY OPTIONAL. If omitted, the image name is derived by
# convention: f"{RYOTENKAI_INFERENCE_IMAGE_REGISTRY:-ryotenkai}/inference-{id}:{version}"
# i.e. for this manifest: ryotenkai/inference-vllm:1.0.0
#
# Override only when you need a non-conventional name (community fork,
# custom registry, multi-arch tag, …):
#   [image]
#   default = "ghcr.io/myorg/custom-vllm:nightly"

[entry_points.runtime]
module = "ryotenkai_engines.vllm.runtime"
class  = "VLLMEngineRuntime"

[entry_points.config_schema]
module = "ryotenkai_engines.vllm.config"
class  = "VLLMEngineConfig"
```

`EngineManifest` Pydantic schema mirrors `ProviderManifest` (~600 LOC). Cross-field invariants (e.g. `supports_quantization=False ⇒ supported_quantizations=()`).

### 6.4 `EngineRegistry`

**File:** `packages/engines/src/ryotenkai_engines/registry.py` (NEW). Verbatim structural copy of `ProviderRegistry`:

- `from_filesystem()` walks `packages/engines/src/ryotenkai_engines/*/engine.toml`.
- `get_manifest(engine_id) → EngineManifest`.
- `get_runtime(engine_id) → type[IInferenceEngine]` (lazy importlib).
- `get_config_class(engine_id) → type[BaseEngineConfig]` (used by union builder).
- `get_image(engine_id, *, provider_overrides, env)` → resolution chain.
- `failures()` → `tuple[LoadFailure, ...]` defensive collection.
- Lock-protected module-level singleton.

### 6.5 Discriminated config (Tag-based, day one)

**File:** `packages/engines/src/ryotenkai_engines/_config_union.py` (NEW).

```python
from typing import Annotated
from pydantic import Discriminator, Tag
from ryotenkai_engines.registry import EngineRegistry


def build_engine_config_union():
    """Build the Tag-based discriminated union over all registered engine
    configs. Works with one or many — Tag/Discriminator API supports it.
    Adding an engine = drop folder, no edits to PipelineConfig."""
    registry = EngineRegistry.from_filesystem()
    members = []
    for engine_id in registry.list():
        cfg_cls = registry.get_config_class(engine_id)
        members.append(Annotated[cfg_cls, Tag(engine_id)])
    if len(members) == 1:
        return Annotated[members[0], Discriminator("kind")]
    from typing import Union
    return Annotated[Union[tuple(members)], Discriminator("kind")]


EngineConfigUnion = build_engine_config_union()
```

Used in `InferenceConfig`:

```python
# packages/shared/src/ryotenkai_shared/config/inference/schema.py — REWRITTEN
from ryotenkai_engines._config_union import EngineConfigUnion

class InferenceConfig(StrictBaseModel):
    enabled: bool = False
    provider: str | None = None
    engine: EngineConfigUnion       # Tag-based discriminated
    common: InferenceCommonConfig = Field(default_factory=InferenceCommonConfig)
```

### 6.6 Provider DI

**Before** (`single_node/inference/provider.py:108-150`):
```python
self._engine_cfg = self._inf_cfg.engines.vllm   # bypass abstraction
self._engine = VLLMEngine()                      # hardcoded
```

**After:**
```python
SUPPORTED_ENGINES: ClassVar[frozenset[str]] = frozenset({"vllm"})

def __init__(self, ctx: ProviderContext) -> None:
    super().__init__(ctx)
    self._inf_cfg = ctx.config.inference
    self._engine_cfg: BaseEngineConfig = self._inf_cfg.engine

    if self._engine_cfg.kind not in self.SUPPORTED_ENGINES:
        raise ProviderError(
            f"engine {self._engine_cfg.kind!r} not supported by "
            f"{type(self).__name__}; supported: {sorted(self.SUPPORTED_ENGINES)}",
            code="PROVIDER_ENGINE_NOT_SUPPORTED",
        )

    runtime_cls = engine_registry.get_runtime(self._engine_cfg.kind)
    self._engine: IInferenceEngine = runtime_cls()

    # Engine validates its own config (replaces inline merge_before_deploy check).
    validate_result = self._engine.validate_config(self._engine_cfg)
    if validate_result.is_err():
        raise ProviderError.from_app_error(validate_result.unwrap_err())
```

### 6.7 Provider ↔ Engine compatibility (in `provider.toml`)

```toml
[capabilities.inference]
supported_engines = ["vllm"]   # explicit list; NO wildcards

[capabilities.inference.engine_overrides.vllm]
image = "ryotenkai/runpod-vllm:cuda-12.4"   # optional per-engine override
```

Cross-validated at PipelineConfig load in `validate_inference_provider_engine_compatibility(cfg)`.

### 6.8 `deploy()` API change — drop `quantization` kwarg

**Protocol:** `packages/providers/src/ryotenkai_providers/inference/interfaces.py:135` — remove `quantization: str | None = None`.
**Caller:** `inference_deployer.py:158-167` — drop the kwarg.
**Provider:** reads `getattr(self._engine_cfg, "quantization", None)` from its own typed engine config.

### 6.9 Image resolution chain (with convention default)

**File:** `packages/engines/src/ryotenkai_engines/images.py` (NEW).

```python
def resolve_image(
    *,
    engine_id: str,
    provider_overrides: dict[str, EngineOverride] | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve image name. Order (first match wins):

    1. ENV override:       RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_<ENGINE_UPPER>
                           — for CI swaps, dev overrides, A/B tests.
    2. Provider override:  provider.toml [capabilities.inference.engine_overrides.<id>].image
                           — for provider-specific builds (e.g. RunPod CUDA-12.4).
    3. Manifest explicit:  engine.toml [image].default
                           — author opt-out from convention (forks, custom registries).
    4. CONVENTION default: f"{prefix}/inference-{id}:{version}"
                           where prefix = env RYOTENKAI_INFERENCE_IMAGE_REGISTRY or "ryotenkai"
                                 version = engine.toml [engine].version.
    """
    env = env or os.environ
    env_key = f"RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_{engine_id.upper()}"
    if env_key in env:
        return env[env_key]

    if provider_overrides and (override := provider_overrides.get(engine_id)) and override.image:
        return override.image

    manifest = EngineRegistry.from_filesystem().get_manifest(engine_id)
    if manifest.image and manifest.image.default:
        return manifest.image.default

    # Convention fallback — author dropped Dockerfile + engine.toml, no [image] block.
    prefix = env.get("RYOTENKAI_INFERENCE_IMAGE_REGISTRY", "ryotenkai")
    return f"{prefix}/inference-{engine_id}:{manifest.engine.version}"
```

### 6.10 Inference deployment flow (end-to-end)

```
1. User edits YAML:
       inference:
         enabled: true
         provider: single_node
         engine:
           kind: vllm                  ← Tag-based discriminator
           max_model_len: 8192
           quantization: awq

2. PipelineConfig.model_validate(yaml):
       • Pydantic discriminator: kind="vllm" → VLLMEngineConfig
       • Typo (kind=vlmm) → ValidationError at YAML load (no later)
       • Cross-validator: provider.toml [capabilities.inference]
         supported_engines must contain "vllm"

3. InferenceDeployer.execute(ctx):
       • image = resolve_image(engine_id="vllm", provider_overrides=...)
         → checks env, provider, manifest, falls to convention
         → "ryotenkai/inference-vllm:1.0.0"
       • provider = ProviderRegistry.create_inference("single_node", ctx)

4. SingleNodeInferenceProvider.__init__:
       • self._engine_cfg = inf_cfg.engine                  ← typed VLLMEngineConfig
       • runtime_cls = engine_registry.get_runtime("vllm")  ← lazy import
       • self._engine = runtime_cls()                       ← VLLMEngineRuntime instance
       • self._engine.validate_config(cfg) → Result          ← engine validates own invariants

5. provider.deploy(model_source, run_id, base_model_id):
       launch_spec = self._engine.build_launch_spec(
           cfg=self._engine_cfg,
           image="ryotenkai/inference-vllm:1.0.0",
           container_name=f"infer_{run_id}",
           port=8000,
           workspace_host_path="/workspace",
           model_path_in_container="/workspace/model",
       )
       # → LaunchSpec(image, args=("serve", "/workspace/model", "--port", "8000",
       #                           "--max-model-len", "8192", "--quantization", "awq"),
       #              env={"HF_HOME": "/workspace/hf_cache"},
       #              port=8000,
       #              volumes=(("/workspace", "/workspace"),))

       # Provider wraps it:
       cmd = format_docker_run(launch_spec)   ← "docker run --gpus all -p 8000:8000 ..."
       ssh_client.exec_command(cmd)

       healthcheck = self._engine.build_healthcheck_command(host="...", port=8000)
       wait_until_healthy(healthcheck)

       return EndpointInfo(
           url=self._engine.build_default_endpoint_url(host="...", port=8000),
           engine="vllm",      ← informational string, free-form
           ...,
       )

6. ModelEvaluator picks up endpoint:
       client = ModelClientFactory.build(engine="vllm", endpoint_url=...)
       # api_dialect="openai_compatible" → OpenAICompatibleInferenceClient
       run_eval_plugins(client, plugins, dataset)
```

### 6.11 Adding a new engine — full flow

1. Author drops folder `packages/engines/src/ryotenkai_engines/sglang/`:
   ```
   sglang/
   ├── engine.toml                 # ~25 lines, no [image] block needed
   ├── runtime.py                  # SGLangEngineRuntime(IInferenceEngine)
   ├── config.py                   # SGLangEngineConfig (kind="sglang")
   └── Dockerfile                  # FROM lmsysorg/sglang:v0.4.0
   ```

2. CI/CD pipeline (out-of-scope but enabled by convention):
   - Walks `packages/engines/src/ryotenkai_engines/*/Dockerfile`.
   - Reads `engine.toml [engine].version`.
   - Builds + tags + pushes by convention name.
   - Result: `ryotenkai/inference-sglang:1.0.0` available.

3. Author adds `"sglang"` to `provider.toml [capabilities.inference] supported_engines`
   for any provider that should support it.

4. Done. Zero edits to PipelineConfig, validators, control plane, or other engines.

`packages/shared/src/ryotenkai_shared/inference/__about__.py` — **DELETED**.

---

## 7. Concern B: Training Adapter discriminated union

### 7.1 Today (the anti-pattern)

**File:** `packages/shared/src/ryotenkai_shared/config/training/schema.py:48-83`:
```python
class TrainingOnlyConfig(StrictBaseModel):
    type: str = Field(TRAINING_TYPE_QLORA, description="Training type: qlora, lora, adalora")
    # ... 25 lines later ...
    lora: LoraConfig | None = Field(None)
    qlora: QloraConfig | None = Field(None)
    adalora: AdaLoraConfig | None = Field(None)
```

**File:** `packages/shared/src/ryotenkai_shared/config/validators/training.py:21-26`:
```python
if cfg.type == "lora" and cfg.lora is None:
    raise ValueError("training.type='lora' requires 'training.lora:' section")
if cfg.type == "qlora" and cfg.qlora is None: ...
if cfg.type == "adalora" and cfg.adalora is None: ...
```

**File:** `packages/shared/src/ryotenkai_shared/config/training/schema.py:174-185`:
```python
def get_adapter_config(self) -> AdapterBaseConfig:
    if self.type == "lora": return self.lora
    if self.type == "qlora": return self.qlora
    if self.type == "adalora": return self.adalora
    raise ValueError(...)
```

### 7.2 After refactor

**File:** `packages/shared/src/ryotenkai_shared/config/training/adapters/` (NEW directory):
```
adapters/
├── __init__.py
├── _union.py          # AdapterConfigUnion = Annotated[…, Discriminator("kind")]
├── base.py            # BaseAdapterConfig
├── lora.py            # LoraConfig (kind="lora")
├── qlora.py           # QloraConfig (kind="qlora")
└── adalora.py         # AdaLoraConfig (kind="adalora")
```

Each config class:
```python
# packages/shared/src/ryotenkai_shared/config/training/adapters/qlora.py
from typing import Literal
from pydantic import Field
from ryotenkai_shared.config.training.adapters.base import BaseAdapterConfig

class QloraConfig(BaseAdapterConfig):
    kind: Literal["qlora"] = "qlora"
    r: int = Field(8, ge=1)
    lora_alpha: int = Field(16, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    bias: str = "none"
    target_modules: str | list[str] = "all-linear"
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: str = "gaussian"
```

**File:** `packages/shared/src/ryotenkai_shared/config/training/adapters/_union.py`:
```python
from typing import Annotated, Union
from pydantic import Discriminator, Tag
from .lora import LoraConfig
from .qlora import QloraConfig
from .adalora import AdaLoraConfig

AdapterConfigUnion = Annotated[
    Union[
        Annotated[LoraConfig, Tag("lora")],
        Annotated[QloraConfig, Tag("qlora")],
        Annotated[AdaLoraConfig, Tag("adalora")],
    ],
    Discriminator("kind"),
]
```

**File:** `packages/shared/src/ryotenkai_shared/config/training/schema.py` — REWRITTEN:
```python
class TrainingOnlyConfig(StrictBaseModel):
    provider: str | None = Field(None)
    hyperparams: GlobalHyperparametersConfig
    strategies: list[StrategyPhaseConfig]
    adapter: AdapterConfigUnion       # Tag-based discriminated
    # type: str — DELETED
    # lora / qlora / adalora flat fields — DELETED

    def get_adapter_config(self) -> BaseAdapterConfig:
        return self.adapter   # already typed via discriminator
```

### 7.3 YAML before/after

**Before:**
```yaml
training:
  provider: single_node
  type: qlora
  qlora:
    r: 8
    lora_alpha: 16
```

**After:**
```yaml
training:
  provider: single_node
  adapter:
    kind: qlora
    r: 8
    lora_alpha: 16
```

### 7.4 Touched files

- `packages/shared/.../config/training/schema.py` — strip `type` field, replace `lora|qlora|adalora` with `adapter: AdapterConfigUnion`.
- `packages/shared/.../config/validators/training.py:21-26` — DELETE (validator now redundant).
- `packages/shared/.../config/training/schema.py:174-185` `get_adapter_config()` — simplify to `return self.adapter`.
- All callers of `cfg.training.type` → `cfg.training.adapter.kind`.
- All callers of `cfg.training.qlora`/`.lora`/`.adalora` → `cfg.training.adapter`.
- All test fixtures and example YAML configs.
- Web UI ConfigBuilder schema rendering.

---

## 8. Concern C: Dataset Source discriminated union

### 8.1 Today

**File:** `packages/shared/src/ryotenkai_shared/config/datasets/schema.py:37-72`:
```python
class DatasetConfig(StrictBaseModel):
    source_type: Literal["local", "huggingface"] | None = Field(None)
    source_local: DatasetSourceLocal | None = Field(None)
    source_hf: DatasetSourceHF | None = Field(None)
```

**File:** `packages/shared/src/ryotenkai_shared/config/validators/datasets.py:12-26`:
```python
st = cfg.get_source_type()
if st == "huggingface":
    if cfg.source_hf is None: raise ValueError(...)
elif st == "local":
    if cfg.source_local is None: raise ValueError(...)
```

### 8.2 After refactor

**Files:**
```
packages/shared/.../config/datasets/sources/
├── __init__.py
├── _union.py
├── base.py
├── local.py            # DatasetSourceLocal (kind="local")
└── huggingface.py      # DatasetSourceHF (kind="huggingface")
```

```python
# sources/local.py
from typing import Literal
from pydantic import Field
from .base import BaseDatasetSourceConfig

class DatasetSourceLocal(BaseDatasetSourceConfig):
    kind: Literal["local"] = "local"
    local_paths: DatasetLocalPaths
    # ... existing fields ...

# sources/_union.py
from typing import Annotated, Union
from pydantic import Discriminator, Tag

DatasetSourceUnion = Annotated[
    Union[
        Annotated[DatasetSourceLocal, Tag("local")],
        Annotated[DatasetSourceHF, Tag("huggingface")],
    ],
    Discriminator("kind"),
]
```

**File:** `packages/shared/.../config/datasets/schema.py` — REWRITTEN:
```python
class DatasetConfig(StrictBaseModel):
    source: DatasetSourceUnion         # Tag-based discriminated
    # source_type: ... — DELETED
    # source_local / source_hf — DELETED

    def get_source_type(self) -> str:
        return self.source.kind
```

**File:** `packages/shared/.../config/validators/datasets.py:12-26` — DELETED.

### 8.3 YAML before/after

**Before:**
```yaml
datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/train.jsonl
```

**After:**
```yaml
datasets:
  default:
    source:
      kind: local
      local_paths:
        train: data/train.jsonl
```

### 8.4 Touched files

- `packages/shared/.../config/datasets/schema.py`
- `packages/shared/.../config/validators/datasets.py`
- All callers of `cfg.datasets[name].source_type` → `cfg.datasets[name].source.kind`.
- All callers of `cfg.datasets[name].source_local` → `cfg.datasets[name].source` (typed via discriminator).
- Test fixtures + example YAML.

---

## 9. Concern D: Auto-IDs for plugin instances

### 9.1 Affected configs

| File:line | Class | Today |
|---|---|---|
| `packages/shared/.../config/evaluation/schema.py:43-46` | `EvaluatorPluginConfig` | `id: str = Field(...)` (required) |
| `packages/shared/.../config/datasets/validation.py:14-39` | `DatasetValidationPluginConfig` | `id: str = Field(...)` (required) |

### 9.2 Refactor — make `id` optional with hash-based default

```python
# packages/shared/.../config/evaluation/schema.py — REWRITTEN

import hashlib
import json
from pydantic import Field, model_validator
from ryotenkai_shared.config.base import StrictBaseModel


class EvaluatorPluginConfig(StrictBaseModel):
    id: str | None = Field(
        default=None,
        description=(
            "Optional instance id. Auto-generated as "
            "f'{plugin}_{md5(params)[:8]}' if not supplied. Override "
            "only if you need a stable human-readable name across configs."
        ),
    )
    plugin: str = Field(...)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _autofill_id(self) -> "EvaluatorPluginConfig":
        if not self.id:
            self.id = _make_plugin_id(self.plugin, self.params)
        return self


def _make_plugin_id(plugin: str, params: dict[str, Any]) -> str:
    """Stable id from plugin name + params hash. Identical params produce
    identical ids — MLflow artifact paths stay stable across runs."""
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    return f"{plugin}_{hashlib.md5(payload, usedforsecurity=False).hexdigest()[:8]}"
```

### 9.3 Collision handling

If a parent list has duplicate (plugin, params) → duplicate hashed ids. Resolution at parent-level validator:

```python
# packages/shared/.../config/evaluation/schema.py — EvaluatorsConfig
class EvaluatorsConfig(StrictBaseModel):
    plugins: list[EvaluatorPluginConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve_duplicate_ids(self) -> "EvaluatorsConfig":
        seen: dict[str, int] = {}
        for plugin in self.plugins:
            base = plugin.id
            n = seen.get(base, 0)
            if n > 0:
                plugin.id = f"{base}_{n + 1}"   # base_2, base_3, ...
            seen[base] = n + 1
        return self
```

User explicit ids take precedence (already filled by per-instance validator) and are checked for uniqueness as before.

### 9.4 Migration

`scripts/migrate_config_to_discriminated_unions.py` includes a pass that **strips** explicit `id:` lines from existing YAML iff the value matches the auto-generated hash. This keeps user YAMLs clean post-migration but is opt-in — the user can run the script or leave their explicit ids.

### 9.5 Touched files

- `packages/shared/.../config/evaluation/schema.py:43-73`
- `packages/shared/.../config/datasets/validation.py:14-83`
- Test fixtures (most have explicit `id:`; can leave unchanged or rewrite).
- Web UI: form rendering — `id` becomes optional input with placeholder showing the auto-generated value.

---

## 10. NOT migrating: Providers

### 10.1 Why providers don't need `kind` — concrete YAML comparison

**Today's shape (and why it works):**

```yaml
providers:
  single_node:           ← dict key IS the implicit discriminator
    connect: ...
  runpod:
    connect: ...

training:
  provider: single_node   ← string reference to dict key
```

The dict key (`single_node`) selects which Pydantic class validates the block.
`provider.toml [entry_points.config_schema]` resolves it to `SingleNodeProviderConfig`.
Cross-validator at `validators/cross.py:92-98` checks `training.provider`
appears in the dict.

**If we added `kind` (NOT recommended, here's why):**

```yaml
providers:
  single_node:           ← (1) dict key
    kind: single_node    ← (2) duplicates the dict key
    connect: ...
```

Three sources of truth for ONE fact ("this block is for single_node"):

1. `provider.toml [provider].id = "single_node"` — manifest declaration
2. YAML dict key — `providers.single_node`
3. Field — `kind: "single_node"`

If author makes a typo in any one of them — which is "right"? Resolution
requires extra cross-validator. Pure regression vs today.

### 10.2 When `kind` WOULD make sense — the only justifying scenario

Switch from dict to LIST of provider configs:

```yaml
providers:
  - kind: single_node
    name: my_lab
    connect: { ssh: { alias: lab1 } }
  - kind: single_node            ← multiple instances of same provider type
    name: my_other_lab
    connect: { ssh: { alias: lab2 } }
  - kind: runpod
    name: prod
    connect: { ... }

training:
  provider: my_lab               ← reference by `name`
```

This enables N instances of the same provider type with different settings
(e.g., two single_node hosts, two RunPod accounts). Today's dict-by-id
shape forbids this — the key must be unique.

**Is this a real need?** Marginal:
- Two single_node machines: rare, doable today via separate config files.
- Two RunPod accounts: extremely rare (one API key per account is the norm).

### 10.3 What providers DO get from this work

Even though provider configs aren't refactored, providers participate in the
inference engine system:

- `[capabilities.inference]` block in `provider.toml` (NEW, required for
  any provider with `inference` role) — declares which engines that provider
  can launch.
- `engine_overrides.<id>.image` (NEW, optional) — per-engine image override
  for provider-specific builds (e.g., RunPod-tuned CUDA).
- Cross-validation `validate_inference_provider_engine_compatibility(cfg)`
  rejects (provider, engine) pairs the manifest doesn't allow.

Provider config classes stay structurally as-is; only manifests gain
inference-engine awareness.

### 10.4 Future: when to revisit

If/when we need multiple-instances-per-provider-type — separate plan,
which would migrate `providers: dict` → `providers: list[ProviderUnion]`
with `kind` discriminator. Today: deferred.

### 10.2 What providers DO get from this work

- `[capabilities.inference]` block in `provider.toml` (NEW, required for any provider with `inference` role) — gives provider-side declaration of which engines it supports.
- Cross-validation `validate_inference_provider_engine_compatibility(cfg)` (NEW) — rejects (provider, engine) pairs the manifest doesn't allow.

Providers stay structurally as-is; only manifests and validators gain inference-engine awareness.

### 10.3 Open question: future provider `kind` migration?

If we ever want providers to live in a single discriminated union (`provider: ProviderConfigUnion`), that's a separate, larger refactor. Today's dict + manifest pattern is OCP-clean (drop a folder, no central edits) and that's the actual property we want from any plugin system. Discriminated union vs dict + manifest is a design taste choice; current shape works.

---

## 11. Migration plan — 10 PRs, no backwards compat

| # | PR | What lands | Reversible? |
|---|---|---|---|
| 1 | **Engines scaffolding** | Create `packages/engines/`, empty modules, importlinter contracts (`engines_is_leaf`, `generic_no_concrete_engine_imports`). | yes |
| 2 | **Engine Protocol + Manifest + Registry** | `interfaces.py`, `manifest.py`, `registry.py`, `images.py`, `_config_union.py`, `errors.py`, `capabilities.py`. Tests scaffold. | yes |
| 3 | **vLLM port** | Move `VLLMEngine` → `packages/engines/.../vllm/` as `VLLMEngineRuntime`. Move config to `packages/engines/.../vllm/config.py` with `kind="vllm"`. Write `engine.toml`. Old code stays as thin shim until PR-7. | yes |
| 4 | **Adapter discriminated union** | Move `LoraConfig`/`QloraConfig`/`AdaLoraConfig` to `config/training/adapters/`, add `kind`, build `AdapterConfigUnion`. Refactor `TrainingOnlyConfig`. Update validators. | partly (config schema breaking) |
| 5 | **Dataset source discriminated union** | Move `DatasetSourceLocal`/`DatasetSourceHF` to `config/datasets/sources/`, add `kind`, build `DatasetSourceUnion`. Refactor `DatasetConfig`. Update validators. | partly |
| 6 | **Inference engine config wiring** | Rewrite `InferenceConfig.engine` as `EngineConfigUnion`. Drop `InferenceEnginesConfig`. Update YAML examples + Web UI ConfigBuilder. | partly |
| 7 | **Provider DI + manifest extension** | Refactor `single_node` and `runpod` inference providers to use registry + DI. Drop `quantization` from `deploy()`. Add `[capabilities.inference]` to provider manifests. Validator `validate_inference_provider_engine_compatibility`. | yes |
| 8 | **Auto-id for plugin instances** | Make `id` optional with hash-based default in `EvaluatorPluginConfig` + `DatasetValidationPluginConfig`. Collision suffix. Migration script pass. | yes |
| 9 | **Eliminate leaks + delete legacy** | `model_evaluator.py:150` (kill default), `inference_deployer.py:164` (kill kwarg), delete `INFERENCE_IMAGES`, delete constants, delete shims from PR-3. | partly (deletions) |
| 10 | **Drift detector + sentinel + docs** | `scripts/check_engine_manifests.py`, sentinel tests, importlinter contracts active in CI. README for engine + adapter + source authors. Migration script `scripts/migrate_config_to_discriminated_unions.py`. | yes |

PR-4, 5, 6 are the riskiest (config schema changes). Each gates with full test sweep + e2e. Migration script provided in PR-10.

### Migration script outline

`scripts/migrate_config_to_discriminated_unions.py`:
- Walks `tests/fixtures/configs/*.yaml` and any user-supplied path.
- Rewrites `training.type: X` + `training.X: {...}` → `training.adapter: {kind: X, ...}`.
- Rewrites `datasets.<name>.source_type: X` + `datasets.<name>.source_X: {...}` → `datasets.<name>.source: {kind: X, ...}`.
- Rewrites `inference.engine: vllm` + `inference.engines.vllm: {...}` → `inference.engine: {kind: vllm, ...}`.
- Strips `id:` lines from `evaluators.plugins[]` + `validations.plugins[]` IFF value matches auto-hash.
- Idempotent — running twice on a migrated config is a no-op.
- Diff-mode (`--dry-run`) to preview changes.

---

## 12. Tests (all 8 categories per concern)

### 12.1 Inference engine (per §6)

Same as the previous engine-only plan version — see §16.1-16.8 in transcript history. Summary by category:

| Category | Coverage |
|---|---|
| **Positive** | Manifest loads, registry discovers vLLM, runtime methods produce correct output. |
| **Negative** | Missing manifest blocks, unknown engine, schema_version too high, runtime not implementing Protocol. |
| **Boundary** | `default_port` 1/65535, `tensor_parallel_size` 1/8, `gpu_memory_utilization` 0.0/1.0. |
| **Invariant** | `supports_quantization=False ⇒ supported_quantizations=()`, registry sorted, `engine_id == kind`. |
| **Dependency-error** | TOML syntax error → LoadFailure, import error → LoadFailure. |
| **Regression** | Existing deploy/health-check/activate/deactivate flows pass after migration. |
| **Logic-specific** | `--quantization` flag iff cfg.quantization is not None; `LaunchSpec` → docker shell formatting unchanged. |
| **Combinatorial** | (provider, engine) compatibility matrix; (quantization, enforce_eager) × (with/without LoRA). |

### 12.2 Adapter discriminated union

| Category | Cases |
|---|---|
| **Positive** | (1) `{kind: qlora, r: 8}` parses to `QloraConfig`. (2) `cfg.training.adapter.kind == "qlora"`. (3) `get_adapter_config()` returns the typed instance. |
| **Negative** | (1) Missing `kind`. (2) Unknown `kind`. (3) `kind: lora` + qlora-only field (extra="forbid"). (4) Old shape `type: qlora` + `qlora: {…}` rejected. |
| **Boundary** | (1) Each adapter type's required-field minimum config. (2) Empty adapter (rejected — adapter is required). |
| **Invariant** | (1) `Tag(kind)` matches `Literal[kind]` on config class — drift detector. (2) Each adapter Pydantic class subclasses `BaseAdapterConfig`. |
| **Dependency-error** | (1) Adapter config import fails → ValidationError surfaced. |
| **Regression** | (1) Existing trainer code (`get_peft_config()` etc.) works unchanged after migration. |
| **Logic-specific** | (1) `adapter.kind` field is the discriminator — switching kinds at runtime requires reconstructing the model. |
| **Combinatorial** | (1) For each adapter kind: round-trip serialize/deserialize is identity. |

### 12.3 Dataset source discriminated union

Symmetric to §12.2.

| Category | Cases |
|---|---|
| **Positive** | `{kind: local, local_paths: {...}}` → `DatasetSourceLocal`. `cfg.datasets["d"].source.kind == "local"`. |
| **Negative** | Missing `kind`, unknown `kind`, kind/payload mismatch (extra="forbid"). Old shape rejected. |
| **Boundary** | Empty `local_paths.train`. |
| **Invariant** | `Tag(kind) ⇔ Literal[kind]` parity. |
| **Dependency-error** | Import error → ValidationError. |
| **Regression** | Existing dataset loader code works unchanged. |
| **Logic-specific** | `get_source_type()` delegates to `self.source.kind`. |
| **Combinatorial** | All source kinds: serialize/deserialize identity. |

### 12.4 Auto-id for plugin instances

| Category | Cases |
|---|---|
| **Positive** | (1) Omitting `id:` produces `{plugin}_{md5(params)[:8]}`. (2) Explicit `id:` survives validation unchanged. (3) Same params + plugin → same id (deterministic). |
| **Negative** | (1) Two plugins with identical (plugin, params) — second gets `_2` suffix. (2) Mix of explicit id matching auto-id of another (rare but possible) — explicit takes precedence; collision suffix on the auto-generated. |
| **Boundary** | (1) Empty params dict. (2) Params with non-JSON-serializable types (e.g., Pydantic model — covered by `default=str`). (3) `id` length 1 (allowed). (4) Plugin name with special chars (slash, etc.) — not blocked but flagged in id. |
| **Invariant** | (1) Auto-id length ≤ 50 chars (plugin name ≤ ~40 + 9-char hash suffix). (2) Auto-id contains only `[a-zA-Z0-9_]`. (3) `params` reordering doesn't change id (sort_keys=True). |
| **Dependency-error** | (1) Hashing fails on un-serializable param — caught with explicit error message. |
| **Regression** | (1) Existing configs with explicit ids work unchanged. (2) MLflow artifact paths under `evaluation_results.json` stable across runs with same params. |
| **Logic-specific** | (1) Suffix `_2`/`_3` only applied when collision detected. (2) Explicit id overrides auto-fill (already-filled `id` field skips the validator). |
| **Combinatorial** | (1) Matrix: (explicit id present, params identical) × (parent has 0/1/2 dups) = 8 cases. |

### 12.5 Sentinel + drift tests (added to existing CI)

- `engines_is_leaf` importlinter contract.
- `generic_code_no_concrete_engine_imports` importlinter contract.
- `test_engine_protocol_parity.py` — every registered engine has `engine_id`, `config_class`, all `IInferenceEngine` methods, capabilities mirror manifest.
- `check_engine_manifests.py` — manifest ↔ runtime drift detector.
- `test_discriminator_kind_uniformity.py` — every Tag-based union has `Discriminator("kind")` (not "type", "engine", etc.) — enforces AD-6.

### 12.6 End-to-end

`packages/control/tests/e2e/test_full_pipeline_discriminated_e2e.py`:
- YAML with new shapes for engine + adapter + dataset source.
- Walk PipelineConfig → DatasetValidator → GPUDeployer → TrainingMonitor → InferenceDeployer → ModelEvaluator.
- Assert all dispatches happen via discriminator, no string-match fallbacks.

---

## 13. Risk analysis (3 iterations)

### Iteration 1 — Architectural risks

**R1. `kind` field name collision.**
Engine configs might want a config field literally named `kind`.
*Mitigation:* document `kind` as reserved in author guide; drift detector flags any non-Literal `kind` field.

**R2. Cross-package import cycle (shared → engines).**
Tag-based union builder lives in `engines`, imported by `shared.config.inference`.
*Mitigation:* `engines` becomes the new leaf — depends only on stdlib + pydantic + tomllib. `shared` may depend on `engines`. Importlinter contract `engines_is_leaf` enforces this.

**R3. Drift between manifest and runtime.**
Author bumps `[capabilities].supports_lora = true` but `runtime.py` doesn't accept LoRA path.
*Mitigation:* drift detector in PR-10 runs in CI.

**R4. Web UI ConfigBuilder breakage.**
Frontend renders forms by walking JSON Schema. Discriminated unions yield `oneOf`.
*Mitigation:* check ConfigBuilder support in PR-6. Most Pydantic-aware form builders handle `oneOf`. If not, frontend gets a small refactor (~1 day).

**R5. `EngineRegistry.from_filesystem()` performance at config load.**
Walking filesystem + parsing TOML adds latency. Today: 1 engine, ~10ms. With 20 engines: ~200ms.
*Mitigation:* lock-protected singleton; first call pays cost, rest O(1).

**R6. Single-PR train migration risk.**
10 PRs touching shared schemas — broken intermediate state.
*Mitigation:* each PR keeps `tests/` green by virtue of staged shims (PR-3 leaves shim in providers/; PR-9 deletes shim). CI runs full test sweep on every PR.

### Iteration 2 — Operational / migration risks

**R7. User YAMLs break.**
After PRs 4/5/6 land, all old shapes raise ValidationError.
*Mitigation:* `scripts/migrate_config_to_discriminated_unions.py` rewrites old YAMLs. Idempotent. CHANGELOG and README updated.

**R8. RunPod image rebuild not needed for engine config changes alone.**
Image stays `ryotenkai/inference-vllm:v1.0.0`.
*Mitigation:* none — image is unchanged.

**R9. Test fixture explosion.**
~50 test fixtures touch one or more of the four shapes.
*Mitigation:* helper `make_pipeline_config(...)` in `tests/conftest.py` builds the new shapes. Migration script runs over fixtures in PR-4, 5, 6.

**R10. `quantization` removal breaks downstream.**
Only one caller (`inference_deployer.py:164`).
*Mitigation:* grep verified.

**R11. Auto-id MLflow artifact path stability.**
If a user changes a param, the auto-id changes — old artifacts orphaned in MLflow.
*Mitigation:* this IS the desired behaviour (param change → different id → distinct artifact). Document in user guide. User who wants stability across param tweaks supplies explicit id.

**R12. Auto-id collision in unrelated parents.**
Two distinct configs (Pipeline A, Pipeline B) both contain `evaluator: {plugin: X, params: {…}}` → identical auto-ids. MLflow stores them separately by run, so no actual collision in artifact storage.
*Mitigation:* MLflow naturally namespaces by run_id. Auto-id collision across runs is intentional — same plugin+params SHOULD compare.

### Iteration 3 — Conformance / future-proofing risks

**R13. Strategies aren't migrated to `kind`.**
Today `strategy_type: str` + generic `params: dict[str, Any]`. Pattern is intentional (avoids per-strategy boilerplate).
*Mitigation:* document in §10 — migrate later if/when strategies diverge in required fields. Current generic `params` works.

**R14. Future K8s provider compat.**
`LaunchSpec` shape (image, args, env, volumes) is docker-shaped. K8s wants ContainerSpec + PodSpec.
*Mitigation:* `LaunchSpec` extensible; K8s provider when added translates `LaunchSpec` → `ContainerSpec` itself.

**R15. Resume/restart engine drift.**
`mem_ddaff0385be4` (helixir): training engine is high-importance config-hash field; on resume, mismatch raises `RestartError`. Same for inference engine.
*Mitigation:* `cfg.inference.engine.kind`, `cfg.training.adapter.kind`, `cfg.datasets[*].source.kind` participate in `pipeline_state` hash. Catches "user changed inference engine mid-run" footgun.

**R16. Pydantic v2 `Tag` API stability.**
`Annotated[T, Tag("…")]` is a Pydantic v2.5+ feature.
*Mitigation:* pyproject pins `pydantic>=2.5`. CI tests on min and current.

**R17. Engine version pinning.**
`engine.toml [image].default` is mutable; CI image floats unless tagged.
*Mitigation:* drift detector regex-checks no `:latest` tag.

---

## 14. Deep-think on remaining open questions

### Q1: Where do engine configs live — in `engines/` or `shared/`?

**Iter 1:** Engine configs are pure Pydantic — no runtime deps. Pod imports `PipelineConfig` (and transitively engine configs).
**Iter 2:** Pod doesn't use inference at runtime; only PipelineConfig parsing. Importing engine configs is cheap.
**Iter 3:** **Decision: configs in `engines/<id>/config.py`** (with the runtime). Co-location matches engine author mental model.

### Q2: Should `EndpointInfo.engine: str` become typed?

**Iter 1:** `EndpointInfo` crosses serialization boundaries (manifest reports, MLflow tags).
**Iter 2:** Strict types break round-trip if a deployment has an engine the consumer doesn't know about.
**Iter 3:** **Decision: `engine: str` stays.** Validation lives at PipelineConfig load, not at EndpointInfo boundary.

### Q3: What's the failure mode if a manifest is malformed in production?

**Iter 1:** Two modes: hard-fail (block all engines) vs defensive collection (one engine unavailable, others work).
**Iter 2:** ProviderRegistry uses defensive collection. Same here.
**Iter 3:** **Decision: defensive collection.** `validate_inference_enabled_is_supported` validator raises if user picks an engine in `failures()`.

### Q4: Should adapter `kind` field allow extra adapters from community plugins?

**Iter 1:** Today adapters are static (LoraConfig/QloraConfig/AdaLoraConfig).
**Iter 2:** Adding adapter via community plugin pattern is a separate, larger refactor.
**Iter 3:** **Decision: adapters stay static for now.** Same shape as inference engines initially — Tag-based union with hardcoded members. If community adapters become a need, lift to `packages/community/` plugin pattern (separate plan).

### Q5: Should we extend auto-id to other "instance" places (e.g., strategy phases)?

**Iter 1:** Strategy phases don't have explicit `id` field — they're identified positionally.
**Iter 2:** Adding auto-id to strategies would change a contract (phases become named).
**Iter 3:** **Decision: scoped to evaluator + validator only.** Strategy phase identification can be revisited separately.

---

## 15. Best-practices conformance check

| Principle | Status | Evidence |
|---|---|---|
| **SRP** | ✅ | Engine = command-builder. Provider = compute. ModelClient = wire. Adapter = peft config. Dataset source = data origin. Each Pydantic class owns one variant. |
| **OCP** | ✅ | Adding engine = drop folder. Adding adapter = new class + 1 union edit. Adding dataset source = same. |
| **LSP** | ✅ | Discriminated unions enforce substitutability via Pydantic discriminator dispatch. |
| **ISP** | ✅ | Protocols minimal — `IInferenceEngine` 6 methods, no fat interface. |
| **DIP** | ✅ | Provider depends on `IInferenceEngine` Protocol (resolved by registry), not concrete `VLLMEngine`. |
| **KISS** | ⚠️ trade-off | Plan A engine plugin system is heavier than minimum. User chose for marketplace-readiness. |
| **DRY** | ✅ | Image registry: one source per engine. Engine name: discriminator only. Adapter selector: `kind` only. Dataset source selector: `kind` only. Auto-id: hash function once. |
| **YAGNI** | ⚠️ trade-off | Engine plugin system, adapter discriminated union, dataset source discriminated union — all justified by current pain. Auto-id by current UX friction. |
| **Boy scout** | ✅ | Eliminates 4+ leak sites (model_evaluator, inference_deployer, single_node, runpod) + 5 stale constants + 12 if/elif branches across 4 validators. |
| **Clean architecture** | ✅ | engines = leaf; shared depends on engines; providers/control depend on shared+engines. Acyclic. |
| **Observability** | ✅ | `kind` in EndpointInfo, log lines, MLflow tags. Auto-id in artifact keys. Registry failures surfaced. Drift detector in CI. |
| **Security** | ✅ | Engines never see secrets. Provider holds secrets, passes only what the engine needs (env vars in LaunchSpec). |
| **Reliability** | ✅ | Discriminated union catches typos at config load. Provider validates SUPPORTED_ENGINES ClassVar. Engine validates own config. Three-layer fail-fast for engine concern; analogous for adapter/source concerns. |
| **Rollback** | ✅ | Each PR `git revert`-able. PR-4/5/6 (config schema breaks) require migration script — provided in PR-10. |
| **Testability** | ✅ | Protocol + Tag-discriminator + registry → deterministic, mockable. Fake engines in tests synthesized via Pydantic. Auto-id deterministic (hash) — easy assertions. |

---

## 16. Architectural Decision Records (codified)

- **AD-1 / RD-1:** Tag-based discriminator from day one. Pydantic v2.5+ pinned in pyproject.
- **AD-2 / RD-2:** New workspace member `packages/engines/`. NOT applying same pattern to providers (provider has manifest+dict-key already). NOT promoting adapters/sources to community plugins (no need today).
- **AD-3 / RD-3:** Auto-id is **hash-based** (`{plugin}_{md5(params)[:8]}`). Collisions resolved with `_2`/`_3` suffix at parent validator.
- **AD-4 / RD-4:** Discriminator field name uniformly `kind`. Old field names (`type`, `source_type`, `engine`) deleted.
- **AD-5 / RD-5:** `LaunchSpec` is the engine→provider contract (forward-compatible to k8s).
- **AD-6 / RD-6:** `quantization` kwarg removed from `IInferenceProvider.deploy()`.

---

## 17. Critical files

### To create

```
packages/engines/                                        # NEW workspace member (entire tree)
packages/shared/.../config/training/adapters/
├── __init__.py
├── _union.py
├── base.py
├── lora.py
├── qlora.py
└── adalora.py
packages/shared/.../config/datasets/sources/
├── __init__.py
├── _union.py
├── base.py
├── local.py
└── huggingface.py
scripts/migrate_config_to_discriminated_unions.py        # YAML migration
packages/engines/scripts/check_engine_manifests.py       # CI drift detector
```

### To modify

```
packages/shared/.../config/inference/schema.py           ← engine: EngineConfigUnion
packages/shared/.../config/training/schema.py            ← adapter: AdapterConfigUnion
packages/shared/.../config/datasets/schema.py            ← source: DatasetSourceUnion
packages/shared/.../config/evaluation/schema.py          ← id: optional, autofill validator
packages/shared/.../config/datasets/validation.py        ← id: optional, autofill validator
packages/shared/.../config/validators/inference.py       ← rewritten for engine union + provider compat
packages/shared/.../config/validators/training.py        ← drop type-presence validator (now redundant)
packages/shared/.../config/validators/datasets.py        ← drop source-presence validator (now redundant)
packages/providers/.../manifest.py                       ← +InferenceCapabilitiesBlock
packages/providers/.../inference/interfaces.py           ← drop quantization kwarg
packages/providers/.../single_node/inference/provider.py ← DI
packages/providers/.../single_node/provider.toml         ← +supported_engines
packages/providers/.../runpod/inference/pods/provider.py ← DI
packages/providers/.../runpod/provider.toml              ← +supported_engines
packages/control/.../pipeline/stages/inference_deployer.py ← drop kwarg, .kind
packages/control/.../pipeline/stages/model_evaluator.py    ← .kind, no fallback
.importlinter                                             ← +engines contracts + kind discriminator uniformity
pyproject.toml (root)                                     ← register packages/engines
```

### To delete

```
packages/shared/.../inference/__about__.py                                 (image dict)
packages/shared/.../config/inference/engines/vllm.py                       (moved)
packages/shared/.../config/inference/__init__.py InferenceEnginesConfig    (deleted block)
packages/providers/.../inference/vllm/engine.py                            (moved)
constants in packages/shared/.../constants.py:52-62                        (deleted block)
training.lora / training.qlora / training.adalora flat fields              (deleted; moved into discriminated union)
training.type field                                                         (deleted)
dataset.source_type / source_local / source_hf flat fields                 (deleted)
```

---

## 18. Verification

### Static checks

```bash
uv run lint-imports                     # all 11 contracts (8 existing + 3 new) KEPT
uv run python packages/engines/scripts/check_engine_manifests.py
uv run mypy packages/engines/ packages/shared/ packages/providers/ packages/control/
uv run ruff check .
```

### Unit tests

```bash
uv run pytest packages/engines/tests/                       # new
uv run pytest packages/shared/tests/unit/config/            # all four refactors
uv run pytest packages/providers/tests/unit/providers/      # DI changes
uv run pytest packages/control/tests/unit/pipeline/stages/  # leak fixes
```

### Sentinel + contract tests

```bash
uv run pytest packages/engines/tests/sentinel/
uv run pytest packages/engines/tests/contract/
uv run pytest packages/control/tests/sentinel/
```

### End-to-end smoke

```bash
uv run pytest -m e2e packages/control/tests/e2e/test_full_pipeline_discriminated_e2e.py
uv run pytest -m e2e packages/control/tests/e2e/test_inference_deploy_e2e.py
```

### Migration script smoke

```bash
# Dry-run on all fixture configs.
uv run python scripts/migrate_config_to_discriminated_unions.py --dry-run \
    packages/control/tests/fixtures/configs/

# Apply, then sanity-check each migrated config loads.
uv run python scripts/migrate_config_to_discriminated_unions.py \
    packages/control/tests/fixtures/configs/
uv run pytest packages/control/tests/integration/test_fixture_configs_load.py
```

### Acceptance criteria

- [ ] Adding a stub `tgi` engine takes ≤ 4 file edits (engine.toml without `[image]` block + runtime.py + config.py + 1 line in provider.toml).
- [ ] Stub engine's image name is auto-derived by convention (`ryotenkai/inference-tgi:1.0.0`) — verified by `resolve_image()` unit test.
- [ ] Adding a stub `dora` adapter takes ≤ 2 file edits (new class + 1 line in `_union.py`).
- [ ] Adding a stub `s3` dataset source takes ≤ 2 file edits.
- [ ] `grep -rn '"vllm"' packages/control packages/shared` returns 0 hits outside discriminator declarations.
- [ ] `grep -rn 'engines.vllm\|training.qlora\|source_local\|source_hf\|cfg.training.type' packages/` returns 0 hits.
- [ ] `grep -rn 'if.*engine.*==\|if.*source_type.*==\|if.*\.type.*==' packages/shared/.../config/validators/` returns 0 hits.
- [ ] `grep -rn 'INFERENCE_IMAGES\|resolve_inference_image' packages/` returns 0 hits.
- [ ] All evaluator/validator plugin instances in fixtures load without explicit `id:` and produce stable hash-based ids.
- [ ] Image tag verification: `engine.toml [engine].version` matches the tag of every image referenced by the provider — drift detector enforces.

---

## 19. Out-of-scope (deferred)

- Custom (non-OpenAI) `api_dialect` engines in `ModelClientFactory`.
- K8s inference provider.
- Per-engine LoRA hot-swap (vLLM API).
- Engine version negotiation (multiple `engine.toml` per id).
- Web UI engine catalogue endpoint (`/api/engines`).
- Community-contributed engines marketplace (signing, sandboxing).
- Strategy phase discriminated union (deferred — strategies stay generic until divergence).
- Provider discriminated union (deferred — current manifest+dict pattern works).
  - Migration trigger: only if multiple-instances-per-provider-type becomes a real need.
- Multiple-instances-per-provider-type (`providers: list[…]` with `kind`).
- Auto-id for strategy phases (deferred).
- Adapter migration to community-plugin pattern.
- Engine image build CI/CD pipeline (`.github/workflows/build-engine-images.yml`):
  walks `packages/engines/src/ryotenkai_engines/*/Dockerfile`, builds + pushes
  by convention. Convention naming enables this; pipeline itself is separate.

---

## 20. Helixir memory updates

Save after approval (in main session, not in plan mode):

```python
add_memory("RyotenkAI 2026-05-06 architectural stabilization plan: discriminated
unions + auto-id. Three concerns unified under Tag-based Pydantic discriminator
with `kind` field: (1) inference engine — new packages/engines/ workspace
member with engine.toml manifests, EngineRegistry filesystem discovery,
IInferenceEngine Protocol returning structured LaunchSpec; (2) training adapter —
LoraConfig/QloraConfig/AdaLoraConfig moved to config/training/adapters/ with
kind discriminator, type+lora|qlora|adalora flat fields deleted; (3) dataset
source — DatasetSourceLocal/HF moved to config/datasets/sources/ with kind,
source_type+flat fields deleted. Plus auto-id for EvaluatorPluginConfig and
DatasetValidationPluginConfig: id becomes optional, default is f'{plugin}_
{md5(params)[:8]}', collision suffix _2/_3 at parent validator. Providers NOT
migrated (manifest+dict-key already type-safe). Strategies NOT migrated (stay
generic). 10-PR train, no backwards compat, single migration script
scripts/migrate_config_to_discriminated_unions.py. Boy scout cleanup eliminates
12+ if/elif branches and 5 stale constants.")
```

---

**End of plan.**
