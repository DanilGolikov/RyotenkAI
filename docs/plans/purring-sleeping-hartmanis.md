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

# Addendum: PR-16 — `prepare_model()` lifecycle hook

**Date added:** 2026-05-06 (post PR-15 merge)
**Owner:** daniil
**Status:** PLANNED — awaiting approval.
**Strategy:** Single PR (user choice). Multi-step return shape from day 1 (user choice).

---

## A1. Context

PR-1..15 made the inference engine a clean plugin (engine.toml + IInferenceEngine + LaunchSpec). But **one concern leaked**: the LoRA-merge step still lives inside `SingleNodeInferenceProvider._run_merge_container()` (~330 lines, `provider.py:760-1088`). That's a SOLID violation:

1. The provider is **not** engine-agnostic — it hardcodes vLLM-specific knowledge: which Docker image, what CLI flags (`--base-model`, `--adapter`, `--output`, `--cache-dir`, `--trust-remote-code`), what success marker (`MERGE_SUCCESS`), what artifact must exist (`config.json`), what timeout (3600s).
2. **Two reasons to change** a single class: infrastructure changes (SSH, k8s) **and** vLLM merge-script changes. Classic SRP violation.
3. **Adding any second engine with non-trivial preparation** (llama.cpp GGUF conversion, TensorRT-LLM compilation, MLX conversion, AWQ pre-quantization) cannot be done by "drop a folder" — it requires editing the provider. **This breaks the plug-in promise** PR-1..15 made.

The fix: a new **`prepare_model()` hook** on `IInferenceEngine` that returns a structured `PreparePlan`. The engine declares **WHAT** to prepare; the provider executes **HOW** on its fabric (Docker today, k8s tomorrow). Same split that already worked for `build_launch_spec` / `LaunchSpec` / `format_docker_run`.

---

## A2. Industry validation

The pattern matches established practice (web research 2026-05):

- **KServe `ServingRuntime`** — predict containers + init containers, one PreLaunch lifecycle phase per runtime.
- **TensorRT-LLM + Triton** — Alibaba Cloud reference: `command: /mnt/models/trtllm-llama-2-7b.sh` runs as init script before serve.
- **Open Model Engine (OME, NVIDIA)** — Kubernetes operator unifying SGLang/vLLM/TensorRT-LLM via shared lifecycle hooks.
- **Pluggy / Wan2GP plugin lifecycles** — multi-phase plugin systems with named hook points (`register_data_hook`, `on_tab_select`).
- **Kubernetes container lifecycle hooks** — `PostStart` / `PreStop` are the conceptual analogue.

Our `prepare_model()` is the **engine-author-side declaration** of a PreLaunch hook. None of the references invented a richer abstraction; we're consistent with the field.

---

## A3. Design decisions (codified)

| ID | Decision | Rationale |
|---|---|---|
| **AD-A1** | `prepare_model() -> Result[PreparePlan, AppError]` returns a `PreparePlan(steps=tuple[PrepareStep, ...])`, not a flat spec | User choice: multi-step support from day 1. The return *shape* is the most expensive thing to change later. tuple of 1 step today; tuple of 3 tomorrow (e.g., GGUF→Q4→optimize). No breaking change when 2nd engine ships. |
| **AD-A2** | Engine is pure description; provider executes | Hard invariant from PR-1..15. Engine has NO IO (no SSH, no subprocess, no fs). It declares image, args, env, volumes, success_marker, success_artifact, timeout. Provider runs the spec. |
| **AD-A3** | `merge_lora.py` lives **inside the image** (`/opt/helix/merge_lora.py`) | Already true post-PR-15 (Dockerfile `COPY merge_lora.py /opt/helix/merge_lora.py`). Engine spec just emits `args=("/opt/helix/merge_lora.py", ...)`. **No `files_to_upload` field** — kills an extension point we don't need; the legacy filesystem-walk hack is deleted. |
| **AD-A4** | `image: str | None = None` on `PrepareStep`; None ⇒ provider uses serve image | TensorRT-LLM future case wants different image. vLLM today wants same. None as default keeps current case clean. |
| **AD-A5** | `outputs: tuple[str, ...]` (multi-output from day 1) | TensorRT produces `.engine` + calibration cache. Tuple of 1 today is ergonomic. |
| **AD-A6** | `success_marker: str | None` and `success_artifact: str | None` as plain fields on PrepareStep | Today this covers vLLM (marker="MERGE_SUCCESS", artifact="…/config.json"). For 2nd engine with different success contract — promote to discriminated union (`PrepareValidation = MarkerInLogs | FileExists | ExitCodeOnly`). YAGNI today: 2 plain fields. |
| **AD-A7** | `entrypoint: tuple[str, ...] | None` on `PrepareStep` | vLLM merge needs `--entrypoint python3` (image's default ENTRYPOINT is vLLM serve). Future engines may need `bash`, `node`, `bin/convert`. Optional, None = use image default. |
| **AD-A8** | `EngineCapabilities.requires_prepare: bool` declared in `engine.toml [capabilities]` | Static visibility for operators + drift-detector enforcement (manifest ↔ runtime parity). vLLM=True, future SGLang=False. |
| **AD-A9** | `NoPrepareMixin` ships in `interfaces.py` | Engines without prep get `return Ok(PreparePlan.empty())` for free. vLLM does NOT use the mixin (it overrides). |
| **AD-A10** | `PreparePlan.spec_version: int = 1` | Forward-compat: provider rejects unknown versions with explicit error code. Today free-of-charge insurance. |
| **AD-A11** | `merge_before_deploy=False` rejection stays in `validate_config` (vLLM today's MVP gate) | Single canonical validation point. `prepare_model` is contractually only called when `validate_config()` returned Ok. |
| **AD-A12** | All `SINGLENODE_MERGE_*` error codes renamed to `SINGLENODE_PREPARE_*` | Per user constraint #1 (no backward compat). `INFERENCE_MERGE_INVALID_PATH` → `SINGLENODE_PREPARE_INVALID_PATH`. |
| **AD-A13** | MLflow event messages change "Merge…" → "Prepare…" | Per user constraint #1. Structure (kwargs, category, source) preserved; only message string changes. |
| **AD-A14** | New importlinter contract `engines do not import providers/control/pod/community` | Pins the boundary that's currently only by convention. Cheap insurance against drift. |
| **AD-A15** | Single PR (no split, no feature flag) | User choice. Mitigated by golden-parity tests captured as fixture files. |

---

## A4. Data shape (load-bearing)

```python
# packages/engines/src/ryotenkai_engines/interfaces.py — additions

class PrepareStep(BaseModel):
    """One ordered preparation step (e.g., 'merge_lora', 'convert_gguf').

    Pure data. Engine describes; provider executes via format_prepare_step().
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Stable id within plan; used in logs/MLflow tags.")
    image: str | None = Field(
        default=None,
        description="Container image. None = provider uses engine's serve image.",
    )
    entrypoint: tuple[str, ...] | None = Field(
        default=None,
        description="Override image ENTRYPOINT (e.g., ('python3',)). None = image default.",
    )
    args: tuple[str, ...] = Field(description="CLI args, pre-split (no shell quoting).")
    env: dict[str, str] = Field(default_factory=dict)
    volumes: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="(host_path, container_path) bind mounts.",
    )
    inputs: tuple[str, ...] = Field(
        default=(),
        description="Container paths this step READS. PreparePlan validator checks "
                    "they appear in earlier steps' outputs OR are engine inputs.",
    )
    outputs: tuple[str, ...] = Field(
        description="Container paths this step PRODUCES. Provider verifies post-step.",
    )
    success_marker: str | None = Field(
        default=None,
        description="Optional substring that must appear in stdout. None = exit-code-only check.",
    )
    success_artifact: str | None = Field(
        default=None,
        description="Optional container path that must exist after success. None = skip check.",
    )
    timeout_seconds: int = Field(default=3600, ge=1)


class PreparePlan(BaseModel):
    """Engine's preparation plan. Empty plan = no work needed."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: int = Field(default=1, description="Bumped when shape changes.")
    steps: tuple[PrepareStep, ...] = Field(default=())
    final_model_path: str | None = Field(
        default=None,
        description="Container path the provider mounts as model for serve. "
                    "When steps non-empty, MUST be set (usually steps[-1].outputs[0]).",
    )

    @model_validator(mode="after")
    def _validate(self) -> "PreparePlan":
        if self.steps and self.final_model_path is None:
            raise ValueError("PreparePlan with steps must set final_model_path")
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(f"step names must be unique: {names}")
        # Cross-step: every input must be produced by an earlier step OR be an
        # explicit engine input (provider-supplied). Defer cross-input check
        # to the provider (it knows what's external).
        return self

    @classmethod
    def empty(cls) -> "PreparePlan":
        return cls()


# Protocol method addition:

class IInferenceEngine(Protocol):
    ...
    def prepare_model(
        self,
        *,
        cfg: BaseEngineConfig,
        base_model: str,
        adapter_path_in_container: str | None,
        workspace_host_path: str,
        run_id: str,
        trust_remote_code: bool,
    ) -> Result[PreparePlan, AppError]: ...


class NoPrepareMixin:
    """Default for engines with no prep step (SGLang, future LiveLoRA-vLLM)."""
    def prepare_model(self, **_: Any) -> Result[PreparePlan, AppError]:
        return Ok(PreparePlan.empty())
```

`EngineCapabilities` adds one field:
```python
requires_prepare: bool = Field(
    default=False,
    description="True if this engine returns a non-empty PreparePlan. Mirrors engine.toml.",
)
```

---

## A5. vLLM implementation (content)

`packages/engines/src/ryotenkai_engines/vllm/runtime.py`:

```python
# Module-level constants — engine concerns
_MERGE_TIMEOUT_S = 3600
_MERGE_SUCCESS_MARKER = "MERGE_SUCCESS"
_MERGE_SCRIPT_PATH = "/opt/helix/merge_lora.py"  # baked in image (Dockerfile COPY)


class VLLMEngineRuntime:
    ...
    def prepare_model(
        self,
        *,
        cfg: BaseEngineConfig,
        base_model: str,
        adapter_path_in_container: str | None,
        workspace_host_path: str,
        run_id: str,
        trust_remote_code: bool,
    ) -> Result[PreparePlan, AppError]:
        if not isinstance(cfg, VLLMEngineConfig):
            return Err(AppError(
                message=f"prepare_model expected VLLMEngineConfig, got {type(cfg).__name__}",
                code="VLLM_CONFIG_TYPE_MISMATCH",
            ))

        # No adapter ⇒ nothing to merge. Provider serves base model directly.
        if adapter_path_in_container is None or not cfg.merge_before_deploy:
            return Ok(PreparePlan.empty())

        output_in_container = f"/workspace/runs/{run_id}/model"
        cache_in_container = "/workspace/hf_cache"

        args: list[str] = [
            _MERGE_SCRIPT_PATH,
            "--base-model", base_model,
            "--adapter", adapter_path_in_container,
            "--output", output_in_container,
            "--cache-dir", cache_in_container,
        ]
        if trust_remote_code:
            args.append("--trust-remote-code")

        merge_step = PrepareStep(
            name="merge_lora",
            image=None,  # use serve image (provider resolves)
            entrypoint=("python3",),
            args=tuple(args),
            env={
                "HF_HOME": cache_in_container,
                "HUGGINGFACE_HUB_CACHE": cache_in_container,
                "TRANSFORMERS_CACHE": cache_in_container,
            },
            volumes=((workspace_host_path, "/workspace"),),
            inputs=(adapter_path_in_container,),  # adapter must exist
            outputs=(output_in_container,),
            success_marker=_MERGE_SUCCESS_MARKER,
            success_artifact=f"{output_in_container}/config.json",
            timeout_seconds=_MERGE_TIMEOUT_S,
        )

        return Ok(PreparePlan(
            steps=(merge_step,),
            final_model_path=output_in_container,
        ))
```

---

## A6. Provider refactor

`packages/providers/src/ryotenkai_providers/inference/launch.py`:

Add sibling helper (mirror of `format_docker_run`):

```python
def format_prepare_step(
    step: PrepareStep,
    *,
    image: str,                       # provider resolves (None on step ⇒ serve image)
    container_name: str,
    extra_env: Mapping[str, str] | None = None,  # HF_TOKEN injected by provider
    gpus_all: bool = True,
) -> str:
    """LaunchSpec sibling for ephemeral PrepareStep → docker run …"""
    parts: list[str] = ["docker run", "--detach"]
    parts.append(f"--name {shlex.quote(container_name)}")
    if gpus_all:
        parts.append("--gpus all")
    for h, c in step.volumes:
        parts.append(f"-v {shlex.quote(h)}:{shlex.quote(c)}")
    merged_env = {**step.env, **(extra_env or {})}
    for k, v in merged_env.items():
        parts.append(f"-e {shlex.quote(f'{k}={v}')}")
    if step.entrypoint:
        ep = " ".join(shlex.quote(a) for a in step.entrypoint)
        parts.append(f"--entrypoint {shlex.quote(step.entrypoint[0])}")
        # Note: docker --entrypoint takes single token; multi-token entrypoints
        # are handled by prepending the rest of entrypoint to args. (KISS: for
        # now we only support 1-token entrypoint per AD-A7. tuple-typed for
        # forward compat.)
    parts.append(shlex.quote(image))
    parts.extend(shlex.quote(a) for a in step.args)
    return " ".join(parts)
```

`packages/providers/src/ryotenkai_providers/single_node/inference/provider.py`:

- **Delete** `_run_merge_container` (lines ~760-1088, ~330 lines).
- **Delete** `_merge_adapter_remote` (deprecated host-merge fallback, ~95 lines).
- **Delete** `MERGE_TIMEOUT` constant (moves to engine).
- **Delete** `here.parents` filesystem walk for `merge_lora.py` (script is in image).
- **Add** `_run_prepare_plan(ssh, plan, run_id, workspace_host_path) -> Result[None, InferenceError]`:
  - Iterates `plan.steps` sequentially.
  - For each step: resolve `image := step.image or self._engine_serve_image`; call `format_prepare_step(step, image=..., container_name=f"helix-prepare-{run_id}-{step.name}", extra_env={"HF_TOKEN": ...})`; ensure docker image; clean output dirs (`rm -rf` then `mkdir -p` host-side for each output via volume mapping); `docker run`; poll logs (interval 2s, timeout from step); on timeout `docker rm -f`; check exit code; check `success_marker` in logs; check `success_artifact` exists on host.
  - On any failure: emit MLflow `prepare_step_failed` event, return `Err` with one of: `SINGLENODE_PREPARE_IMAGE_PULL_FAILED`, `SINGLENODE_PREPARE_CONTAINER_START_FAILED`, `SINGLENODE_PREPARE_CONTAINER_FAILED`, `SINGLENODE_PREPARE_NO_SUCCESS_MARKER`, `SINGLENODE_PREPARE_ARTIFACTS_NOT_FOUND`, `SINGLENODE_PREPARE_TIMEOUT`, `SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED`.
  - On success of a step: `docker rm -f` cleanup, MLflow `prepare_step_complete`, proceed.
- **Add** `_host_to_container(host_path, workspace_host_path)` helper (extracted from existing inline `_to_container_path`) — provider keeps path validation; engine never sees host paths.
- **Refactor** `deploy()`:

```python
# (existing: SSH connect, healthchecks, dir setup, adapter upload — unchanged)

# Reserve SHA-stable run dir; provider concern.
run_dir_name = f"run_{run_id}"
adapter_path_in_container = self._host_to_container(adapter_path_on_host, workspace)

prep_result = self._engine.prepare_model(
    cfg=self._engine_cfg,
    base_model=base_model_id,
    adapter_path_in_container=adapter_path_in_container,
    workspace_host_path=workspace,
    run_id=run_id,
    trust_remote_code=trust_remote_code,
)
if prep_result.is_err():
    return Err(InferenceError.from_app_error(prep_result.unwrap_err()))
plan = prep_result.unwrap()

# Reject unknown future spec versions with a loud error
if plan.spec_version != 1:
    return Err(InferenceError(
        message=f"Provider does not support PreparePlan spec_version={plan.spec_version}",
        code="SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED",
    ))

if plan.steps:
    run_res = self._run_prepare_plan(
        ssh=ssh, plan=plan, run_id=run_id, workspace_host_path=workspace,
    )
    if run_res.is_err():
        return run_res
    model_path_in_container = plan.final_model_path
else:
    # No prep — serve original model_source path (translated to container)
    model_path_in_container = self._host_to_container(model_source, workspace)

# Cross-validation A4: if engine declared final_model_path, build_launch_spec
# MUST consume it. Belt-and-suspenders.
launch = self._engine.build_launch_spec(
    cfg=self._engine_cfg,
    image=self._engine_serve_image,
    container_name=self._CONTAINER_NAME,
    port=port,
    workspace_host_path=workspace,
    model_path_in_container=model_path_in_container,
)
# (existing: format_docker_run(launch) → ssh.exec_command — unchanged)
```

- **Rename** `_start_vllm_container` → `_start_engine_container` (already engine-agnostic; "vllm" was vestigial).

---

## A7. File-by-file changes

### Create
- `packages/engines/tests/unit/test_prepare_plan_invariants.py` — `PreparePlan` validators (unique names, final_model_path consistency, spec_version=1).
- `packages/engines/tests/unit/vllm/test_prepare_model.py` — vLLM `prepare_model()` tests (positive, negative, boundary, invariant, logic, regression, combinatorial).
- `packages/providers/tests/unit/providers/inference/test_format_prepare_step.py` — shell-formatter tests (incl. shell-injection safety).
- `packages/providers/tests/unit/providers/single_node/test_run_prepare_plan.py` — provider's plan-runner tests (positive, all error paths preserved, multi-step combinatorial).
- `packages/control/tests/golden/merge_cmd_v1.txt` — captured pre-PR docker-run command for byte-equal parity.
- `packages/control/tests/golden/mlflow_events_v1.json` — captured pre-PR MLflow event sequence.
- `packages/engines/tests/sentinel/test_no_io_in_engine_prepare.py` — sentinel: AST-scan that engine `prepare_model` doesn't import paramiko / subprocess / os.system / Path.write_text.
- `packages/engines/tests/contract/test_prepare_model_signature_uniformity.py` — sentinel: every registered engine has `prepare_model` with the exact signature.

### Modify
- `packages/engines/src/ryotenkai_engines/interfaces.py` — add `PrepareStep`, `PreparePlan`, `NoPrepareMixin`, `prepare_model` to `IInferenceEngine` Protocol; export.
- `packages/engines/src/ryotenkai_engines/__init__.py` — export `PrepareStep`, `PreparePlan`.
- `packages/engines/src/ryotenkai_engines/capabilities.py` — add `requires_prepare: bool` field.
- `packages/engines/src/ryotenkai_engines/manifest.py` — accept new `requires_prepare` in `[capabilities]` block; drift-detector validates manifest ↔ runtime parity.
- `packages/engines/src/ryotenkai_engines/vllm/engine.toml` — add `requires_prepare = true` in `[capabilities]`.
- `packages/engines/src/ryotenkai_engines/vllm/runtime.py` — implement `prepare_model()`; add module-level constants; `get_capabilities()` returns `requires_prepare=True`.
- `packages/engines/scripts/check_engine_manifests.py` — extend drift detector with `requires_prepare` field.
- `packages/providers/src/ryotenkai_providers/inference/launch.py` — add `format_prepare_step()`.
- `packages/providers/src/ryotenkai_providers/single_node/inference/provider.py` — DELETE `_run_merge_container`, `_merge_adapter_remote`, `MERGE_TIMEOUT`, filesystem walk; ADD `_run_prepare_plan`, `_host_to_container`; rename `_start_vllm_container` → `_start_engine_container`; refactor `deploy()` per §A6; rename all `SINGLENODE_MERGE_*` codes to `SINGLENODE_PREPARE_*`.
- `pyproject.toml` (importlinter section) — add contract: `engines do not import providers/control/pod/community`.
- `packages/providers/tests/unit/providers/single_node/test_inference_provider.py::TestRunMergeContainerErrors` → rename `TestRunPreparePlanErrors`; update to drive `_run_prepare_plan(plan)` with synthetic `PreparePlan` fixture; assert renamed error codes; **delete** `TestMergeAdapterRemote` (the deprecated method is gone).
- `packages/control/tests/unit/pipeline/inference/test_single_node_provider_regressions.py` — `TestDockerPullTimeout` keeps assertion (timeout=1200 still on provider side); `TestMergeCommandFormatting` (3 tests) move to `test_format_prepare_step.py` + engine `test_prepare_model.py` per shell-string vs args-tuple split; `TestMergePathMapping` (2 tests) stay (provider concern).

### Delete
- `_run_merge_container` method (~330 lines in `provider.py`)
- `_merge_adapter_remote` method (~95 lines in `provider.py`)
- `MERGE_TIMEOUT = 3600` constant in `provider.py`
- `here.parents` script-discovery walk
- `TestMergeAdapterRemote` (3 tests, deprecated path)
- `test_fails_when_merge_image_not_configured` (impossible after registry resolution)

---

## A8. Error code mapping

All renames per AD-A12 (no backward compat per user constraint):

| Old | New |
|---|---|
| `SINGLENODE_MERGE_IMAGE_PULL_FAILED` | `SINGLENODE_PREPARE_IMAGE_PULL_FAILED` |
| `SINGLENODE_MERGE_CONTAINER_START_FAILED` | `SINGLENODE_PREPARE_CONTAINER_START_FAILED` |
| `SINGLENODE_MERGE_CONTAINER_FAILED` | `SINGLENODE_PREPARE_CONTAINER_FAILED` |
| `SINGLENODE_MERGE_NO_SUCCESS_MARKER` | `SINGLENODE_PREPARE_NO_SUCCESS_MARKER` |
| `SINGLENODE_MERGE_ARTIFACTS_NOT_FOUND` | `SINGLENODE_PREPARE_ARTIFACTS_NOT_FOUND` |
| `SINGLENODE_MERGE_SCRIPT_UPLOAD_FAILED` | DELETED (script in image, no upload) |
| `SINGLENODE_MERGE_SCRIPT_NOT_FOUND` | DELETED (filesystem walk gone) |
| `INFERENCE_MERGE_INVALID_PATH` | `SINGLENODE_PREPARE_INVALID_PATH` |
| `SINGLENODE_LORA_MERGE_REQUIRED` | unchanged (vLLM MVP gate, fires from `validate_config`) |
| — (new) | `SINGLENODE_PREPARE_TIMEOUT` |
| — (new) | `SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED` |

MLflow events: `"Merge started"` → `"Prepare started"` (and analogues). Structure (kwargs, category, source) preserved. CHANGELOG note.

---

## A9. Tests (all 8 categories)

### `packages/engines/tests/unit/vllm/test_prepare_model.py`

| Category | Cases |
|---|---|
| **Positive** | `merge_before_deploy=True` + adapter present → `Ok(PreparePlan(steps=(merge_step,), final_model_path=...))`. Args contain `--base-model`, `--adapter`, `--output`, `--cache-dir` in order. `outputs[0] == final_model_path`. `entrypoint == ("python3",)`. `image is None`. HF env vars present. `volumes` correct. |
| **Negative** | wrong cfg type → `Err(VLLM_CONFIG_TYPE_MISMATCH)`. |
| **Boundary** | `merge_before_deploy=False` → `Ok(PreparePlan.empty())`. `adapter_path_in_container=None` → `Ok(PreparePlan.empty())`. |
| **Invariant** | `success_marker == "MERGE_SUCCESS"` always. `success_artifact` ends with `/config.json`. `timeout_seconds == 3600`. PreparePlan validators reject inconsistent shapes (steps without final_model_path, duplicate names). spec_version=1. |
| **Dependency-error** | `prepare_model` doesn't import `paramiko`, `subprocess`, `pathlib.Path.write_text` (sentinel test). |
| **Regression** | `--trust-remote-code` flag conditional on kwarg (absorbed from `test_merge_command_with_trust_remote_code_includes_flag`). args is flat tuple, no embedded `\n`, no embedded `\\` (absorbs single-line / no-trailing-backslash regression). |
| **Logic-specific** | output_path computed as `/workspace/runs/{run_id}/model`. Cache path always `/workspace/hf_cache`. |
| **Combinatorial** | `(merge_before_deploy, adapter_present, trust_remote_code) ∈ {True, False}³` → 8 cases produce expected plan shape. |

### `packages/providers/tests/unit/providers/inference/test_format_prepare_step.py`

| Category | Cases |
|---|---|
| **Positive** | full step → `docker run --detach --name X --gpus all -v ... -e HF_HOME=... --entrypoint python3 IMAGE arg1 arg2`. |
| **Shell safety** | image with spaces quoted; args with `;rm -rf /` quoted; env values with `$` / `;` quoted; volume paths with spaces quoted. |
| **Boundary** | empty `env` → no `-e`. empty `volumes` → no `-v`. empty `args` → image is last token. `entrypoint=None` → no `--entrypoint`. |
| **Invariant** | image always after all flags. arg order preserved. |
| **Logic-specific** | `gpus_all=False` → no `--gpus all`. `extra_env` overrides step env on key collision (HF_TOKEN injection contract). |

### `packages/providers/tests/unit/providers/single_node/test_run_prepare_plan.py` (new)

| Category | Cases |
|---|---|
| **Positive** | 1-step plan succeeds → returns Ok, calls `format_prepare_step` once, exit code 0 + marker present + artifact exists. |
| **Negative** | image pull fails → `SINGLENODE_PREPARE_IMAGE_PULL_FAILED`. container start fails → `SINGLENODE_PREPARE_CONTAINER_START_FAILED`. exit code != 0 → `SINGLENODE_PREPARE_CONTAINER_FAILED`. marker missing → `SINGLENODE_PREPARE_NO_SUCCESS_MARKER`. artifact missing → `SINGLENODE_PREPARE_ARTIFACTS_NOT_FOUND`. timeout → `SINGLENODE_PREPARE_TIMEOUT`. unknown spec_version → `SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED`. |
| **Boundary** | empty plan → returns Ok immediately, no docker calls. step with no `success_marker` and no `success_artifact` → exit-code-only check works. |
| **Invariant** | output dirs cleaned (`rm -rf` then `mkdir -p`) BEFORE container run. Container `docker rm -f` AFTER run regardless of success/failure. log file `prepare.log` written. |
| **Dependency-error** | SSH disconnects mid-poll → graceful error (not hang). |
| **Regression** | docker pull timeout=1200 (preserved from `TestDockerPullTimeout`). poll interval=2s. /workspace path mapping correct (preserved from `TestMergePathMapping`). |
| **Logic-specific** | per-step MLflow events: `prepare_step_started`, `prepare_step_complete` / `prepare_step_failed`, `prepare_complete`. step.image=None → uses `engine_serve_image`. |
| **Combinatorial** | 2-step plan: first succeeds, second fails → first step's outputs remain on disk (no rollback per AD); second's error returned. 3-step plan: all succeed → all 3 events emitted in order. |

### Golden / parity tests

- `test_merge_cmd_byte_equal_v1_fixture` — captures the pre-PR-16 docker-run command for the merge step into `tests/golden/merge_cmd_v1.txt` (a snapshot). Post-refactor: `format_prepare_step(vllm_engine.prepare_model(...).unwrap().steps[0], image="ryotenkai/inference-vllm:1.0.0", container_name="helix-prepare-test_run-merge_lora", extra_env={"HF_TOKEN": "TEST"})` must produce a string differing from the snapshot ONLY by the container-name suffix (`merge` → `prepare-merge_lora`). All other tokens byte-equal. The diff is committed in the PR.
- `test_mlflow_events_byte_equal_v1_fixture` — captures the pre-PR sequence of MLflow `log_event_*` calls (with kwargs) into `tests/golden/mlflow_events_v1.json`. Post-refactor: events MUST keep the same `category`, `source`, kwargs structure; only message strings change (`"Merge"` → `"Prepare"`).

### Sentinel tests

- `test_no_io_in_engine_prepare.py` — AST-scan: `import paramiko`, `subprocess`, `os.system`, `pathlib.Path.write_text` MUST NOT appear in the engine `prepare_model` call graph.
- `test_prepare_model_signature_uniformity.py` — `inspect.signature(EngineCls().prepare_model)` matches the Protocol signature for every registered engine.
- `test_engines_do_not_import_providers.py` — already covered by importlinter contract; assert importlinter passes.

---

## A10. Risk analysis (3 iterations consolidated)

### Iteration 1 — architectural

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| AR1 | Engine forced to do IO breaks the "pure description" contract | HIGH | `prepare_model` returns `PreparePlan`; provider executes. Sentinel test enforces. |
| AR2 | success-detection contract straddles boundary | HIGH | `success_marker` + `success_artifact` are engine-declared FIELDS on PrepareStep; provider executes the checks. |
| AR3 | path-mapping leaks into engine | MEDIUM | Provider does host→container mapping BEFORE calling `prepare_model`. Engine sees only container paths. |
| AR4 | `final_model_path` ≠ `build_launch_spec.model_path_in_container` | HIGH | PreparePlan validator + provider passes `plan.final_model_path` directly into `build_launch_spec`. |
| AR5 | docker-run shell formatting drift | MEDIUM | Single sibling helper `format_prepare_step()`. Golden parity test captures byte-equal pre/post. |
| AR6 | Engine vs provider error layering | MEDIUM | Engine returns `Result[..., AppError]` with `VLLM_*` codes; provider wraps into `SINGLENODE_PREPARE_*` for wire-level error codes. |
| AR7 | Log-path ownership | LOW | Provider keeps `prepare.log` writes. Engine never touches fs. |
| AR8 | `requires_prepare` capability/manifest drift | MEDIUM | Drift detector enforces manifest ↔ runtime parity. |

### Iteration 2 — migration & operational

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| MR1 | Single-PR rollback burden | HIGH | Golden parity tests (docker-cmd snapshot, MLflow event snapshot) so any deviation visible in `git diff`. `git revert <sha>` is the rollback. |
| MR2 | Test fragility on docker-cmd strings | HIGH | Golden snapshot files (`tests/golden/`) — diffs visible at PR review. shlex-quote contract pinned by tests. |
| MR3 | Engine tests must stay IO-free | HIGH | Sentinel `test_no_io_in_engine_prepare.py`. Engine tests use synthetic configs, no SSH/docker mocks. |
| MR4 | MLflow event drift (silent telemetry breakage) | HIGH | `mlflow_events_v1.json` snapshot test. Approved diffs only. |
| MR5 | Performance/timeouts drift | MEDIUM | Constants in engine module (`_MERGE_TIMEOUT_S`); provider asserts via spec field. Regression test asserts pull_timeout=1200, poll=2s. |
| MR6 | Cross-validation prep_path == launch_path | HIGH | PreparePlan validator + integration test `test_prepare_launch_path_consistency`. |
| MR7 | `validate_config` vs `prepare_model` validation overlap | LOW | `validate_config` is the only gate for config-level errors; `prepare_model` only fails on type-mismatch (raises) or returns `Empty`. |
| MR8 | `merge_lora.py` CLI flag drift | MEDIUM | CI smoke test: `python -m ryotenkai_engines.vllm.merge_lora --help` asserts expected flag names. Lives in `packages/engines/tests/`. |

### Iteration 3 — conformance & future-proofing

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| CR1 | 2nd engine has different success contract | MEDIUM | Today: `success_marker` + `success_artifact` as plain fields (covers MarkerInLogs + FileExists). When 3rd contract arrives → promote to discriminated union. Additive evolution. |
| CR2 | engines→providers boundary unpinned | HIGH | New importlinter contract `engines is leaf`. Already passes; this just locks it. |
| CR3 | Protocol signature drift | HIGH | Sentinel `test_prepare_model_signature_uniformity.py`. Calls every registered engine's `prepare_model` with required kwargs. |
| CR4 | spec_version skew between engine and provider | MEDIUM | `PreparePlan.spec_version: int = 1`; provider rejects unknown with `SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED`. |
| CR5 | Future engine adds 3-step plan; provider's runner not tested for chains | MEDIUM | Combinatorial test of provider's `_run_prepare_plan` with 0/1/2/3 step synthetic plans. |
| CR6 | `requires_prepare` ↔ runtime parity | LOW | Drift detector + capability declared in `engine.toml`. |
| CR7 | Cache reuse across engine versions | LOW | Defer (.merge_metadata.json marker). Today: every prepare run fresh-merges (current behaviour preserved). |
| CR8 | Calibration dataset future param | LOW | Add `calibration_dataset: CalibrationDatasetRef \| None = None` to `prepare_model` kwargs when AWQ/GPTQ engine ships. Optional kwarg = additive evolution. |

---

## A11. Open questions / deferred items

| Item | Trigger to add | Why deferred today |
|---|---|---|
| `PrepareStep.cache_key: str | None` | 2nd engine + user request for cross-run reuse | Speculative. Today's behavior `rm -rf` before each merge is correct. |
| `prepare_model(..., calibration_dataset=...)` | AWQ/GPTQ engine ships | No engine consumes it; param shape unclear without concrete need. |
| `PrepareStep.depends_on: tuple[str, ...]` | Engine wants parallel steps | Linear sequencing covers vLLM + llama.cpp + TRT-LLM cases. |
| `PrepareStep.cleanup_on_failure: tuple[str, ...]` | Step where partial output is corrupting | Today partial outputs are observable + `rm -rf` cleans them on rerun. |
| `PrepareStep.resource_requirements` | Engine needs constraint provider can't infer (GPU class, VRAM min) | No use case yet; k8s-future. |
| `PrepareValidation` discriminated union | 3rd engine with success contract not covered by marker+artifact | YAGNI; 2 fields cover today. |
| Async `prepare_model` | Engine genuinely needs IO at plan-construction time | Probably never; would expose `probe_model()` instead. |
| `.merge_metadata.json` provenance marker | Operator rolls back engine but keeps merged artifacts | Not a current pain point. |

---

## A12. Verification plan

### Static
```bash
uv run lint-imports                                              # 10 contracts (9 prior + 1 new "engines is leaf to providers/control/pod/community")
uv run python packages/engines/scripts/check_engine_manifests.py # drift detector incl. requires_prepare parity
uv run mypy packages/engines/ packages/providers/ packages/shared/ packages/control/
uv run ruff check .
```

### Unit + sentinel
```bash
uv run pytest packages/engines/tests/                                    # incl. test_prepare_plan_invariants, test_prepare_model, sentinels
uv run pytest packages/providers/tests/unit/providers/inference/         # incl. test_format_prepare_step
uv run pytest packages/providers/tests/unit/providers/single_node/       # incl. test_run_prepare_plan, renamed TestRunPreparePlanErrors
```

### Golden parity (the load-bearing tests)
```bash
uv run pytest packages/control/tests/golden/test_merge_cmd_v1_byte_equal.py
uv run pytest packages/control/tests/golden/test_mlflow_events_v1.py
```

### Acceptance criteria
- [ ] Adding a stub 2nd engine with `requires_prepare=True` (e.g., a dummy `gguf` engine returning a 1-step PreparePlan) takes ≤ 4 file edits: `engine.toml`, `runtime.py`, `config.py`, optional `Dockerfile`.
- [ ] Adding a stub 2nd engine with `requires_prepare=False` (SGLang-shape) takes ≤ 3 file edits — `prepare_model` inherits `NoPrepareMixin`.
- [ ] `grep -rn "_run_merge_container\|_merge_adapter_remote\|MERGE_SCRIPT_NOT_FOUND" packages/` returns **0 hits**.
- [ ] `grep -rn "SINGLENODE_MERGE_" packages/` returns **0 hits** (all renamed to `_PREPARE_`).
- [ ] `grep -rn "MERGE_SUCCESS" packages/` returns hits **only** in `engines/vllm/runtime.py` and `engines/vllm/merge_lora.py` and `tests/`.
- [ ] importlinter passes with the new "engines is leaf to providers/control/pod/community" contract added.
- [ ] golden parity test for docker-run cmd: only the container-name token differs (`merge_*` → `helix-prepare-*-merge_lora`); all other tokens identical.
- [ ] golden parity test for MLflow events: structure (kwargs, category, source) byte-equal; messages renamed `Merge*` → `Prepare*` per AD-A13.
- [ ] No net new test failures vs PR-15 baseline.

---

## A13. Helixir memory updates (post-approval)

```python
add_memory("RyotenkAI 2026-05-06 PR-16: prepare_model() lifecycle hook on
IInferenceEngine. Engine returns Result[PreparePlan, AppError]; PreparePlan
carries tuple[PrepareStep, ...] (multi-step from day 1) + final_model_path +
spec_version. PrepareStep fields: name, image (None=use serve), entrypoint,
args, env, volumes, inputs, outputs, success_marker, success_artifact,
timeout_seconds. vLLM merge moves out of single_node provider into
VLLMEngineRuntime.prepare_model() returning 1-step plan. merge_lora.py stays
baked in image at /opt/helix/merge_lora.py (PR-15 already did COPY). Provider
gets sibling format_prepare_step() (mirror of format_docker_run) and
_run_prepare_plan() generic runner. NoPrepareMixin for engines without prep.
EngineCapabilities.requires_prepare:bool added; drift detector enforces
manifest↔runtime parity. New importlinter contract: engines do not import
providers/control/pod/community. All SINGLENODE_MERGE_* error codes renamed
to SINGLENODE_PREPARE_*. MLflow events 'Merge*' → 'Prepare*'. _run_merge_container
(330 lines) and _merge_adapter_remote (95 lines) DELETED. Single PR strategy
with golden parity tests as snapshot fixtures. No feature flag, no
backward compat. Industry alignment: KServe ServingRuntime, TensorRT-LLM init
script, OME unified lifecycle, Pluggy hooks. Deferred: cache_key (trigger:
2nd engine), calibration_dataset (trigger: AWQ/GPTQ), depends_on (trigger:
parallel steps), cleanup_on_failure, resource_requirements,
PrepareValidation discriminated union, async prepare_model.")
```

---

**End of plan.**
