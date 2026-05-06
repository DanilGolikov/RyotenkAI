# Adding a New Inference Engine — Step-by-Step

This guide walks an engineer (or LLM agent) through adding a new
inference engine to the RyotenkAI platform. The system is plugin-based
and discriminator-driven — most of the work is **drop a folder**.

> **Status check first.** Before you start, run
> `uv run python packages/engines/scripts/check_engine_manifests.py`
> and `uv run pytest packages/engines/tests/` — both must be green.
> If they're red, fix the existing engines before adding new ones.

---

## TL;DR (the contract)

To add an engine called `<engine_id>` (snake_case, alphanumeric):

1. Create folder `packages/engines/src/ryotenkai_engines/<engine_id>/`.
2. Write **4 files** in that folder:
   - `engine.toml` — declarative manifest (~30 lines).
   - `runtime.py` — `<EngineId>EngineRuntime(IInferenceEngine)` (~80 lines).
   - `config.py` — `<EngineId>EngineConfig(BaseEngineConfig)` (~30 lines).
   - `Dockerfile` — base image + RyotenkAI customizations (~30 lines).
3. Add `"<engine_id>"` to `[capabilities].supported_engines` in any
   `provider.toml` that should launch it (`single_node`, `runpod`, …).
4. Build & push image: convention name = `ryotenkai/inference-<id>:<version>`.

That's it. Pydantic discriminator picks up the new variant automatically.
No edits to `InferenceConfig`, validators, control plane, or other engines.

---

## Step 1 — Pick an `engine_id`

Rules:
- snake_case, lowercase, starts with a letter, `[a-z0-9_]+`.
- Must equal the folder name under `packages/engines/src/ryotenkai_engines/`.
- Stable forever — used in user YAML (`inference.engine.kind: <id>`),
  image tag (`ryotenkai/inference-<id>:<v>`), MLflow tags. Renaming
  is a **contract break**.

**Examples that work:** `vllm`, `sglang`, `tgi`, `ollama`, `llama_cpp`.
**Examples that don't:** `vLLM` (uppercase), `tgi-fork` (hyphen),
`1_engine` (digit-first), `vllm/community` (path).

For this guide we'll use **`sglang`** as a running example.

---

## Step 2 — Create the folder

```bash
mkdir -p packages/engines/src/ryotenkai_engines/sglang
```

The discovery walker (`EngineRegistry.from_filesystem()`) picks up any
folder under `packages/engines/src/ryotenkai_engines/` that contains an
`engine.toml`.

---

## Step 3 — Write `engine.toml` (manifest)

This is the **single declarative source of truth** for the engine's
identity and capabilities. The runtime class MUST mirror it exactly
(drift detector enforces).

```toml
# packages/engines/src/ryotenkai_engines/sglang/engine.toml

schema_version = 1

[engine]
id               = "sglang"               # MUST match folder name
name             = "SGLang"               # display name for UI
version          = "1.0.0"                # YOUR integration contract semver — drives the image tag
upstream_version = "0.4.0"                # informational: actual upstream SGLang release
description      = "RadixAttention-based OpenAI-compatible LLM server."
stability        = "experimental"         # "stable" | "beta" | "experimental"
homepage         = "https://github.com/sgl-project/sglang"

[capabilities]
api_dialect              = "openai_compatible"   # or "custom" (will fail validation today; ModelClientFactory only knows OpenAI)
supports_lora            = true                  # accepts LoRA adapter paths at launch
supports_quantization    = true                  # accepts a quantization mode CLI flag
supports_streaming       = true                  # streams tokens via SSE / chunked HTTP
supports_tensor_parallel = true                  # multi-GPU TP sharding
supported_quantizations  = ["awq", "gptq", "fp8"]  # MUST be empty if supports_quantization=false
supported_dtypes         = ["bfloat16", "float16"] # MUST be non-empty
default_port             = 30000                 # engine's natural bind port

# [image] block is OPTIONAL.
# If omitted, the image auto-derives via convention:
#   ``{prefix}/inference-<id>:<version>``
# i.e. ``ryotenkai/inference-sglang:1.0.0``.
# Override here ONLY if you need a non-conventional name (forks, custom
# registries, multi-arch tags). Floating tags (:latest, :dev, :main)
# are rejected by the manifest validator.

[entry_points.runtime]
module = "ryotenkai_engines.sglang.runtime"
class  = "SGLangEngineRuntime"

[entry_points.config_schema]
module = "ryotenkai_engines.sglang.config"
class  = "SGLangEngineConfig"
```

**Hard invariants the validator enforces** (`EngineManifest`):

| Rule | Why |
|---|---|
| `engine.id` matches folder name | Discovery walker pairs them |
| `engine.id` matches `[a-z][a-z0-9_]*` | Stable-id contract |
| `engine.version` is valid PEP 440 / SemVer | Drives image tag — needs ordering |
| `schema_version = 1` (current LATEST) | Future schemas will bump this; older loaders reject newer schemas |
| `supports_quantization=False` ⇒ `supported_quantizations` empty | No phantom modes |
| No duplicates in `supported_dtypes` / `supported_quantizations` | Data integrity |
| `supported_dtypes` is non-empty | Empty would mean nothing can run |
| `supported_quantizations` ∩ `supported_dtypes` = ∅ | Distinct concepts |
| `default_port` ∈ [1, 65535] | TCP range |
| `[image].default` (if present) NOT a floating tag | Reproducibility |

---

## Step 4 — Write `config.py`

Pydantic class for engine-specific runtime tuning knobs. Subclasses
`BaseEngineConfig`, overrides `kind` with the engine id Literal:

```python
# packages/engines/src/ryotenkai_engines/sglang/config.py

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ryotenkai_engines.interfaces import BaseEngineConfig


class SGLangEngineConfig(BaseEngineConfig):
    """SGLang runtime config.

    The container image is resolved by the engine registry (convention
    default: ``ryotenkai/inference-sglang:{engine_version}``;
    overridable via env / provider / manifest). This class carries
    only runtime tuning knobs.
    """

    kind: Literal["sglang"] = "sglang"   # MUST match engine.toml [engine].id

    # --- engine-specific knobs ---
    # Only fields the engine actually accepts as CLI args. NO image,
    # NO model paths — those are plumbed by the provider via LaunchSpec.

    tensor_parallel_size: int = Field(
        1,
        ge=1,
        description="Number of GPUs for tensor parallelism.",
    )
    context_length: int = Field(
        4096,
        ge=256,
        description="Maximum sequence length the engine will accept.",
    )
    mem_fraction_static: float = Field(
        0.85,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU memory reserved for the model.",
    )
    quantization: str | None = Field(
        None,
        description=(
            "Optional quantization mode (awq / gptq / fp8). "
            "Validated against capabilities at engine.validate_config()."
        ),
    )


__all__ = ("SGLangEngineConfig",)
```

**Conventions:**
- `kind` ALWAYS first field, ALWAYS `Literal["<engine_id>"]` with default.
- Use Pydantic `Field(...)` with `ge=` / `le=` / `gt=` constraints — fail-fast
  beats runtime errors.
- DON'T add `image:` / `model_path:` fields — those flow through `LaunchSpec`
  from the provider.
- DON'T add `quantization` cross-validation in Pydantic — do it in
  `runtime.validate_config()` (engine-level invariants).

---

## Step 5 — Write `runtime.py`

The actual `IInferenceEngine` implementation. **Pure command-builder**:
no IO, no async, no side effects. Returns data; provider does the launch.

```python
# packages/engines/src/ryotenkai_engines/sglang/runtime.py

from __future__ import annotations

from typing import ClassVar

from ryotenkai_engines.capabilities import EngineCapabilities
from ryotenkai_engines.interfaces import (
    BaseEngineConfig,
    IInferenceEngine,  # noqa: F401  — runtime-checkable; isinstance check uses it
    LaunchSpec,
)
from ryotenkai_engines.sglang.config import SGLangEngineConfig
from ryotenkai_shared.utils.result import AppError, Err, Ok, Result


class SGLangEngineRuntime:
    """SGLang OpenAI-compatible server runtime.

    See :class:`IInferenceEngine` for the Protocol contract.
    """

    engine_id: ClassVar[str] = "sglang"                    # MUST match engine.toml
    config_class: ClassVar[type[BaseEngineConfig]] = SGLangEngineConfig

    # ---- capabilities ----

    def get_capabilities(self) -> EngineCapabilities:
        """MUST mirror engine.toml [capabilities] exactly.

        Drift detector cross-checks this against the manifest in CI.
        Adding a field here without updating the .toml (or vice versa)
        fails the build.
        """
        return EngineCapabilities(
            api_dialect="openai_compatible",
            supports_lora=True,
            supports_quantization=True,
            supports_streaming=True,
            supports_tensor_parallel=True,
            supported_quantizations=("awq", "gptq", "fp8"),
            supported_dtypes=("bfloat16", "float16"),
            default_port=30000,
        )

    # ---- launch spec ----

    def build_launch_spec(
        self,
        *,
        cfg: BaseEngineConfig,
        image: str,
        container_name: str,
        port: int,
        workspace_host_path: str,
        model_path_in_container: str,
    ) -> LaunchSpec:
        """Build the structured launch description.

        Provider receives this and decides HOW to launch (docker run /
        k8s Pod / systemd / RunPod API). Engine just describes WHAT
        to run.
        """
        if not isinstance(cfg, SGLangEngineConfig):
            raise TypeError(
                f"SGLangEngineRuntime expected SGLangEngineConfig, got {type(cfg).__name__}"
            )

        args: list[str] = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path_in_container,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--tp-size", str(cfg.tensor_parallel_size),
            "--context-length", str(cfg.context_length),
            "--mem-fraction-static", str(cfg.mem_fraction_static),
        ]
        if cfg.quantization:
            args.extend(["--quantization", cfg.quantization])

        env = {
            "HF_HOME": "/workspace/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/workspace/hf_cache",
            "TRANSFORMERS_CACHE": "/workspace/hf_cache",
        }

        return LaunchSpec(
            image=image,
            container_name=container_name,
            args=tuple(args),
            env=env,
            port=port,
            volumes=((workspace_host_path, "/workspace"),),
        )

    # ---- healthcheck ----

    def build_healthcheck_command(self, *, host: str, port: int) -> str:
        """Shell snippet whose exit-0 means engine is ready.

        Provider runs this remotely (SSH / docker exec).
        Echoes ``1`` on success, ``0`` on failure (legacy convention).
        """
        url = f"http://{host}:{port}/v1/models"
        return f"curl -s -f -m 5 {url} >/dev/null 2>&1 && echo 1 || echo 0"

    # ---- endpoint url ----

    def build_default_endpoint_url(self, *, host: str, port: int) -> str:
        """OpenAI-compatible base URL — what ModelClientFactory dials."""
        return f"http://{host}:{port}/v1"

    # ---- validate_config ----

    def validate_config(self, cfg: BaseEngineConfig) -> Result[None, AppError]:
        """Engine-side invariants (post-Pydantic).

        Cross-field rules that don't fit cleanly in Pydantic ``@model_validator``.
        Today: quantization mode whitelist (caught here instead of
        loose-typed Pydantic ``str``).
        """
        if not isinstance(cfg, SGLangEngineConfig):
            return Err(
                AppError(
                    message=f"Expected SGLangEngineConfig, got {type(cfg).__name__}",
                    code="SGLANG_CONFIG_TYPE_MISMATCH",
                )
            )

        if cfg.quantization is not None:
            caps = self.get_capabilities()
            if cfg.quantization not in caps.supported_quantizations:
                return Err(
                    AppError(
                        message=(
                            f"SGLang does not support quantization "
                            f"{cfg.quantization!r}. Supported: "
                            f"{list(caps.supported_quantizations)!r}."
                        ),
                        code="SGLANG_QUANTIZATION_UNSUPPORTED",
                        details={
                            "requested": cfg.quantization,
                            "supported": list(caps.supported_quantizations),
                        },
                    )
                )

        return Ok(None)


__all__ = ("SGLangEngineRuntime",)
```

**Hard rules:**
- `engine_id` ClassVar MUST equal `engine.toml [engine].id`.
- `config_class` ClassVar MUST equal the class from `config.py`.
- `get_capabilities()` MUST return the exact same `EngineCapabilities`
  values as the manifest. Drift detector cross-checks 1:1.
- All 6 methods (`get_capabilities`, `build_launch_spec`,
  `build_healthcheck_command`, `build_default_endpoint_url`,
  `validate_config`, plus the ClassVars) — required.
- The class MUST be **zero-arg constructible** (`runtime_cls()` works).
  Don't add `__init__` parameters.
- NO IO. NO async. NO HTTP calls. NO SSH. NO subprocess.
  Engine = command-builder. Provider = launcher.

---

## Step 6 — Write `Dockerfile`

```dockerfile
# packages/engines/src/ryotenkai_engines/sglang/Dockerfile

FROM lmsysorg/sglang:v0.4.0

# RyotenkAI customizations
WORKDIR /workspace

ENV HF_HOME=/workspace/hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache

# (Optional) Add SSH server, merge utilities, healthcheck binaries —
# whatever your launcher provider needs INSIDE the container.

CMD ["bash", "-lc", "tail -f /dev/null"]
```

**Convention:**
- Base on the upstream engine's official image when possible.
- Pin the upstream version in `FROM` (matches `engine.toml [engine].upstream_version`).
- The CI build pipeline reads `engine.toml [engine].version` and tags
  the image as `ryotenkai/inference-sglang:<version>`.

---

## Step 7 — Wire it into a provider

Add `"<engine_id>"` to the `supported_engines` list in any
`provider.toml` that should launch it:

```toml
# packages/providers/src/ryotenkai_providers/single_node/provider.toml
# (or runpod/provider.toml, or any other provider's manifest)

[capabilities]
# ... existing fields ...
supported_engines = ["vllm", "sglang"]   # ← add the new one
```

Also update the provider class's `SUPPORTED_ENGINES` ClassVar (defensive
mirror for direct callers):

```python
# packages/providers/src/ryotenkai_providers/single_node/inference/provider.py
SUPPORTED_ENGINES: ClassVar[frozenset[str]] = frozenset({"vllm", "sglang"})
```

The cross-validator (`validate_inference_enabled_is_supported`)
automatically rejects (provider, engine) pairs the manifest doesn't
allow.

---

## Step 8 — Verify

```bash
# 1. Manifest loads + drift detector clean
uv run python packages/engines/scripts/check_engine_manifests.py
# → "OK — N engine(s) checked: sglang, vllm"

# 2. Contract parity tests pass for the new engine
uv run pytest packages/engines/tests/contract/

# 3. Discriminator naming sentinel green
uv run pytest packages/engines/tests/sentinel/

# 4. Importlinter contracts hold
uv run lint-imports
# → "engines is a leaf (depends only on shared) KEPT"
# → "generic code must not import concrete engine modules KEPT"

# 5. End-to-end smoke
cat <<'EOF' > /tmp/test_sglang.yaml
inference:
  enabled: true
  provider: single_node
  engine:
    kind: sglang
    tensor_parallel_size: 1
    context_length: 8192
EOF
uv run python -c "
import yaml
from ryotenkai_shared.config.inference.schema import InferenceConfig
data = yaml.safe_load(open('/tmp/test_sglang.yaml'))['inference']
cfg = InferenceConfig.model_validate(data)
print('kind:', cfg.engine.kind)
print('image:', __import__('ryotenkai_engines').get_registry().get_image(cfg.engine.kind))
"
```

If all 5 checks are green — the engine is ready to ship.

---

## Common pitfalls (and how the system catches them)

| Mistake | Caught by | Error message |
|---|---|---|
| Folder name ≠ `engine.toml` `[engine].id` | Registry walker | `"folder name {x} does not match engine.id {y}"` |
| Capability values don't match manifest | Drift detector + contract test | `"capability drift for {id}: manifest != runtime"` |
| `kind: Literal["wrong_id"]` | Registry `_extract_kind_literal` | `"ENGINE_CONFIG_KIND_DRIFT"` |
| `engine_id` ClassVar wrong | Registry runtime check | `"ENGINE_RUNTIME_ID_DRIFT"` |
| Discriminator named `"type"` instead of `"kind"` | AST sentinel | `"AD-6: all discriminators must be named 'kind'"` |
| Import error in `runtime.py` | Registry `LoadFailure` | `"locator … could not be resolved: {ImportError}"` |
| Floating image tag (`:latest`) | Manifest validator | `"image.default uses a floating tag"` |
| Generic code does `from ryotenkai_engines.sglang import …` | Importlinter | `"generic code must not import concrete engine modules"` |

---

## Where each piece is enforced

```
   Author writes:
     engine.toml          ← declarative metadata
     runtime.py           ← IInferenceEngine impl
     config.py            ← BaseEngineConfig subclass
     Dockerfile           ← base image + RyotenkAI tweaks

                       ↓
   Manifest validator (Pydantic, EngineManifest):
     extra="forbid", schema_version, snake_case id,
     PEP 440 version, capability invariants, no floating tags

                       ↓
   Registry validator (runtime, EngineRegistry.get_runtime):
     class is IInferenceEngine instance,
     engine_id ClassVar matches manifest,
     kind: Literal[...] matches manifest

                       ↓
   Contract parity test (pytest, test_engine_protocol_parity):
     7 checks per engine — Protocol, ClassVars, kind Literal,
     capabilities ≡ manifest

                       ↓
   AST sentinel (pytest, test_discriminator_uniformity):
     every Discriminator(...) literal arg is "kind"

                       ↓
   Drift detector (CI script, check_engine_manifests):
     same checks as contract test, but standalone for CI
```

If you change the contract (rename a method, add a required field),
**all five layers fire on the same drift** — no silent regressions.

---

## When NOT to add an engine

- The engine speaks a non-OpenAI protocol (`api_dialect = "custom"`):
  the validator will reject `enabled=true` because `ModelClientFactory`
  only knows OpenAI. Adding a custom dialect is a separate, larger
  refactor — file an ADR first.
- The engine is a Python library (BentoML-style embedded engine), not
  a container. Our launcher contract is "container with HTTP server".
  Embedded engines need a different abstraction.
- The engine has incompatible launch semantics (e.g., requires an
  external orchestrator like Ray Serve). Wrap it in a container that
  hides those details, or design a new provider type.

---

## Reference

- Plan ADR: `docs/plans/purring-sleeping-hartmanis.md`
- Protocol: `packages/engines/src/ryotenkai_engines/interfaces.py`
- Manifest schema: `packages/engines/src/ryotenkai_engines/manifest.py`
- Registry: `packages/engines/src/ryotenkai_engines/registry.py`
- Image resolution: `packages/engines/src/ryotenkai_engines/images.py`
- Existing engine (reference impl): `packages/engines/src/ryotenkai_engines/vllm/`
