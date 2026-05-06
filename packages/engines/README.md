# ryotenkai-engines

Inference engine plugin system for the RyotenkAI platform.

This package owns the **engine** axis of inference: the runtime that crunches
tokens (vLLM, SGLang, MAX, Ollama, …). The orthogonal **provider** axis (where
that runtime is launched — single_node, RunPod, k8s) lives in
`packages/providers/`.

## Status

**SCAFFOLDING (PR-1 of 10).** Module stubs only — public API is empty.
See `docs/plans/purring-sleeping-hartmanis.md` for the full design.

## Layout

```
src/ryotenkai_engines/
├── interfaces.py           # IInferenceEngine Protocol, BaseEngineConfig, LaunchSpec
├── manifest.py             # EngineManifest pydantic schema for engine.toml
├── registry.py             # EngineRegistry — filesystem discovery, lazy resolution
├── images.py               # resolve_image() — convention + override chain
├── capabilities.py         # EngineCapabilities pydantic
├── _config_union.py        # Tag-based discriminated union builder
├── errors.py               # EngineRegistryError, EngineConfigError
└── <engine_id>/            # one subfolder per shipped engine
    ├── engine.toml
    ├── runtime.py          # <Engine>EngineRuntime(IInferenceEngine)
    ├── config.py           # <Engine>EngineConfig (kind="<id>")
    └── Dockerfile          # optional — convention-named image build
```

## Adding a new engine (after PR-2/PR-3 land)

1. `mkdir packages/engines/src/ryotenkai_engines/<engine_id>/`
2. Write `engine.toml` (declarative metadata + capabilities).
3. Implement `<Engine>EngineRuntime(IInferenceEngine)` in `runtime.py`.
4. Define `<Engine>EngineConfig(BaseEngineConfig)` with `kind: Literal["<id>"]`.
5. List `"<engine_id>"` in `provider.toml [capabilities.inference] supported_engines`
   for any provider that should launch this engine.

That's it. Image name auto-derives via convention
(`ryotenkai/inference-{id}:{version}`); registry discovers the manifest at
process start; Pydantic discriminator routes user YAML to the right
config class.

## Image naming convention

Default: `f"{registry_prefix}/inference-{engine_id}:{engine_version}"`
where:
- `registry_prefix` defaults to `"ryotenkai"`, overridable via env
  `RYOTENKAI_INFERENCE_IMAGE_REGISTRY`.
- `engine_version` is `engine.toml [engine].version` (semver of OUR
  integration contract, NOT upstream version of the engine).

Override chain (first match wins):
1. Env var `RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_<ENGINE_UPPER>`.
2. Provider-side override: `provider.toml [capabilities.inference.engine_overrides.<id>].image`.
3. Manifest explicit: `engine.toml [image].default`.
4. Convention.

## Versioning policy

- `engine.toml [engine].version` follows semver of OUR integration contract.
- `engine.toml [engine].upstream_version` is informational (the upstream
  engine release we currently package). Bumping it without touching our
  contract is allowed and triggers an image rebuild without a manifest
  schema bump.
- `engine.toml schema_version` bumps only on breaking changes to the
  manifest schema itself.
