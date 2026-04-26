# Community plugin authoring guide

A `community/` plugin is a self-contained folder under one of four kind
directories that the host imports at runtime via the community catalog.
This guide is the single source of truth for the cross-cutting parts:
lifecycle, secrets, env contract, the registry API, and the deprecation
policy. Per-kind specifics (the actual ABC each kind subclasses, what
the runtime passes to the entry point, expected return shape) live in
the kind directory's own `README.md`.

## Layout

```
community/<kind>/<plugin_id>/
├── manifest.toml          # required — single source of truth for metadata
├── plugin.py              # required — single-file form, OR
└── plugin/                # required — package form (multi-file)
    ├── __init__.py
    └── …
```

Bootstrap a fresh skeleton with the CLI — it generates a working
folder you can load through the catalog without further edits:

```sh
ryotenkai plugin scaffold validation hello_world
```

The four supported kinds are `validation`, `evaluation`, `reward`,
and `reports`. Each ships its own README under
`community/<kind>/README.md` covering the kind-specific contract.

## Manifest schema

Manifests declare `schema_version` (currently **4**). Older versions
are still accepted; future ones get rejected with a clear "upgrade
the host" error. The minimum manifest is just identity + entry point:

```toml
schema_version = 4

[plugin]
id = "my_plugin"
kind = "validation"
name = "My Plugin"
version = "1.0.0"
description = "One-line description shown in the UI catalog."

[plugin.entry_point]
module = "plugin"
class = "MyPlugin"
```

`params_schema` and `thresholds_schema` describe the per-instance
config the user fills in via the Configure modal (or hand-edits in
YAML). Field names must be **snake_case Python identifiers** — the
loader rejects `min-samples` or `MaxLength` to keep TypedDict /
attribute-access codegen unambiguous downstream.

```toml
[params_schema.timeout_seconds]
type = "integer"
min = 1
max = 60
default = 30
title = "Timeout (seconds)"
description = "How long to wait per upstream call."
```

`suggested_params` / `suggested_thresholds` pre-fill the form when a
user first adds the plugin to a project.

## Lifecycle per kind

The community catalog drives the same lifecycle for every kind:

1. **Discovery** — catalog scans `community/<kind>/` once per process
   (and again when on-disk fingerprint changes). Each entry is parsed,
   the manifest validated, and the entry-point class imported.
2. **Registration** — the class is attached to a per-kind
   `PluginRegistry`. The loader stamps `name`, `version`,
   `_required_secrets` (derived from `[[required_env]]`), and
   `_community_manifest` onto the class.
3. **Instantiation** — when a stage actually needs the plugin, it
   calls `registry.instantiate(plugin_id, …)`. The registry handles
   secret + env injection (see below) and forwards kind-specific
   kwargs to your `__init__`.
4. **Execution** — the kind's runtime hands a context-specific input
   to your method (`validate(dataset)`, `evaluate(samples)`,
   `build_trainer_kwargs(...)`, `render(ctx)`).

What the runtime passes to your `__init__`:

| Kind | Constructor signature |
|---|---|
| `validation` | `(params: dict, thresholds: dict)` |
| `evaluation` | `(params: dict, thresholds: dict)` |
| `reward` | `(params: dict)` |
| `reports` | `()` (no kwargs) |

## `[[required_env]]` contract

Every env var your plugin needs at runtime is declared in the manifest:

```toml
[[required_env]]
name = "EVAL_CEREBRAS_API_KEY"
description = "API key for the upstream judge service"
optional = false
secret = true
managed_by = ""
```

Field semantics:

- `optional = false` (default) — the **preflight gate** refuses to
  launch the pipeline if the value isn't set in process env, project
  `env.json`, or `secrets.env`. By the time your code runs, the value
  is guaranteed available.
- `secret = true` — UI renders a password-style input, value is
  injected into `self._secrets[name]` automatically. Plugins access it
  through the typed helper (see below).
- `managed_by = "integrations" | "providers"` — the value is owned
  by a Settings workspace token (HF integration, RunPod provider). The
  preflight gate skips this entry; the Settings layer checks it
  per-resource. Use the `_env(name)` helper at runtime.

The loader derives `cls._required_secrets` from entries with
`secret=true, optional=false` and the registry resolves their values
at instantiate time.

### Cross-checking with `BasePlugin.REQUIRED_ENV`

Authors who prefer code as the source of truth can declare the env
contract on the class:

```python
from src.community.manifest import RequiredEnvSpec

class MyPlugin(EvaluatorPlugin):
    REQUIRED_ENV = (
        RequiredEnvSpec(name="EVAL_API_KEY", optional=False, secret=True),
    )
```

The loader then **cross-checks** the Python tuple against the
manifest's `[[required_env]]` block on `(name, optional, secret,
managed_by)`. Drift raises a precise diff at load time. Keep both
sides aligned with `ryotenkai community sync-envs <plugin>`, which
rewrites the manifest from the ClassVar.

Leaving `REQUIRED_ENV = ()` (the default) opts out of the cross-check
— the manifest is the only source of truth.

## Secrets vs envs

The four kinds use different namespace prefixes so a plugin can't
read another kind's secrets by accident:

| Kind | Prefix |
|---|---|
| `validation` | `DTST_*` |
| `evaluation` | `EVAL_*` |
| `reward` | `RWRD_*` |
| `reports` | `RPRT_*` |

System secrets (`HF_TOKEN`, `RUNPOD_API_KEY`) live in
`src/config/secrets/model.py`'s typed fields and are **not** accessible
to plugins — they're routed through the integration / provider layer.

## Runtime helpers on `BasePlugin`

Plugin classes inherit two helpers that the rest of the runtime
populates:

```python
class MyPlugin(EvaluatorPlugin):
    def evaluate(self, samples):
        api_key = self._secret("EVAL_API_KEY")          # required, fail loud
        timeout = int(self._env("EVAL_TIMEOUT", "30"))  # optional, with default
        ...
```

- `self._secret(name)` — returns a value from the resolved
  `_secrets` dict. Raises `KeyError` with a clear hint if the key
  isn't in the dict (which means the manifest's `[[required_env]]`
  doesn't declare it). **Never falls back to `os.environ`** — that
  would mask manifest contract violations.
- `self._env(name, default=None)` — reads from the registry's
  injected env dict first, then falls back to `os.environ`. Empty
  strings are treated as unset so operators can blank a var with
  `MY_VAR=`.

Authors should NOT poke `os.environ` or `self._secrets` directly.
The helpers give us a place to add validation, telemetry, and per-test
mocking without touching every plugin folder.

## Sharing code across plugins — `community/libs/`

When two or more plugins in **different kinds** need the same domain
code (e.g. a HelixQL compiler wrapper used by validation, reward, and
evaluation), put it in `community/libs/<lib>/` rather than
copy-pasting or pulling into `src/`. The catalog automatically
registers each subpackage as `community_libs.<lib>` in `sys.modules`
before any plugin loads:

```python
# inside any community/<kind>/<plugin>/plugin.py
from community_libs.helixql.compiler import get_compiler
from community_libs.helixql.semantics import semantic_match_details
```

Single-plugin helpers stay in the plugin's own folder. Generic
platform utilities (string/dict/retry) belong in `src/utils/`. See
[`community/libs/README.md`](libs/README.md) for the full contract,
authoring rules, and an example layout.

## Reward batch-kwargs contract

The reward kind is the only one whose runtime invocation passes
trainer-specific keyword arguments through to your reward callback.
TRL's `GRPOTrainer` / `SAPOTrainer` invoke each reward function with:

```python
def my_reward(prompts, completions, **kwargs) -> list[float]:
    schema_context  = kwargs.get("schema_context")    # str | None
    reference_answer = kwargs.get("reference_answer") # str | None
    ...
```

- `prompts: list[str]` — input prompts for the current batch.
- `completions: list[str]` — model-generated completions.
- `schema_context: str | None` — the dataset row's schema field
  (used by HelixQL-style domain rewards).
- `reference_answer: str | None` — the ground-truth answer for the
  prompt, when the dataset includes it.

Return a list of floats (one per completion) in `[0.0, 1.0]` range.
Anything outside that range gets clipped by TRL's normalisation and
is usually a sign of a buggy reward function.

## Testing

Each plugin folder ships a `tests/` directory. The scaffold CLI
emits a smoke test that imports the class — a useful regression
guard. Real coverage is the author's responsibility:

```python
# community/<kind>/<id>/tests/test_plugin.py
def test_score_returns_in_range(...):
    plugin = MyPlugin(params={"timeout": 5}, thresholds={"min_score": 0.5})
    plugin._secrets = {"EVAL_API_KEY": "test"}  # bypass the registry
    result = plugin.evaluate([sample])
    assert 0.0 <= result.metrics["mean_score"] <= 1.0
```

The shared community fixtures in `src/tests/unit/community/conftest.py`
(`tmp_community_root`, `make_plugin_dir`, `mock_catalog`,
`fake_secrets`) help when you need to exercise the loader/registry
end-to-end from a unit test.

## Deprecation policy

Schema changes follow a one-minor-release rule:

1. **Announce** — a deprecation note lands in the manifest schema's
   docstring history and CHANGELOG when a field/feature gets slated
   for removal.
2. **Warn** — the next release continues to load the deprecated
   shape, logging a `DeprecationWarning`. CI tests start failing
   when authors run with `-W error::DeprecationWarning` so internal
   plugins migrate first.
3. **Remove** — the release after that drops the field. Manifests
   still using it fail load with a clear "removed in v<N>" error.

The legacy `[secrets].required` block followed this pattern: announced
in v3, dropped in v4. New authors should never see it.

## CLI reference

Two complementary command groups under `ryotenkai`:

| Command | Use when |
|---|---|
| `plugin scaffold <kind> <id>` | Bootstrapping a brand-new plugin folder. |
| `community scaffold <path>` | Re-deriving the manifest for an existing folder (you wrote `plugin.py` first). |
| `community sync <path>` | Re-running inference on the code and 3-way-merging the result into `manifest.toml`. |
| `community sync-envs <path>` | Aligning the manifest's `[[required_env]]` with the class's `REQUIRED_ENV` ClassVar. |
| `community pack <path>` | Producing a distributable `.zip`. |

Every command accepts `--help` for its own flags.
