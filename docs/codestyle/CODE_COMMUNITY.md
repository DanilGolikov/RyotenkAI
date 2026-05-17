# Community plugins & libs in this project

Guidelines for writing, packaging, and consuming code under `community/` —
plugins (validation / evaluation / reward / reports), presets, and shared
libs. The contract here is the foundation that future authors and LLM
agents build on; getting it wrong fragments the catalog and produces
plugins that load locally but fail in CI.

---

## Contents

- [Instructions for LLMs](#instructions-for-llms)
- [Layering rule — what lives where](#layering-rule--what-lives-where)
- [Plugin manifest (schema v5)](#plugin-manifest-schema-v5)
- [Lib manifest (schema v1)](#lib-manifest-schema-v1)
- [Lib distribution: folder OR zip](#lib-distribution-folder-or-zip)
- [`community_libs.*` import contract](#community_libs-import-contract)
- [Plugin → lib version constraints](#plugin--lib-version-constraints)
- [REQUIRED_LIBS / REQUIRED_ENV cross-check](#required_libs--required_env-cross-check)
- [Runtime helpers — `_env` / `_secret`](#runtime-helpers--_env--_secret)
- [CLI surface — `ryotenkai plugin <verb>`](#cli-surface--ryotenkai-plugin-verb)
- [Anti-patterns](#anti-patterns)
- [Common LLM mistakes](#common-llm-mistakes)
- [Tests](#tests)

---

## Instructions for LLMs

> This document is for developers and LLM agents. When generating or
> refactoring code that touches `community/`, `src/community/`, or any
> plugin author surface — read and apply the rules below.

**Quick reference:** `src/` = platform, `community/` = domain | every
plugin/lib has a `manifest.toml` | plugins import shared code via
`from community_libs.<lib>.<module> import …` | version constraints
are PEP 440 specifiers in `[[lib_requirements]]` | `REQUIRED_LIBS`
mirrors the manifest on the class for cross-check | NO backwards-compat
shims — schema bumps remove the old shape cleanly.

**When editing code in this project, follow these rules.**

### Mandatory rules (MUST)

1. **Never put domain code in `src/`.** HelixQL, Cypher, SPARQL — any
   DSL or domain-specific helper — lives under `community/libs/<name>/`.
   `src/` is platform infrastructure (loader, registries, base classes,
   generic utilities).
2. **Never `[plugin].libs = [...]`.** That shorthand was removed in
   schema v5. Use top-level `[[lib_requirements]]` blocks with a `name`
   and optional PEP 440 `version` specifier.
3. **Never `from src.utils.domains.<x>` or any other `src.*` domain
   import.** Plugins import from `community_libs.<lib>.<module>`.
4. **Never edit a community plugin's runtime contract without bumping
   `[plugin].version`.** Run `ryotenkai plugin sync community/<kind>/<id>`
   — that bumps by default. Same rule for libs (`[lib].version`).
5. **Never silently catch `os.environ.get(...)` in plugin bodies.** Use
   `self._env(name, default)` and `self._secret(name)`. The latter
   raises if the key isn't in `[[required_env]]` — that's the contract
   that keeps env declarations honest.
6. **Never put eager `from community_libs.<x>.<y> import …` at the
   top of a lib's `__init__.py`.** Pytest collection imports the
   package before any fixture runs, so eager re-exports break
   discovery. Use PEP 562 `__getattr__` (see
   `community/libs/helixql/__init__.py` for the pattern).
7. **Never add backwards-compat shims for removed manifest fields.**
   Project rule: when a schema version bumps, the old shape is
   deleted. Stale manifests get a plain Pydantic `extra="forbid"`
   error — that's intentional.
8. **Never write a separate `sync-<aspect>` CLI command.** The
   unified `ryotenkai plugin sync <path>` classifies the leaf
   (plugin / preset / lib) and dispatches. Adding `sync-libs` /
   `sync-deps` / `sync-author` would fragment the contract — extend
   the existing `sync_plugin_manifest` instead.
9. **Never declare cross-plugin helpers in a single plugin folder.**
   The moment a helper is used by ≥2 plugins (in any kind), extract
   to `community/libs/<name>/`. Single-plugin helpers stay in the
   plugin's own folder.

### Quick choice tree

| Situation | Action |
|-----------|--------|
| Writing a new validation/evaluation/reward/reports plugin | `ryotenkai plugin scaffold <kind> <id>` — generates minimum-valid skeleton |
| Adding a shared helper used by ≥2 plugins across kinds | Create `community/libs/<name>/` with `manifest.toml` + `__init__.py` (lazy re-exports) |
| Plugin needs `os.environ["FOO"]` | Declare in `[[required_env]]`; read via `self._env("FOO")` |
| Plugin needs a secret credential | `[[required_env]]` with `secret=true, optional=false`; read via `self._secret("FOO")` |
| Plugin depends on `community_libs.helixql` | Add `[[lib_requirements]]` with optional PEP 440 `version`; mirror on class with `REQUIRED_LIBS = (("helixql", ">=1.0.0"),)` |
| Cleaning up after a class-level contract edit | `ryotenkai plugin sync community/<kind>/<id>` — re-renders `[[required_env]]`, `[[lib_requirements]]`, params/thresholds schema, bumps version |
| Shipping a lib for distribution | `ryotenkai plugin pack community/libs/<name>` — produces `<name>.zip` next to the source folder |
| Stale `[[lib_requirements]]` entry pointing at a deleted lib | Loader fails the plugin with a precise message. Either restore the lib or remove the entry — don't comment it out |

---

## Layering rule — what lives where

```
src/                       ← platform (knows nothing about HelixQL, Cypher, …)
├── community/             ← loader, catalog, manifest schemas, registry base
├── data/validation/       ← ValidationPlugin ABC + registry
├── evaluation/plugins/    ← EvaluatorPlugin ABC + registry
├── training/reward_plugins/ ← RewardPlugin ABC + registry
├── reports/plugins/       ← ReportPlugin ABC + registry
└── utils/                 ← generic helpers (string/dict/retry — NOT domain)

community/                 ← content (every domain plugin + lib)
├── validation/<id>/       ← validation plugins
├── evaluation/<id>/
├── reward/<id>/
├── reports/<id>/
├── presets/<id>/
├── libs/<id>/             ← shared domain code (manifest + Python pkg)
└── libs/<id>.zip          ← OR distributed as zip
```

**Why this matters.** When RyotenkAI gets used for a non-HelixQL
workload tomorrow, the platform shouldn't carry an unused
`src/utils/domains/helixql.py`. The split also makes future
PyPI-distributed third-party packs trivially possible — `community/`
is the marketplace surface, `src/` is the framework.

If you ever feel tempted to add `src/utils/domains/<thing>.py`,
**stop**: that's a layering violation. Find the right
`community/libs/<thing>/` instead.

---

## Plugin manifest (schema v5)

Every plugin has a `manifest.toml` next to `plugin.py`. Minimum form:

```toml
schema_version = 5

[plugin]
id = "my_plugin"                  # snake_case, matches the folder name
kind = "validation"               # validation | evaluation | reward | reports
name = "My Plugin"
version = "1.0.0"                 # PEP 440 — bump via `plugin sync --bump`
category = "basic"
stability = "stable"              # alpha | beta | stable | deprecated
description = "..."
author = "Name <email>"           # free-form; "Name <email>" recommended

[plugin.entry_point]
module = "plugin"                 # plugin.py (file) or plugin/ (package)
class = "MyPlugin"
```

**Optional sections:**

- `params_schema.<field>` / `thresholds_schema.<field>` — JSON Schema
  for the UI Configure modal. `ryotenkai plugin sync` infers these
  from the class's `_validate_contract` body.
- `suggested_params` / `suggested_thresholds` — initial values shown
  in the modal.
- `supported_strategies = ["grpo", "sapo"]` — **required** for
  `kind = "reward"`, **forbidden** for everything else.
- `[[required_env]]` — env vars the plugin needs at runtime.
- `[[lib_requirements]]` — `community_libs.*` deps with optional
  PEP 440 version constraints (see below).

**Hard rules:**

- `id` must match the folder name.
- `version` must be valid PEP 440 (`SpecifierSet` rejects garbage).
- `[plugin]` has `extra="forbid"` — unknown fields fail loading with
  a precise error.

---

## Lib manifest (schema v1)

Every `community/libs/<name>/` carries its own minimal manifest:

```toml
schema_version = 1

[lib]
id = "helixql"                    # matches folder/zip stem
version = "1.0.0"                 # PEP 440
description = "..."
author = "Name <email>"
```

Lib manifests evolve on their own `schema_version` axis (currently
`LATEST_LIB_SCHEMA_VERSION = 1`). Plugin manifest bumps don't affect
lib schema and vice versa.

**Required fields:** `id`, `version`. Everything else is optional.

---

## Lib distribution: folder OR zip

A lib lives at one of two paths — same precedence semantics as
plugin packs:

| Form | Use case |
|------|----------|
| `community/libs/<name>/`     | Dev-time iteration. Folder wins on collision. |
| `community/libs/<name>.zip`  | Distribution. Auto-extracted to `community/.cache/<hash>/<name>/` on first load. |

Build the archive with `ryotenkai plugin pack community/libs/<name>`
— filters `__pycache__`/build artifacts, validates the manifest,
drops `<name>.zip` next to the source folder.

A stale zip next to a live folder triggers a warning log
(`[COMMUNITY_LIBS] X shadows Y — folder wins`) — don't leave both
in the same tree long-term.

---

## `community_libs.*` import contract

The catalog registers each loaded lib in `sys.modules` as
`community_libs.<id>` **before** any plugin imports. Plugins consume
the lib like a normal pip package:

```python
# inside community/validation/my_plugin/plugin.py
from community_libs.helixql.compiler import get_compiler
from community_libs.helixql.extract import extract_query_text
```

**Lib `__init__.py` MUST use lazy re-exports.** Pytest's collection
phase imports `__init__.py` *before* any conftest fixture has fired;
eager `from community_libs.<lib>.<sub> import …` at the top of the
file would crash discovery. Use PEP 562 `__getattr__`:

```python
# community/libs/helixql/__init__.py
_LAZY_EXPORTS = {
    "get_compiler": ("community_libs.helixql.compiler", "get_compiler"),
    ...
}

def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"no such symbol: {name}")
    import importlib
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value  # cache for subsequent reads
    return value
```

---

## Plugin → lib version constraints

A plugin that imports from a lib declares the dependency in its
manifest as a top-level **array of tables**:

```toml
[[lib_requirements]]
name = "helixql"
version = ">=1.0.0,<2.0.0"        # PEP 440 specifier; omit for "any"
```

The loader cross-checks every requirement against the catalog of
loaded libs **before** registering the plugin:

- Missing lib → `ValueError: plugin 'X' requires lib 'Y' but no such
  lib is loaded`.
- Version mismatch → `ValueError: lib 'Y' is at version 1.0.0 but
  plugin requires '>=2.0'`.
- Empty version → presence check only.

Names must be unique within a single plugin's `[[lib_requirements]]`
(combine constraints with a comma instead of two separate blocks).

---

## REQUIRED_LIBS / REQUIRED_ENV cross-check

If your plugin class subclasses `BasePlugin` (every kind's ABC does),
opt into a load-time cross-check by declaring `REQUIRED_LIBS` and/or
`REQUIRED_ENV` as class-level tuples:

```python
from src.community.manifest import LibRequirement, RequiredEnvSpec

class MyPlugin(ValidationPlugin):
    REQUIRED_LIBS = (("helixql", ">=1.0.0,<2.0.0"),)   # (name, version) tuple
    # OR bare names:
    REQUIRED_LIBS = ("helixql",)
    # OR explicit:
    REQUIRED_LIBS = (LibRequirement(name="helixql", version=">=1.0.0"),)

    REQUIRED_ENV = (
        RequiredEnvSpec(name="MY_KEY", optional=False, secret=True, managed_by=""),
    )
```

The loader compares Python and TOML **set-keyed by name**, with
versions compared byte-for-byte. Mismatch raises `ValueError` with
a precise diff (`code='>=2.0' vs toml='>=1.0'`).

**Empty tuples skip the check** — that's the right default for plugins
where the manifest stays authoritative. Set them only when the class
should drive the contract.

Sync code → TOML with `ryotenkai plugin sync community/<kind>/<id>`.
Existing TOML versions are preserved when code is silent (bare
`("helixql",)` doesn't wipe a hand-typed `">=1.0.0"`).

---

## Runtime helpers — `_env` / `_secret`

`BasePlugin` exposes two accessors. **Use them, never `os.environ`
directly:**

```python
class MyPlugin(EvaluatorPlugin):
    def evaluate(self, samples):
        api_key = self._secret("EVAL_API_KEY")          # required, raises if missing
        timeout = int(self._env("EVAL_TIMEOUT", "30"))  # optional, fallback default
```

- `self._secret(name)` — returns from the resolved `_secrets` dict
  (populated by the registry's resolver). Raises `KeyError` with a
  clear hint if the key isn't declared in `[[required_env]]` with
  `secret=true, optional=false`. **Never falls back to `os.environ`**
  — that would mask manifest contract violations.
- `self._env(name, default=None)` — reads from the registry's
  injected env dict first, then `os.environ`. Empty strings count as
  unset so operators can blank a var with `MY_VAR=`.

The helpers give us a place to add validation, telemetry, and per-
test mocking later without touching every plugin folder.

---

## CLI surface — `ryotenkai plugin <verb>`

The unified surface lives at `src/cli/commands/plugin.py`. Every
author workflow goes through it:

| Verb | Purpose |
|------|---------|
| `ls --kind <kind>` | List installed plugins |
| `show <kind> <id>` | Full ui_manifest of one plugin |
| `scaffold <kind> <id>` | Bootstrap a new plugin folder |
| `sync <path> [--bump …]` | Re-render manifest from code (plugins, presets, AND libs) |
| `sync-envs <path>` | Narrow shortcut: only `[[required_env]]` (rarely needed) |
| `pack <path>` | Zip a plugin/preset/lib for distribution |
| `validate <path>` | Pure TOML+Pydantic validation, no Python import |
| `install <source>` | Copy/clone/unzip a plugin into `community/<kind>/` |
| `preflight --config <file>` | Pre-launch env + instance-shape gate |
| `stale --config <file>` | Find references to plugins missing from catalog |

**`sync` is THE universal sync** — it classifies the leaf path
(plugin / preset / lib by parent directory + manifest shape) and
dispatches. Do not add `sync-libs` / `sync-author` / `sync-deps` —
those would fragment the surface. Extend `sync_plugin_manifest`
itself.

---

## Anti-patterns

### ❌ Domain code in `src/`

```python
# src/utils/domains/helixql.py  ← VIOLATION
def extract_query_text(...): ...
```

Move to `community/libs/helixql/extract.py`. Plugins import via
`community_libs.helixql.extract`.

### ❌ Inline `from src.utils.domains.helixql import …`

```python
# community/validation/my_plugin/plugin.py
from src.utils.domains.helixql import extract_query_text  # ← VIOLATION
```

Use `from community_libs.helixql.extract import extract_query_text`.

### ❌ Per-plugin compiler re-implementation

```python
# Three plugins, each carrying:
class MyPlugin(ValidationPlugin):
    def _get_compiler(self):
        if self._compiler is None:
            self._compiler = HelixCompiler(timeout_seconds=...)
        return self._compiler
```

Extract once into `community/libs/<lib>/compiler.py` with a
`get_compiler(timeout_seconds=...)` factory keyed by config. Plugins
become one-liners.

### ❌ `[plugin].libs = ["helixql"]` (v4 shape)

```toml
[plugin]
libs = ["helixql"]   # ← removed in schema v5
```

Use top-level `[[lib_requirements]]` blocks (see manifest section).
Loader rejects v4 with `ValidationError: extra fields not permitted`.

### ❌ Backwards-compat shims for removed manifest fields

```python
@model_validator(mode="before")
def _migrate_v4_libs(cls, data):
    if "libs" in data:
        data["lib_requirements"] = [{"name": x} for x in data.pop("libs")]
    return data
```

Project rule: **no legacy support**. When a schema bumps, the old
shape is deleted cleanly. Authors migrate the manifest in the same
PR that bumps `schema_version`.

### ❌ `os.environ.get("FOO")` inside a plugin's `evaluate`/`validate`/`__call__`

Use `self._env("FOO")` so the registry can inject test-time
overrides. The static analysis lint will flag this in CI.

### ❌ Plugin classes declaring `plugin_id` / `title` / `order` ClassVars (reports)

```python
class MyReportPlugin(ReportPlugin):
    plugin_id = "my_block"     # ← duplicates manifest.plugin.id
    title = "My Block"         # ← duplicates manifest.plugin.name
    order = 50                 # ← removed (per-instance, injected by build_report_plugins)
```

`ReportPlugin` exposes them as manifest-backed `@property`s. Subclass
body should contain only `render()` and helpers.

### ❌ Lib `__init__.py` with eager submodule imports

```python
# community/libs/helixql/__init__.py
from community_libs.helixql.compiler import get_compiler   # ← breaks pytest collection
```

Use `__getattr__` lazy re-exports (see `community_libs.*` section).

### ❌ New `sync-<aspect>` CLI command

If you're tempted to add `ryotenkai plugin sync-author` or
`ryotenkai plugin sync-deps`, **extend the unified `sync` command
instead**. The classify-and-dispatch pattern in `sync_cmd` is the
correct extension point.

---

## Common LLM mistakes

1. **Adding `src/utils/domains/<thing>.py` because the platform layer
   "needs to know about" a DSL.** Wrong — the platform never knows
   about specific DSLs. Move to `community/libs/`.

2. **Updating `manifest.toml` by hand without `plugin sync`.** That
   skips the version bump and risks Python ↔ TOML drift. Always use
   the CLI for content changes.

3. **Catching `ImportError` when importing from `community_libs.*`.**
   Don't — declare `[[lib_requirements]]` instead. The loader checks
   presence and version before the plugin runs.

4. **Writing a v4-shaped manifest because that's what the LLM saw in
   an older example.** Check `LATEST_SCHEMA_VERSION` in
   `src/community/manifest.py` and use the current shape.

5. **Treating the `community/libs/<name>/__init__.py` as a regular
   `__init__.py`.** It MUST be lazy (PEP 562 `__getattr__`).

6. **Putting `description` text on the wrong side.** TOML descriptions
   are operator-facing (what does this plugin do?); Python docstrings
   are developer-facing (how does it work?). Sync writes TOML
   `description` from the class docstring's first line if absent.

7. **Forgetting `author` on new manifests.** Optional but recommended
   — surfaced through `GET /plugins/{kind}` for catalog attribution.

8. **Distributing a lib without a `manifest.toml`.** The loader
   rejects it (`FileNotFoundError: manifest.toml not found …`).
   Every lib needs at least `id` and `version`.

9. **Hardcoding a lib path like `community/libs/helixql/compiler.py`
   in tests.** Use `from community_libs.helixql.compiler import …` —
   the test conftest preloads the namespace.

10. **Asking "where should I put this cross-plugin helper?" and
    landing on `src/utils/`.** The answer is `community/libs/<name>/`
    — unless the helper is truly generic platform code (string ops,
    retry loops, dict merging), in which case `src/utils/` is right.

---

## Tests

### File layout

- Plugin tests: `community/<kind>/<id>/test_*.py` (per-plugin
  `conftest.py` rebinds `plugin.py` as a unique module so identically-
  named plugin modules across plugins coexist).
- Lib tests: `community/libs/<id>/tests/test_*.py` — **no
  `__init__.py`** in `tests/` (pytest `--import-mode=importlib`
  relies on its absence).
- Catalog/loader/registry tests: `src/tests/unit/community/`.

### Required tests when adding a new plugin

- Smoke: `_validate_contract` accepts valid inputs, rejects invalid.
- End-to-end: `validate(...)` / `evaluate(...)` / `__call__(...)`
  produces the expected result shape for a representative sample.
- Registration: the plugin appears in `<kind>_registry.list_ids()`
  after `catalog.reload()`.

### Required tests when adding a new lib

- Manifest round-trip: schema validates.
- Public API smoke: `from community_libs.<id> import <symbol>`
  works after `preload_community_libs(libs_root_for(COMMUNITY_ROOT))`.
- Per-submodule unit tests for behaviour.

### Pre-commit checklist

- [ ] `pytest src/tests/unit/community/ community/` green.
- [ ] `ryotenkai plugin validate community/<kind>/<id>` passes
  without warnings.
- [ ] No new `from src.utils.domains.*` imports.
- [ ] No new `[plugin].libs = [...]` (use `[[lib_requirements]]`).
- [ ] `author` field set on new manifests.
- [ ] If `REQUIRED_LIBS` / `REQUIRED_ENV` changed: `ryotenkai plugin
  sync community/<kind>/<id>` run and committed.
