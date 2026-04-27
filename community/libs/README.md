# `community/libs/` — shared domain libraries for plugins

Drop a Python package here when **two or more community plugins**
across **two or more kinds** need the same domain code (e.g. a query
compiler wrapper, a parser, a semantic-similarity scorer). The
catalog automatically registers each subpackage as
`community_libs.<id>` in `sys.modules` before any plugin loads, so
plugins can import without thinking about it:

```python
# inside community/validation/my_plugin/plugin.py
from community_libs.helixql.compiler import get_compiler
from community_libs.helixql.extract import extract_schema_and_query
```

## Layout

```
community/libs/
├── README.md
└── <lib_name>/                 # OR <lib_name>.zip — both work
    ├── manifest.toml           # required — declares id, version, author, …
    ├── __init__.py             # required — every lib is a Python package
    ├── <module>.py             # implementation modules
    ├── ...
    └── tests/                  # unit tests for the lib (no __init__.py)
        ├── conftest.py         # preloads the namespace at collection time
        └── test_*.py
```

## Manifest

Every lib needs a `manifest.toml` with at least `id` (matching the
folder/zip stem) and a PEP 440 `version`:

```toml
schema_version = 1

[lib]
id = "helixql"
version = "1.0.0"
description = "HelixQL compiler wrapper, …"
author = "Daniil Golikoff <daniil.golikoff@gmail.com>"
```

The loader fails the load with a precise per-failure message when:
- `manifest.toml` is missing,
- `[lib].id` doesn't match the folder/zip stem,
- `version` isn't valid PEP 440,
- the package has no `__init__.py`.

## Distribution: folder OR zip

Same precedence rule as plugins:

| Form | When to use |
|---|---|
| `community/libs/<name>/`     | dev-time iteration. Folder wins on collision. |
| `community/libs/<name>.zip`  | distribution. Auto-extracted to `community/.cache/<hash>/<name>/` on first load and re-used until the zip's mtime changes. |

Build the zip with `ryotenkai community pack community/libs/<name>`
(filters out `__pycache__`, validates the manifest, drops the
archive next to the source).

## What belongs here

- **Cross-kind helpers** — code used by ≥2 plugins in different
  kinds. Single-plugin helpers stay in the plugin's own folder.
- **Domain knowledge** — anything that names domain concepts (HelixQL
  queries, Cypher patterns, SQL dialects, …). The whole point of
  `community/libs/` is to keep `src/` free of domain coupling.
- **A stable public surface** — what `__init__.py` re-exports is the
  contract; internal modules can change. Bump the lib's docstring
  with the change rationale when you alter the public API.

## What does **not** belong here

- ❌ **Generic platform utilities** — string manipulation, retry
  loops, dict-merging helpers. Those go in `src/utils/`.
- ❌ **Framework code** — manifest schema, registry base classes,
  loader internals. That's `src/community/`.
- ❌ **Single-plugin helpers** — code only one plugin imports.
  Promote it here once a second plugin needs it; not before.

## Authoring rules

1. **Lazy `__init__.py`.** Use PEP 562 `__getattr__` to defer
   submodule imports. Pytest's collection phase imports `__init__.py`
   *before* any conftest fixture runs, so eager
   `from community_libs.<lib>.foo import …` at module level breaks
   test discovery. See `community/libs/helixql/__init__.py` for the
   pattern.
2. **No state at the package level.** Each submodule must be
   independently importable. Inter-submodule imports are fine, but
   never lean on side-effects from `__init__.py`.
3. **Tests live in `<lib>/tests/`.** Pytest collects them
   automatically (it walks `community/` per `pytest.ini`). Do
   **not** add `__init__.py` to the `tests/` dir — `--import-mode=importlib`
   relies on its absence to avoid path-shadow issues.
4. **`tests/conftest.py` preloads the namespace.** Test modules do
   `from community_libs.<lib> import …` at import time, which
   collection phase resolves before any session fixture runs. Have
   the conftest call `preload_community_libs(libs_root_for(COMMUNITY_ROOT))`
   at module top-level (not in a fixture).

## Declaring lib usage on the consumer side

Plugins that import from `community_libs.<lib>` MUST list each
dependency as a top-level `[[lib_requirements]]` block in their
`manifest.toml`:

```toml
[[lib_requirements]]
name = "helixql"
version = ">=1.0.0,<2.0.0"   # PEP 440 specifier; omit for "any version"
```

The loader pre-validates each requirement against the catalog of
loaded libs:

- **Missing lib** — clear "requires lib X but no such lib is
  loaded" error.
- **Version mismatch** — "lib X is at version 1.0.0 but plugin
  requires `>=2.0`". Plugin won't register.
- **Empty `version`** — only presence is checked.

For plugins that subclass `BasePlugin`, mirror the declaration on
the class to opt into a strict cross-check. Three shapes work,
pick whichever reads best:

```python
class MyPlugin(ValidationPlugin):
    REQUIRED_LIBS = ("helixql",)                              # name only
    REQUIRED_LIBS = (("helixql", ">=1.0.0,<2.0.0"),)          # (name, version)

    from src.community.manifest import LibRequirement
    REQUIRED_LIBS = (LibRequirement(name="helixql", version=">=1.0.0"),)
```

Cross-check is set-keyed by name; when both sides supply a
`version`, they must match byte-for-byte. Empty `REQUIRED_LIBS = ()`
skips the check.

After editing `REQUIRED_LIBS`, run `ryotenkai community sync
community/<kind>/<plugin>` to re-render `[[lib_requirements]]` from
the ClassVar (along with the rest of the manifest sync). Pair with
`--dry-run` to preview the diff first.

## Contract: how the loader sees libs

- `community/libs/` is **not** a plugin kind — `PLUGIN_KIND_DIRS` in
  `src/community/constants.py` whitelists only
  `validation / evaluation / reward / reports`. Anything under `libs/`
  is invisible to `load_all_plugins`. (Tests in
  `src/tests/unit/community/test_libs_isolation.py` pin this
  invariant.)
- Loading runs once per `catalog.ensure_loaded()`, **before** plugins
  load. Each lib's `manifest.toml` is parsed, validated against
  :class:`LibManifest`, and the package is registered under
  `community_libs.<id>` in `sys.modules`. Same root → idempotent.
  Different root → namespace replaced, cached subpackages purged.
- Empty/absent `community/libs/` → no namespace registered. Existing
  plugins keep working; they just don't have any shared lib to
  import.
- Folder vs zip: folder beats zip on collision (a warning is logged).
  Zip libs are extracted to `community/.cache/<hash>/<id>/` on first
  load and re-used until the zip's mtime changes.
- Fingerprint covers each lib's `manifest.toml` + every direct
  `*.py` at the lib root, plus the `*.zip` archive itself. Edits to
  deeper files (e.g. `lib/sub/foo.py`) require a manual backend
  restart, same rule as `src/`.
- Failures (missing manifest, mismatched id, bad version, no
  `__init__.py`) are captured as :class:`LibLoadFailure` and
  surfaced via `catalog.lib_failures()` so the UI catalogue can
  render an actionable error banner.

## Migration from `src/utils/domains/`

The `src/utils/domains/` directory was the previous home for
HelixQL helpers. It violated the "no domain code in `src/`" rule and
has been removed. If you find lingering imports of
`src.utils.domains.*` anywhere, replace with
`community_libs.<lib>.<module>` and verify the lib re-exports the
symbol you need.
