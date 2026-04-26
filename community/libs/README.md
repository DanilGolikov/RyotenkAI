# `community/libs/` — shared domain libraries for plugins

Drop a Python package here when **two or more community plugins**
across **two or more kinds** need the same domain code (e.g. a query
compiler wrapper, a parser, a semantic-similarity scorer). The
catalog automatically registers each subpackage as
`community_libs.<name>` in `sys.modules` before any plugin loads, so
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
└── <lib_name>/
    ├── __init__.py        # public re-exports (lazy, see below)
    ├── <module>.py        # implementation modules
    ├── ...
    └── tests/             # unit tests for the lib (no __init__.py)
        ├── conftest.py    # preloads the namespace at collection time
        └── test_*.py
```

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

## Contract: how the loader sees libs

- `community/libs/` is **not** a plugin kind — `PLUGIN_KIND_DIRS` in
  `src/community/constants.py` whitelists only
  `validation / evaluation / reward / reports`. Anything under `libs/`
  is invisible to `load_all_plugins`. (Tests in
  `src/tests/unit/community/test_libs_isolation.py` pin this
  invariant.)
- Preload runs once per `catalog.ensure_loaded()`, before plugins are
  loaded. Same root → no-op. Different root → namespace replaced and
  cached subpackages purged from `sys.modules`.
- Empty/absent `community/libs/` → no namespace registered. Existing
  plugins keep working; they just don't have any shared lib to
  import.
- Fingerprint covers `__init__.py` + every direct `*.py` of every
  lib. Edits to deeper files (e.g. `lib/sub/foo.py`) require a
  manual backend restart, same rule as `src/`.

## Migration from `src/utils/domains/`

The `src/utils/domains/` directory was the previous home for
HelixQL helpers. It violated the "no domain code in `src/`" rule and
has been removed. If you find lingering imports of
`src.utils.domains.*` anywhere, replace with
`community_libs.<lib>.<module>` and verify the lib re-exports the
symbol you need.
