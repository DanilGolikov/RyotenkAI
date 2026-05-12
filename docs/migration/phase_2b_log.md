# Phase 2B Log — `patch.dict("os.environ")` → `monkeypatch.setenv` codemod

> Phase 2B of the mock-elimination plan
> ([docs/plans/mock-elimination-architecture.md](../plans/mock-elimination-architecture.md)):
> mechanically convert `patch.dict("os.environ", {...})` to pytest's
> native `monkeypatch.setenv(...)` fixture.

## TL;DR

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `patch.dict("os.environ", ...)` hits in `tests/` | 3 | 1 | **−2** |
| Files modified by codemod (kept) | — | 2 | +2 |
| Files reverted by auto-verifier | — | 0 | (lane stayed green) |
| Files annotated with TODO marker (skipped) | — | 1 | +1 |
| Codemod meta-tests (Phase 2B) | — | 9 passed | new |
| Codemod meta-tests (total Phase 2A+2B) | 18 | 27 | +9 |
| Lane status | 6823 passed | 6825 passed | unchanged (no regressions) |

## Key finding: scope was 3 hits, not the inventory's "68"

The mock inventory at
[docs/migration/mock_inventory.md](mock_inventory.md) reports
`patch_dict_context_manager: 68`, but that bucket conflates **all**
`patch.dict(...)` context-manager usages, not just `os.environ`.  An
AST-level disambiguation shows:

| Target of `patch.dict(...)` | Count |
|---|---:|
| `patch.dict(sys.modules, ...)` (attr) | 62 |
| `patch.dict("sys.modules", ...)` (str) | 3 |
| `patch.dict("os.environ", ...)` (str) | **3** |

The 65 `sys.modules` patches are an entirely different pattern (they
mask a real Python import at test time, e.g. `with
patch.dict(sys.modules, {"mlflow": mock_mlflow}):`) and are out of scope
for Phase 2B per the plan, which targets only `os.environ`.

In particular:

* `tests/unit/pod/test_phase_executor.py` (the "33-hit" file the prompt
  flagged as the dominant target) uses `patch.dict(sys.modules, ...)`
  exclusively — zero `os.environ` patches.
* The actual `os.environ` patch.dict hits are:
  | File | Line | Form | Outcome |
  |---|---:|---|---|
  | `tests/unit/pod/test_dataset_loaders.py` | 289 | `with patch.dict("os.environ", {"K": "v"}):` | CONVERTED |
  | `tests/unit/control/pipeline/test_run_inspector.py` | 733 | compound `with`, item with `clear=False` | CONVERTED |
  | `tests/unit/control/pipeline/test_run_inspector.py` | 761 | `clear=True` + variable second arg | SKIPPED (TODO annotated) |

This is documented up front so a future operator running the codemod
on a re-indexed inventory doesn't get confused by the row.

## Codemod design

### File layout

```
scripts/
├── __init__.py
└── codemods/
    ├── __init__.py
    ├── magicmock_to_simplenamespace.py     # Phase 2A
    ├── patchdict_environ_to_monkeypatch.py # Phase 2B (this)
    ├── apply_with_revert.py                # safety wrapper (now codemod-agnostic)
    ├── test_cases/
    │   └── patchdict_environ_<scenario>/   # 9 before/after pairs
    └── tests/
        └── test_patchdict_environ_to_monkeypatch.py
```

### Detection rules — what gets CONVERTED

A `patch.dict(...)` call is convertible iff **all** of the following hold:

1. The call shape is `patch.dict(...)` (i.e. `patch` name in scope,
   `.dict` attribute) — matches the Phase 2A precedent of recognising
   the canonical `unittest.mock.patch` API.
2. The first positional argument is `"os.environ"` (string literal) or
   `os.environ` (attribute) — anything else (e.g. `sys.modules`,
   `"some.module.CONST"`) is out of scope.
3. The second positional argument is a **plain `Dict` literal** with
   string-literal keys.  A variable like `patch.dict("os.environ",
   env_without_hf)` cannot be converted because we don't know its keys
   at codemod time.
4. No `clear=True` keyword (see Pattern 3 in the prompt).
5. If used as a `with`-item, no `as <name>:` binding (see Pattern 5).

When all conditions hold, the codemod emits one
`monkeypatch.setenv(<key>, <value>)` statement per dict entry.

### Patterns supported

| # | Pattern | Conversion |
|---|---|---|
| 1 | `with patch.dict("os.environ", {"K": "v"}): body` | dissolve `with`, prepend `setenv`, dedent body |
| 2 | `@patch.dict("os.environ", {"K": "v"})` on a function | drop decorator, prepend `setenv` to body |
| 4 | `with patch.dict("os.environ", {"A": "1"}), patch.dict("os.environ", {"B": "2"}):` | both items dissolved, `body` dedented |
| 4-mixed | `with patch("X"), patch.dict("os.environ", {...}):` | drop only the dict item, prepend `setenv` |

For pattern 1/4, when the `with`'s `items` list becomes empty after
removal, the entire `with`-block is flattened into the parent
(`FlattenSentinel`) and the original `with`'s `leading_lines` are
preserved on the first emitted `setenv` line.

### Patterns SKIPPED (annotated only)

| # | Pattern | Why skipped |
|---|---|---|
| 3 | `patch.dict(..., clear=True)` | needs context-specific cleanup logic; not mechanical |
| 5 | `with patch.dict(...) as env_dict:` | the bound name is referenced inside the body; `monkeypatch.setenv` is statement-based, doesn't naturally bind a name |
| — | second arg is not a `Dict` literal (e.g. a variable) | keys are unknown at codemod time |

Skipped occurrences receive a comment `# TODO(codemod): manual review
needed for clear=True / with-as binding` on the preceding line.  The
annotation is **idempotent**: re-running the codemod does not add a
second copy.

### Imports

If `patch` is no longer referenced anywhere in the module after
conversion (free-occurrence scan, excluding the imports themselves),
the codemod drops `patch` from `from unittest.mock import patch, ...`.
If `patch` is the only name in the import, the whole `import` line is
removed.

### Function signature

If a converted function does not already have a `monkeypatch`
parameter (either in `params`, `posonly_params`, `kwonly_params`,
`star_arg`, or `star_kwarg`), the codemod appends
`monkeypatch` to the positional parameter list, after the existing
positionals but before `*args` / `**kwargs`.  Annotations on prior
parameters are preserved.

### Trailing-comma cleanup

When the codemod drops items from a compound `with (A, B, C):`,
libcst leaves the prior item's `Comma` referencing the dropped
item's layout, which generates a blank line with trailing
whitespace (ruff `W293`).  The codemod normalises the last
surviving item's `comma` to `MaybeSentinel.DEFAULT`, eliminating the
stray whitespace.

### Idempotency

* Files with no convertible / skip-candidate `patch.dict(os.environ,
  ...)` calls produce zero output (no-op).
* Files already converted produce zero output on a second pass
  (planner returns no plan).
* Files with skipped-but-annotated calls retain the same annotation
  count after re-runs — the annotation helper guards against
  duplicate TODO insertion.

Confirmed empirically by running the codemod three times on each
test case fixture; output stabilised after iteration 1.

## Test cases (9)

| Scenario | Pattern | Verifies |
|---|---|---|
| `patchdict_environ_simple_context_manager` | 1 | `with` → setenv + dedented body |
| `patchdict_environ_decorator_style` | 2 | `@patch.dict` → setenv at body top, decorator removed |
| `patchdict_environ_multiple_envs_in_one_dict` | 1 (multi-key) | one setenv per dict entry, ordered |
| `patchdict_environ_nested_patches` | 4 | compound `with A, B:` both dissolved |
| `patchdict_environ_skip_clear_true` | 3 | `clear=True` → TODO annotation, no rewrite |
| `patchdict_environ_skip_with_as_binding` | 5 | `with patch.dict(...) as env:` → TODO annotation |
| `patchdict_environ_existing_monkeypatch_fixture` | — | function already has `monkeypatch`, no double-add |
| `patchdict_environ_removes_unused_patch_import` | — | drop `patch` from import after conversion |
| `patchdict_environ_keeps_patch_import_if_other_patches_remain` | — | keep `patch` import when other uses survive |

Each scenario has paired `before.py` / `after.py` fixtures under
`scripts/codemods/test_cases/`, asserted via libcst's `CodemodTest`.

## Application results

### Single-file safety run (test_dataset_loaders.py)

```
[KEEP]   tests/unit/pod/test_dataset_loaders.py: 1 conversions
Summary: 1 kept (1 conversions), 0 reverted, 0 untouched.
```

### Single-file safety run (test_run_inspector.py)

```
[KEEP]   tests/unit/control/pipeline/test_run_inspector.py: 1 conversions
Summary: 1 kept (1 conversions), 0 reverted, 0 untouched.
```

### Tree-wide pass (tests/)

```
Summary: 0 kept (0 conversions), 0 reverted, 391 untouched.
```

391 test files scanned, 0 additional conversions — confirms only the
two known sites were convertible, no spurious rewrites elsewhere.

## `apply_with_revert.py` generalisation

The Phase 2A safety wrapper was hard-wired to the MagicMock codemod
via a top-level `from scripts.codemods.magicmock_to_simplenamespace
import ...`.  Phase 2B extends it with:

* `--codemod <module_path>` flag — dotted Python module to import;
  defaults to `scripts.codemods.magicmock_to_simplenamespace` for
  backwards compatibility with Phase 2A invocations.
* `_resolve_codemod_class(module_path)` helper — imports the module
  and locates the single `libcst.codemod.Codemod` subclass it defines;
  raises if the module defines zero or more than one.
* `_run_codemod` now takes the resolved class and uses
  `getattr(codemod, "changed_count", 1)` for the conversion count,
  so future codemods only need to expose `changed_count` to
  participate in the per-file counter (or skip it entirely; the
  wrapper falls back to "1" so the lane sees a non-zero conversion).

Both the legacy positional form (`python -m scripts.codemods.apply_with_revert tests/unit/`)
and the new flag form
(`python -m scripts.codemods.apply_with_revert --codemod scripts.codemods.patchdict_environ_to_monkeypatch --paths tests/ --apply`)
work.

## Lane status

```
6825 passed, 291 skipped, 88 xfailed, 7 xpassed, 701 warnings in 371.16s
```

0 failures.  6825 passed — slightly above the Phase 2A baseline of
6823 (suggesting more tests have been added since), but importantly:
**no test was broken by Phase 2B**.

## Surprises

1. **Inventory scope mismatch.** The `patch_dict_context_manager: 68`
   inventory row covers all `patch.dict` call sites, not just
   `os.environ`.  Phase 2B target was 3, not 68.
2. **No big-win single file.** The prompt highlighted
   `test_phase_executor.py` as a "33-hit dominant single file" — but
   those 33 hits are all `patch.dict(sys.modules, …)`, which is a
   different conversion entirely (potentially Phase 2F).
3. **Trailing-whitespace lint regression on compound `with`.**
   libcst's `Comma.whitespace_after` on the surviving item retained
   the dropped item's layout, generating a stray `    \n` line with
   trailing whitespace.  Fixed by normalising the last item's comma
   to `MaybeSentinel.DEFAULT` after item removal.  Caught by ruff
   `W293`, not by pytest.

## Open issues

* **1 hit needs manual review** —
  `tests/unit/control/pipeline/test_run_inspector.py:761`:
  ```python
  env_without_hf = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
  with patch("...load_pipeline_config", return_value=mock_cfg), \
       patch.dict("os.environ", env_without_hf, clear=True):
      ...
  ```
  The codemod annotates this with a TODO comment because:
  * `clear=True` requires deleting every existing env key, which the
    codemod cannot enumerate.
  * The second arg `env_without_hf` is a comprehension result, not a
    `Dict` literal — keys aren't statically known.

  Manual conversion: use `monkeypatch.delenv("HF_TOKEN",
  raising=False)` (the only key the test cares about removing); the
  monkeypatch fixture's automatic teardown handles per-test scoping.
  Skipped from this batch to keep Phase 2B purely mechanical.

* **Possible future Phase 2F**: 65
  `patch.dict(sys.modules, ...)` hits are a separate pattern that
  could be converted to e.g. `monkeypatch.setitem(sys.modules, ...)`
  or to dependency injection.  Not in the current Phase 2 plan.

## Verification commands

```bash
# Meta-tests (9 cases)
.venv/bin/python -m pytest scripts/codemods/tests/test_patchdict_environ_to_monkeypatch.py -v

# Dry-run on the affected files
.venv/bin/python -m scripts.codemods.patchdict_environ_to_monkeypatch \
  tests/unit/pod/test_dataset_loaders.py \
  tests/unit/control/pipeline/test_run_inspector.py \
  --dry-run

# Apply via safety wrapper (per-file revert on test failure)
.venv/bin/python -m scripts.codemods.apply_with_revert \
  --codemod scripts.codemods.patchdict_environ_to_monkeypatch \
  --paths tests/ \
  --apply

# Confirm lane is GREEN
.venv/bin/python -m pytest -c tests/pytest.ini tests/
```
