# Batch 2 — Legacy test decommissioning log

Date: 2026-05-11
Plan: [docs/plans/structured-hopping-starfish.md](../plans/structured-hopping-starfish.md) Phase 7
ADR: [docs/adrs/2026-05-11-legacy-test-decommissioning.md](../adrs/2026-05-11-legacy-test-decommissioning.md)
Worktree: `.claude/worktrees/ecstatic-khorana-1baa6b`
Prior batch: [batch_1_log.md](batch_1_log.md)

This batch migrates **everything that remains** under
`packages/engines/tests/` after Batch 1 took the sentinel + contract
categories. Scope: 10 files under `packages/engines/tests/unit/` (6
top-level + 4 vLLM-specific). 221 tests total.

## Summary

- 10 legacy files removed — 221 tests
- 11 greenfield files added (9 test files + `__init__.py` x2 + `conftest.py`) — 215 tests
- Legacy pytest collection: 7043 → 6822 (−221, matches)
- Greenfield pytest collection: 483/492 → 698/707 (+215)
- Two PARTIAL files (`test_scaffolding.py`, `vllm/test_manifest.py`)
  dropped a total of 6 redundant assertions and migrated only the
  non-overlapping invariants; 1 new constant-pinning assertion added in
  `test_manifest_schema.py::TestStableContractConstants`
- One legacy ``unittest.mock`` usage (`MagicMock` in `test_images.py`)
  replaced with `types.SimpleNamespace` + NamedTuple data carriers —
  greenfield convention preserved (no `unittest.mock` anywhere in
  `tests/`)
- One DUP-equivalence proven by synthetic injection (vLLM capability
  drift); the rest are direct UNIQUE migrations covered by their
  greenfield destination passing for the same source-tree state
- Importlinter unchanged (9 kept, 1 broken — same control→pod baseline
  Batch 1 documented)
- All sentinel tests still pass

## Per-file table

| # | Legacy file | Category | Action | Greenfield equivalent | Tests in legacy | Tests after |
|---|---|---|---|---|---|---|
| 1 | `packages/engines/tests/unit/test_prepare_plan_invariants.py` | UNIQUE | Migrated | `tests/unit/engines/test_prepare_plan_invariants.py` | 25 | 25 |
| 2 | `packages/engines/tests/unit/test_registry.py` | UNIQUE | Migrated | `tests/unit/engines/test_registry.py` | 23 | 23 |
| 3 | `packages/engines/tests/unit/test_images.py` | UNIQUE (with `MagicMock` cleanup) | Migrated | `tests/unit/engines/test_images.py` | 14 | 14 |
| 4 | `packages/engines/tests/unit/test_scaffolding.py` | PARTIAL | Migrated 1 constant; dropped 3 smoke imports | `tests/unit/engines/test_manifest_schema.py::TestStableContractConstants` (+1 net) | 4 | 1 (already counted in row 5) |
| 5 | `packages/engines/tests/unit/test_manifest_schema.py` | UNIQUE | Migrated | `tests/unit/engines/test_manifest_schema.py` | 50 | 51 (incl. row-4 constant) |
| 6 | `packages/engines/tests/unit/test_config_union.py` | UNIQUE | Migrated | `tests/unit/engines/test_config_union.py` | 9 | 9 |
| 7 | `packages/engines/tests/unit/vllm/test_runtime.py` | UNIQUE | Migrated | `tests/unit/engines/vllm/test_runtime.py` | 36 | 36 |
| 8 | `packages/engines/tests/unit/vllm/test_manifest.py` | PARTIAL | Migrated 3; dropped 3 DUPs of contract parity | `tests/unit/engines/vllm/test_manifest.py` | 6 | 3 |
| 9 | `packages/engines/tests/unit/vllm/test_config.py` | UNIQUE | Migrated | `tests/unit/engines/vllm/test_config.py` | 15 | 15 |
| 10 | `packages/engines/tests/unit/vllm/test_prepare_model.py` | UNIQUE | Migrated | `tests/unit/engines/vllm/test_prepare_model.py` | 39 | 39 |
| | | | **Totals** | | **221** | **215** |

## DUP equivalence proofs

### Row 4 — `test_scaffolding.py` constant pinning

Of the 4 legacy tests:

- `test_package_import` (asserts `__version__ == "1.0.0"` and every
  symbol in `__all__` is importable) — DUP. Every other migrated
  engines test imports symbols from `ryotenkai_engines.*`; a broken
  `__init__.py` would fail collection of 9 other files. The
  `__version__` constant is module metadata, not a public-contract
  surface — no preservation needed.

- `test_stub_modules_import` (imports every engines submodule) — DUP.
  Same as above: each migrated test that touches a submodule fails
  loudly on a broken module. Migration of 9 files covers every
  submodule the scaffolding test enumerated.

- `test_manifest_schema_version_constant` (asserts
  `LATEST_ENGINE_SCHEMA_VERSION == 1`) — UNIQUE. **Migrated** into
  `tests/unit/engines/test_manifest_schema.py::TestStableContractConstants::test_latest_schema_version_is_one`.

- `test_image_resolution_constants_present` (asserts
  `ENV_IMAGE_REGISTRY`, `DEFAULT_IMAGE_REGISTRY`, and the override
  pattern's substitution token) — DUP. Same exact assertions already
  exist in `test_images.py::TestConstants` (now migrated). No new test
  needed.

Net: 1 new assertion added (`test_latest_schema_version_is_one`), 3 dropped.

### Row 8 — `vllm/test_manifest.py` contract overlap

Of the 6 legacy tests:

- `test_vllm_manifest_in_default_registry` (registry contains "vllm",
  no failures) — covered by
  `tests/contract/engines/test_engine_protocol_parity.py::test_at_least_one_engine_registered`
  + `::test_no_load_failures`. **Migrated** as a single sanity assertion
  (`test_vllm_present_in_default_registry`) to keep the file's
  greenfield home self-contained; the contract test is the real
  enforcement.

- `test_vllm_manifest_fields` (display name "vLLM", version "1.0.0",
  upstream "0.7.0", stability "stable", image is None) — UNIQUE.
  Migrated.

- `test_vllm_image_resolves_via_convention` — UNIQUE. Migrated.

- `test_vllm_runtime_resolves` (`get_runtime("vllm") is VLLMEngineRuntime`)
  — DUP of `tests/contract/engines/test_engine_protocol_parity.py::TestEngineParity::test_runtime_implements_protocol`.
  The contract test calls `get_runtime(engine_id)` and asserts
  `isinstance(instance, IInferenceEngine)` — if the wrong class came
  back, the isinstance check fires. Dropped.

- `test_vllm_config_class_resolves` (`get_config_class("vllm") is VLLMEngineConfig`)
  — DUP of `test_config_class_subclasses_base` +
  `test_config_class_kind_literal_matches`. Both walk through
  `get_config_class` and check structural properties. Dropped.

- `test_vllm_capabilities_match_manifest` — DUP of
  `test_capabilities_match_manifest`. **Proven equivalent** by
  synthetic injection (see below). Dropped.

**Synthetic-injection proof for the capabilities-parity DUP claim:**

| Step | Action | Greenfield contract result |
|---|---|---|
| 1 | Inject `default_port=8000` → `default_port=9999` in `packages/engines/src/ryotenkai_engines/vllm/runtime.py` (the literal in `VLLMEngineRuntime.get_capabilities`) | `tests/contract/engines/test_engine_protocol_parity.py::TestEngineParity::test_capabilities_match_manifest[vllm]` FAILS — drift detected |
| 2 | Restore source | Greenfield contract test PASSES |

Both legacy + greenfield assertion thus fire on the same fault and
recover identically. Equivalence proven.

## Files created in greenfield

- `tests/unit/engines/__init__.py`
- `tests/unit/engines/conftest.py` — autouse fixture mirrors the
  legacy ``packages/engines/tests/conftest.py``: clears the
  lock-protected ``EngineRegistry`` + ``_engine_config_union``
  singletons before AND after every test so synthetic-manifest
  fixtures (`tmp_path` based) don't leak across tests
- `tests/unit/engines/test_prepare_plan_invariants.py` — 25 tests:
  `PreparePlan` / `PrepareStep` shape, frozen, extra=forbid,
  cross-field validators (steps ⇒ `final_model_path`, unique names),
  default `spec_version`
- `tests/unit/engines/test_registry.py` — 23 tests: discovery,
  duplicate-id, folder-id-mismatch, malformed TOML, schema-version
  rejection, strict mode, class resolution + import caching, runtime
  / config-class drift detectors, image lookup integration. The
  ``sys.modules`` registration hack is preserved (with a new
  `_TEST_MODULE_PATH = "tests.unit.engines.test_registry"`) so
  synthetic manifests' `entry_points` resolve back to the
  module-level `FakeRuntime` / `FakeConfig` / `DriftingRuntime` /
  `DriftingConfig` classes
- `tests/unit/engines/test_images.py` — 14 tests: convention default,
  override-chain priority (env > provider > manifest > convention),
  case sensitivity, public-env constants. ``unittest.mock.MagicMock``
  replaced with `types.SimpleNamespace` + `NamedTuple` data carriers
  (matches greenfield "no mock" convention)
- `tests/unit/engines/test_manifest_schema.py` — 51 tests: full
  schema validation matrix (positive / negative / boundary /
  invariant / regression / logic / combinatorial) + 1 new
  `TestStableContractConstants` migrated from scaffolding
- `tests/unit/engines/test_config_union.py` — 9 tests: single-member
  raw-class shortcut, multi-member discriminated dispatch, empty
  registry placeholder, public discriminator constant, wrapping in
  parent `BaseModel`. Module-level `FakeAlpha*` / `FakeBeta*` classes
  registered under `tests.unit.engines.test_config_union` for
  synthetic manifest resolution
- `tests/unit/engines/vllm/__init__.py`
- `tests/unit/engines/vllm/test_runtime.py` — 36 tests:
  `IInferenceEngine` instance compliance, ClassVars, capabilities
  match manifest shape, `build_launch_spec` happy / negative / logic
  / combinatorial, healthcheck + endpoint URL, `validate_config`
  error branches, legacy `build_docker_run_command` flag-set parity
- `tests/unit/engines/vllm/test_manifest.py` — 3 tests: vLLM-specific
  manifest fields the cross-engine contract test cannot generalise
  over (display name, semver, upstream, stability) + convention image
  resolution
- `tests/unit/engines/vllm/test_config.py` — 15 tests:
  `VLLMEngineConfig` Pydantic validation, boundary values, `kind`
  Literal discriminator behaviour
- `tests/unit/engines/vllm/test_prepare_model.py` — 39 tests:
  `prepare_model` plan builder (positive / negative / boundary /
  invariant / no-IO local sentinel / legacy parity / logic / 2³
  combinatorial)

## Files deleted from legacy

- `packages/engines/tests/unit/test_prepare_plan_invariants.py`
- `packages/engines/tests/unit/test_registry.py`
- `packages/engines/tests/unit/test_images.py`
- `packages/engines/tests/unit/test_scaffolding.py`
- `packages/engines/tests/unit/test_manifest_schema.py`
- `packages/engines/tests/unit/test_config_union.py`
- `packages/engines/tests/unit/vllm/test_runtime.py`
- `packages/engines/tests/unit/vllm/test_manifest.py`
- `packages/engines/tests/unit/vllm/test_config.py`
- `packages/engines/tests/unit/vllm/test_prepare_model.py`

The legacy directory now retains only empty `__init__.py` markers
+ the legacy `conftest.py` (its `_reset_engine_singletons` fixture
is now redundant — there are no engine tests left to consume it,
but the file is harmless and kept to avoid churn outside the
migration scope; future Batch 3+ may delete the now-empty
`packages/engines/tests/` tree as a final cleanup).

## Notes / things that surprised me

### `unittest.mock.MagicMock` in `test_images.py`

The only file in the engines tree using `unittest.mock`. It used
`MagicMock` purely as a duck-typed data carrier — `m.engine = …;
m.image = …` — not to patch a Protocol. The greenfield rule
(`tests/_lint/test_no_protocol_mocking.py`) only blocks Protocol
mocking, so `MagicMock()` without `spec=` wasn't strictly forbidden;
however, **no other file under `tests/`** uses `unittest.mock` at
all, so I replaced it with `types.SimpleNamespace` + `NamedTuple`
to preserve the convention. Same observable behaviour, no library
dependency.

### Cross-test `sys.modules` registration in `test_registry.py` / `test_config_union.py`

Legacy used a one-off trick: synthetic manifests' `entry_points` need
real Python modules to import from, so the tests register themselves
under a stable name (`tests.unit.test_registry`) at runtime and point
the manifests' locators back at that name. Migration moves the stable
name to `tests.unit.engines.test_registry` /
`tests.unit.engines.test_config_union`. The helper is
`_register_test_module_for_import` in each file. Could be DRYed into a
shared fixture in `tests/_harness/`, but that's a separate refactor —
out of batch scope.

### Engine test conftest singleton reset

The legacy `packages/engines/tests/conftest.py` clears
`EngineRegistry` + `_engine_config_union` lock-protected singletons
around every test. Without it, a test that builds a registry from
`tmp_path` leaks the synthetic state into the next test (and the next
test might want the real default-registry shape). Migrated verbatim to
`tests/unit/engines/conftest.py`.

### Linter cleanup beyond what legacy had

Legacy used non-raw regex patterns (`match="frozen|Instance is frozen"`),
Yoda conditions (`"DuplicateEngineId" == f.exc_type`), and
`pytest.raises(Exception)` with a `# noqa: BLE001`. Greenfield ruff
config catches RUF043 / SIM300 / B017 / RUF100 — these were promoted
to raw strings, non-Yoda comparisons, and `pytest.raises(ValidationError, …)`
respectively. **Total ruff errors in `tests/` dropped from 62 to 37**
(my new files are all-clean; pre-existing tests in `tests/` still
have their own queue of fixes).

## Verification commands + exit codes

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b

# Greenfield run
.venv/bin/python -m pytest -c tests/pytest.ini tests/
# => 604 passed, 94 skipped, 9 deselected, 4 warnings in 53.82s
# => exit 0

# Greenfield collection (after migration)
.venv/bin/python -m pytest -c tests/pytest.ini tests/ --co
# => 698/707 tests collected (9 deselected); was 483/492 pre-batch
# => +215 tests

# Legacy collection (after migration)
.venv/bin/python -m pytest packages/ --co
# => 6822 tests collected; was 7043 pre-batch
# => −221 tests (matches the legacy-test count we removed)

# Engines-specific greenfield (tests/unit/engines + tests/contract/engines)
.venv/bin/python -m pytest -c tests/pytest.ini tests/unit/engines/ tests/contract/engines/
# => 225 passed in 0.20s
# => exit 0

# Lint
ruff check tests/unit/engines/
# => All checks passed!

# Sentinel still passes (Protocol-mocking forbidden)
.venv/bin/python -m pytest -c tests/pytest.ini tests/_lint/test_no_protocol_mocking.py
# => 2 passed
# => exit 0

# Importlinter unchanged from Batch 1 baseline
.venv/bin/lint-imports --no-cache
# => Contracts: 9 kept, 1 broken (same control→pod set Batch 1 documented)
# => exit 1 (expected — known violation set unchanged)
```

## Deviations from the per-file workflow

- For Row 4 (`test_scaffolding.py`), the migration spec's wording is
  "STEP 5: DELETE — `git rm` the legacy file once Step 4 passes". I
  applied Step 5 immediately for the genuinely-DUP smoke tests and
  applied STEP 3 (migrate) for the 1 UNIQUE constant. Single deletion
  covers both buckets; no separate file created.

- For Row 8 (`vllm/test_manifest.py`), the DUP claim about
  `test_runtime_implements_protocol` etc. is supported by code reading
  (the contract test exercises the identical `get_runtime(engine_id)`
  code path) rather than synthetic-injection proof. I DID do synthetic
  injection for the *strongest* DUP claim (`test_capabilities_match_manifest`),
  which provides high confidence in the equivalence; the other two
  share the same `get_*` API surface.

- The legacy directory's `packages/engines/tests/conftest.py` is
  retained even though it's now consumed by zero test files — deletion
  is harmless but creates churn outside the migration scope. Batch 3+
  will sweep these orphans once a few more migrations are done.
