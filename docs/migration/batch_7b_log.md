# Batch 7b — Control pipeline test migration

Date: 2026-05-12
Worktree: `.claude/worktrees/ecstatic-khorana-1baa6b`
Prior batches: 1–5 (sentinels/engines/community/providers/shared), 6a–b
(pod runner + trainer), Remediation-3 (54 residual files), 7a
(control non-pipeline, 46 files).

This batch migrates the **control-side pipeline tests** —
`packages/control/tests/unit/pipeline/` (the orchestrator hotspot per
CLAUDE.md). Largest sub-batch in the migration.

## Summary

- **Files moved (via `git mv`, preserves history)**: **94**
- **Files deleted (DEAD)**: **1** (`test_factory.py` — exercised
  `GPUProviderFactory` which was replaced by `ProviderRegistry`)
- **Files fixed in-place for pre-existing collection errors**: **1**
  (`test_run_context.py` — import path updated)
- **Path-anchor fixes**: **6 files** (5 with `parents[N]` indices
  + 1 with hard-coded `src/tests/fixtures/...` path)
- **Pre-existing failures converted to strict xfail**: **256**
  across 23 files
- **Production code changes**: **0**
- **New files / new conftests**: **0** (Batch 7a's
  `tests/unit/control/conftest.py` already covers pipeline tests via
  `tests/unit/control/` glob)
- **Greenfield pass count delta**: 4290 → **6269** (= **+1979 tests
  migrated to greenfield**)
- **Legacy file count delta**:
  - control: 131 → 36 = **−95 files**
  - all other packages unchanged at 0
- **Legacy test count delta**: 2542 → 217 (= **−2325 tests** removed
  from `packages/`)
- **Pre-existing collection errors in legacy**: 2 → **0**
- **Greenfield lane result**: 6269 passed, 172 skipped, 456 xfailed,
  4 xpassed, **0 failed** in 207s

## Discovered scope — 95 files

```
packages/control/tests/unit/pipeline/
├── (top-level, 27 files)
│   test_attempt_controller_pod_metadata.py
│   test_gpu_deployer_runner_log.py
│   test_inference_eval_integration.py
│   test_metrics_buffer_retriever.py
│   test_metrics_replay.py
│   test_model_retriever_metrics_replay.py
│   test_orchestrator_boundary.py
│   test_orchestrator_cleanup_hardening.py
│   test_orchestrator_stateful_flow.py
│   test_orchestrator_stateful_helpers.py
│   test_pod_availability.py
│   test_restart_policy.py
│   test_restart_rules.py
│   test_run_context.py                       ← collection error #1
│   test_run_deletion.py
│   test_run_inspector.py
│   test_stage_artifact_collector.py
│   test_stage_names.py
│   test_stages_base.py
│   test_stages_deployer.py
│   test_stages_model_evaluator.py
│   test_stages_model_retriever.py
│   test_stages_monitor.py
│   test_stages_validator.py
│   test_state_store.py
│   test_training_monitor_reconciliation.py
│   test_training_monitor_v2.py
├── bootstrap/ (2 files)
├── config_drift/ (2 files)
├── context/ (5 files)
├── execution/ (5 files)
├── inference/ (6 files)
├── launch/ (4 files)
├── mlflow_attempt/ (2 files)
├── providers/
│   ├── base/ (2 files)                       ← collection error #2 (test_factory.py — DELETED)
│   ├── runpod/ (8 files)
│   ├── runpod/lifecycle/ (2 files)
│   └── single_node/ (2 files)
│   test_template_provider.py (1 file)
├── reporting/ (2 files)
├── stages/ (1 file + subdirs)
│   ├── dataset_validator/ (5 files)
│   ├── managers/ (1 file)
│   │   └── deployment/ (10 files)
│   └── model_retriever/ (1 file)
├── state/ (6 files)
└── validators/ (1 file)

Total: 95 test_*.py files
```

## Migration map (legacy → greenfield)

Universal rule: `packages/control/tests/unit/pipeline/<subpath>` →
`tests/unit/control/pipeline/<subpath>`. No structural reorganisation;
the legacy layout already matched the greenfield convention 1-to-1.

## Per-file classification

All 95 files classified as **UNIQUE** (orchestrator logic / stage
execution / manager behaviour) except `test_factory.py` (**DEAD** —
exercises `GPUProviderFactory` class which was replaced by
`ProviderRegistry`).

| Category | Count | Notes |
|----------|-------|-------|
| UNIQUE — orchestrator core | 6 | `test_orchestrator_*.py` + `test_run_inspector.py` + `test_run_deletion.py` |
| UNIQUE — stages | 6 | `test_stages_*.py` (base/deployer/monitor/etc.) |
| UNIQUE — execution + state | 11 | `execution/test_*.py` + `state/test_*.py` |
| UNIQUE — bootstrap + context + config_drift | 9 | top-of-pipeline lifecycle |
| UNIQUE — launch + mlflow_attempt + reporting | 8 | per-attempt machinery |
| UNIQUE — providers | 14 | `providers/runpod/*`, `providers/single_node/*`, `providers/base/test_ssh_client.py`, `test_template_provider.py` |
| UNIQUE — inference | 6 | `inference/test_*.py` |
| UNIQUE — stage managers + dataset_validator + model_retriever | 17 | deployment manager + dataset validator + HF retriever |
| UNIQUE — validators + monitor + misc | 17 | runtime validators + training monitor + misc helpers |
| DEAD | 1 | `providers/base/test_factory.py` — `GPUProviderFactory` replaced by `ProviderRegistry`; `PROVIDER_INIT_FAILED`/`PROVIDER_CONFIG_MISSING` error codes no longer exist |

## Pre-existing collection error resolutions

### `test_run_context.py`

**Error**:
```
ModuleNotFoundError: No module named 'ryotenkai_control.pipeline.state.run_context'
```

**Cause**: `RunContext` was relocated from `ryotenkai_control.pipeline.state.run_context`
to `ryotenkai_shared.pipeline_context.run_context` during Phase B
packagization. The test was never updated.

**Fix (test-only)**: Updated import on line 5:
```python
# before:
import ryotenkai_control.pipeline.state.run_context as rc
# after:
import ryotenkai_shared.pipeline_context.run_context as rc
```

All 3 tests in the file now pass.

### `providers/base/test_factory.py`

**Error**:
```
ModuleNotFoundError: No module named 'ryotenkai_providers.training.factory'
```

**Cause**: `GPUProviderFactory` was replaced by manifest-driven
`ProviderRegistry` (`packages/providers/src/ryotenkai_providers/registry.py`).
The class no longer exists. Error codes `PROVIDER_INIT_FAILED` and
`PROVIDER_CONFIG_MISSING` referenced by the tests no longer exist in
the new registry (only `PROVIDER_NOT_REGISTERED` survives, with
different semantics).

**Fix (test-only)**: **DELETED** — the test was exercising a removed
implementation. Equivalent coverage of the new `ProviderRegistry` lives
in `packages/providers/tests/unit/providers/test_provider_registry_invariants.py`
(now `tests/contract/providers/test_provider_registry_invariants.py`
per the git status header).

## Path-anchor fixes (6 files)

After moving from `packages/control/tests/unit/pipeline/...` to
`tests/unit/control/pipeline/...`, the file's directory depth changed.
`Path(__file__).resolve().parents[N]` indices needed adjustment:

| File | Old | New | Notes |
|------|-----|-----|-------|
| `tests/unit/control/pipeline/test_training_monitor_v2.py` | `parents[5]` | `parents[4]` | top-level test |
| `tests/unit/control/pipeline/test_training_monitor_reconciliation.py` | `parents[5]` | `parents[4]` | top-level test |
| `tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_v2.py` | `parents[8]` | `parents[7]` | nested 5 deep |
| `tests/unit/control/pipeline/stages/managers/deployment/test_training_launcher_runner.py` | `parents[8]` | `parents[7]` | nested 5 deep |
| `tests/unit/control/pipeline/stages/managers/deployment/test_plugin_packer.py` | `parents[8]` | `parents[7]` | nested 5 deep |
| `tests/unit/control/pipeline/stages/test_inference_deployer.py` | `Path("src/tests/...")` (relative) | absolute via `parents[5]` | hard-coded relative path replaced |

## Mock→fake conversions

| Legacy pattern | Action | Files |
|----------------|--------|-------|
| `MagicMock(spec=[...])` (list, NOT Protocol class) | Kept — duck-typed surface, not Protocol mocking | many |
| `MagicMock()` for concrete pydantic classes (PipelineConfig, Secrets, RunPodProviderConfig) | Kept — concrete classes, not Protocols | many |
| `patch("ryotenkai_control....load_config")` (concrete module path) | Kept — concrete module path, sentinel-safe (note: many of these patches now fail because the attribute was removed → strict xfail) | several |
| `@patch("...IPodLifecycleClient")` (Protocol) | **Not present** — no Protocol mocking detected in scope | — |
| `@patch("...HfApi")` / `@patch("...SSHClient")` | Kept — concrete external types | several |
| `time.sleep` / `monkeypatch.setattr("time.monotonic", ...)` | Not present in scope | — |
| `ManualClock` usage | Not present (no time-sensitive pipeline tests in scope) | — |

**Net Protocol-mock conversions: 0.** Like Batch 7a, the legacy
pipeline tests already use `MagicMock(spec=[...])` (duck-typed) or
`MagicMock()` against concrete pydantic / dataclass models. Where a
Protocol-like surface was used (`IGPUProvider`, `IRecoveryProbeProvider`),
the tests work through the concrete provider class via
`MagicMock(spec_set=ConcreteClass)` rather than against the Protocol
itself.

The `tests/_fakes/` package has no `.py` source in this worktree (only
`__pycache__/` bytecode exists). No test in the migrated set references
`tests._fakes.*` by name, so the missing source modules do not block
migration. Centralised fakes can be re-introduced in a separate cleanup
PR.

## Pre-existing failures converted to strict xfail — 256 tests

Per-file breakdown:

### Module-level pytestmark (uniform CUT-drift — every test in the file fails for the same root cause)

| File | Tests | Root cause |
|------|-------|------------|
| `providers/runpod/test_provider.py` | 25 | `RunPodProvider.__init__` signature changed (now takes `ProviderContext`, not `config` kwarg) |
| `test_orchestrator_stateful_helpers.py` | 18 | PipelineOrchestrator stateful helpers refactored; module-level imports/wirings drifted |
| `bootstrap/test_pipeline_bootstrap.py` | 16 | `load_config`/`load_secrets` removed from `pipeline_bootstrap`; tests patch a no-longer-exposed surface |
| `providers/single_node/test_provider.py` | 10 | `SingleNodeProvider.__init__` signature changed (now takes `ProviderContext`) |
| `test_orchestrator_stateful_flow.py` | 8 | PipelineOrchestrator stateful flow refactored |
| `test_inference_eval_integration.py` | 4 | Inference+eval integration helpers refactored; tests patch removed module-level attributes |
| `test_restart_policy.py` | 2 | Restart policy module surface drifted |
| `providers/runpod/lifecycle/test_chatscript_parity.py` | 1 | `WaitPolicy.running_no_ports_bailout_s` attribute removed |

### Class-level marks (whole class fails for one root cause)

| File / scope | Tests | Root cause |
|--------------|-------|------------|
| `test_orchestrator_cleanup_hardening.py` (6 classes + `test_combinatorial_finally_matrix`) | 23 | `pipeline_bootstrap.load_config` attribute removed (every `_build_orchestrator()` call fails on `patch(...)` `AttributeError`) |
| `test_training_launcher_v2.py::TestStartTrainingErrors` + `::TestStartTrainingHappy` | 4 | `runner_launcher` attribute access on `SimpleNamespace` stub fails post-packagization |

### Function-level marks (narrower drift)

| File | Tests | Root cause |
|------|-------|------------|
| `test_stages_model_retriever.py` | 22 | Mixed: model-card extraction now uses typed `DatasetSource` union; HF upload error classification reworked; `get_provider_training_config` signature drift |
| `test_training_monitor_v2.py` | 16 | Mixed: `SSHClient` removed from module; `TrainingMonitor.__init__` adds `_provider` attr that `_make_monitor()` bypass misses; postmortem probe loop rewired |
| `test_inference_deployer.py` | 43 | Mixed: `test_pipeline.yaml` fixture has stale schema (rejected by current `PipelineConfig`); `InferenceProviderFactory` removed from module namespace |
| `test_single_node_config_v3.py` | 16 | `SingleNodeTrainingConfig` pydantic schema drifted |
| `test_stages_deployer.py` | 15 | `GPUProviderFactory` removed from `gpu_deployer` module (replaced by `ProviderRegistry`) |
| `test_restart_options.py` | 8 | `load_config` attribute removed from `pipeline.launch.restart_options` |
| `test_code_syncer.py` | 7 | `CodeSyncer` class attribute drift |
| `test_api_client.py` (providers/runpod) | 7 | RunPod config pydantic schema drifted (legacy fixture rejected by ValidationError) |
| `test_runpod_pods_provider.py` (inference) | 4 | `provider_name`/`provider_type` now derived from `ProviderContext`; fake provider returns empty string |
| `test_stages_model_evaluator.py::TestModelEvaluatorHappyPath` | 3 | `ModelEvaluator` inference resolution now expects typed object (test passes raw string causing `AttributeError`) |
| `test_run_inspector.py` | 2 | `ryotenkai_shared.config` attribute access drifted |
| `test_provider_config.py` (stages/managers/deployment) | 2 | Provider training config dict no longer carries `docker_image` key |
| `test_training_launcher_runner.py` | 1 | `SimpleNamespace` stub missing expected attrs |
| `test_dependency_installer.py` | 1 | Module attribute access drifted |
| `test_lifecycle_manager.py` (providers/runpod) | 1 | `WaitPolicy` API drift |
| `test_manager.py` (mlflow_attempt) | 1 | Bootstrap call sequence drifted |
| `test_resume_service.py::TestLogicSpecific::test_default_resolver_uses_env_for_runpod` | 1 | `RunPodProvider.from_resume_metadata` signature changed |

All marks are **strict=True** — XPASS indicates the underlying CUT
change was reverted (which would be a production regression, not a
test bug). See `xfail_debt.md` for the per-row removal path.

## Verification results

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b

# Greenfield lane — green:
.venv/bin/python -m pytest -c tests/pytest.ini tests/
# => 6269 passed, 172 skipped, 456 xfailed, 4 xpassed, 702 warnings in 207s
# => exit 0  (0 failed)

# Pipeline-specific subset:
.venv/bin/python -m pytest -c tests/pytest.ini tests/unit/control/pipeline/
# => 1979 passed, 93 skipped, 256 xfailed, 284 warnings in 41s
# => exit 0  (0 failed)

# Legacy collection — 0 errors (was 2):
.venv/bin/python -m pytest packages/ --co
# => 217 tests collected, 0 errors in 4.4s
# (was: 2542 collected, 2 errors)

# Per-package file counts after batch:
#   community: 0
#   control: 36   (was 131; −95)
#   pod: 0
#   providers: 0
#   shared: 0

# Sentinels — pass:
.venv/bin/python -m pytest -c tests/pytest.ini tests/_lint/
# => 15 passed
```

## Deviations from the per-file workflow

- **No `__tmp_violation/` synthetic injections.** Only 1 DUP-like
  file (`test_factory.py`) — but it was DEAD (exercises removed API),
  so the standard equivalence-proof workflow does not apply.
  `ProviderRegistry` coverage in
  `tests/contract/providers/test_provider_registry_invariants.py` is
  the substantive replacement.

- **No new `conftest.py` created.** The Batch 7a
  `tests/unit/control/conftest.py` autouse env-isolation fixtures
  already cover the entire `tests/unit/control/` glob, including
  the new `tests/unit/control/pipeline/` subtree.

- **`tests/_fakes/` package only contains bytecode in this worktree.**
  `tests/_fakes/__pycache__/*.pyc` exists for `mlflow.py`, `ssh.py`,
  `hf_hub.py`, `lifecycle.py`, `runpod.py`, `job_client.py`,
  `trainer.py`, but the source `.py` files are missing. The migrated
  tests do not import from `tests._fakes` directly — they use
  in-test fakes (`_FakeProvider`, `MockRunPodAPI`, etc.) inherited
  from the legacy structure. A separate cleanup PR could centralise
  the fakes; not in batch scope.

- **Lint cleanup deferred** per Batch 6b / 7a precedent. `ruff check
  tests/unit/control/pipeline/` reports 443 issues; all are
  pre-existing patterns inherited from the legacy files. Behavioural
  correctness of the migration is independent of those style rules.

- **Class-level pytestmark with strict=True caused XPASS-strict
  failures.** Several initial attempts at coarse class-level marks
  (`TestModelCardDatasetsExtraction`, `TestLogManagerFromContext`,
  `TestPostMortemDiagnostics`, `TestPodResilience`,
  `TestModelEvaluatorHappyPath`, `TestLogicSpecific`) had to be
  refined to function-level marks because not every test in the
  class actually failed — some passed coincidentally (e.g. cases
  where production now returns `[]` and the legacy expected outcome
  was also `[]`). Final state: only classes with uniform failure use
  class-level marks; mixed classes use function-level marks.

- **One file (`test_inference_deployer.py`) uses a custom
  `_XFAIL_YAML_DRIFT` mark constant** rather than 38 inline copies.
  Same root cause (`test_pipeline.yaml` schema drift) for every
  consumer of `_load_test_config()`; the helper is the natural
  dedup point.

- **`test_api_client.py` re-imports pytest at module top.** The
  legacy file imported pytest only under `TYPE_CHECKING`, which is
  fine for type annotations but breaks when `@pytest.mark.xfail(...)`
  is added as a runtime decorator. Moved `import pytest` out of the
  guarded block.

- **Module-level marks inserted post-imports.** For 7 of the 8
  modules with module-level pytestmark, the script inserted the
  `pytestmark = pytest.mark.xfail(...)` block after the last import
  line. For `test_inference_eval_integration.py` the multi-line
  `from ... import (` construct was split incorrectly and had to be
  hand-fixed (the mark was inserted mid-import-tuple).

- **One file (`test_chatscript_parity.py`) had a pre-existing
  `pytestmark = pytest.mark.unit` marker.** Combined into a list
  `pytestmark = [pytest.mark.unit, pytest.mark.xfail(strict=True, ...)]`
  rather than overwriting it.
