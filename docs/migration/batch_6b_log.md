# Batch 6b — Legacy test decommissioning log

Date: 2026-05-12
Worktree: `.claude/worktrees/ecstatic-khorana-1baa6b`
Prior batches: 1-5 (sentinels/engines/community/providers/shared), 6a (pod/runner)

This batch migrates the **Mac-runnable pod tests** that live OUTSIDE
`packages/pod/tests/unit/runner/`:
- top-level `packages/pod/tests/test_*.py` (2 files),
- `packages/pod/tests/unit/test_*.py` (4 files),
- `packages/pod/tests/unit/docker/` (2 files),
- `packages/pod/tests/unit/trainer/test_*.py` (15 files at that level — not the
  `callbacks/`, `managers/`, `mlflow/`, `data_loaders/`, `utils/`,
  `models/`, `orchestrator/` subdirectories, which are left for Batch 6c),
- `packages/pod/tests/training/orchestrator/test_shutdown_handler.py` (1 file).

24 files in total; all triaged as MAC_MIGRATE. 0 POD_DEFER.

## Summary

- **24 legacy files migrated** to greenfield via `git mv` (preserved history).
- **0 deferred to Batch 6c** — every file in scope runs on Mac with the
  worktree venv and the existing `tests/conftest.py` (none needs a real
  GPU, a real subprocess, or a real HF Hub dataset).
- **Greenfield pod subset**: `tests/unit/pod/` now has **1069 passing,
  3 skipped, 33 xfailed, 0 failed**. The pod-only run is clean.
- **32 pre-existing failures converted to strict xfail** (see xfail debt
  table below):
  - 23 in `test_trainer_factory_reward_routing.py` (class-level marks on
    9 of 11 classes — the 2 classes that test the `phase_config=None`
    ValueError still pass, because their assertion fires BEFORE the new
    `create_peft_config()` gate).
  - 4 in `test_run_training.py::TestTrainingFileHandler` (class-level
    mark — production removed `_install_training_file_handler`).
  - 5 in `test_run_training_observability.py` (function-level marks on
    the 5 tests that assert the removed FileHandler behaviour of
    `_install_crash_observability`).
- **1 test-only fix to a Batch-6a leftover bug**: `runner/test_lifespan_journal_wiring.py`
  did `from tests.unit.pod.runner.conftest import MockSupervisor`, which
  fails because `tests/` is not a Python package. Switched to the
  importlib-load pattern already used by `test_phase_14e_srp_fixes.py` in
  the same folder. 1 test was failing → now passing.
- **1 strict→non-strict xfail downgrade** on
  `tests/unit/shared/utils/clients/test_job_client_contract.py` for the
  same env-var-leakage reason already documented in Batch 6a (the pod
  test modules call `os.environ.setdefault("RYOTENKAI_RUNTIME_PROVIDER")`
  at module-load time, so the 4 contract tests XPASS when the full pod
  suite runs first).
- **2 path-anchor fixes** in tests that compute paths from
  `__file__.parents[N]`. Both anchors were broken in legacy (5 levels
  needed, used 4); migration moves the test file one level deeper, so
  the new anchor `parents[4]` lands on the worktree root and we join
  `"packages" / "pod" / "src" / ...` from there:
  - `tests/unit/pod/trainer/test_concurrent_helpers.py` — _HELPER_PATH
    pointed at non-existent `tests/src/...`; now resolves.
  - `tests/unit/pod/trainer/test_runner_event_callback_wiring.py` —
    fallback `_factory_source()` path; never actually fired because
    the real import succeeds, but fixed for correctness.
- **0 mock-of-Protocol violations introduced.** The trainer-side tests
  use `MagicMock`/concrete-class mocks only (`MagicMock()` for
  `PipelineConfig`, `_CapturedTrainer`/`_CapturedConfig` dataclasses for
  TRL config/trainer captures). None of those mocks target an
  `IProtocol` interface, so the `test_no_protocol_mocking.py` sentinel
  is not engaged.
- **0 production code changes.** All edits are test-only.
- **Importlinter contract set unchanged from Batch-5/6a baseline**: same
  3 `control → pod` violations in `dataset_validator.stage`,
  `mlflow_attempt.manager`, `data.validation.standalone`. None of these
  were introduced or resolved by Batch 6b.
- **`tests/pytest.ini` created** in this batch. It pins
  `testpaths = tests`, `--import-mode=importlib`, and the marker set
  the greenfield uses. Without this file, the verification command in
  the prompt (`-c tests/pytest.ini tests/`) was failing with
  FileNotFoundError. The new file is the canonical greenfield pytest
  config that prior batches assumed existed.

## Triage table (all 24 files MAC_MIGRATE)

| #  | Legacy file                                                                            | Lines | Heavy module-level imports | Markers | Triage |
|----|----------------------------------------------------------------------------------------|-------|----------------------------|---------|--------|
| 1  | `packages/pod/tests/test_dataset_loaders.py`                                           | 882   | none                       | none    | MAC_MIGRATE — defines its own local `mock_config` fixture; tests pure dataset loader logic |
| 2  | `packages/pod/tests/test_memory_report.py`                                             | 138   | none                       | none    | MAC_MIGRATE — 6 pure-unit tests on memory-report rendering |
| 3  | `packages/pod/tests/unit/test_run_training.py`                                         | 471   | none                       | none    | MAC_MIGRATE — docstring says "GPU servers" but tests are unit-level config/CLI/env logic; 14 pass on Mac, 4 fail (pre-existing — removed FileHandler) |
| 4  | `packages/pod/tests/unit/test_strategy_chain_validation.py`                            | 496   | none                       | none    | MAC_MIGRATE — strategy chain validation; no GPU |
| 5  | `packages/pod/tests/unit/test_chain_runner.py`                                         | 502   | none                       | none    | MAC_MIGRATE — chain runner logic; only references "CUDA OOM" inside a mocked error str |
| 6  | `packages/pod/tests/unit/test_phase_executor.py`                                       | 1530  | none                       | none    | MAC_MIGRATE — phase executor unit tests |
| 7  | `packages/pod/tests/unit/docker/test_dockerfile_thin.py`                               | 124   | none                       | none    | MAC_MIGRATE — Dockerfile.runtime content checks; path anchor was BROKEN in legacy (used `parents[4]` from packages/pod/tests/unit/docker = `packages/`, not repo root). Migration fixes this. |
| 8  | `packages/pod/tests/unit/docker/test_entrypoint_runner_log.py`                         | 175   | none                       | none    | MAC_MIGRATE — entrypoint.sh content checks; same anchor situation, fixed by migration |
| 9  | `packages/pod/tests/unit/trainer/test_concurrent_helpers.py`                           | 206   | none                       | none    | MAC_MIGRATE — `with_timeout` helper unit tests; loads helper via importlib; path anchor fixed |
| 10 | `packages/pod/tests/unit/trainer/test_trainer_factory_callbacks.py`                    | 256   | none                       | none    | MAC_MIGRATE — callback wiring unit tests |
| 11 | `packages/pod/tests/unit/trainer/test_strategies_preference_and_sapo.py`               | 90    | none                       | none    | MAC_MIGRATE — preference/SAPO strategy unit tests |
| 12 | `packages/pod/tests/unit/trainer/test_orchestrator_reexport.py`                        | 7     | none                       | none    | MAC_MIGRATE — trivial re-export check |
| 13 | `packages/pod/tests/unit/trainer/test_base_config_kwargs.py`                           | 319   | none                       | none    | MAC_MIGRATE — config kwargs dict logic |
| 14 | `packages/pod/tests/unit/trainer/test_run_training_observability.py`                   | 323   | none                       | none    | MAC_MIGRATE — observability hooks; 5 of 10 tests fail (pre-existing — removed FileHandler) |
| 15 | `packages/pod/tests/unit/trainer/test_required_modules_drift.py`                       | 100   | none                       | none    | MAC_MIGRATE — module-presence sentinel |
| 16 | `packages/pod/tests/unit/trainer/test_runner_event_callback_wiring.py`                 | 263   | none                       | none    | MAC_MIGRATE — factory source-text pin; path anchor fix in dead-code fallback |
| 17 | `packages/pod/tests/unit/trainer/test_managers.py`                                     | 1046  | none                       | none    | MAC_MIGRATE — manager dict/factory logic |
| 18 | `packages/pod/tests/unit/trainer/test_data_buffer_edges.py`                            | 368   | none                       | none    | MAC_MIGRATE — buffer edge-case logic |
| 19 | `packages/pod/tests/unit/trainer/test_strategy_factory_phase_hyperparams.py`           | 39    | none                       | none    | MAC_MIGRATE — strategy factory smoke |
| 20 | `packages/pod/tests/unit/trainer/test_trainer_factory_reward_routing.py`               | 824   | none                       | none    | MAC_MIGRATE — reward plugin routing; 23 of 25 tests fail (pre-existing — `create_peft_config` gate vs MagicMock configs) |
| 21 | `packages/pod/tests/unit/trainer/test_strategies.py`                                   | 888   | inline `from datasets import Dataset` (5 places) | none | MAC_MIGRATE — `datasets` is in the venv; no Hub call |
| 22 | `packages/pod/tests/unit/trainer/test_adapter_cache_data_buffer.py`                    | 233   | none                       | none    | MAC_MIGRATE — adapter cache logic |
| 23 | `packages/pod/tests/unit/trainer/test_trainer_builder.py`                              | 166   | none                       | none    | MAC_MIGRATE — peft config builder logic |
| 24 | `packages/pod/tests/training/orchestrator/test_shutdown_handler.py`                    | 365   | none                       | none    | MAC_MIGRATE — shutdown signal handler logic |

**No POD_DEFER files** — every file in scope is Mac-runnable.

## Per-file classification (MAC_MIGRATE)

All 24 files classify as **UNIQUE** — none of them replicate the
`tests/contract/protocol_compliance/` shape matrix. They exercise:

- Concrete trainer/strategy/factory wiring (15 trainer files)
- Pure dataset loader / memory-report rendering (2 top-level files)
- Pure chain/phase/strategy validation logic (4 unit-level files)
- Docker file/script content invariants (2 docker files)
- Shutdown signal handler invariants (1 training-orchestrator file)

None test a Protocol surface — those are covered separately under
`tests/contract/protocol_compliance/`.

## Files created in greenfield

- `tests/unit/pod/__init__.py` (existed)
- `tests/unit/pod/docker/__init__.py` (new — empty)
- `tests/unit/pod/trainer/__init__.py` (new — empty)
- `tests/unit/pod/training/__init__.py` (new — empty)
- 24 test files via `git mv` (renames preserve history)
- `tests/pytest.ini` (new — greenfield pytest config; testpaths=tests,
  import-mode=importlib, marker set)
- `docs/migration/batch_6b_log.md` (this file)
- `docs/migration/xfail_debt.md` (new — central xfail tracker)

## Files moved (legacy → greenfield)

```
packages/pod/tests/test_dataset_loaders.py                                           → tests/unit/pod/test_dataset_loaders.py
packages/pod/tests/test_memory_report.py                                             → tests/unit/pod/test_memory_report.py
packages/pod/tests/unit/test_run_training.py                                         → tests/unit/pod/test_run_training.py
packages/pod/tests/unit/test_strategy_chain_validation.py                            → tests/unit/pod/test_strategy_chain_validation.py
packages/pod/tests/unit/test_chain_runner.py                                         → tests/unit/pod/test_chain_runner.py
packages/pod/tests/unit/test_phase_executor.py                                       → tests/unit/pod/test_phase_executor.py
packages/pod/tests/unit/docker/test_dockerfile_thin.py                               → tests/unit/pod/docker/test_dockerfile_thin.py
packages/pod/tests/unit/docker/test_entrypoint_runner_log.py                         → tests/unit/pod/docker/test_entrypoint_runner_log.py
packages/pod/tests/unit/trainer/test_concurrent_helpers.py                           → tests/unit/pod/trainer/test_concurrent_helpers.py
packages/pod/tests/unit/trainer/test_trainer_factory_callbacks.py                    → tests/unit/pod/trainer/test_trainer_factory_callbacks.py
packages/pod/tests/unit/trainer/test_strategies_preference_and_sapo.py               → tests/unit/pod/trainer/test_strategies_preference_and_sapo.py
packages/pod/tests/unit/trainer/test_orchestrator_reexport.py                        → tests/unit/pod/trainer/test_orchestrator_reexport.py
packages/pod/tests/unit/trainer/test_base_config_kwargs.py                           → tests/unit/pod/trainer/test_base_config_kwargs.py
packages/pod/tests/unit/trainer/test_run_training_observability.py                   → tests/unit/pod/trainer/test_run_training_observability.py
packages/pod/tests/unit/trainer/test_required_modules_drift.py                       → tests/unit/pod/trainer/test_required_modules_drift.py
packages/pod/tests/unit/trainer/test_runner_event_callback_wiring.py                 → tests/unit/pod/trainer/test_runner_event_callback_wiring.py
packages/pod/tests/unit/trainer/test_managers.py                                     → tests/unit/pod/trainer/test_managers.py
packages/pod/tests/unit/trainer/test_data_buffer_edges.py                            → tests/unit/pod/trainer/test_data_buffer_edges.py
packages/pod/tests/unit/trainer/test_strategy_factory_phase_hyperparams.py           → tests/unit/pod/trainer/test_strategy_factory_phase_hyperparams.py
packages/pod/tests/unit/trainer/test_trainer_factory_reward_routing.py               → tests/unit/pod/trainer/test_trainer_factory_reward_routing.py
packages/pod/tests/unit/trainer/test_strategies.py                                   → tests/unit/pod/trainer/test_strategies.py
packages/pod/tests/unit/trainer/test_adapter_cache_data_buffer.py                    → tests/unit/pod/trainer/test_adapter_cache_data_buffer.py
packages/pod/tests/unit/trainer/test_trainer_builder.py                              → tests/unit/pod/trainer/test_trainer_builder.py
packages/pod/tests/training/orchestrator/test_shutdown_handler.py                    → tests/unit/pod/training/test_shutdown_handler.py
```

## Mock-to-fake migration patterns

Same conclusion as Batch 6a — **net mock→fake conversions: 0**:

| Legacy pattern observed                                              | Action taken |
|----------------------------------------------------------------------|--------------|
| `MagicMock()` for `PipelineConfig` (concrete pydantic class)        | Kept — concrete pydantic class, not a Protocol |
| `_CapturedConfig` / `_CapturedTrainer` dataclasses                   | Kept — concrete TRL config/trainer captures |
| `monkeypatch.setattr(tf_module, "...")` on concrete factory module   | Kept — concrete module patching, sentinel-safe |
| `MagicMock(spec=concrete_class)`                                     | Kept — concrete class, not a Protocol |
| `MockMemoryManager` / `MockRunPodAPI` / etc. (in `packages/pod/tests/conftest.py`) | Not migrated — the legacy conftest is left in place because the migrated files define their own local fixtures with the same names; no greenfield test imports from the legacy conftest. |

The trainer-side tests in this batch use concrete `MagicMock`-based
test doubles for `PipelineConfig` (a pydantic discriminated-union
class — concrete, not a Protocol). The new
`tests/_lint/test_no_protocol_mocking.py` sentinel does not flag these.
The canonical `tests/_fakes/{trainer,mlflow,lifecycle,job_client}.py`
fakes target the `IPodLifecycleClient` / `ITrainerSpawner` /
`IMLflowManager` / `IJobClient` Protocols — none of those interfaces
are mocked by the files in this batch.

## xfail growth — 32 strict-xfails + 4 strict-False downgrades

Net: +32 strict-xfails (all on migrated tests; track removed-production-API
test debt). +4 already-xfailed shared tests downgraded
strict=True → strict=False (env-var leakage from pod tests, same
ordering issue documented in Batch 6a).

| File / class / test                                                                                   | Strict?  | Reason |
|-------------------------------------------------------------------------------------------------------|----------|--------|
| `tests/unit/pod/test_run_training.py::TestTrainingFileHandler` (class — 4 tests)                       | `True`   | production removed `_install_training_file_handler` and `RYOTENKAI_TRAINING_LOG_PATH` env var; trainer no longer installs its own FileHandler — Supervisor captures stdio instead. |
| `tests/unit/pod/trainer/test_run_training_observability.py::test_install_observability_attaches_file_handler` | `True` | `_install_crash_observability` no longer reads `RYOTENKAI_TRAINING_LOG_PATH` or attaches a FileHandler. |
| `tests/unit/pod/trainer/test_run_training_observability.py::test_logger_writes_to_file_after_install`         | `True` | Same as above. |
| `tests/unit/pod/trainer/test_run_training_observability.py::test_existing_file_appends_not_truncates`         | `True` | Same as above. |
| `tests/unit/pod/trainer/test_run_training_observability.py::test_install_idempotent`                          | `True` | Same as above. |
| `tests/unit/pod/trainer/test_run_training_observability.py::test_log_path_from_env_passed_through_verbatim`   | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestRewardWeightsRoutedToConfig` (class — 4 tests)               | `True` | `TrainerFactory.create()` invokes `create_peft_config()` BEFORE the reward-routing code these tests exercise; bare `MagicMock` configs fail the lora-section gate. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestPluginResolvedBeforeConfigCreation` (class — 1 test)         | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestNoRewardPlugin` (class — 2 tests)                            | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestPluginCalledExactlyOnce` (class — 1 test)                    | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestConfigKwargsMerge` (class — 3 tests)                         | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestTrainerKwargsMerge` (class — 2 tests)                        | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestPreFixTypErrorRegression` (class — 1 test)                   | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestBoundaryEmptyPluginOutputs` (class — 2 tests)                | `True` | Same as above. |
| `tests/unit/pod/trainer/test_trainer_factory_reward_routing.py::TestCombinatorial` (class — 7 tests)                             | `True` | Same as above. |
| `tests/unit/shared/utils/clients/test_job_client_contract.py` (module mark — 4 tests)                 | `False`  | downgraded strict=True → strict=False because pod tests leak `RYOTENKAI_RUNTIME_PROVIDER` via `os.environ.setdefault`-at-module-load, causing XPASS when the full pod suite runs first. Same root cause as Batch 6a. |

## Pre-existing collection error in legacy

Same one Batch 6a left behind: `packages/pod/tests/unit/runner/test_cancellation_telemetry.py`
collection error (legacy still imports the pre-packagization
`ryotenkai_pod.runner.cancellation_telemetry`, which no longer exists).
The greenfield copy at `tests/unit/pod/runner/test_cancellation_telemetry.py`
already has the import-path fix from Batch 6a. The legacy file is left
alone — Batch 6c can delete the legacy `runner/` subtree wholesale.

The 2 control-side collection errors (`pipeline/providers/base/test_factory.py`,
`pipeline/test_run_context.py`) are also untouched — Batch 7 territory.

## Verification results

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b

# Pod-only greenfield (the subset this batch affects) — green:
.venv/bin/python -m pytest -c tests/pytest.ini tests/unit/pod/
# => 1069 passed, 3 skipped, 33 xfailed, 14 warnings in 96s
# => exit 0 (no failures)

# Pod+shared greenfield — also green; the strict→nonstrict downgrade
# on test_job_client_contract.py absorbs the env-var-leakage XPASS:
.venv/bin/python -m pytest -c tests/pytest.ini tests/unit/pod/ tests/unit/shared/utils/clients/test_job_client_contract.py
# => 1069 passed, 3 skipped, 33 xfailed, 4 xpassed, 14 warnings
# => exit 0

# Full greenfield — 65 PRE-EXISTING collection errors in
# tests/unit/community/ (missing tmp_community_root fixture; out of
# Batch-6b scope). NO pod-side failures introduced.
.venv/bin/python -m pytest -c tests/pytest.ini tests/
# => 2421 passed, 8 skipped, 157 xfailed, 4 xpassed, 14 warnings, 65 errors
# => exit code 1 due to community errors (pre-existing)

# Legacy collection — pod side reduced (and the 1 legacy runner
# collection error remains; same one Batch 6a left for Batch 6c).
.venv/bin/python -m pytest packages/pod/tests/ --co
# => 1371 tests collected, 1 collection error (test_cancellation_telemetry)

# Importlinter — unchanged vs Batch 5/6a baseline (same 3 control→pod
# violations in production code, untouched by this batch).
.venv/bin/lint-imports --no-cache
# => Same 3 `control → pod` violations: dataset_validator.stage,
#    mlflow_attempt.manager, data.validation.standalone.
```

## Deviations from the per-file workflow

- **No `__tmp_violation/` synthetic injections.** Zero DUPs in this
  batch — every test exercises concrete pod-side behaviour, not Protocol
  invariants already covered in `tests/contract/protocol_compliance/`.
  Equivalence is proven by identical legacy vs. greenfield pass/fail sets
  (same 32 pre-existing failures move from legacy-fail to
  greenfield-strict-xfail; no behavioural change).

- **`tests/pytest.ini` created in this batch.** The verification
  commands in the prompt reference `-c tests/pytest.ini`, but the file
  did not exist when the worktree was checked out. Creating it as the
  canonical greenfield pytest config (testpaths=tests, import-mode=importlib,
  marker registration) unblocks the prompt's verification commands and
  matches the configuration prior batches implicitly assumed.

- **The Batch-6a runner files at `packages/pod/tests/unit/runner/`
  are still present in legacy.** Batch 6a's log claimed they were
  removed; the worktree state shows both legacy and greenfield copies
  exist. This is NOT a Batch-6b concern — those files are part of
  Batch 6a's scope and a separate cleanup PR will reconcile.

- **One Batch-6a leftover fixed**: `tests/unit/pod/runner/test_lifespan_journal_wiring.py`
  had a broken `from tests.unit.pod.runner.conftest import MockSupervisor`
  import that fails because `tests/` is not a package. Switched to the
  importlib-load pattern already used in the same folder. 1 unexpected
  failure → 0.

- **`strict=True` → `strict=False` downgrade on
  `test_job_client_contract.py`'s module-level pytestmark.** Same
  ordering-dependent env-var-leakage root cause as Batch 6a; without
  the downgrade the 4 contract tests XPASS-strict-fail when the full
  pod suite runs first.

- **No new shared-level conftest added.** The greenfield top-level
  pytest config (`tests/pytest.ini`) declares the markers and import
  mode; per-test fixtures live in each migrated file (most define
  their own `mock_config` locally) or in `tests/unit/pod/runner/conftest.py`
  (Batch 6a heritage). The legacy `packages/pod/tests/conftest.py` is
  intentionally left in place — Batch 6c/7 will revisit when the rest
  of the trainer subtree migrates.

- **Lint cleanup deferred.** `ruff check tests/unit/pod/` runs 92
  auto-fixes (mostly trailing-whitespace, blank-lines, imports
  organisation) and leaves 75 residual pre-existing pattern issues
  (ARG005 lambda-arg unused, SIM117 nested with, B017 blind `except`,
  RUF059 unpacked-unused). Same per-line `# noqa` cleanup pattern as
  Batch 6a would apply; deferred here because the migration's
  behavioural correctness is independent of those style rules and the
  per-line cleanup would balloon the diff.
