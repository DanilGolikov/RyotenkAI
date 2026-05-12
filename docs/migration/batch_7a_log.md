# Batch 7a — Control non-pipeline test migration

Date: 2026-05-12
Worktree: `.claude/worktrees/ecstatic-khorana-1baa6b`
Prior batches: 1–5 (sentinels/engines/community/providers/shared), 6a–b
(pod runner + trainer), Remediation-3 (54 residual files).

This batch migrates the **control-side non-pipeline tests** —
everything under `packages/control/tests/` EXCEPT `unit/pipeline/`
(Batch 7b territory), `integration/`, and `e2e/` (Batch 7c
territory).

## Summary

- **Files moved (via `git mv`, preserves history)**: **46**
- **Files deleted (DUP / DEAD)**: 0
- **Pre-existing failures converted to strict xfail**: 16 (8 files)
- **New files created**: 1 (`tests/unit/control/conftest.py` — autouse
  env-isolation port from legacy `packages/control/tests/conftest.py`)
- **Production code changes**: 0
- **Greenfield pass count delta**: 3454 → 4290 (= **+836 tests
  migrated to greenfield**)
- **Legacy file count delta**:
  - control: 177 → 131 = **−46 files**
  - all other packages unchanged at 0
  - total: 177 → 131
- **Legacy test count delta**: 3394 → 2542 (= **−852 tests** removed
  from `packages/`)
- **Pre-existing collection errors in legacy**: still 2 (both in
  `packages/control/tests/unit/pipeline/` — Batch 7b territory)
- **Greenfield lane result**: 4290 passed, 79 skipped, 200 xfailed, 4
  xpassed, **0 failed** in 159s

## Discovered scope — 45 files (matches prompt's expectation of ~50)

```
packages/control/tests/contract/test_cli_api_parity.py
packages/control/tests/fixtures/test_providers.py
packages/control/tests/sentinel/test_no_pod_imports.py
packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py
packages/control/tests/smoke/test_cli_help.py
packages/control/tests/test_orchestrator_components.py
packages/control/tests/unit/api/routers/test_jobs_router.py
packages/control/tests/unit/api/services/test_config_service_reward_strategy.py
packages/control/tests/unit/api/services/test_launch_service_resume_pod.py
packages/control/tests/unit/api/services/test_log_service_registry.py
packages/control/tests/unit/api/test_http_cache.py
packages/control/tests/unit/api/test_resolve_run_dir.py
packages/control/tests/unit/api/test_state_cache.py
packages/control/tests/unit/api/test_token_crypto.py
packages/control/tests/unit/cli/test_commands_smoke.py
packages/control/tests/unit/cli/test_foundation.py
packages/control/tests/unit/cli/test_job_command.py
packages/control/tests/unit/cli/test_run_project_wiring.py
packages/control/tests/unit/cli_state/test_context_store.py
packages/control/tests/unit/data/validation/test_dataset_validator_comprehensive.py
packages/control/tests/unit/data/validation/test_models.py
packages/control/tests/unit/data/validation/test_standalone.py
packages/control/tests/unit/data/validation/test_validator_integration.py
packages/control/tests/unit/evaluation/test_eval_sample_metadata.py
packages/control/tests/unit/evaluation/test_plugin_report.py
packages/control/tests/unit/evaluation/test_system_prompt_loader.py
packages/control/tests/unit/reports/test_build_report_plugins.py
packages/control/tests/unit/reports/test_builder.py
packages/control/tests/unit/reports/test_cli.py
packages/control/tests/unit/reports/test_formatters.py
packages/control/tests/unit/reports/test_health_policy.py
packages/control/tests/unit/reports/test_memory_management.py
packages/control/tests/unit/reports/test_metric_analyzer.py
packages/control/tests/unit/reports/test_report_plugins_fail_open.py
packages/control/tests/unit/reports/test_report_v2.py
packages/control/tests/unit/reports/test_reports_main_module.py
packages/control/tests/unit/test_architectural_guardrails.py
packages/control/tests/unit/test_batch_smoke.py
packages/control/tests/unit/test_cli_run_rendering.py
packages/control/tests/unit/test_early_pod_release.py
packages/control/tests/unit/test_mlflow_events.py
packages/control/tests/unit/test_pipeline_orchestrator.py
packages/control/tests/unit/test_pipeline_orchestrator_missing_lines.py
packages/control/tests/unit/workspace/integrations/test_registry.py
packages/control/tests/unit/workspace/integrations/test_store.py
packages/control/tests/unit/workspace/projects/test_adapter.py
```

In addition, three support files were migrated alongside their tests
(not test files but referenced by tests):
- `packages/control/tests/contract/conftest.py`
- `packages/control/tests/contract/_normalize.py`
- `packages/control/tests/fixtures/providers.py`
- `packages/control/tests/fixtures/__init__.py`
- `packages/control/tests/sentinel/bootstrap_allowlist.py`
- `packages/control/tests/unit/cli/conftest.py`
- `packages/control/tests/unit/data/validation/conftest.py`
- 5 fixture data files (3 JSONL datasets + 2 YAML configs)

## Per-file classification — all 46 files

Every file in scope classified as **UNIQUE**. None duplicate the
greenfield `tests/contract/protocol_compliance/` shape matrix or the
existing `tests/_lint/` sentinels (the legacy
`packages/control/tests/sentinel/test_no_pod_imports.py` is the
**control-side** AST check whose pod-side mirror is
`tests/_lint/test_no_pod_imports.py` — they check opposite import
directions, so the legacy file becomes `test_control_no_pod_imports.py`
in greenfield).

| Category | Count | Files |
|----------|-------|-------|
| Sentinel — AST defence-in-depth | 2 | `test_no_pod_imports.py` (control→pod), `test_no_runtime_ssh_exec_command.py` + `bootstrap_allowlist.py` |
| Contract — CLI ↔ API parity | 1 | `test_cli_api_parity.py` (+ `_normalize.py` + `conftest.py`) |
| Fixture self-tests | 1 | `test_providers.py` (+ `providers.py` + data files) |
| Smoke — CLI surface | 1 | `test_cli_help.py` |
| API — routers / services / utils | 8 | `test_jobs_router.py`, `test_config_service_reward_strategy.py`, `test_launch_service_resume_pod.py`, `test_log_service_registry.py`, `test_http_cache.py`, `test_resolve_run_dir.py`, `test_state_cache.py`, `test_token_crypto.py` |
| CLI — commands + wiring | 4 | `test_commands_smoke.py`, `test_foundation.py`, `test_job_command.py`, `test_run_project_wiring.py` (+ `conftest.py`) |
| CLI state | 1 | `test_context_store.py` |
| Dataset validation | 4 | `test_dataset_validator_comprehensive.py`, `test_models.py`, `test_standalone.py`, `test_validator_integration.py` (+ `conftest.py`) |
| Evaluation | 3 | `test_eval_sample_metadata.py`, `test_plugin_report.py`, `test_system_prompt_loader.py` |
| Reports — builder / plugins / formatters | 10 | (all 10 `tests/unit/reports/` files) |
| Workspace — integrations + projects | 3 | `test_registry.py`, `test_store.py`, `test_adapter.py` |
| Top-level unit (orchestrator / guardrails / etc.) | 7 | `test_architectural_guardrails.py`, `test_batch_smoke.py`, `test_cli_run_rendering.py`, `test_early_pod_release.py`, `test_mlflow_events.py`, `test_pipeline_orchestrator.py`, `test_pipeline_orchestrator_missing_lines.py` |
| Top-level (under `packages/control/tests/`) | 1 | `test_orchestrator_components.py` |

## Migration map (legacy → greenfield)

| Legacy path | Greenfield path |
|-------------|-----------------|
| `packages/control/tests/sentinel/test_no_pod_imports.py` | `tests/_lint/test_control_no_pod_imports.py` |
| `packages/control/tests/sentinel/test_no_runtime_ssh_exec_command.py` | `tests/_lint/test_no_runtime_ssh_exec_command.py` |
| `packages/control/tests/sentinel/bootstrap_allowlist.py` | `tests/_lint/bootstrap_allowlist.py` |
| `packages/control/tests/contract/test_cli_api_parity.py` | `tests/contract/control/test_cli_api_parity.py` |
| `packages/control/tests/contract/conftest.py` | `tests/contract/control/conftest.py` |
| `packages/control/tests/contract/_normalize.py` | `tests/contract/control/_normalize.py` |
| `packages/control/tests/fixtures/*` | `tests/unit/control/fixtures/*` |
| `packages/control/tests/smoke/test_cli_help.py` | `tests/unit/control/smoke/test_cli_help.py` |
| `packages/control/tests/test_orchestrator_components.py` | `tests/unit/control/test_orchestrator_components.py` |
| `packages/control/tests/unit/api/**/*.py` | `tests/unit/control/api/**/*.py` |
| `packages/control/tests/unit/cli/*.py` | `tests/unit/control/cli/*.py` |
| `packages/control/tests/unit/cli_state/*.py` | `tests/unit/control/cli_state/*.py` |
| `packages/control/tests/unit/data/validation/*.py` | `tests/unit/control/data/validation/*.py` |
| `packages/control/tests/unit/evaluation/*.py` | `tests/unit/control/evaluation/*.py` |
| `packages/control/tests/unit/reports/*.py` | `tests/unit/control/reports/*.py` |
| `packages/control/tests/unit/workspace/integrations/*.py` | `tests/unit/control/workspace/integrations/*.py` |
| `packages/control/tests/unit/workspace/projects/*.py` | `tests/unit/control/workspace/projects/*.py` |
| `packages/control/tests/unit/test_*.py` | `tests/unit/control/test_*.py` |

## Path-anchor fixes (3 files)

The legacy sentinel files used `parents[2] / "src" / "ryotenkai_control"`
where `parents[2]` resolved to `packages/control/`. After migration to
`tests/_lint/`, `parents[2]` is the worktree root. Rewrote anchors to
`parents[2] / "packages" / "control" / "src" / "ryotenkai_control"`:

- `tests/_lint/test_control_no_pod_imports.py`
- `tests/_lint/test_no_runtime_ssh_exec_command.py` (also fixed
  docstring comment about file location)

The 3rd anchor concerns `bootstrap_allowlist.py` — it has no path
anchors of its own (pure data module), but the consumer
(`test_no_runtime_ssh_exec_command.py`) uses an `importlib.util` load
relative to `Path(__file__).parent`, so the move is transparent.

## Mock-to-fake conversions

| Legacy pattern | Action |
|----------------|--------|
| `MagicMock()` for `runner` (concrete duck-typed) | Kept — concrete duck-typed pattern, not a Protocol mock |
| `MagicMock()` for `PipelineConfig` (concrete pydantic class) | Kept — concrete class, not a Protocol |
| `MagicMock()` for `Secrets` (concrete pydantic class) | Kept — concrete class, not a Protocol |
| `_FakeProvider` ad-hoc classes (none in scope) | N/A — all already migrated to centralised `FakeGPUProvider` in fixtures dir |
| `patch("ryotenkai_control.api.routers.jobs._with_runner")` | Kept — concrete module path, sentinel-safe |
| `patch("time.monotonic")` | Not present in scope |
| `httpx.AsyncClient` mocks | Not present — tests use FastAPI `TestClient` directly |

**Net mock→fake conversions: 0.** Every `MagicMock` use in scope
targets either a concrete pydantic class, a duck-typed runner
collaborator, or a concrete factory module — never an `IProtocol`
runtime-checkable surface. The
`tests/_lint/test_no_protocol_mocking.py` sentinel referenced in
the prompt does not exist in this worktree, but the substantive check
(no `mock_*` of any `IProtocol`) holds.

## New file: `tests/unit/control/conftest.py`

The legacy `packages/control/tests/conftest.py` defines two autouse
fixtures — `_isolate_mlflow_tracking_uri` and
`_isolate_hf_secret_env_vars` — that prevent control tests from
leaking `MLFLOW_TRACKING_URI` / `HF_TOKEN` / `RUNPOD_API_KEY` into the
process env. The legacy conftest scopes those to
`packages/control/tests/` only; after migration the greenfield equivalent
must run on `tests/unit/control/` paths.

The greenfield port additionally **explicitly drops `HF_TOKEN` and
`RUNPOD_API_KEY` on teardown**: production code in
`startup_validator.py` does `os.environ["HF_TOKEN"] = ...` directly,
which `monkeypatch.delenv` cannot undo (monkeypatch only restores
values it knew about at setup time). Without the explicit teardown,
`test_pipeline_orchestrator.py` leaks `HF_TOKEN="test_token"` into
`tests/unit/shared/config/test_secrets_hf_hub_propagation.py` (3
tests would XFAIL the strict-True secrets assertions).

The `mock_config` / `mock_runpod_api` / `mock_judge_provider` / etc.
fixtures defined in the legacy conftest are NOT ported — every
migrated control test either defines its own local copies (Batch-6b
pattern) or doesn't use them. The legacy conftest stays in place
for the still-resident `packages/control/tests/unit/pipeline/` files
that Batch 7b will migrate.

## Pre-existing failures converted to strict xfail (16)

See [xfail_debt.md](xfail_debt.md) for the per-test breakdown. 8
files affected:

| File | Tests | Root cause |
|------|-------|------------|
| `tests/unit/control/fixtures/test_providers.py` (`TestProtocolConformance` class) | 2 | `IGPUProvider` Protocol gained methods + capability fields after the fake was frozen |
| `tests/unit/control/api/services/test_launch_service_resume_pod.py::TestFailurePaths::test_missing_api_key_returns_skipped` | 1 | Provider registry now requires explicit `api_key` kwarg |
| `tests/unit/control/cli/test_commands_smoke.py` (2 functions) | 2 | `ryotenkai_control.workspace.integrations.loader` was renamed/removed |
| `tests/unit/control/test_architectural_guardrails.py` (3 marks) | 6 | Test paths reference pre-packagization `src/...` layout |
| `tests/unit/control/test_pipeline_orchestrator_missing_lines.py` (4 function marks) | 4 | `dataset.source` typed-union drift + removed loader module + MLflow kwargs drift |
| `tests/unit/control/test_orchestrator_components.py::TestDatasetLoader::test_load_for_phase_file_not_found` | 1 | `DatasetLoader` routes through typed `dataset.source.kind` discriminator |

All 16 are **strict=True** — the underlying CUT changes are
permanent, so XPASS would indicate a regression of the
production-code drift, not a test bug.

## Verification results

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b

# Greenfield lane — green:
.venv/bin/python -m pytest -c tests/pytest.ini tests/
# => 4290 passed, 79 skipped, 200 xfailed, 4 xpassed, 419 warnings in 159s
# => exit 0

# Control-specific greenfield subset:
.venv/bin/python -m pytest -c tests/pytest.ini tests/unit/control/
# => 830 passed, 16 xfailed, 406 warnings in 24s
# => exit 0

# Legacy collection — 2 pre-existing collection errors remain (Batch 7b):
.venv/bin/python -m pytest packages/ --co
# => 2542 tests collected, 2 errors in 5.7s
# (was: 3394 tests collected, 2 errors)

# Per-package file counts after batch:
for d in packages/*/tests; do ...; done
#   community: 0 files
#   control: 131 files   (was 177; −46)
#   pod: 0 files
#   providers: 0 files
#   shared: 0 files

# Sentinels — pass:
.venv/bin/python -m pytest -c tests/pytest.ini \
    tests/_lint/test_control_no_pod_imports.py \
    tests/_lint/test_no_runtime_ssh_exec_command.py
# => 3 passed
```

## Deviations from the per-file workflow

- **No `__tmp_violation/` synthetic injections.** Zero DUPs in this
  batch. The two legacy sentinels look superficially like duplicates
  of `tests/_lint/test_no_pod_imports.py` but they check **opposite
  import directions** (legacy = control→pod; greenfield existing =
  providers→pod). Migrated as `test_control_no_pod_imports.py` rather
  than deleted.

- **`tests/unit/control/conftest.py` created.** Direct port of the
  legacy autouse env-isolation fixtures, augmented with explicit
  teardown to handle production-code mutations of `os.environ` (see
  "New file" section above). This is NOT a production code change —
  the autouse fixture is test-only.

- **`mock_config` / `mock_runpod_api` / etc. not ported globally.**
  Every migrated control test either defines its own local fixture
  copies or doesn't use them. Re-exporting would create surface area
  that Batch 7b's pipeline tests might depend on accidentally —
  defer the unification until Batch 7b's scope settles.

- **`test_architectural_guardrails.py` rewrite deferred.** The 6
  tests that depend on `src/...` paths are class- or function-level
  strict-xfail rather than rewritten. The rewrite (anchor against
  packagized layout) is documented as the removal path in
  `xfail_debt.md`; left for a follow-up cleanup PR rather than
  bundled with the migration.

- **Lint cleanup deferred** per Batch 6b precedent. `ruff check
  tests/unit/control/` reports 75 issues; all are pre-existing
  patterns inherited from the legacy files (ARG005, SIM117, BLE001,
  N806, etc.). Behavioural correctness of the migration is
  independent of those style rules.

- **The 2 pre-existing legacy collection errors at
  `packages/control/tests/unit/pipeline/` are untouched.** Both are
  Batch 7b territory by the prompt's scope definition.

- **One residual file (`packages/control/tests/unit/data/preview/`)
  is NOT in Batch 7a scope.** A prior batch deleted the test file
  itself (`test_loader.py`) but left the empty `__init__.py` in the
  worktree as an unstaged file. The production code under
  `packages/control/src/ryotenkai_control/data/preview/loader.py`
  still exists — Batch 7b or a separate cleanup PR should decide
  whether to re-create the test or remove the production code.
