# Batch 7c — Control integration + e2e test migration (FINAL BATCH)

Date: 2026-05-12
Worktree: `.claude/worktrees/ecstatic-khorana-1baa6b`
Prior batches: 1–5 (sentinels/engines/community/providers/shared), 6a–b
(pod runner + trainer), Remediation-3 (54 residual files), 7a (control
non-pipeline, 46 files), 7b (control unit/pipeline, 95 files).

This is the **final batch** of the legacy → greenfield test migration.
Scope: `packages/control/tests/integration/` (29 files) + 
`packages/control/tests/e2e/` (7 files). After this batch the legacy
`packages/*/tests/` tree contains zero `test_*.py` files.

## Summary

- **Files moved (via `git mv`, preserves history)**: **35**
- **Files deleted (DEAD)**: **1** (`integration/training/test_memory_margin_in_training.py`)
- **Conftest moved**: 1 (`packages/control/tests/conftest.py` →
  `tests/integration/control/conftest.py`, copied to `tests/e2e/control/conftest.py`)
- **Production code changes**: **0**
- **Per-file path-anchor fixes**: 2
  (`tests/integration/control/api/test_plugins.py`,
  `tests/integration/control/api/test_projects.py` — fixture YAML lives
  under `tests/unit/control/fixtures/`)
- **Per-file env fixes**: 2 fixtures in
  `tests/integration/control/runner/conftest.py` (added
  `RYOTENKAI_RUNTIME_PROVIDER=single_node`) + 1 in
  `tests/integration/control/runner/test_plugin_payload.py` (rewired
  `MockSupervisor` import to the greenfield pod conftest location)
- **Pre-existing failures converted to strict xfail**: **22**
  across 6 files (well under the <30 budget)
- **Greenfield pass count delta**: 6269 → **6429** (= **+160 tests
  migrated to greenfield**)
- **xfail growth**: 456 → **478** (= **+22 new strict-True xfails**)
- **Legacy file count delta**:
  - control: 36 → **0 files** (final cleanup; all 5 packages now empty)
- **Legacy test count delta**: 217 → 0 (= **−217 tests** removed from `packages/`)
- **Greenfield lane result**: 6429 passed, 203 skipped, 478 xfailed,
  4 xpassed, **0 failed** in 364s

## Discovered scope — 36 files

```
packages/control/tests/
├── integration/
│   ├── api/ (13 files)
│   │   test_attempts_and_logs.py
│   │   test_conditional_get.py
│   │   test_delete_and_config.py
│   │   test_health.py
│   │   test_integrations_router.py
│   │   test_launch_and_interrupt.py
│   │   test_logs_ws.py
│   │   test_no_secret_leaks.py
│   │   test_plugins.py
│   │   test_project_runs.py
│   │   test_projects.py
│   │   test_providers.py
│   │   test_runs.py
│   ├── runner/ (6 files)
│   │   test_detach_reattach.py
│   │   test_e2e_happy.py
│   │   test_plugin_payload.py
│   │   test_preflight_blocks_submit.py
│   │   test_stale_plugin_blocks.py
│   │   test_stop_with_checkpoint.py
│   ├── training/ (2 files; 1 DEAD)
│   │   test_memory_manager_restart_flow.py
│   │   test_memory_margin_in_training.py            ← DELETED (DEAD)
│   └── (top-level, 8 files)
│       test_cerebras_api_live.py
│       test_eval_runner_offline.py
│       test_llm_pipeline_datas_contract.py
│       test_mlflow_stack.py
│       test_phase_executor_restart_flow.py
│       test_provider_env_requirements.py
│       test_strategy_orchestrator_restart_flow.py
│       test_training_config_reaches_trainer.py
└── e2e/ (7 files)
    ├── api/ (1 file)
    │   test_full_launch_cycle.py
    ├── reports/ (1 file)
    │   test_report_e2e.py
    └── (top-level, 5 files)
        test_dataset_flow_new_schema_e2e.py
        test_full_pipeline_e2e.py
        test_report_generation.py
        test_stages_integration.py
        test_strategy_orchestrator_e2e.py

Total: 36 test_*.py files (35 migrated + 1 DEAD)
```

## Per-file triage

Every file was triaged L4 / L5 / L6 / DEAD per the workflow rule.
**No file required L6 (full Stack subprocess)** — all "real-component"
tests already use in-process ASGI transports (`httpx.ASGITransport` /
`fastapi.testclient.TestClient`) or work against fakes / mocks. The
existing `tests/stack/` infrastructure is unchanged.

| Source | Class | Destination | Notes |
|--------|-------|-------------|-------|
| `integration/api/test_attempts_and_logs.py` | L4 | `tests/integration/control/api/` | FastAPI TestClient + StateStore fixtures |
| `integration/api/test_conditional_get.py` | L4 | `tests/integration/control/api/` | ETag / Last-Modified e2e via TestClient |
| `integration/api/test_delete_and_config.py` | L4 | `tests/integration/control/api/` | RunDeleter monkeypatched |
| `integration/api/test_health.py` | L4 | `tests/integration/control/api/` | Plain health check |
| `integration/api/test_integrations_router.py` | L4 | `tests/integration/control/api/` | Token crypto via tmp master key |
| `integration/api/test_launch_and_interrupt.py` | L4 | `tests/integration/control/api/` | `_fake_spawn` injected via monkeypatch |
| `integration/api/test_logs_ws.py` | L4 | `tests/integration/control/api/` | WebSocket via TestClient |
| `integration/api/test_no_secret_leaks.py` | L4 | `tests/integration/control/api/` | OpenAPI invariant |
| `integration/api/test_plugins.py` | L4 | `tests/integration/control/api/` | Path anchor fix (parents[2] → parents[3]); 3 xfails (fixture YAML legacy schema) |
| `integration/api/test_project_runs.py` | L4 | `tests/integration/control/api/` | StateStore seeding |
| `integration/api/test_projects.py` | L4 | `tests/integration/control/api/` | Path anchor fix; 1 xfail (fixture YAML legacy schema) |
| `integration/api/test_providers.py` | L4 | `tests/integration/control/api/` | CRUD on provider configs |
| `integration/api/test_runs.py` | L4 | `tests/integration/control/api/` | Run listing |
| `integration/runner/test_detach_reattach.py` | L4 | `tests/integration/control/runner/` | ASGITransport + WebSocket |
| `integration/runner/test_e2e_happy.py` | L4 | `tests/integration/control/runner/` | ASGITransport JobClient |
| `integration/runner/test_plugin_payload.py` | L4 | `tests/integration/control/runner/` | MockSupervisor import rewired + env fix |
| `integration/runner/test_preflight_blocks_submit.py` | L4 | `tests/integration/control/runner/` | Preflight gate |
| `integration/runner/test_stale_plugin_blocks.py` | L4 | `tests/integration/control/runner/` | Stale plugin gate |
| `integration/runner/test_stop_with_checkpoint.py` | L4 | `tests/integration/control/runner/` | Stop FSM transitions |
| `integration/test_cerebras_api_live.py` | L4 (auto-skip) | `tests/integration/control/` | Skipped without `EVAL_CEREBRAS_API_KEY` |
| `integration/test_eval_runner_offline.py` | L4 (auto-skip) | `tests/integration/control/` | Skipped without external eval dataset |
| `integration/test_llm_pipeline_datas_contract.py` | L4 (auto-skip) | `tests/integration/control/` | Skipped without sibling `llm_pipeline_datas`; 1 xfail (fallback test) |
| `integration/test_mlflow_stack.py` | L4 (auto-skip) | `tests/integration/control/` | Skipped without MLflow stack on :5002 |
| `integration/test_phase_executor_restart_flow.py` | L4 | `tests/integration/control/` | Real DataBuffer + MagicMock trainer |
| `integration/test_provider_env_requirements.py` | L4 | `tests/integration/control/` | Module-level xfail (`PipelineOrchestrator` constructor signature drift) — 3 tests |
| `integration/test_strategy_orchestrator_restart_flow.py` | L4 | `tests/integration/control/` | Real DataBuffer + MagicMock orchestrator |
| `integration/test_training_config_reaches_trainer.py` | L4 | `tests/integration/control/` | Module-level xfail (legacy YAML schema) — 3 tests |
| `integration/training/test_memory_manager_restart_flow.py` | L4 | `tests/integration/control/training/` | OOM recovery contract |
| `integration/training/test_memory_margin_in_training.py` | **DEAD** | DELETED | Script-style with `print` "assertions"; same coverage in `test_memory_manager_restart_flow.py` |
| `e2e/api/test_full_launch_cycle.py` | L5 | `tests/e2e/control/api/` | `subprocess.Popen` for real fork/exec boundary; 1 xfail (`spawn_launch_detached` removed) |
| `e2e/reports/test_report_e2e.py` | L5 | `tests/e2e/control/reports/` | ReportBuilder + community catalog plugins |
| `e2e/test_dataset_flow_new_schema_e2e.py` | L5 | `tests/e2e/control/` | 1 xfail (legacy YAML schema) |
| `e2e/test_full_pipeline_e2e.py` | L5 | `tests/e2e/control/` | Module-level xfail (`pipeline_bootstrap.load_config` removed) — 10 tests |
| `e2e/test_report_generation.py` | L5 (always-skip) | `tests/e2e/control/` | Always-skipped — outdated MLflow run ID |
| `e2e/test_stages_integration.py` | L5 | `tests/e2e/control/` | Pure mock-stage propagation tests |
| `e2e/test_strategy_orchestrator_e2e.py` | L5 | `tests/e2e/control/` | Strategy chain validation + DataBuffer integration |

## Migration map (legacy → greenfield)

Universal rule: `packages/control/tests/integration/<subpath>` →
`tests/integration/control/<subpath>`. Same for e2e.

| Legacy root | Greenfield root |
|-------------|-----------------|
| `packages/control/tests/integration/api/` | `tests/integration/control/api/` |
| `packages/control/tests/integration/runner/` | `tests/integration/control/runner/` |
| `packages/control/tests/integration/training/` | `tests/integration/control/training/` |
| `packages/control/tests/integration/test_*.py` | `tests/integration/control/test_*.py` |
| `packages/control/tests/e2e/api/` | `tests/e2e/control/api/` |
| `packages/control/tests/e2e/reports/` | `tests/e2e/control/reports/` |
| `packages/control/tests/e2e/test_*.py` | `tests/e2e/control/test_*.py` |
| `packages/control/tests/conftest.py` | `tests/integration/control/conftest.py` + copied to `tests/e2e/control/conftest.py` |

## Conftest strategy

The 811-line legacy `packages/control/tests/conftest.py` provides
`mock_config`, `mock_secrets`, `test_fixtures_dir`, `mock_runpod_api`,
`mock_judge_provider`, autouse env-isolation fixtures, etc. After the
migration multiple migrated test modules in **both** `tests/integration/control/`
and `tests/e2e/control/` reference these fixtures.

Approach: `git mv` the legacy conftest to
`tests/integration/control/conftest.py` (preserves history), then copy
the same content to `tests/e2e/control/conftest.py`. Duplication is
intentional — pytest's conftest scoping requires the file at each
subtree root, and `pytest_plugins` cannot reach a `tests._fixtures.X`
plugin module from outside the `tests` package root (which has no
`__init__.py` in this importlib-mode layout).

Trade-off accepted: two copies of the same 800-line file. Alternative
(extract to `tests/_fixtures/*.py` + add `tests/conftest.py` enabling
`pytest_plugins`) would be cleaner but is a separate refactor; out of
scope for this batch.

The autouse env-isolation fixtures (`_isolate_mlflow_tracking_uri`,
`_isolate_hf_secret_env_vars`, `_cleanup_repo_root_test_side_effect_dirs`)
do NOT collide with the `tests/unit/control/conftest.py` autouse
fixtures of the same name — pytest applies each scope's fixtures
independently, and the inner-scope fixture wins (pytest resolves by
proximity to the test).

## Path-anchor fixes (test-only)

After moving from `packages/control/tests/integration/api/test_X.py`
to `tests/integration/control/api/test_X.py`, the `parents[N]` index
for relative path resolution changes. Two files needed the fix:

| File | Old | New |
|------|-----|-----|
| `tests/integration/control/api/test_plugins.py` | `parents[2] / "fixtures/configs/test_pipeline.yaml"` | `parents[3] / "unit" / "control" / "fixtures" / "configs" / "test_pipeline.yaml"` |
| `tests/integration/control/api/test_projects.py` | `parents[2] / "fixtures/configs/test_pipeline.yaml"` | `parents[3] / "unit" / "control" / "fixtures" / "configs" / "test_pipeline.yaml"` |

## Runner conftest / MockSupervisor relocation

`packages/control/tests/integration/runner/conftest.py` loaded
`MockSupervisor` via `importlib.util.spec_from_file_location` from
`packages/pod/tests/unit/runner/conftest.py`. After migration both
files live under `tests/`:

| File | Old anchor | New anchor |
|------|-----------|-----------|
| `tests/integration/control/runner/conftest.py` | `parents[4] / "pod" / "tests" / "unit" / "runner" / "conftest.py"` | `parents[3] / "unit" / "pod" / "runner" / "conftest.py"` |
| `tests/integration/control/runner/test_plugin_payload.py` | `from tests.unit.runner.conftest import MockSupervisor` | importlib spec load from `parents[3] / "unit" / "pod" / "runner" / "conftest.py"` |

Both fixtures in the runner conftest also gained
`monkeypatch.setenv("RYOTENKAI_RUNTIME_PROVIDER", "single_node")` —
without this the FastAPI lifespan fails to boot because the new
runner runtime rejects an unset provider.

## Mock→fake conversions

| Legacy pattern | Action | Files |
|----------------|--------|-------|
| `MagicMock()` against concrete pydantic classes (`PipelineConfig`, `Secrets`) | Kept — concrete classes, not Protocols | many |
| `MagicMock(spec=PipelineStage)` (concrete base class, not Protocol) | Kept | `e2e/test_full_pipeline_e2e.py`, `e2e/test_stages_integration.py` |
| `MockSupervisor` (concrete fake class defined in `tests/unit/pod/runner/conftest.py`) | Reused via importlib | runner integration |
| `patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_config")` | Marked xfail (attr removed) | `e2e/test_full_pipeline_e2e.py` |
| `monkeypatch.setattr(launch_service, "spawn_launch_detached", ...)` | Marked xfail (attr removed) | `e2e/api/test_full_launch_cycle.py` |
| `time.sleep` / busy loops | Not present in scope | — |

**Net Protocol-mock conversions: 0.** Like Batches 7a/7b, the
integration suite already uses `MagicMock(spec=ConcreteClass)`,
fake-class instances, or duck-typed `MagicMock()` against concrete
pydantic models. No Protocol-against-MagicMock patterns surfaced.

## Pre-existing failures converted to strict xfail — 22 tests

### Module-level pytestmark (uniform CUT-drift)

| File | Tests | Root cause |
|------|-------|------------|
| `tests/integration/control/test_provider_env_requirements.py` | 3 | `PipelineOrchestrator.__init__()` removed positional `cfg_path` arg post-Phase B; every test instantiates the legacy way |
| `tests/integration/control/test_training_config_reaches_trainer.py` | 3 | Inline YAML uses legacy `datasets.<name>.source_type` / `source_local`; current `PipelineConfig` requires typed `source` discriminated union |
| `tests/e2e/control/test_full_pipeline_e2e.py` | 10 | Fixture patches `pipeline_bootstrap.load_config` which no longer exists; every test errors at fixture setup |

### Function-level marks (narrower drift)

| File | Tests | Root cause |
|------|-------|------------|
| `tests/integration/control/api/test_plugins.py` | 3 | `_minimal_preflight_config_payload()` reads the canonical fixture YAML `tests/unit/control/fixtures/configs/test_pipeline.yaml` which uses the legacy dataset schema; API returns 422 |
| `tests/integration/control/api/test_projects.py` | 1 | Same fixture YAML, PUT `/projects/{id}/config` returns 422 |
| `tests/integration/control/test_llm_pipeline_datas_contract.py::test_deployment_manager_missing_dataset_returns_clear_err` | 1 | Inline YAML uses legacy dataset schema (the 4 happy-path tests in this file are SKIPPED — sibling repo missing) |
| `tests/e2e/control/test_dataset_flow_new_schema_e2e.py::test_yaml_to_deploy_mapping_to_runtime_load_chain` | 1 | Inline YAML uses legacy dataset schema |
| `tests/e2e/control/api/test_full_launch_cycle.py::test_launch_then_poll_state_reflects_completion` | 1 | `launch_service.spawn_launch_detached` removed; `monkeypatch.setattr` raises AttributeError |

All marks are **strict=True**. XPASS would indicate the CUT was reverted
to the legacy shape (a production regression, not a test bug).

## Verification results

```bash
cd /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/ecstatic-khorana-1baa6b

# Greenfield lane — green:
.venv/bin/python -m pytest -c tests/pytest.ini tests/
# => 6429 passed, 203 skipped, 478 xfailed, 4 xpassed, 703 warnings in 364s
# => exit 0  (0 failed)

# Integration + e2e subset:
.venv/bin/python -m pytest -c tests/pytest.ini tests/integration/control/ tests/e2e/control/
# => 160 passed, 31 skipped, 22 xfailed, 2 warnings in 158s
# => exit 0  (0 failed)

# Legacy collection — 0 tests, 0 errors (was 217 tests, 0 errors):
.venv/bin/python -m pytest packages/ --co
# => no tests collected in 0.35s

# Per-package file counts after batch:
#   community: 0
#   control: 0   (was 36; −36)
#   pod: 0
#   providers: 0
#   shared: 0

# Sentinels — pass:
.venv/bin/python -m pytest -c tests/pytest.ini tests/_lint/
# => 15 passed
```

## Deviations from the per-file workflow

- **No `__tmp_violation/` synthetic injections.** Only 1 DUP-like file
  (`test_memory_margin_in_training.py`) — but it was DEAD (script-style
  with `print` "assertions" rather than real pytest asserts), and its
  behavioral coverage is preserved by the migrated
  `test_memory_manager_restart_flow.py`. No equivalence proof needed.

- **Conftest duplicated rather than centralised.** The 811-line legacy
  conftest is needed under both `tests/integration/control/` and
  `tests/e2e/control/`. Centralising via a `tests/_fixtures/*.py`
  plugin module would require either:
  1. Adding `tests/__init__.py` (breaks importlib-mode), OR
  2. Adding `tests/conftest.py` with `pytest_plugins = [...]` (changes
     plugin loading globally; out of scope).

  Two copies is the lowest-risk migration. Future cleanup: extract the
  fixture set into a proper plugin once the centralisation strategy
  is settled.

- **No L6 (Stack subprocess) migrations.** Every "integration" /
  "e2e" file already operates in-process — either via
  `fastapi.testclient.TestClient` (synchronous, supports WebSocket) or
  `httpx.ASGITransport` (async, no WS). None spawn a real control-plane
  or runner subprocess. `tests/stack/` infrastructure is unchanged.

- **`runner/test_plugin_payload.py` inline import fix.** The test
  imports `MockSupervisor` directly from the legacy
  `tests.unit.runner.conftest` path (broken post-Phase-B). Rather than
  changing the test logic, the import was rewired to the same
  importlib spec mechanism the sibling conftest already uses. The
  test body is unchanged.

- **22 xfails added, within the <30 budget.** Per the batch hard rule.
  Breakdown:
  - 16 are module-level (3 files: provider_env_requirements,
    training_config_reaches_trainer, full_pipeline_e2e) — uniform CUT
    drift, every test in the module fails for the same reason.
  - 6 are function-level — pin specific drift points (fixture YAML
    legacy schema; `spawn_launch_detached` removed).
  - All marks are `strict=True`. Each row in `xfail_debt.md` documents
    the removal path.

- **No `tests/integration/control/__init__.py` or
  `tests/e2e/control/__init__.py` created.** Pytest importlib mode
  resolves modules by path, not package; the existing tree has no
  `__init__.py` files anywhere under `tests/`.

- **Skipped `test_cerebras_api_live`, `test_eval_runner_offline`,
  `test_llm_pipeline_datas_contract`, `test_mlflow_stack`,
  `test_report_generation` collected but skip automatically** — they
  gate on credentials, external dataset, MLflow stack on :5002, or
  hard-coded outdated run-IDs. These are not failures, they are
  intentional environment gates inherited from the legacy suite.
