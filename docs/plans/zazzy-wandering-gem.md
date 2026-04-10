# Plan: RyotenkAI Codebase Improvement

## Context

Repowise MCP analysis revealed 10 systemic problems in the RyotenkAI codebase (130K LOC, 599 files).
The most critical: 3 files without tests (test_gap), a monolithic orchestrator (2266 LOC, hotspot 95%, bug-prone),
co-change synchronization gaps that cause silent runtime failures, and architectural fragilities.

This plan addresses the **top 6 actionable problems** in 4 phases, ordered by risk/effort/impact.
Circular dependency fix and RunPod SDK migration are excluded — they work as-is and require separate planning.

**Active research context**: SAPO vs GRPO vs DPO experiments are in progress.
Phases 1-2 are safe to run anytime. Phases 3-4 require a pause between experiments.

---

## Phase 1: Close Test Gaps (Safety Net)

**Goal**: Cover 3 untested modules so subsequent refactoring has a safety net.
**Effort**: ~3h | **Risk**: LOW | **Blocks**: Nothing

### 1A. `resilient_transport.py` — new dedicated test file

**Create**: `src/tests/unit/training/mlflow/test_resilient_transport.py` (~12-15 tests)

Tests:
- `MLflowTransportCircuitBreaker` state machine: CLOSED→OPEN after 3 failures, OPEN→HALF_OPEN after cooldown, HALF_OPEN→CLOSED on success, HALF_OPEN→OPEN on failure
- `_is_transport_exception()`: positive cases for each `_TRANSPORT_MESSAGE_MARKERS`, negative for unrelated exceptions
- `install()` / `uninstall()`: monkey-patches and restores fluent + client methods
- Metric buffering: buffer metrics when circuit open, flush on recovery
- Double-install guard (marker check)

**Pattern to reuse**: Fake MLflow module from `src/tests/unit/training/managers/test_mlflow_manager_more_coverage.py` (lines 94-141).
**Critical**: Use fake mlflow module, NOT real mlflow. Use `tmp_path` fixture for MetricsBuffer (default `/workspace` doesn't exist locally).

### 1B. `mlflow_manager/setup.py` — extend existing tests

**Extend**: `src/tests/unit/training/managers/test_mlflow_manager_more_coverage.py` (~6-8 new tests)

Tests:
- `_build_subcomponents()`: all 5 attributes wired after setup()
- URI resolution: `local_tracking_uri` precedence over `tracking_uri`
- Connectivity retry: respects max attempts
- `MetricsBuffer` attachment: created and linked

### 1C. `training_monitor.py` — extend existing tests

**Extend**: `src/tests/unit/pipeline/test_stages_monitor.py` (~6-8 new tests)

Tests:
- `_parse_cgroup_ram()`: valid v1/v2, missing file, garbage (it's a @staticmethod — no SSH mock needed)
- `_parse_meminfo_ram()`: valid output, partial, missing MemTotal
- `_download_metrics_buffer()`: success, connection refused, timeout
- Callback invocation order validation

### Verification
```bash
pytest src/tests/unit/training/mlflow/test_resilient_transport.py -v  # 12+ tests
pytest src/tests/unit/training/managers/test_mlflow_manager_more_coverage.py -v  # 44+ (was 38)
pytest src/tests/unit/pipeline/test_stages_monitor.py -v  # 64+ (was 58)
pytest src/tests/ --tb=short  # 3600+ tests, zero regressions
```

---

## Phase 2: Fix Co-change Synchronization

**Goal**: Eliminate 2 highest-risk silent breakage patterns.
**Effort**: ~1.5h | **Risk**: LOW | **Soft dep**: Phase 1

### 2A. Unify MLflowDatasetLogger construction

**Problem**: 3 separate `MLflowDatasetLogger(...)` constructor calls in setup.py and manager.py.
If constructor signature changes, cleanup path breaks silently.

**Fix**: Extract `_make_dataset_logger()` factory method on MLflowManager class.
All 3 call sites delegate to it.

**Files**:
- `src/training/managers/mlflow_manager/manager.py` — add factory method, refactor line 87 and line 607
- `src/training/managers/mlflow_manager/setup.py` — refactor line 259

**Test**: Assert cleanup() leaves `_dataset_logger` in valid state (same type, same constructor args as init).

### 2B. Add contract test for HelixQL reward function signatures

**Problem**: Tests verify count of `reward_funcs`/`reward_weights` but not function parameter signatures.

**Fix**: Add parametrized test in `src/training/reward_plugins/plugins/test_helixql_compiler_semantic.py` that validates:
- Each reward function accepts `(completions, prompts=None, **kwargs)` signature
- reward_weights length == reward_funcs length (existing)
- Functions have `__name__` attribute set (not generic closure name)

### Verification
```bash
grep -c "MLflowDatasetLogger(" src/training/managers/mlflow_manager/*.py  # 1 per file (factory)
pytest -k "test_helixql_reward_function_contract" -v  # passes
pytest src/tests/ --tb=short  # zero regressions
```

---

## Phase 3: Orchestrator Quick-Win Decomposition

**Goal**: Reduce `orchestrator.py` from 2266 to ~1775 LOC (23% reduction) by extracting 3 self-contained method groups.
**Effort**: ~3h | **Risk**: MEDIUM | **Requires**: Phase 1 complete, pause between experiments

**Reference pattern**: `src/training/orchestrator/` — 7 files, 3007 LOC distributed (facade + specialized managers).

### 3A. Extract ValidationArtifactManager (~170 LOC)

**New file**: `src/pipeline/validation/artifact_manager.py`

Extract 9 callback methods (lines 2077-2255):
`_on_dataset_scheduled`, `_on_dataset_loaded`, `_on_validation_completed`, `_on_validation_failed`,
`_on_plugin_start`, `_on_plugin_complete`, `_on_plugin_failed`,
`_flush_validation_artifact`, `_build_dataset_validation_state_outputs`

Plus state: `_validation_accumulator`, `_validation_plugin_descriptions`

**Constructor**: receives `collectors: dict[str, StageArtifactCollector]`, `run_directory: Path`
**New test**: `src/tests/unit/pipeline/validation/test_artifact_manager.py` (8-10 tests)

### 3B. Extract MLflowAttemptManager (~220 LOC)

**New file**: `src/pipeline/mlflow/attempt_manager.py`

Extract 6 methods (lines 1219-1375 + 1917-2036):
`_setup_mlflow_for_attempt`, `_ensure_mlflow_preflight`, `_open_existing_root_run`,
`_teardown_mlflow_attempt`, `_aggregate_training_metrics`, `_collect_descendant_metrics`

**Constructor**: receives `mlflow_manager: IMLflowManager`, `config: PipelineConfig`
**New test**: `src/tests/unit/pipeline/mlflow/test_attempt_manager.py` (6-8 tests)

### 3C. Extract StateTransitioner (~140 LOC)

**New file**: `src/pipeline/state/transitioner.py`

Extract 7 methods (lines 1121-1218):
`_mark_stage_running`, `_mark_stage_completed`, `_mark_stage_failed`,
`_mark_stage_skipped`, `_mark_stage_interrupted`, `_finalize_attempt_state`, `_save_state`

**Pattern**: Pure functions on `PipelineAttemptState` + `PipelineState` args, single `PipelineStateStore` dependency.
**Extend test**: `src/tests/unit/pipeline/test_orchestrator_stateful_helpers.py` (+8-10 tests)

### Verification
```bash
wc -l src/pipeline/orchestrator.py  # ~1775 lines
pytest src/tests/unit/pipeline/ -v  # all green including new tests
pytest src/tests/ --tb=short  # zero regressions
```

**Each deliverable (3A, 3B, 3C) is a separate commit**, independently revertable.

---

## Phase 4: Architectural Guardrails

**Goal**: Prevent regression through automated checks.
**Effort**: ~1.5h | **Risk**: LOW | **Requires**: Phases 2-3 complete

### 4A. Import cycle detection test

**New file**: `src/tests/unit/test_import_cycles.py`

AST-based import graph walker (excludes TYPE_CHECKING blocks). Asserts zero cycles.
Should run in <2s on 599 files.

### 4B. File size guardrail

**New file**: `src/tests/unit/test_code_health.py`

- No `.py` file under `src/` (excluding tests) exceeds 800 LOC
- Exceptions list: `training_monitor.py` (833 LOC, grandfathered with target date)

### 4C. Co-change contract assertion

Add to `test_code_health.py`:
- Assert `MLflowDatasetLogger` constructor parameter count == 3
- Assert `_make_dataset_logger` exists in manager.py

### Verification
```bash
pytest src/tests/unit/test_import_cycles.py src/tests/unit/test_code_health.py -v
```

---

## Summary

| Phase | Goal | Effort | Tests Added | Risk |
|-------|------|--------|-------------|------|
| 1 | Test gaps | 3h | +26-31 | LOW |
| 2 | Co-change sync | 1.5h | +4-6 | LOW |
| 3 | Orchestrator decomposition | 3h | +22-28 | MEDIUM |
| 4 | Guardrails | 1.5h | +5-8 | LOW |
| **Total** | | **~9h** | **+57-73** | |

**Execution order**: 1 → 2 → 3 → 4 (strict). Phase 2 can overlap with Phase 1.

## Explicitly Excluded (Separate Planning Needed)

- **Phase 4 deep extraction** (`_run_stateful` → `StageExecutionOrchestrator`): needs characterization tests and clear state ownership model. Do after Phase 3 stabilizes.
- **Circular dependency fix** (container ↔ orchestrator): works via TYPE_CHECKING guards, runtime safe.
- **RunPod SDK migration**: 26 files, `proposed` status, separate initiative.
- **Bus factor**: documentation/decision records, separate initiative.

## Key Files

| File | Role in Plan |
|------|-------------|
| `src/pipeline/orchestrator.py` | Phase 3: decompose |
| `src/training/mlflow/resilient_transport.py` | Phase 1: add tests |
| `src/training/managers/mlflow_manager/manager.py` | Phase 2: factory method |
| `src/training/managers/mlflow_manager/setup.py` | Phase 1: add tests, Phase 2: refactor |
| `src/pipeline/stages/training_monitor.py` | Phase 1: add tests |
| `src/tests/unit/training/managers/test_mlflow_manager_more_coverage.py` | Phase 1: extend |
| `src/tests/unit/pipeline/test_stages_monitor.py` | Phase 1: extend |
| `src/training/reward_plugins/plugins/helixql_compiler_semantic.py` | Phase 2: contract test |

## Risk Register

| # | Risk | Severity | Mitigation |
|---|------|----------|-----------|
| R1 | MetricsBuffer default `/workspace` in tests | LOW | Use `tmp_path` pytest fixture |
| R2 | resilient_transport patches global mlflow | MEDIUM | Use fake mlflow module, not real |
| R3 | Phase 3 breaks active experiment | HIGH | Feature branch per deliverable, only between experiments |
| R4 | Half-done Phase 3 blocks develop | MEDIUM | Each deliverable independent commit, revertable |
| R5 | _run_stateful extraction (future) double-mutates state | HIGH | Deferred to separate plan; requires characterization tests + PipelineExecutionContext dataclass |
