# Pipeline Orchestrator — Architectural Refactor (Final)

## Context

The orchestrator reached 1385 LOC, 50 methods after 9 decomposition commits.
Root cause of remaining size: **state ownership is fragmented** (eight write-sites
for `_pipeline_state`, two for `_current_attempt`, lineage updates in three
places), and ~200 LOC of thin delegates exist solely so 46 test-callsites keep
probing private internals. These are not "just tech debt" — they are structural
problems that will bite hard when we add async stages, parallel execution, new
state backends or new restart semantics.

Goal: **long-term architecture without regrets** — each invariant guarded by
one component, each piece of state has one writer, each collaborator is
test-in-isolation without 10 mocks of orchestrator internals.

Non-goal: cosmetic LOC reduction. Net LOC will stay ~1500, spread across
~12 files with max-file ~220 LOC and mean ~125 LOC.

## Target architecture

```
PipelineOrchestrator (facade, ~180 LOC, 5 public methods)
│
├── PipelineBootstrap (~150 LOC)            — loads config/secrets, builds stages, StageRegistry
│     └── StartupValidator                  — HF_TOKEN / RunPod / strategy_chain / eval plugin secrets
│
├── RunSession (~60 LOC, disposable per run())
│     ├── LaunchPreparator (~160 LOC)       — bootstrap state, drift, resume/restart, lineage restore
│     ├── RunLockGuard (~40 LOC, CM)        — invariant #1: run_lock always released
│     ├── AttemptController (~220 LOC)      — SOLE owner of PipelineState + current_attempt + lineage
│     ├── StageExecutionLoop (~180 LOC)     — main loop body + 6 exception/outcome handlers
│     └── MLflowRunHierarchy (~90 LOC)      — wraps MLflowAttemptManager + preflight + lifecycle callbacks
│
├── StageRegistry (~140 LOC)                — stages + collectors + reverse-cleanup + early-release
├── RestartPointsInspector (~140 LOC)       — standalone, zero coupling
├── LineageManager (~80 LOC)                — pure lineage funcs, called by AttemptController
├── PipelineContext (~90 LOC)               — typed value-object over dict, .fork(attempt_id)
│
└── Keep-as-is (already extracted, now pure collaborators — never mutate orchestrator state):
      StagePlanner, ConfigDriftValidator, ContextPropagator, StageInfoLogger,
      ExecutionSummaryReporter, ValidationArtifactManager, transitioner funcs
```

**Core principle**: collaborators never import `PipelineOrchestrator`. No
collaborator sees `self.orchestrator`. State handles flow in through ctors.

## State ownership map

| State | Owner (sole writer) | Read-only viewers |
|---|---|---|
| `PipelineState` (full) | `AttemptController` | `StageExecutionLoop` via `AttemptController.snapshot()` (frozen copy) |
| `current_attempt` | `AttemptController` | `StageExecutionLoop` via snapshot |
| `current_output_lineage` | `AttemptController` (delegates to `LineageManager` pure funcs) | — |
| `run_lock` | `RunLockGuard` (context manager) | nobody; lifetime = `with` block |
| `context: PipelineContext` | `StageExecutionLoop` at stage boundaries; stages write via injected handle | `AttemptController` reads immutable snapshot for lineage outputs |
| `attempt_directory`, `_log_layout` | `LaunchPreparator` produces `PreparedAttempt`; `RunSession` holds | `AttemptController`, `StageExecutionLoop` |
| `state_store` | `LaunchPreparator` creates; `AttemptController` keeps ref for `save()` | — |
| `stages`, `collectors` | `StageRegistry` (immutable after ctor) | everybody reads |
| MLflow nested runs | `MLflowRunHierarchy` (wraps existing `MLflowAttemptManager`) | `StageExecutionLoop` via callback interface |
| `_shutdown_signal_name` | `PipelineOrchestrator` (facade, signal-handler bridge) | `StageRegistry` reads for cleanup policy |
| `secrets`, `config`, `settings` | `PipelineOrchestrator` | passed by value to collaborators |

## Invariants → components map

| # | Invariant | Guarantor |
|---|---|---|
| 1 | `run_lock` acquired ⇒ ALWAYS released on every exit path | `RunLockGuard.__exit__` via `with RunLockGuard(path):` |
| 2 | `attempt_directory` exists before any stage runs | `LaunchPreparator.prepare()` returns `PreparedAttempt` or raises — no partial state visible downstream |
| 3 | `state.attempts` and `active_attempt_id` coherent | `AttemptController.begin_attempt()` — single atomic mutation |
| 4 | Lineage updated after each stage (skip/fail/success) | `AttemptController.record_stage_*()` internally calls `LineageManager.apply(...)` + `state_store.save()`; loop cannot bypass |
| 5 | MLflow nested run hierarchy (root → attempt → strategy → phase) | `MLflowRunHierarchy` with stack-based enter/exit |
| 6 | Cleanup runs in reverse order, exactly once | `StageRegistry.cleanup_in_reverse(signal_name)` — idempotent flag inside |
| 7 | Collectors flushed even on crash | Loop finally → `StageRegistry.flush_pending_collectors(context)` |

## Public API (after refactor)

```python
class PipelineOrchestrator:
    def __init__(self, config_path: Path, run_directory: Path | None = None,
                 settings: RuntimeSettings | None = None) -> None: ...

    def run(self, *, run_dir: Path | None = None, resume: bool = False,
            restart_from_stage: str | int | None = None,
            ) -> Result[PipelineContext, AppError]: ...

    def list_restart_points(self, run_dir: Path) -> list[RestartPoint]: ...

    def list_stages(self) -> list[str]: ...
    def get_stage_by_name(self, name: str) -> PipelineStage | None: ...

    def notify_signal(self, *, signal_name: str) -> None: ...
```

Instance attributes collapse to: `config`, `secrets`, `settings`, `run_ctx`,
`_bootstrap`, `_shutdown_signal_name`. Everything else lives inside a
`RunSession` created per `run()` call and discarded afterwards.

## Extraction sequence (11 PRs, each atomically revertible)

Each PR ports its own test-callsites to the new public API **within the same PR** —
no "final sweep". Rationale: no intermediate state with thin delegates that might
never be cleaned up; each PR's review checks the full contract, not half.

### PR-A1: `RunLockGuard` (context manager)
- New: [src/pipeline/state/run_lock_guard.py](src/pipeline/state/run_lock_guard.py) (~40 LOC).
- Wraps existing [`acquire_run_lock`](src/pipeline/state/store.py:44) in a context manager.
- Orchestrator `_run_stateful` wraps its `try/except/finally` in `with RunLockGuard(...)` — removes manual release in finally.
- Tests: migrate 1 test-callsite from `orchestrator._run_lock` to public contract ("acquiring guard → lock file exists").
- Behavior change: zero.

### PR-A2: `PipelineContext` value object
- New: [src/pipeline/context/pipeline_context.py](src/pipeline/context/pipeline_context.py) (~90 LOC).
- Typed facade around the current `dict`. Keeps `__getitem__`/`__setitem__`/`__contains__` so existing stages keep working.
- Adds `.fork(attempt_id, attempt_dir) -> PipelineContext` (used by `LaunchPreparator`).
- Replaces `self.context: dict` everywhere it's used internally. Stages still receive a dict-compatible object.
- Tests: new unit tests for `PipelineContext` API; no existing tests need migration (dict-compat).

### PR-A3: `LineageManager` (pure module)
- New: [src/pipeline/state/lineage_manager.py](src/pipeline/state/lineage_manager.py) (~80 LOC).
- Moves `_invalidate_lineage_from`, `_restore_reused_context`, and the three `update_lineage` call sites into cohesive pure functions.
- Already-extracted `invalidate_lineage_from`/`restore_reused_context` in [transitioner.py](src/pipeline/state/transitioner.py) are re-homed here.
- `transitioner.py` stays focused on stage-state transitions only.
- Tests: existing 13 lineage tests migrate to `lineage_manager` module; no orchestrator test changes.

### PR-A4: `AttemptController` — SOLE owner of PipelineState + current_attempt
- New: [src/pipeline/state/attempt_controller.py](src/pipeline/state/attempt_controller.py) (~220 LOC).
- Ctor: `(state_store: PipelineStateStore, state: PipelineState, run_ctx: RunContext)`.
- Public methods:
  - `begin_attempt(requested_action, effective_action, restart_from_stage, enabled_stage_names, config_hashes) -> AttemptHandle`
  - `record_running(stage_name, started_at)`
  - `record_completed(stage_name, outputs)`
  - `record_failed(stage_name, error, failure_kind, outputs=None)`
  - `record_skipped(stage_name, reason, outputs=None)`
  - `record_interrupted(stage_name, started_at)`
  - `finalize(status, completed_at=None)`
  - `snapshot() -> AttemptSnapshot` (frozen/deepcopy)
  - `mlflow_run_id: str | None` (getter/setter — thin)
- Each method: mutates state internally, calls `LineageManager.apply(...)` if relevant, calls `state_store.save()` at the end. Single-point-of-mutation invariant.
- Orchestrator drops `_pipeline_state`, `_current_attempt`, all `_mark_stage_*` delegates, all direct `update_lineage` calls.
- Tests: 9 tests patching `_mark_stage_*` + 4 tests on `_bootstrap_pipeline_state` migrate to `AttemptController` public API.

### PR-A5: `LaunchPreparator`
- New: [src/pipeline/launch/launch_preparator.py](src/pipeline/launch/launch_preparator.py) (~160 LOC).
- Absorbs `_bootstrap_pipeline_state`, `_prepare_stateful_attempt`, `_record_launch_rejection_attempt`.
- Ctor: `(config, run_ctx, state_store_factory, stage_planner, config_drift, lineage_manager)`.
- Public method: `prepare(run_dir, resume, restart_from_stage, config_hashes) -> PreparedAttempt` — returns a frozen value object with `(state, attempt, start_idx, stop_idx, start_stage_name, enabled_stages, attempt_directory, log_layout, state_store)`; or raises `LaunchPreparationError`.
- Tests: 4 `_bootstrap_pipeline_state` tests + 2 `_record_launch_rejection_attempt` tests migrate to `LaunchPreparator`.

### PR-A6: `StageExecutionLoop`
- New: [src/pipeline/execution/stage_execution_loop.py](src/pipeline/execution/stage_execution_loop.py) (~180 LOC).
- Ctor: `(attempt_controller, context, mlflow_hierarchy, stage_registry, stage_planner, context_propagator, stage_info_logger, validation_artifact_mgr)`.
- Single public method: `run_attempt(prepared: PreparedAttempt) -> Result[PipelineContext, AppError]`.
- Body: the 185-LOC loop + 3 outcome handlers (failure/success/interrupt) + 4 exception handlers.
- Mutations: only via `attempt_controller.record_*(...)` — no direct state writes.
- Tests: `test_orchestrator_stateful_flow.py` migrates to directly construct `StageExecutionLoop` with mocks. Orchestrator-level tests keep the same public `run()` contract.

### PR-A7: `MLflowRunHierarchy`
- New: [src/pipeline/mlflow_attempt/run_hierarchy.py](src/pipeline/mlflow_attempt/run_hierarchy.py) (~90 LOC).
- Wraps existing [`MLflowAttemptManager`](src/pipeline/mlflow_attempt/manager.py) — subsumes `_setup_mlflow_for_attempt`, `_ensure_mlflow_preflight`, `_open_existing_root_run`, `_teardown_mlflow_attempt`.
- Exposes stage-lifecycle callbacks: `on_stage_start(stage_name, idx, total)`, `on_stage_complete(...)`, `on_stage_failed(...)`, `on_pipeline_complete(duration)`, `on_pipeline_interrupted()`, `on_pipeline_failed(exc)`.
- `StageExecutionLoop` consumes them — no direct `_mlflow_manager.log_event_*()` in the loop.
- Tests: 6 MLflow-patching test sites migrate to inject `MLflowRunHierarchy` directly or patch `MLflowAttemptManager` at its module.

### PR-A8: `StartupValidator` + `PipelineBootstrap`
- New: [src/pipeline/bootstrap/pipeline_bootstrap.py](src/pipeline/bootstrap/pipeline_bootstrap.py) (~150 LOC).
- `StartupValidator.validate(config, secrets)` — runs HF_TOKEN check, RunPod check, strategy_chain validation, `validate_eval_plugin_secrets`.
- `PipelineBootstrap.load(config_path, settings)` — loads config/secrets via `load_config`/`load_secrets`, validates, builds `StageRegistry`, returns frozen `BootstrapResult(config, secrets, run_ctx, stage_registry, stage_planner, config_drift, context_propagator, stage_info_logger, summary_reporter, lineage_manager)`.
- Orchestrator `__init__` shrinks from 125 LOC → ~25 LOC.
- Tests: `__init__` validation tests migrate to `StartupValidator`.

### PR-A9: `RestartPointsInspector`
- New: [src/pipeline/execution/restart_inspector.py](src/pipeline/execution/restart_inspector.py) (~140 LOC).
- Takes the 66-LOC body of `list_restart_points` + the `_is_inference_runtime_healthy` callable.
- Returns typed `RestartPoint` dataclasses (currently returns `list[dict]`).
- Consolidates with thin existing [src/pipeline/run_inspector.py](src/pipeline/run_inspector.py) (27 LOC) and [src/pipeline/restart_points.py](src/pipeline/restart_points.py) (131 LOC) — three files → one inspector.
- Orchestrator `list_restart_points` becomes a 3-line delegate → kept as public method, forwards to inspector.
- Tests: `test_restart_policy.py` tests migrate to directly construct `RestartPointsInspector`.

### PR-A10: `StageRegistry`
- New: [src/pipeline/execution/stage_registry.py](src/pipeline/execution/stage_registry.py) (~140 LOC).
- Consolidates `stages`, `_collectors`, `_init_stages`, `_init_collectors`, `_cleanup_resources`, `_flush_pending_collectors`, `_maybe_early_release_gpu`, `get_stage_by_name`, `list_stages`.
- Ctor called once by `PipelineBootstrap`.
- Exposes: `stages`, `collectors`, `cleanup_in_reverse(signal_name, success)`, `flush_pending_collectors(context)`, `maybe_early_release_gpu(config)`, `get_stage_by_name(name)`, `list_stages()`.
- Tests: 2 `_cleanup_resources` tests migrate; early-release tests stay (already testing public protocol).

### PR-A11: `RunSession` aggregate + orchestrator facade shrink
- New: [src/pipeline/runtime/run_session.py](src/pipeline/runtime/run_session.py) (~60 LOC).
- Small aggregate: wires together `LaunchPreparator`, `RunLockGuard`, `AttemptController`, `StageExecutionLoop`, `MLflowRunHierarchy` for one run.
- `PipelineOrchestrator.run()` becomes:
  ```python
  def run(self, *, run_dir=None, resume=False, restart_from_stage=None):
      config_hashes = self._bootstrap.config_drift.build_config_hashes()
      with RunLockGuard(lock_path) as lock:
          session = RunSession.create(self._bootstrap, lock, self._shutdown_signal_name)
          return session.execute(run_dir, resume, restart_from_stage, config_hashes)
  ```
  ~15 LOC. Facade total: ~180 LOC.
- Final test sweep: remove all lingering thin delegates from orchestrator; verify no private-attr access in tests via a grep-based architectural test.

## Test migration strategy

**Per-PR, not deferred.** Each extraction PR lands with its own test migration.
If PR-A4 introduces `AttemptController`, PR-A4 **also** ports the 13 test-callsites
for `_mark_stage_*` / `_bootstrap_pipeline_state` to `AttemptController`'s public
API. No thin delegates left behind.

Architectural guardrail test (added in PR-A1, enforced onward):
```python
# src/tests/unit/test_architectural_guardrails.py
def test_no_test_accesses_orchestrator_privates():
    """Tests must not probe orchestrator._<private> — use public component API instead."""
    # grep src/tests for orchestrator\._ (excluding _mlflow_manager property during migration)
```
Adjusted progressively: each PR that migrates a test family tightens the regex.

## Critical files

**Modified heavily:**
- [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) — 1385 → ~180 LOC
- [src/pipeline/state/transitioner.py](src/pipeline/state/transitioner.py) — lineage funcs move to new `lineage_manager.py`

**Created:**
- `src/pipeline/state/run_lock_guard.py`
- `src/pipeline/state/attempt_controller.py`
- `src/pipeline/state/lineage_manager.py`
- `src/pipeline/context/pipeline_context.py`
- `src/pipeline/launch/launch_preparator.py`
- `src/pipeline/execution/stage_execution_loop.py`
- `src/pipeline/execution/stage_registry.py`
- `src/pipeline/execution/restart_inspector.py`
- `src/pipeline/mlflow_attempt/run_hierarchy.py`
- `src/pipeline/bootstrap/pipeline_bootstrap.py`
- `src/pipeline/runtime/run_session.py`

**Deleted (consolidated):**
- [src/pipeline/run_inspector.py](src/pipeline/run_inspector.py) (27 LOC) — into `restart_inspector.py`
- [src/pipeline/restart_points.py](src/pipeline/restart_points.py) (131 LOC) — into `restart_inspector.py`

**Test files — migrate in PR of corresponding extraction:**
- [test_orchestrator_stateful_flow.py](src/tests/unit/pipeline/test_orchestrator_stateful_flow.py) — PR-A6
- [test_orchestrator_stateful_helpers.py](src/tests/unit/pipeline/test_orchestrator_stateful_helpers.py) — PR-A4/A5
- [test_restart_policy.py](src/tests/unit/pipeline/test_restart_policy.py) — PR-A9
- [test_pipeline_orchestrator.py](src/tests/unit/test_pipeline_orchestrator.py) — PR-A4/A6/A7
- [test_pipeline_orchestrator_missing_lines.py](src/tests/unit/test_pipeline_orchestrator_missing_lines.py) — PR-A7/A8
- [test_early_pod_release.py](src/tests/unit/test_early_pod_release.py) — PR-A10

## Trade-offs (honest)

- **More files, more navigation.** 12 modules vs 1. Mitigated by `src/pipeline/` package-level docstring with a 10-line "where things live" map.
- **Stack traces 2–3 frames deeper.** Accepted — each frame is now meaningful.
- **`PipelineContext` rollout is viral.** Stages receive `dict` today; migrating them needs downstream PRs. `PipelineContext` keeps dict-compat during transition.
- **`RunSession` risks becoming a new god-object.** Strict discipline: zero conditional logic, zero state mutations — it only wires.
- **Time cost: ~2 weeks** including tests + reviews. Refactor-in-place (no worktrees) — each PR small enough to review in <30 min.

## Red flags (what will go wrong if done badly)

1. **Passing `self` (orchestrator) into a collaborator** — recreates vampire ref. Rule: collaborators never import `PipelineOrchestrator`. Enforced by `test_architectural_guardrails`.
2. **`AttemptController` >350 LOC.** Split lineage off into its own sub-component before that.
3. **`AttemptController.snapshot()` returning a live reference.** Must be `deepcopy` or frozen dataclass view, else invariants collapse silently.
4. **`RunLockGuard` called non-contextually (manual acquire/release).** Make `acquire_run_lock` module-private in `run_lock_guard.py`.
5. **Big-bang test rewrite.** Each extraction PR ports its own tests — no batch at the end.
6. **MLflow hierarchy regression.** Add an integration test BEFORE PR-A7 that asserts nested-run count at each lifecycle point — catch regressions early.
7. **Signal handling split across layers.** `notify_signal` stays on facade. `StageRegistry.cleanup_in_reverse(signal_name=...)` receives the value explicitly — registry does NOT call back into orchestrator.
8. **Circular imports** likely between `AttemptController ↔ LineageManager`. `LineageManager` stays pure functions at bottom of import graph.
9. **Missing `save_state()` on a mutation path.** In new design, `AttemptController` is the ONLY writer and calls `save()` at the end of every public method. Invariant test: every public method of `AttemptController` produces a file-write (verified in unit tests).
10. **"Temporary" thin delegates that stay forever.** Zero thin delegates at any point — tests port in the same PR as the extraction.

## Verification (per-PR + final)

**Per-PR checklist:**
1. New component has unit tests with ≥90% coverage (per-category: positive, negative, boundary, invariants, dependency errors, regressions, combinatorial).
2. `pytest src/tests/unit/pipeline/ src/tests/unit/test_pipeline_orchestrator*.py src/tests/unit/test_early_pod_release.py --ignore=...` — green.
3. `ruff check src/pipeline/` — zero errors in touched files.
4. `grep -rn "src.pipeline.orchestrator\._" src/tests/` — no new matches.
5. Functional test (`test_full_pipeline_e2e.py`, `test_stages_integration.py`) — green.
6. `mcp__repowise__update_decision_records(action="update", ...)` with the new consequences.

**Final verification (after PR-A11):**
1. `wc -l src/pipeline/orchestrator.py` → ≤200 LOC.
2. Max file in `src/pipeline/` → ≤250 LOC.
3. `grep -c "def " src/pipeline/orchestrator.py` → ≤8 (5 public + `__init__` + property + setter).
4. `mcp__repowise__get_risk(targets=["src/pipeline/orchestrator.py"])` → `risk_type` no longer "bug-prone" or "churn-heavy"; hotspot_score drops below 0.75 over the following 30 days.
5. `test_architectural_guardrails` passes strictest form — zero `orchestrator._<private>` references in tests.
6. `mcp__repowise__update_decision_records(action="update", decision_id="0b4a32a9707f4f78ab052ae77e762f1c", status="completed")`.

## Success metrics

| Metric | Before | Target |
|---|---|---|
| orchestrator.py LOC | 1385 | ≤ 200 |
| Max file LOC in pipeline/ | 1385 | ≤ 250 |
| Methods on `PipelineOrchestrator` | 50 | ≤ 8 |
| Public API methods | 6 | 5 (unchanged, plus internal getters removed) |
| Test-callsites to `orchestrator._<private>` | 46 | 0 |
| State write-sites for `_pipeline_state` | 8 | 1 (`AttemptController`) |
| Components test-in-isolation | 9 | 12 |
| Cyclomatic complexity `_run_stateful` / replacement | ~15 | ≤ 8 (`StageExecutionLoop.run_attempt`) |

## Rollback plan

Each of PR-A1…A11 is an independent commit on branch `RESEACRH`. Any PR can
be reverted individually via `git revert <sha>`. No PR spans more than 3 days
of work; if a PR sits open longer than that, it's too big — split it.
