# Phase 4B/4C — orchestrator bootstrap seam (partial)

## Goal
Eliminate `patch.object(PipelineOrchestrator, "_setup_mlflow")` (51 hits) and `patch.object(StageRegistry, "_build_stages")` (62 hits) in `tests/unit/control/test_pipeline_orchestrator.py`.

## Outcome

**Production seam established. Test migration deferred.**

### Production changes (additive only)
- `PipelineOrchestrator.__init__` now accepts `mlflow_manager: IMLflowManager | None = None` kwarg
  - When provided, sets via existing `_mlflow_manager` setter after bootstrap
  - When omitted (current callers), behavior unchanged
- `PipelineOrchestrator._setup_mlflow` now **idempotent**:
  - If `_mlflow_attempt.manager` is already set → return it (no bootstrap call)
  - Else → call `_mlflow_attempt.bootstrap()` (existing behavior)

This is the **clean architectural injection point** going forward. New tests:
```python
fake_mlflow = FakeMLflowManager()
fake_mlflow.setup()
orchestrator = PipelineOrchestrator(config=cfg, mlflow_manager=fake_mlflow)
```

### Test migration: deferred

Bulk migration of the existing 51 `_setup_mlflow` patches + 54 `_build_stages` patches **was attempted** but reverted because:
- Patches usage patterns vary: some are `return_value = None`, some `return_value = <mock>`, some `assert_called_once()`, some load-bearing for `mock_stages` injection
- `_build_stages` patches are LOAD-BEARING in 11/58 tests (they substitute real PipelineStage instances with `MagicMock` stages — required because `mock_config` is itself a `MagicMock(spec=PipelineConfig)` that real stages choke on)
- A proper migration requires also `stages_override` kwarg in `PipelineOrchestrator.__init__` AND refactoring `PipelineBootstrap.build()` to accept and pass through `stages_override` — substantial production work outside this batch's scope

### Recommended follow-up (separate PR)

**Phase 4-followup**: Bootstrap injection completeness
1. Add `stages_override: Sequence[PipelineStage] | None = None` to `PipelineOrchestrator.__init__`
2. Thread through `PipelineBootstrap.build(...)` → when provided, skip `StageRegistry.build()` and use override
3. Migrate `mock_init_stages.return_value = mock_stages` test pattern to `PipelineOrchestrator(config=cfg, stages_override=mock_stages)`
4. Simultaneously: replace `mock_config = MagicMock(spec=PipelineConfig)` with `make_pipeline_config()` factory (Phase 3B already created `tests/_factories/`)
5. Once both done, ALL 51+54 patches can be removed cleanly

Total impact when followup completes: -105 patches.

## Files modified
- `packages/control/src/ryotenkai_control/pipeline/orchestrator.py` (additive: kwarg + idempotent method)

## Verification
- Lane: **6855 passed, 0 failed**, 88 xfailed (unchanged from Phase 4A baseline)
- Production change is backwards-compatible: every existing call site works without modification

## Mock count delta
- `patch.object(*, "_setup_mlflow")`: 51 → 51 (production seam available, test migration deferred)
- `patch.object(StageRegistry, "_build_stages")`: 62 → 62 (requires bootstrap refactor)
- Net Phase 4BC: -0 patches, +1 production injection seam

## Architectural value
- The mlflow injection mechanism is now **documented in production source**: any future test or advanced caller has a clear API instead of having to patch internal methods
- Establishes precedent for `stages_override` extension when followup PR lands
- Phase 4A docker refactor + Phase 4B/4C production seams give a 3-axis injection completeness:
  - Docker operations: `IDockerClient` (full refactor done in 4A)
  - MLflow setup: `mlflow_manager` kwarg (seam done in 4B; test migration deferred)
  - Stage building: pending followup
