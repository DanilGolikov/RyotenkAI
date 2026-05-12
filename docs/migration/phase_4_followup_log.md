# Phase 4-followup — orchestrator bootstrap injection completeness

## Goal

Complete the orchestrator-test mock-elimination strategy started in
Phase 4B/4C ([phase_4bc_log.md](phase_4bc_log.md)): close the second
injection seam (`stages_override`) and add a real `PipelineConfig`
factory so the 105 deferred patches in
`tests/unit/control/test_pipeline_orchestrator.py` can be retired.

## Outcome

**All 105 patches removed.** Lane unaffected: every test in
`tests/unit/control/` still passes (2931 / 0 failed).

| Metric | Before | After |
|---|---:|---:|
| `patch.object(StageRegistry, "_build_stages")` in `test_pipeline_orchestrator.py` | 54 | 0 |
| `patch.object(PipelineOrchestrator, "_setup_mlflow")` in `test_pipeline_orchestrator.py` | 51 | 0 |
| Total removed | — | **105** |
| Production constructor seams | 1 (`mlflow_manager`) | 2 (`mlflow_manager` + `stages_override`) |
| Real-value test factories | 1 (`make_run_data`) | 2 (+ `make_pipeline_config`) |

## 1 — `stages_override` production seam

### `PipelineBootstrap.build`

`packages/control/src/ryotenkai_control/pipeline/bootstrap/pipeline_bootstrap.py`

```python
@classmethod
def build(
    cls,
    *,
    config: PipelineConfig,
    secrets: Secrets | None = None,
    run_ctx: RunContext,
    settings: RuntimeSettings,
    attempt_controller: AttemptController,
    on_stage_completed: Callable[[str], None],
    on_shutdown_signal: Callable[[str], None],
    stages_override: Sequence[PipelineStage] | None = None,  # NEW
) -> BootstrapResult:
    ...
    if stages_override is not None:
        stages_list = list(stages_override)
    else:
        stages_list = StageRegistry._build_stages(
            config=config,
            secrets=secrets,
            validation_artifact_mgr=validation_artifact_mgr,
        )
    registry = StageRegistry(config=config, stages=stages_list, collectors=collectors)
```

When omitted (every production call site), behaviour is unchanged.
When supplied (tests / advanced callers), the canonical
`StageRegistry._build_stages` path is **bypassed** — no
`DatasetValidator`/`GPUDeployer`/… imports run, no validation
artifact wiring happens for stages that the test never exercises.

### `PipelineOrchestrator.__init__`

`packages/control/src/ryotenkai_control/pipeline/orchestrator.py`

```python
def __init__(
    self,
    *,
    config: PipelineConfig,
    run_directory: Path | None = None,
    settings: RuntimeSettings | None = None,
    mlflow_manager: IMLflowManager | None = None,
    stages_override: Sequence[PipelineStage] | None = None,  # NEW
):
    ...
    bootstrap = PipelineBootstrap.build(
        config=config,
        ...,
        stages_override=stages_override,
    )
```

Pass-through: the orchestrator does no work on the override list itself;
the bootstrap owns the policy.

## 2 — `make_pipeline_config` factory

`tests/_factories/pipeline_config.py`

Pure-function builder that returns a real `PipelineConfig` with sensible
test defaults:

* `model=test/model`, `torch_dtype=bfloat16`
* QLoRA adapter, batch_size=1, LR=2e-4, 1 epoch
* Single `[SFT]` strategy (no dataset reference → cross-validators pass)
* One `"default"` local-path dataset
* MLflow enabled (`tracking_uri=http://localhost:5002`)
* `_source_path` stamped (so `PipelineBootstrap.build` accepts it)

```python
def make_pipeline_config(
    *,
    source_path: Path | None = None,
    model: ModelConfig | None = None,
    training: TrainingOnlyConfig | None = None,
    providers: dict[str, Any] | None = None,
    datasets: dict[str, DatasetConfig] | None = None,
    integrations: IntegrationsConfig | None = None,
    **extra: Any,
) -> PipelineConfig:
    ...
```

Five unit tests in `tests/_factories/test_pipeline_config.py` cover:

* returns a real `PipelineConfig` (not a mock)
* `_source_path` is stamped + override-able
* default cross-block validators pass (single SFT + populated `datasets`)
* override kwargs reach the constructed config without affecting the rest

## 3 — Test migration

`tests/unit/control/test_pipeline_orchestrator.py` — 58 test methods,
~25 individual patch-stack blocks updated.

### Pattern transform

```python
# BEFORE
with (
    patch("...load_secrets") as mock_load_secrets,
    patch("...validate_strategy_chain") as mock_validate,
    patch.object(StageRegistry, "_build_stages") as mock_init_stages,
    patch.object(PipelineOrchestrator, "_setup_mlflow") as mock_setup_mlflow,
):
    mock_load_secrets.return_value = mock_secrets
    mock_validate.return_value = Ok(None)
    mock_init_stages.return_value = mock_stages
    mock_setup_mlflow.return_value = mock_mlflow_manager  # or None

    orchestrator = PipelineOrchestrator(config=mock_config)
    result = orchestrator.run()

# AFTER
with (
    patch("...load_secrets") as mock_load_secrets,
    patch("...validate_strategy_chain") as mock_validate,
):
    mock_load_secrets.return_value = mock_secrets
    mock_validate.return_value = Ok(None)

    orchestrator = PipelineOrchestrator(
        config=mock_config,
        stages_override=mock_stages,
        mlflow_manager=mock_mlflow_manager,  # or None
    )
    result = orchestrator.run()
```

The two remaining patches (`load_secrets`, `validate_strategy_chain`)
are legitimate test scaffolding — they keep the orchestrator from
hitting the disk for `secrets.env` and from running the heavy
strategy-chain validator in unit tests.

### Tests legitimately keeping `_setup_mlflow` behaviour

The three `TestPipelineOrchestratorMLflowInternals` tests
(`test_setup_mlflow_returns_none_when_disabled`,
`test_setup_mlflow_handles_setup_exception`,
`test_setup_mlflow_returns_none_when_not_active`) test the **default
no-injection path**: they don't pass `mlflow_manager=` and let the
orchestrator's `_mlflow_attempt.bootstrap()` run. These tests pass `None`
for the mock manager argument as the migration is mechanical — the
underlying code path is exactly what they test.

## 4 — Lane status

Per-package targeted runs (the global lane shows pre-existing failures
in `tests/chaos/` / `tests/contract/openapi_drift/` /
`tests/stack/test_runpod_via_sidecar.py` — environmental, missing
`schemathesis` and `syrupy`):

| Run | Result |
|---|---|
| `tests/unit/control/test_pipeline_orchestrator.py` | **58 passed**, 0 failed |
| `tests/_factories/` | **6 passed**, 0 failed (was 1) |
| `tests/unit/control/` (all) | **2931 passed**, 93 skipped, 37 xfailed, 0 failed |
| `tests/unit/control/pipeline/test_orchestrator_boundary.py` + `test_pipeline_orchestrator_missing_lines.py` | **50 passed**, 2 xfailed (unaffected) |

## 5 — Open items

* `tests/unit/control/pipeline/test_orchestrator_boundary.py` (7
  occurrences of `patch.object(StageRegistry, "_build_stages",
  return_value=[])`) and
  `tests/unit/control/test_pipeline_orchestrator_missing_lines.py` (1)
  use the same pattern and could be migrated with the same recipe in a
  follow-up batch. They were intentionally **out of this phase's
  scope** (the spec focuses on the 105 patches in the main test file)
  and currently still pass.
* `mock_config` in the test fixture stays a `MagicMock(spec=PipelineConfig)`
  for backwards compatibility with tests that mutate config attrs
  (e.g. `mock_config.integrations.mlflow = None`). A follow-up could
  swap individual tests to `make_pipeline_config()` to eliminate the
  remaining `mock_config` MagicMock. Not blocking — the production
  seam works equally well with either input shape.
