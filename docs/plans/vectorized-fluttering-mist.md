# MLflow Integration Redesign — Implementation Plan

**Status:** awaiting implementation
**Created:** 2026-05-18
**Owner:** daniil
**Related ADRs:** ADR-0009 (Unified Event System), Phase B (Monorepo Packagization)

---

## Context

### Why this change is being made

Current MLflow integration in RyotenkAI has accumulated structural debt across all three planes — write path (training-time logging), read path (reports generation), and lifecycle (run boundary management). A multi-agent deep audit identified **30+ architectural problems** across code, **3 critical infrastructure gaps**, and **5 strong assets to preserve**. A subsequent deep audit of *config / URI / trainer subprocess / system_prompt* zones surfaced concrete responsibility-overlap evidence used in §3 of this plan.

**Root causes:**
1. **Responsibility blur** — a single `MLflowManager` class (660 LOC + 6 mixins + facade with 150+ methods) lives on both control-plane and pod via a `runtime_role` flag, mixing SDK wrapping, domain logging, connectivity, lifecycle, monkey-patch transport, autolog and tracing into one God class.
2. **No factoring between write/read/lifecycle** — read path constructs ad-hoc `MlflowClient()` in 6 places, write path goes through SDK monkey-patching, lifecycle is smeared across `MLflowAttemptManager` + `MLflowEnvironment` + signal handlers.
3. **Layering breach** — `control` package imports `pod.trainer.MLflowManager` at runtime ([mlflow_attempt/manager.py:114](packages/control/src/ryotenkai_control/pipeline/mlflow_attempt/manager.py:114)), violating the importlinter contract `control → never pod`. The TODO comment in that file (lines 109–113) acknowledges this — it was supposed to be fixed in Phase B but wasn't.
4. **Trainer opens its OWN top-level MLflow run** ([run_training.py:283](packages/pod/src/ryotenkai_pod/trainer/run_training.py:283)) and only attaches `mlflow.parentRunId` as a tag — it is NOT structurally nested under the control-plane's root run. This is the source of double-writes (see §3).
5. **Inconsistent taxonomy** — 4 different conventions for the same data: `log_training_config` uses flat snake_case (`model_name`, `lora_r`), `log_pipeline_config` uses dotted namespaces (`config.model.name`, `training.hyperparams.*`), `system_metrics_callback` uses both `system.gpu.0.name` and `gpu/0/utilization`.
6. **Phase 7 migration not completed** — 10+ no-op stubs left in `domain_logger.py:325-401`, `CancellationCallback` ≈ `CompletionCallback` (80% structural duplicate).

### Intended outcome

A cleanly factored MLflow integration that:
- Adopts the **community-standard Pattern A** (control owns parent run, HF Trainer creates *structurally* nested child via `report_to="mlflow"` + `MLFLOW_NESTED_RUN=TRUE` env), eliminating the layering breach, the dual top-level run anti-pattern, and the double-writes.
- Decomposes the God-class into **20 focused components** (each 200–500 LOC, single responsibility) across `shared` / `control` / `pod` packages.
- Provides **6 narrow Protocols** (≤7 methods each, no `Any` leakage) for write/lifecycle/read concerns.
- Unifies the **dual URI policy** behind a single typed `RuntimeUri` (kept `local_tracking_uri` / `tracking_uri` semantics — they map onto real local-loop vs Tailscale-Funnel reachability).
- Relocates `SystemPromptLoader` out of `infrastructure/mlflow/` (it is a domain loader, not infrastructure).
- Unifies **taxonomy** under a single `ryotenkai.<domain>.<area>` dotted namespace with a `ReservedPrefixGuard`.
- Closes critical infrastructure gaps (anonymous MinIO read + Funnel without auth) and adds a thin auth layer.
- Establishes **lint enforcement** to prevent regression.

Backups are out of scope per user direction (workspace is local-only).

---

## Architectural principles

1. **SOLID / KISS / DRY / YAGNI** — each component has one reason to change.
2. **No backwards compatibility** — legacy code is deleted, not deprecated.
3. **Importlinter-enforced layering:** `shared` (leaf) → `community` → `{pod, providers, control}`. Control never imports pod. `shared.infrastructure.mlflow` has no outbound dependencies.
4. **Composition root pattern** — all wiring happens in `control.main` / `pod.runner.main` / `pod.trainer.run_training`. Modules import only Protocols.
5. **Strong assets preserved:** ADR-0009 SSOT journal, `MetricsBuffer + MetricsDecimator`, `_flush_helper.py`, `RunnerEventCallback`, `UriResolver` (renamed to `RuntimeUriResolver` and tightened).
6. **Test discipline:** canonical fakes in `tests/_fakes/`; no Protocol mocking; mutation testing via `scripts/mutation/validate_agent_output.sh`.

---

## Run hierarchy — current (chaos) vs target (clean)

### Current state — 5 distinct runs per pipeline attempt

A single pipeline run with a 2-strategy chain (e.g. SFT → DPO) currently produces **5 distinct MLflow runs** distributed across 2 processes (Mac control + cloud pod-trainer), opened by **3 independent run-management surfaces**:

```
Process: control-plane (Mac)
============================
[#1] Root run                                                     opened by MLflowAttemptManager._open_root_run
     run_name = state.logical_run_id                              (mlflow_attempt/manager.py:190)
     log_system_metrics=True (flag — native sampler unused)
       │
       └─ NATIVE NESTED (mlflow.start_run(nested=True))
       │
   [#2] Attempt run                                               opened by MLflowAttemptManager._open_attempt_run
        run_name = {logical_run_id}_attempt_{N}                   (mlflow_attempt/manager.py:203)
        tags: pipeline.logical_run_id, pipeline.attempt_id,
              pipeline.attempt_no, nested_run_depth=1
             │
             ┊ Process boundary; env MLFLOW_PARENT_RUN_ID=<attempt_run_id>
             ┊ written at training_launcher.py:569-571
             ┊
             ┊ TAG-ONLY linkage via mlflow.parentRunId
             ┊ (MLflow API cannot create native nesting across processes)
             ┊
        Process: pod-trainer subprocess
        ================================
        [#3] Trainer's own run                                    opened by run_training.run_training
             run_name = {model_base}_{strategy_chain}_{YYYYMMDD_HHMM}  (run_training.py:283)
             nested=False (top-level run from MLflow's perspective)
             mlflow.parentRunId tag set explicitly at run_training.py:289
               │
               └─ NATIVE NESTED (mlflow.start_run(nested=True))
               │
          [#4] Phase run × N  (one per strategy in chain)         opened by MlflowPhaseLogger.start_nested_run
               run_name = phase_{idx}_{strategy}                  (phase_executor/mlflow_logger.py:120)
               tags: phase_idx, strategy_type
               ← HF Trainer's auto-MLflow callback writes train/loss/eval/* HERE
               ← because Phase run is the active run when Trainer.train() executes
               ← also: SystemMetricsCallback writes gpu/cpu/ram metrics HERE on every step
                       (active-run-on-on_step_end semantics)

[#5] Reconciliation (not a separate run — set_terminated on existing #3)
     Called by TrainingMonitor._force_mlflow_run_status              (training_monitor.py:626)
     Reads cancelled.marker / completion.marker from pod-side
     Forces #3 to KILLED or FINISHED when trainer subprocess died
     before HF callback could close cleanly
     ⚠ Only #3 is reconciled. #4 (Phase) has NO reconciliation —
       a kill-9 of trainer leaves Phase RUNNING indefinitely.
```

### Where each kind of data actually lands today

| Data | Run | Writer | Discoverability |
|---|---|---|---|
| HF `train/loss`, `train/learning_rate`, `eval/*` | **#4 Phase run** | HF auto-MLflow callback (active-run semantics) | NON-OBVIOUS — looks like per-strategy in UI |
| GPU/CPU/RAM per-step metrics | **#4 Phase run** | `SystemMetricsCallback.on_step_end` (active-run semantics) | NON-OBVIOUS |
| `system.gpu.count`, `system.gpu.{i}.name`, `system.driver.version` | **#4 Phase run** | `SystemMetricsCallback.on_train_begin` | NON-OBVIOUS |
| Pipeline config params (dotted `config.model.name`, `training.hyperparams.*`) | **#2 Attempt** | `MLflowAttemptManager._log_attempt_metadata` → `log_pipeline_config` | obvious if you know |
| Training config params (snake_case `model_name`, `lora_r`) | **#3 Trainer** | `log_training_config` (called from `run_training.py:307`) | DUPLICATE of #2 with different namespace |
| Config YAML file artifact | **BOTH #2 and #3** (bit-identical) | `mlflow_attempt/manager.py:326-327` + `run_training.py:308` | DOUBLE-WRITE |
| Dataset config params | **#2 Attempt only** | `log_dataset_config` | obvious |
| `mlflow.data.Dataset` linkage (log_input) | **#4 Phase only** | `MlflowPhaseLogger.log_dataset` | NON-OBVIOUS |
| Aggregated `final_train_loss`, `total_train_steps`, `total_train_runtime` | **#2 Attempt** | `ExecutionSummaryReporter.aggregate_training_metrics` (BFS descendants → set on parent) | non-obvious |
| `gpu_name`, `gpu_vram_gb`, `gpu_tier` params | **#3 Trainer** | `log_gpu_detection` (run_training.py:357) | duplicates `provider.gpu_type` tag on #2 with different vocabulary |
| `mm.actual_model_size`, `mm.memory_margin_mb`, model loading params | **#3 Trainer** | `run_training.py:367-379, 429-436` | obvious if you know |
| Per-phase effective hyperparams (`training.hyperparams.actual.*`) | **#4 Phase** | `MlflowPhaseLogger.log_phase_start` | obvious |
| Phase completion metrics (`train_runtime`, `global_step`) | **#4 Phase** | `MlflowPhaseLogger.log_completion` | obvious |
| Pipeline `state.json` artifact | **#2 Attempt** | `MLflowAttemptManager.teardown_attempt` | obvious |
| `events.jsonl` (SSOT typed-event journal) + manifest | **#2 Attempt** under `events/` | `MlflowFinalizer.upload` | obvious |
| Legacy `training_events.json` (`log_summary_artifact`) | **#2 Attempt** (when `MLFLOW_PARENT_RUN_ID` env set), else **#3** | `run_training.py:602-607` | confusing — explicit "push to parent" |
| Registered model (`helix-{model}`, alias=`latest`) | **#3 Trainer** | `mlflow_mgr.register_model` (`run_training.py:536-540`) | obvious |
| `status` tag (FINISHED/FAILED/KILLED) | **#3 Trainer** AND **#4 Phase** AND set via reconciliation | 3+ writers per run, last-writer-wins races | DOUBLE-WRITE on #3 (run_training + ChainRunner) |
| `mlflow.parentRunId` tag | **#2** (auto+explicit), **#3** (explicit tag-only), **#4** (auto) | multiple | by design |

### Top-7 "where is this data?" points of confusion

1. **HF `train/loss` lives on Phase run #4**, not on Trainer run #3 or Attempt run #2 — because HF autologging targets the active run at `Trainer.train()` time, which PhaseExecutor has already replaced with the nested phase run.
2. **GPU per-step metrics also live on Phase #4** for the same reason — even though "GPU" feels like a hardware-of-this-trainer fact, the metrics are scoped to the active phase.
3. **Config exists in 3 places**: #2 Attempt (dotted keys via `log_pipeline_config`), #3 Trainer (snake_case keys via `log_training_config`), and as a YAML artifact on both #2 and #3.
4. **`provider.gpu_type` (#2) vs `gpu_name` (#3) vs `system.gpu.{i}.name` (#4)** — three different vocabularies for "what GPU"; you need to know who wrote which.
5. **Trainer #3 → Attempt #2 linkage is tag-only**. MLflow UI shows it as a tree, but the tree is built from `mlflow.parentRunId` tag set by hand at [run_training.py:289](packages/pod/src/ryotenkai_pod/trainer/run_training.py:289). If the tag fails to write, Trainer #3 floats free at the top level.
6. **Phase runs (#4) have NO reconciliation** — only #3 is force-closed by marker reconciliation. A kill-9 mid-phase leaves the phase RUNNING forever (visible as a stuck child).
7. **Aggregated metrics on Attempt #2** are produced by BFS descent at pipeline end (`ExecutionSummaryReporter.aggregate_training_metrics`) — they appear *after* training, and don't update live. A user looking at Attempt #2 during training sees stale/empty metric history.

### Target state — 3-level native-nested hierarchy, NO trainer top-level run

```
Process: control-plane (Mac)
============================
[#1] Root run                          opened by ParentRunOpener (only if multiple attempts)
     run_name = state.logical_run_id
     tags: ryotenkai.lineage.*, ryotenkai.lifecycle.* (full required taxonomy)
     params: pipeline config (dotted ryotenkai.config.* — ONCE, ONE namespace)
     artifacts: config.yaml (ONCE), events.jsonl + manifest at finalize
       │
       └─ NATIVE NESTED (mlflow.start_run(nested=True))
       │
   [#2] Attempt run                    opened by ParentRunOpener (one per retry)
        run_name = {logical_run_id}_attempt_{N}
        tags: ryotenkai.attempt.id, ryotenkai.attempt.no
        env exported to pod:
             MLFLOW_TRACKING_URI / MLFLOW_RUN_ID = <#2 attempt_id>
             MLFLOW_NESTED_RUN = "TRUE"
             MLFLOW_EXPERIMENT_NAME
             MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = "true"
             MLFLOW_TRACKING_USERNAME/PASSWORD (if auth.kind=basic)
             REQUESTS_CA_BUNDLE (if ca_bundle_path set)
             RYOTENKAI_PHASE_INDEX / RYOTENKAI_PHASE_NAME (if chain — see below)
             │
             ┊ Process boundary — env propagated via SSH SendEnv
             ┊
        Process: pod-trainer subprocess
        ================================
        ⛔  NO Trainer top-level run.
            ⛔ NO mlflow.start_run() in trainer code.
            ⛔ NO mlflow.autolog().
            ⛔ NO double-write of log_pipeline_config / log_dataset_config / config artifact.
        
        HFMlflowWiring:
          - TrainingArguments(report_to=["mlflow"])
          - mlflow.set_system_metrics_node_id(rank or 0)
        
        HF MLflowCallback adopts MLFLOW_RUN_ID + MLFLOW_NESTED_RUN=TRUE and
        creates a nested child of Attempt #2 automatically.
        
        For a chain (SFT → DPO → GRPO), ONE OF TWO PATTERNS (decide in M4):
        
        Pattern P-A: One nested child for the whole chain
        --------------------------------------------------
             │
             └─ HF callback opens [#3'] Training run (single)
                run_name = "{model}_{chain}"
                metrics: train/loss/eval/* across all phases, separated by step axis only
                system metrics: gpu/cpu/ram
                phase boundaries encoded as tags ryotenkai.phase.current
                ↳ Pros: simpler hierarchy; matches HF/community convention
                ↳ Cons: per-phase comparison requires step-window filtering
        
        Pattern P-B: One nested grandchild per phase  (RECOMMENDED)
        ------------------------------------------------------------
             │
             └─ For each phase in chain:
                  ChainRunner exports MLFLOW_RUN_ID=<attempt_id>, MLFLOW_NESTED_RUN=TRUE
                  ChainRunner constructs phase-scoped TrainingArguments
                  HF MLflowCallback creates [#3'] Phase run (one per strategy)
                  run_name = "phase_{idx}_{strategy}"
                  Native MLflow nested=True (because MLFLOW_NESTED_RUN=TRUE)
                  metrics: train/loss/eval/* (HF) + gpu/cpu/ram (native system metrics)
                  ↳ Pros: per-phase isolation, per-phase status, mirrors current #4
                  ↳ Cons: ChainRunner must orchestrate phase env

Reconciliation (set_terminated on existing runs):
  - MlflowFinalizer.finalize is the SINGLE close path; idempotent via
    ryotenkai.lifecycle.finalized tag.
  - Marker-based reconciliation in TrainingMonitor extended to:
      (a) #3' (HF child or per-phase children) — close to KILLED/FAILED if stuck RUNNING
      (b) #2 Attempt — close to FAILED/KILLED via Finalizer
      (c) #1 Root — close on pipeline end
  - Crash-recovery sweep: at next CLI invoke, scan workspace for orphan RUNNING runs
    matching state.json's mlflow_run_id and reconcile.
```

### Target write matrix (recommended Pattern P-B)

| Data | Target run | Writer | Notes |
|---|---|---|---|
| Pipeline config (dotted `ryotenkai.config.*` — single namespace) | **#1 Root** | `ParentRunOpener.open` (once on pipeline init) | inherited by all descendants in MLflow UI search |
| Dataset config | **#1 Root** | `ParentRunOpener.open` | once |
| Config YAML artifact | **#1 Root** | `ParentRunOpener.open` | once |
| `ryotenkai.lineage.*` / `ryotenkai.lifecycle.*` required tags | **#1 Root** + **#2 Attempt** | `ParentRunOpener` | lifecycle tags also on attempt for fast queries |
| Attempt-specific metadata (`ryotenkai.attempt.id/no`) | **#2 Attempt** | `ParentRunOpener.open_attempt` | once per attempt |
| HF `train/loss`, `eval/*` | **#3' Phase run (P-B) or #3' Training run (P-A)** | HF MLflowCallback | native, no manual code |
| GPU/CPU/RAM metrics | **#3'** | native MLflow system-metrics sampler (`MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true`) | rank-zero via `set_system_metrics_node_id` |
| Per-phase tags (`ryotenkai.phase.idx`, `ryotenkai.phase.strategy`) | **#3'** | `HFMlflowWiring` (P-B) or `ChainRunner.tag_active_phase` (P-A) | identifies phase |
| Effective hyperparams per phase (`ryotenkai.training.actual.*`) | **#3'** | `HFMlflowWiring` (one set_params at trainer start) | obvious location |
| Dataset linkage (`mlflow.data.Dataset` log_input) | **#3'** | `HFDatasetLogger` called from trainer | once per phase (P-B) or once per training (P-A) |
| Aggregated `ryotenkai.metric.summary.*` | **#2 Attempt** | `MlflowFinalizer` runs aggregation right before close | written ONCE at finalize, not BFS-late |
| `events.jsonl` (SSOT journal) + sha256 manifest | **#2 Attempt** under `events/` | `JournalUploader` (called by `MlflowFinalizer`) | idempotent |
| Final `state.json` | **#2 Attempt** | `MlflowFinalizer.finalize` | idempotent |
| Registered model with alias `@challenger` | (run_id from HF child) | `ModelPublisher.publish` | aliases not stages |
| `ryotenkai.lifecycle.finalized=true`, `ryotenkai.lifecycle.status` | **all closing runs** (#3', #2, #1) | `MlflowFinalizer.finalize` (idempotent tag-guarded) | single close path |

### Why Pattern P-B is recommended over P-A

- Matches current operator mental model (one phase = one MLflow row).
- Per-phase status (FINISHED/FAILED/KILLED) preserved — already useful for chain debugging.
- Same data-flow for chains of 1 strategy (no special-case).
- Native MLflow nesting (via `MLFLOW_NESTED_RUN=TRUE` + re-setting `MLFLOW_RUN_ID` between phases) — no tag-only linkage, no cross-process tagging hacks.
- The trainer becomes a pure compute-engine: it doesn't open or close ANY MLflow runs; HF callback handles all of that based on env vars set by ChainRunner between phases.
- Phase runs are nested children of Attempt #2 directly — no orphan top-level Trainer run #3 anymore.

The cost: ChainRunner re-exports `MLFLOW_RUN_ID` and toggles `MLFLOW_NESTED_RUN=TRUE` between phase boundaries (or, more simply, lets HF close the previous phase's run and opens a new nested one for the next phase by re-invoking the relevant config sequence). This is a 20-line addition to ChainRunner, not a structural rework.

**Decision**: target architecture adopts Pattern P-B. M4 wires it; M5 model publisher writes against the final phase's HF child.

---

## Responsibility zones (current → target)

### A. Configuration (two configs → unified base + project extension)

**Current state (problem):**

Two Pydantic models exist with asymmetric requiredness and no shared base.

| Model | File | `tracking_uri` | `local_tracking_uri` | `experiment_name` | `ca_bundle_path` | `system_metrics` | `run_description_file` | Token |
|---|---|---|---|---|---|---|---|---|
| `MLflowConfig` | [shared/.../config/integrations/mlflow.py:16](packages/shared/src/ryotenkai_shared/config/integrations/mlflow.py:16) | optional | optional | **required** | optional | block | optional | — |
| `MLflowIntegrationConfig` | [shared/.../config/integrations/mlflow_integration.py:26](packages/shared/src/ryotenkai_shared/config/integrations/mlflow_integration.py:26) | **required** | optional | — | optional | block | — | encrypted `token.enc` |

`MLflowConfig` validates "at least one of tracking_uri/local_tracking_uri must be set" via `model_validator` ([mlflow.py:50-57](packages/shared/src/ryotenkai_shared/config/integrations/mlflow.py:50)). Settings-form-bound `MLflowIntegrationConfig` requires `tracking_uri` strictly. They share only `SystemMetricsConfig` by reference. Token lives encrypted in `workspace/integrations/store.py:36` as `token.enc` and is decrypted at point-of-use (only callsite is `api/services/connection_test.py:67-69`).

**Target state:**

Three layers replace the two-config asymmetry:

1. **`MLflowConnectionConfig`** (`packages/shared/src/ryotenkai_shared/infrastructure/mlflow/config.py`) — base reachability fields:

```python
class MLflowConnectionConfig(StrictBaseModel):
    tracking_uri: str | None              # public endpoint (Tailscale Funnel etc.)
    local_tracking_uri: str | None        # local-loop endpoint (Mac docker bridge etc.)
    ca_bundle_path: str | None
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 30.0
    retry_total_budget_s: float = 30.0
    auth: MLflowAuthConfig                # discriminated: none|basic|bearer
    
    @model_validator(mode="after")
    def _at_least_one_uri(self) -> Self: ...
    
    @field_validator("tracking_uri", "local_tracking_uri")
    @classmethod
    def _reject_userinfo(cls, v: str | None) -> str | None: ...  # R-07
```

2. **`MLflowProjectConfig(MLflowConnectionConfig)`** (`packages/shared/src/ryotenkai_shared/config/integrations/mlflow_project.py`) — adds project fields:

```python
class MLflowProjectConfig(MLflowConnectionConfig):
    experiment_name: str   # required: ^[a-z][a-z0-9_-]*__[a-z][a-z0-9_-]*__[a-z][a-z0-9_-]*$
    run_description_file: str | None
    system_metrics: SystemMetricsConfig
    model_registry_name_template: str = "ryotenkai/{experiment}/{model_family}"
    alias_on_success: str = "challenger"
```

3. **`MLflowAuthConfig`** (`packages/shared/src/ryotenkai_shared/infrastructure/mlflow/auth.py`) — Pydantic discriminated union:

```python
class _AuthNone(StrictBaseModel):
    kind: Literal["none"]

class _AuthBasic(StrictBaseModel):
    kind: Literal["basic"]
    username: str
    password_env_var: str  # name of env var holding the password (never inline)

class _AuthBearer(StrictBaseModel):
    kind: Literal["bearer"]
    token_env_var: str  # name of env var holding the bearer token

MLflowAuthConfig = Annotated[_AuthNone | _AuthBasic | _AuthBearer, Field(discriminator="kind")]
```

The Settings UI uses `MLflowConnectionConfig` (no `experiment_name`); pipeline YAML uses `MLflowProjectConfig`. Token-encrypted-on-disk pattern is preserved but moved to `auth.password_env_var` / `auth.token_env_var` references — the orchestrator decrypts and injects into env before subprocess spawn.

**Removed:** `MLflowIntegrationConfig` (replaced by `MLflowConnectionConfig`). `runtime_role` field on subcomponents (mono-class is being removed).

### B. URI resolution — dual-URI policy explained and kept

**Current state:**

`resolve_mlflow_uris` ([uri_resolver.py:29-55](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/uri_resolver.py:29)) produces 5 derived URIs:

```
effective_local_tracking_uri  = local_tracking_uri OR tracking_uri OR ""
effective_remote_tracking_uri = tracking_uri OR local_tracking_uri OR ""
runtime_tracking_uri:
    control_plane → effective_local_tracking_uri  (env var IGNORED)
    training      → env[MLFLOW_TRACKING_URI] OR effective_remote_tracking_uri
```

The audit confirms the **real use case**:
- `local_tracking_uri` = the same MLflow server addressed via local loopback or docker-bridge from Mac (control-plane).
- `tracking_uri` = the same MLflow server addressed via Tailscale Funnel (public) — needed because the cloud GPU pod cannot reach Mac's `localhost`.

Both URIs point at **one MLflow server**, addressed by two different network paths. The control-plane has the fast/cheap local path; the pod has only the public/funneled path.

Callsites (4 places select URI today):
- [pod/.../trainer/managers/mlflow_manager/setup.py:58](packages/pod/src/ryotenkai_pod/trainer/managers/mlflow_manager/setup.py:58)
- [control/.../pipeline/mlflow_attempt/manager.py:116](packages/control/src/ryotenkai_control/pipeline/mlflow_attempt/manager.py:116)
- [shared/.../infrastructure/mlflow/system_prompt.py:119](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/system_prompt.py:119)
- [control/.../pipeline/stages/managers/deployment/training_launcher.py:566](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/training_launcher.py:566)

Additional URI-stamping happens redundantly in:
- `MLflowGateway.load_prompt` re-calls `mlflow.set_tracking_uri(self._uri)` per fetch ([gateway.py:181](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/gateway.py:181)).
- `MLflowEnvironment.activate` calls `mlflow.set_tracking_uri()` ([environment.py:65](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/environment.py:65)), but `deactivate()` does NOT restore — sticky.
- `training_launcher._build_job_env` writes `MLFLOW_TRACKING_URI` into pod env.

**Target state:**

The dual-URI semantic IS kept — it solves a real problem. The four-place ad-hoc resolution is consolidated:

1. **`RuntimeUriResolver`** in `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/uri.py` (rename from `uri_resolver.py`). Same logic, narrower output:

```python
@dataclass(frozen=True)
class RuntimeUri:
    """Resolved URI for one role on one host. Immutable."""
    uri: str
    role: Literal["control_plane", "training"]

class RuntimeUriResolver:
    @staticmethod
    def for_control_plane(cfg: MLflowConnectionConfig) -> RuntimeUri: ...
    @staticmethod
    def for_training(cfg: MLflowConnectionConfig, env_override: str | None) -> RuntimeUri: ...
```

2. **One stamping point per role:**
   - Control: `control.main` composition root calls `RuntimeUriResolver.for_control_plane(cfg)` once; passes `RuntimeUri` into `MlflowTransport` constructor. `MlflowTransport` calls `mlflow.set_tracking_uri()` ONCE on construction. No re-stamping anywhere.
   - Trainer subprocess: `run_training._setup_mlflow` calls `RuntimeUriResolver.for_training(cfg, env_override=os.getenv("MLFLOW_TRACKING_URI"))` once.

3. **`MlflowAuthAdapter`** injects `Authorization` header into the SDK's HTTP session at the same construction point. Eliminates the `MLFLOW_TRACKING_URI=user:pass@host` antipattern (R-07).

4. **CA bundle**: `MLflowConnectionConfig.ca_bundle_path` set in env ONLY by `training_launcher._build_job_env` for the pod subprocess. In control-plane process, the `MlflowTransport` is configured directly with `verify=ca_bundle_path` argument — no `os.environ` mutation. Removes the dual-writer (in-process + subprocess) collision.

5. **`state.mlflow_runtime_tracking_uri` / `state.mlflow_ca_bundle_path`** in `pipeline_state.json` are kept — they're necessary for `RunDeleter._delete_mlflow_for_run` to reach the right server post-pipeline. Source: `RuntimeUriResolver.for_control_plane` output.

### C. Trainer subprocess MLflow logic — Pattern A straightening

**Current state (the core anti-pattern):**

The trainer ([run_training.py](packages/pod/src/ryotenkai_pod/trainer/run_training.py)) does this:

1. Reads env `MLFLOW_PARENT_RUN_ID` ([run_training.py:287](packages/pod/src/ryotenkai_pod/trainer/run_training.py:287)) — set by control via `training_launcher.py:569-571` from `context[MLFLOW_PARENT_RUN_ID]`, which is the attempt-run id.
2. Constructs its own `MLflowManager(runtime_role="training")` via `_setup_mlflow` ([run_training.py:142-160](packages/pod/src/ryotenkai_pod/trainer/run_training.py:142)).
3. Opens a **brand-new top-level run** with `mlflow.start_run(run_name=f"{model}_{strategy}_{ts}")` ([run_training.py:283](packages/pod/src/ryotenkai_pod/trainer/run_training.py:283)) — **NOT** `nested=True`, **NOT** adopting the parent's id.
4. Tags this new run with `mlflow.parentRunId = MLFLOW_PARENT_RUN_ID` ([run_training.py:289](packages/pod/src/ryotenkai_pod/trainer/run_training.py:289)) — purely cosmetic nesting; MLflow UI groups by tag.
5. Logs training metrics / params / domain logs / system metrics / autolog into the **trainer's own run** (not the parent).
6. Special-cases: `log_summary_artifact(events_artifact_name="training_events.json", parent_run_id=mac_parent_run_id)` ([run_training.py:605-608](packages/pod/src/ryotenkai_pod/trainer/run_training.py:605)) → final journal goes to the parent.
7. `register_model("helix-...", alias="latest", tags=...)` ([run_training.py:535-540](packages/pod/src/ryotenkai_pod/trainer/run_training.py:535)) — model registry call on trainer side.

Effects observed:
- **Two distinct top-level runs per pipeline attempt**: control's `attempt_N` run AND trainer's `{model}_{strategy}_{ts}` run. Reports walk by tag (`mlflow.parentRunId`) — fragile.
- **Double-write of `log_pipeline_config` / `log_dataset_config`**: control logs them to attempt-run ([mlflow_attempt/manager.py:221-222](packages/control/src/ryotenkai_control/pipeline/mlflow_attempt/manager.py:221)), trainer logs them again to its own run ([run_training.py:307](packages/pod/src/ryotenkai_pod/trainer/run_training.py:307)) — same data, different run_ids.
- **Double-write of config-file artifact**: control uploads to attempt-run ([mlflow_attempt/manager.py:326-327](packages/control/src/ryotenkai_control/pipeline/mlflow_attempt/manager.py:326)), trainer uploads to its own run ([run_training.py:308](packages/pod/src/ryotenkai_pod/trainer/run_training.py:308)).
- **Phase executor opens parent AGAIN inside trainer** at [phase_executor/mlflow_logger.py:194](packages/pod/src/ryotenkai_pod/trainer/orchestrator/phase_executor/mlflow_logger.py:194) (`mlflow.start_run(run_id=parent_run_id, nested=False)`) — third independent run-management surface in one process.
- **Closure has 4 paths**: control's `teardown_attempt`, trainer's `finally` block, `set_run_terminated` from Mac for reconciliation, atexit hooks.

**Target state (true Pattern A — community standard):**

Trainer-side single rule: **NEVER call `mlflow.start_run` directly. Let HF MLflowCallback do it via env.**

1. Control writes env for pod subprocess (via `training_launcher._build_job_env`):
   ```
   MLFLOW_TRACKING_URI         = RuntimeUriResolver.for_training(cfg).uri
   MLFLOW_EXPERIMENT_NAME      = cfg.experiment_name
   MLFLOW_RUN_ID               = <attempt_run_id from ParentRunOpener>
   MLFLOW_NESTED_RUN           = "TRUE"                          # uppercase canonical (R-29)
   MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = "true"                  # native (BP #4)
   MLFLOW_TRACKING_USERNAME    = <decrypted if auth.kind=basic>
   MLFLOW_TRACKING_PASSWORD    = <decrypted if auth.kind=basic>
   REQUESTS_CA_BUNDLE          = <ca_bundle_path>                 # if set
   ```
   Plus pod env carrier: `SendEnv MLFLOW_*` / explicit `--export` over SSH (R-03).
2. Trainer's `HFMlflowWiring` does **two things only**:
   - `TrainingArguments(report_to=["mlflow"], ...)` — HF MLflowCallback adopts MLFLOW_RUN_ID and creates a nested child (because MLFLOW_NESTED_RUN=TRUE).
   - `mlflow.set_system_metrics_node_id(local_rank if local_rank is not None else 0)` (R-10).
   - **No `mlflow.start_run` call**, **no `mlflow.autolog()` call** (lint-enforced).
3. `phase_executor.mlflow_logger` is deleted; phase artifacts go through `MetricSink` / `IArtifactSink` keyed to the active run (the HF callback's nested child).
4. Control owns `log_pipeline_config`/`log_dataset_config`/config artifact — **one write per pipeline**, set on the **attempt run** (now the parent of the HF child). Trainer no longer logs these (deleted from `run_training.py:307,308`).
5. Final journal upload remains a control-side responsibility via `MlflowFinalizer` against the attempt run.
6. Model publishing moves to `ModelPublisher.publish(run_id=hf_child.run_id, ...)` called from trainer end-of-train hook; sets `set_registered_model_alias("challenger", version)` (BP #3, replaces `register_model(alias="latest")`).
7. Closure: trainer's HF callback closes the nested child on `Trainer.train()` exit. Control's `Finalizer` (idempotent) closes the attempt run and root run. Atexit handlers converge into the same idempotent `_safe_finalize` (single path).

### D. system_prompt.py — relocate (it's a domain loader, not infrastructure)

**Current state:**

[shared/.../infrastructure/mlflow/system_prompt.py](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/system_prompt.py) implements `SystemPromptLoader.load(llm_cfg, mlflow_cfg, gateway)`:
- Mutually exclusive sources: `InferenceLLMConfig.system_prompt_path` (file) OR `InferenceLLMConfig.system_prompt_mlflow_name` (MLflow Prompt Registry).
- MLflow source accepts `'my-prompt'`, `'prompts:/my-prompt/3'`, `'prompts:/my-prompt@production'`.
- Callsites (3, all inference-time):
  - [control/.../pipeline/stages/model_evaluator.py:316-344](packages/control/src/ryotenkai_control/pipeline/stages/model_evaluator.py:316)
  - [providers/.../runpod/inference/pods/provider.py:1013-1019](packages/providers/src/ryotenkai_providers/runpod/inference/pods/provider.py:1013)
  - [providers/.../single_node/inference/provider.py:620-627](packages/providers/src/ryotenkai_providers/single_node/inference/provider.py:620)
- No cache (hits MLflow every call).
- No fallback to file when MLflow Prompt Registry is unreachable — silent degrade to "no system prompt".

**Why it's misplaced:** the loader knows about `InferenceLLMConfig`, builds audit metadata for `manifest["llm"]["system_prompt_source"]`, picks between two unrelated source kinds (file vs MLflow). That's domain logic, not transport.

**Target state:**

1. Move to `packages/shared/src/ryotenkai_shared/inference/prompts/system_prompt_loader.py`.
2. Keep public API identical (`load(llm_cfg, mlflow_cfg=None, gateway=None) -> SystemPromptResult | None`).
3. Inject `IRunQuery` (or a narrower `IPromptRegistry` Protocol) instead of a concrete `IMLflowGateway`. The loader becomes a domain service with a port to MLflow, not a sibling of the gateway.
4. Add **bounded in-memory cache** keyed by `(name, version_or_alias_resolved)` with TTL 300 s. Inference happens many times per pipeline; current implementation hits MLflow every time. (Q-13 — new.)
5. Add **explicit failure mode toggle** to `InferenceLLMConfig.system_prompt_on_mlflow_failure: Literal["fail", "warn", "fallback_to_file"]` (default `"warn"` preserves current behaviour). Eliminates silent degrade.
6. Required version pinning recommendation in docs: use `prompts:/name/N` or `prompts:/name@alias` for reproducibility — version `latest` is documented as not-for-prod.

### E. environment.py — fix sticky URI, delete dead code

**Current state:**

[shared/.../infrastructure/mlflow/environment.py](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/environment.py):
- `activate()` sets `os.environ[REQUESTS_CA_BUNDLE/SSL_CERT_FILE]` and `mlflow.set_tracking_uri()`.
- `deactivate()` restores env vars BUT does NOT restore prior tracking URI (sticky on process).
- `force_unregister_atexit()` is a static escape hatch — **0 callsites** in `packages/` (verified by grep). Dead code.

**Target state:**

- File deleted entirely.
- URI stamping moves to `MlflowTransport.__init__` (one-shot, per `RuntimeUri` value object).
- CA bundle handled via `verify=` parameter on transport, no env-var mutation in control-plane process.
- Atexit unregister moves into `RunLifecycleCoord.__enter__` (one call site, registered idempotently). The "static escape hatch" pattern is eliminated — `RunLifecycleCoord` is always reachable from signal handler via a module-level `weakref` (R-31 mitigation).

### F. Connectivity probes — 3 paths collapse to 1

**Current state:**
- `MLflowGateway.check_connectivity` ([gateway.py:196](packages/shared/src/ryotenkai_shared/infrastructure/mlflow/gateway.py:196)) — HEAD probe via stdlib urllib, never raises, populates `_last_connectivity_error`.
- `MLflowSetupMixin.check_mlflow_connectivity` ([setup.py:155](packages/pod/src/ryotenkai_pod/trainer/managers/mlflow_manager/setup.py:155)) — calls gateway; used by `MLflowAttemptManager.ensure_preflight`.
- `connection_test._test_mlflow` ([api/services/connection_test.py:56-85](packages/control/src/ryotenkai_control/api/services/connection_test.py:56)) — independent HTTP POST to `/api/2.0/mlflow/experiments/search`, with optional bearer token from `token.enc`. Completely different code path.

**Target state:**
- Single `PreflightConnectivityCheck` in `packages/control/src/ryotenkai_control/pipeline/mlflow/lifecycle/preflight.py`.
- Implementation calls `ITrackingClient.ping(timeout_s)` (Protocol method) + auth verification + experiment existence check + write-probe on a dedicated `__preflight__` run (cleaned up after).
- Settings UI's "Test connection" endpoint replaces direct HTTP probe with `PreflightConnectivityCheck.run(...)`.
- `MLflowGateway` is deleted; its timeout enforcement is absorbed into `MlflowTransport` via tenacity per-attempt timeout (BP #8).

### G. Run lifecycle — 4 closure paths collapse to 1

**Current closure paths (audit-confirmed):**
1. Control: `MLflowAttemptManager.teardown_attempt` → `_attempt_run.__exit__`, `manager.end_run(root_status)`, `manager.cleanup()` ([mlflow_attempt/manager.py:342-400](packages/control/src/ryotenkai_control/pipeline/mlflow_attempt/manager.py:342)).
2. Trainer: `run_training.py` finally → `mlflow_mgr.end_run(status)`, `mlflow_run_context.__exit__`, `mlflow_mgr.cleanup()` ([run_training.py:619-627](packages/pod/src/ryotenkai_pod/trainer/run_training.py:619)).
3. Reconciliation: `MLflowManager.set_run_terminated` ([manager.py:175-222](packages/pod/src/ryotenkai_pod/trainer/managers/mlflow_manager/manager.py:175)) — force-close to KILLED from Mac.
4. atexit: `mlflow._safe_end_run` (unregistered by `MLflowEnvironment` / `_signals.py` duplicately).

**Target closure paths:**
1. Trainer's HF nested child: closed by HF MLflowCallback on `Trainer.train()` exit (blocking flush ensures all metrics arrive before close — R-18 mitigation).
2. Control's attempt run + root run: closed by single idempotent `MlflowFinalizer.finalize(status, journal_path)` from `RunLifecycleCoord` — converges atexit, SIGTERM/SIGINT, normal exit, reconciliation into ONE function with mutex + `ryotenkai.lifecycle.finalized` tag guard.

### H. Summary table — current vs target zones

| Concern | Currently lives in | Target ownership |
|---|---|---|
| URI policy (which URI for which role) | 4 callsites + redundant re-stamping in 3 more | `RuntimeUriResolver` (one source) + `MlflowTransport.__init__` (one stamping point per process) |
| CA bundle delivery | `MLflowEnvironment.activate()` (in-process env) + `training_launcher._build_job_env` (subprocess env) — both writing `REQUESTS_CA_BUNDLE`, neither aware of the other | Control: `MlflowTransport(verify=ca_path)` arg, no env. Pod: `training_launcher` writes pod env only |
| Auth secret | Encrypted `token.enc` decrypted only by `connection_test.py` | `MLflowAuthConfig` discriminated union; secrets via env-var-by-name (`*.env_var`); orchestrator decrypts and exports to pod env |
| Connectivity probe | 3 independent code paths | `PreflightConnectivityCheck` via `ITrackingClient.ping` |
| Parent run open | Control's `MLflowAttemptManager._open_root_run` | `ParentRunOpener.open` (same logic, narrower contract) |
| Nested child run open | Trainer's `MLflowManager.start_run` (currently NOT nested!) | HF MLflowCallback via `MLFLOW_NESTED_RUN=TRUE` env; no manual call |
| `log_pipeline_config` / `log_dataset_config` | Written twice (control + trainer) | Written once by `ParentRunOpener` to the attempt run |
| Config-file artifact | Written twice (control + trainer) | Uploaded once by `ParentRunOpener` |
| Training metrics | Trainer-side via `report_to="mlflow"` (HF) + autolog + manual `log_metric` | HF callback only; autolog forbidden by lint (BP #2) |
| System metrics | Custom `SystemMetricsCallback` | Native via `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true` + `set_system_metrics_node_id(rank)` (BP #4) |
| Model registration | Trainer `register_model(alias="latest")` | `ModelPublisher.publish` with `set_registered_model_alias("challenger", v)` (BP #3) |
| Journal-as-artifact | `MlflowFinalizer` (control-side) | KEPT as-is; logic absorbed into new `Finalizer` |
| Run closure | 4 independent paths | `MlflowFinalizer.finalize` (idempotent, converges all paths) |
| Prompt Registry read | `SystemPromptLoader` in `infrastructure/mlflow/`, 3 callsites | Moves to `shared/inference/prompts/`; cached; explicit failure mode |
| Process URI stamping | `MLflowEnvironment.activate` + `MLflowGateway.load_prompt` (redundant) + `MlflowSetupMixin.setup` | `MlflowTransport.__init__` once; `MLflowEnvironment` deleted |

---

## Target architecture

### Topology

```
[packages/shared/src/ryotenkai_shared/infrastructure/mlflow/]    ← Protocols + low-level
  protocols.py        : ITrackingClient, IRunHandle, IMetricSink,
                        IArtifactSink, IRunQuery, IModelRegistry,
                        IJournalUploader, IPromptRegistry   (≤7 methods each)
  config.py           : MLflowConnectionConfig (base)
  auth.py             : MLflowAuthConfig discriminated union + MlflowAuthAdapter
  uri.py              : RuntimeUriResolver + RuntimeUri (renamed from uri_resolver.py)
  taxonomy.py         : ParamKey/TagKey/MetricKey + ReservedPrefixGuard
  transport.py        : MlflowTransport  (tenacity retry; one set_tracking_uri at __init__)
  dataset.py          : HFDatasetLogger  (mlflow.data.huggingface_dataset facade)
  journal_uploader.py : JournalUploader  (idempotent via sha256-marker tag)
  metric_sink.py      : MetricSink  (log_batch(synchronous=False) + MetricsBuffer reuse)
  dead_letter.py      : DeadLetterBuffer  (bounded; replaces leaky transport state)
  run_handle.py       : RunHandle value-object  (immutable)
  errors.py           : MLflowConnectivityError, MLflowFinalizeError, ...

[packages/shared/src/ryotenkai_shared/config/integrations/]
  mlflow_project.py   : MLflowProjectConfig(MLflowConnectionConfig)  
                        +experiment_name +run_description_file +system_metrics
                        +model_registry_name_template +alias_on_success

[packages/shared/src/ryotenkai_shared/inference/prompts/]
  system_prompt_loader.py : SystemPromptLoader  (relocated from infrastructure/mlflow/)
                            + bounded cache + explicit failure mode

[packages/control/src/ryotenkai_control/pipeline/mlflow/]    ← lifecycle + read
  lifecycle/
    preflight.py      : PreflightConnectivityCheck  (BEFORE start_run; absorbs all
                                                      3 current probe paths)
    opener.py         : ParentRunOpener  (opens attempt run, sets required
                                          ryotenkai.lineage.* tags, writes
                                          log_pipeline_config/log_dataset_config ONCE,
                                          uploads config artifact ONCE,
                                          exports env for pod subprocess)
    coord.py          : RunLifecycleCoord  (atexit/SIGTERM, mutex, finalize)
    finalizer.py      : MlflowFinalizer  (idempotent via ryotenkai.lifecycle.finalized tag)
  read/
    client.py         : MlflowReadClient  (DI'd; replaces 6 ad-hoc sites)
    tree_walker.py    : RunTreeWalker  (single BFS with LRU)

[packages/control/src/ryotenkai_control/reports/]
  composer.py         : ReportComposer
  slices/             : header / metrics / loss_curve / eval / inference / artifacts / lineage

[packages/pod/src/ryotenkai_pod/trainer/mlflow/]    ← thin wiring only
  hf_wiring.py        : HFMlflowWiring  (TrainingArguments.report_to=["mlflow"];
                                         set_system_metrics_node_id;
                                         NEVER calls mlflow.start_run / mlflow.autolog)
  model_publisher.py  : ModelPublisher  (log_model + set_registered_model_alias)
```

### Importlinter contract additions

```ini
[importlinter:contract:mlflow-shared-leaf]
type = forbidden
source_modules = ryotenkai_shared.infrastructure.mlflow
forbidden_modules =
    ryotenkai_control
    ryotenkai_pod
    ryotenkai_providers
    ryotenkai_community

[importlinter:contract:mlflow-control-no-pod]
type = forbidden
source_modules = ryotenkai_control.pipeline.mlflow
forbidden_modules = ryotenkai_pod

[importlinter:contract:mlflow-trainer-no-runner]
type = forbidden
source_modules = ryotenkai_pod.trainer.mlflow
forbidden_modules = ryotenkai_pod.runner

[importlinter:contract:system-prompt-domain]
type = forbidden
source_modules = ryotenkai_shared.inference.prompts
forbidden_modules =
    ryotenkai_control
    ryotenkai_pod
    ryotenkai_providers
```

### Protocols (full signatures)

```python
class RunStatus(StrEnum):
    RUNNING = "RUNNING"; FINISHED = "FINISHED"; FAILED = "FAILED"; KILLED = "KILLED"

@runtime_checkable
class IRunHandle(Protocol):
    run_id: str
    experiment_id: str
    parent_run_id: str | None
    tracking_uri: str
    @property
    def status(self) -> RunStatus: ...

@runtime_checkable
class ITrackingClient(Protocol):
    def ping(self, timeout_s: float) -> None: ...
    def start_run(self, experiment: str, name: str,
                  tags: Mapping[str, str], params: Mapping[str, str]) -> IRunHandle: ...
    def adopt_run(self, run_id: str) -> IRunHandle: ...
    def set_terminated(self, run_id: str, status: RunStatus) -> None: ...
    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None: ...

@runtime_checkable
class IMetricSink(Protocol):
    def log(self, run_id: str, metrics: Mapping[str, float], step: int) -> None: ...
    def flush(self, run_id: str, blocking: bool) -> None: ...

@runtime_checkable
class IArtifactSink(Protocol):
    def upload_file(self, run_id: str, local_path: Path, artifact_path: str,
                    checksum_sha256: str | None) -> None: ...

@runtime_checkable
class IRunQuery(Protocol):
    def get_run(self, run_id: str) -> IRunHandle: ...
    def list_children(self, parent_run_id: str) -> Sequence[IRunHandle]: ...
    def search(self, experiment: str, filter_: str, max_results: int) -> Sequence[IRunHandle]: ...

@runtime_checkable
class IModelRegistry(Protocol):
    def register(self, model_uri: str, name: str) -> ModelVersion: ...
    def set_alias(self, name: str, alias: str, version: str) -> None: ...
    def resolve_alias(self, name: str, alias: str) -> ModelVersion: ...

@runtime_checkable
class IJournalUploader(Protocol):
    def upload(self, run_id: str, journal_path: Path, sha256: str) -> None: ...

@runtime_checkable
class IPromptRegistry(Protocol):
    def load(self, name_or_uri: str, timeout_s: float) -> PromptArtifact | None: ...
```

### Write path (Pattern A, straightened)

```
control.RunLifecycleCoord.run():
  ├─ 1. PreflightConnectivityCheck.run()
  │     ITrackingClient.ping() + auth + experiment exists + write-probe
  ├─ 2. handle = ParentRunOpener.open(experiment, name, tags, params)
  │     ├─ start_run on attempt
  │     ├─ set required ryotenkai.lineage.* + ryotenkai.lifecycle.* tags
  │     ├─ log_pipeline_config + log_dataset_config  (ONCE)
  │     └─ upload config artifact                    (ONCE)
  ├─ 3. export env to pod-runner subprocess:
  │       MLFLOW_TRACKING_URI = RuntimeUriResolver.for_training(cfg).uri
  │       MLFLOW_EXPERIMENT_NAME = cfg.experiment_name
  │       MLFLOW_RUN_ID = handle.run_id
  │       MLFLOW_NESTED_RUN = "TRUE"                   ← uppercase (R-29)
  │       MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = "true"
  │       MLFLOW_TRACKING_USERNAME / _PASSWORD         ← decrypted from token store
  │       REQUESTS_CA_BUNDLE = ca_path                 ← if set; pod env only
  ├─ 4. spawn pod-runner; env propagates to trainer subprocess via SSH SendEnv / REST
  ├─ 5. pod.trainer.HFMlflowWiring:
  │       TrainingArguments(report_to=["mlflow"])      ← HF callback creates nested child
  │       mlflow.set_system_metrics_node_id(rank or 0)
  │       (NO mlflow.start_run, NO mlflow.autolog)
  ├─ 6. trainer continues to emit ryotenkai events via RunnerEventCallback → journal SSOT
  └─ 7. Finalizer.finalize(handle.run_id, status, journal_path)
        ├─ idempotency check: ryotenkai.lifecycle.finalized tag
        ├─ JournalUploader.upload(journal_path, sha256)  → ryotenkai.journal.sha256 tag
        ├─ ITrackingClient.set_tags(lifecycle.finalized=true, lifecycle.status, exit.reason)
        └─ ITrackingClient.set_terminated(handle.run_id, status)
```

### Read path

- All read sites receive `MlflowReadClient` via DI from the composition root in `control.main` (lint-enforced: no ad-hoc `MlflowClient()`).
- `RunTreeWalker.walk(parent_run_id) -> RunTree`: single BFS with per-process LRU(maxsize=64). Replaces three independent walks:
  - [mlflow_adapter._get_sorted_children](packages/control/src/ryotenkai_control/reports/adapters/mlflow_adapter.py:358)
  - [deletion._search_child_run_ids](packages/control/src/ryotenkai_control/pipeline/deletion.py:120)
  - [summary_reporter.collect_descendant_metrics](packages/control/src/ryotenkai_control/pipeline/reporting/summary_reporter.py:285)
- Batch/parallel/cache:
  - `IRunQuery.search` paginates with `max_results>=200`.
  - `ThreadPoolExecutor(max_workers=8)` scoped per `ReportGenerator` (context-managed; fixes R-22 FD leak).
  - Per-report LRU cache for immutable FINISHED runs.
  - Target: **≤20 HTTP round-trips per report** (down from 120).
- Schema verification: `Taxonomy.assert_known(tag_key)` on read; unknown required tag missing → graceful degrade with warning slice.

### Taxonomy

Convention: `ryotenkai.<domain>.<area>.<verb_or_noun>` (lowercase dotted) — matches ADR-0009 `kind` format; MLflow UI groups by prefix.

**Required tags (set by `ParentRunOpener`):**
- `ryotenkai.lineage.pipeline_id`, `ryotenkai.lineage.run_id`, `ryotenkai.lineage.config_sha256`, `ryotenkai.lineage.code_commit`
- `ryotenkai.lifecycle.opened_by` (`host:user`)
- `ryotenkai.engine.kind` (`sft|dpo|grpo|sapo`)
- `ryotenkai.provider.kind`, `ryotenkai.provider.gpu`

**Required tags (set by `MlflowFinalizer`):**
- `ryotenkai.lifecycle.finalized=true`, `ryotenkai.lifecycle.status`, `ryotenkai.journal.sha256`, `ryotenkai.exit.reason` (if ≠ FINISHED)

**Reserved prefixes (`ReservedPrefixGuard.assert_safe`):**
- `mlflow.*` — blocked except whitelist: `mlflow.note.content`, `mlflow.runName`, `mlflow.parentRunId`, `mlflow.source.*`, `mlflow.user`.
- `ryotenkai.*` — our namespace, validated against `taxonomy.py` enum.
- `hf.*` — HF Trainer automatic tags, accepted as-is.

Metrics: HF Trainer-emitted (`train/loss`, `eval/*`) accepted. Custom: `ryotenkai.metric.<area>.<name>`.

### Infrastructure (auth-only scope per user direction)

| Item | Status | Action |
|---|---|---|
| Anonymous MinIO read | **fix** | Remove `mc anonymous set download minio/mlflow` from [docker-compose.mlflow.yml:54-61](docker/mlflow/docker-compose.mlflow.yml:54) |
| MLflow without auth in front of Funnel | **fix** | Add Caddy reverse proxy with basic-auth in front of MLflow server; Funnel targets Caddy |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` optional | **fix** | Fail-fast in `entrypoint.mlflow.sh` if unset when `MLFLOW_SERVER_ALLOWED_HOSTS` is non-empty |
| Auth in config | **add** | `MLflowAuthConfig` discriminated union; secrets via env-var-by-name; `MlflowAuthAdapter` injects header |
| Token storage (encrypted `token.enc`) | **keep** | Orchestrator decrypts and exports to pod env; on-disk format unchanged |
| Backup (pg_dump, MinIO versioning) | **out of scope** | Per user direction (local workspace) |
| `mlflow gc` cron | **defer** | Not needed without retention pressure |

---

## Migration phases

Priority per user: **code first, infra parallel**.

### Phase M1 — Shared foundations *(blocks: none)*

Create in `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/`:
- `protocols.py`, `config.py` (`MLflowConnectionConfig`), `auth.py` (config + adapter), `uri.py` (renamed; `RuntimeUriResolver` + `RuntimeUri`), `taxonomy.py`, `transport.py`, `dataset.py`, `journal_uploader.py`, `metric_sink.py`, `dead_letter.py`, `run_handle.py`, `errors.py`.

Create in `packages/shared/src/ryotenkai_shared/config/integrations/`:
- `mlflow_project.py` (`MLflowProjectConfig(MLflowConnectionConfig)`).

Create in `packages/shared/src/ryotenkai_shared/inference/prompts/`:
- `system_prompt_loader.py` (relocated from `infrastructure/mlflow/`; +cache; +failure-mode toggle).

Per-file unit tests (7-class structure following [test_preempt_inference_container.py](tests/unit/providers/single_node/training/test_preempt_inference_container.py)).

Canonical fakes in `tests/_fakes/`: `tracking_client.py`, `metric_sink.py`, `artifact_sink.py`, `run_query.py`, `model_registry.py`, `journal_uploader.py`, `prompt_registry.py`.

Add 4 importlinter contracts (see §Importlinter).

### Phase M2 — Control lifecycle ownership *(blocks: M1; ATOMIC with M4)*

Create:
- `packages/control/src/ryotenkai_control/pipeline/mlflow/__init__.py`
- `pipeline/mlflow/lifecycle/{preflight,opener,coord,finalizer}.py`

Migrate logic:
- `_open_root_run` → `ParentRunOpener.open`
- `_open_attempt_run` → `ParentRunOpener.open_attempt`
- `_log_attempt_metadata` → `ParentRunOpener` (with one-shot log_pipeline_config/log_dataset_config; deduplicated from trainer side)
- `ensure_preflight` → `PreflightConnectivityCheck`
- `teardown_attempt` → `MlflowFinalizer.finalize`
- `MlflowFinalizer` (current) journal upload logic → new `Finalizer.finalize` step

Delete:
- `packages/control/src/ryotenkai_control/pipeline/mlflow_attempt/` (entire directory; closes layering breach)
- `packages/control/src/ryotenkai_control/events/mlflow_finalizer.py` (logic absorbed into new `Finalizer`)
- Backward-compat shim `orchestrator._mlflow_manager` property+setter ([orchestrator.py:241-254](packages/control/src/ryotenkai_control/pipeline/orchestrator.py:241))
- Duplicate atexit-unregister in `cli/_signals.py` (consolidate into `RunLifecycleCoord`)

### Phase M3 — Control read decomposition *(blocks: M1)*

Create:
- `pipeline/mlflow/read/client.py` (`MlflowReadClient`), `pipeline/mlflow/read/tree_walker.py` (`RunTreeWalker`)
- `reports/composer.py` (`ReportComposer`)
- `reports/slices/{header,metrics,loss_curve,eval,inference,artifacts,lineage}.py`

Replace ad-hoc `MlflowClient()`:
- [reports/adapters/mlflow_adapter.py:141,145](packages/control/src/ryotenkai_control/reports/adapters/mlflow_adapter.py:141)
- [reports/report_generator.py:83,88](packages/control/src/ryotenkai_control/reports/report_generator.py:83)
- [pipeline/stages/model_retriever/retriever.py:649-665](packages/control/src/ryotenkai_control/pipeline/stages/model_retriever/retriever.py:649)
- [pipeline/stages/training_monitor.py:605-608](packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py:605)
- [pipeline/deletion.py:120-175](packages/control/src/ryotenkai_control/pipeline/deletion.py:120)
- [pipeline/reporting/summary_reporter.py:285-322](packages/control/src/ryotenkai_control/pipeline/reporting/summary_reporter.py:285)

Delete:
- `_get_sorted_children`, `_search_child_run_ids`, `collect_descendant_metrics` (replaced by `RunTreeWalker`)
- `reports/core/builder.py` god-class (replaced by `ReportComposer + slices/`)
- Phase 7 legacy shims at [builder.py:300,672-697,795](packages/control/src/ryotenkai_control/reports/core/builder.py:300)
- Import of `StageArtifactEnvelope` from pipeline in mlflow_adapter ([mlflow_adapter.py:158](packages/control/src/ryotenkai_control/reports/adapters/mlflow_adapter.py:158)) — move type to `shared.contracts` or `reports.domain`.

### Phase M4 — Pod trainer rewire (Pattern A) *(blocks: M2; ATOMIC PR with M2)*

Critical: M2 + M4 ship in one atomic PR (R-20). Feature flag `RYOTENKAI_MLFLOW_PATTERN_A=1` enables both sides simultaneously.

Create:
- `packages/pod/src/ryotenkai_pod/trainer/mlflow/hf_wiring.py` (`HFMlflowWiring`)
- `packages/pod/src/ryotenkai_pod/trainer/mlflow/model_publisher.py` (`ModelPublisher`; M5 fills out)

Modify:
- [training_launcher.py:566-576](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/training_launcher.py:566) — Pattern A env propagation (add `MLFLOW_RUN_ID`, `MLFLOW_NESTED_RUN=TRUE`, `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true`, `MLFLOW_TRACKING_USERNAME/PASSWORD`)
- [run_training.py](packages/pod/src/ryotenkai_pod/trainer/run_training.py) — remove `_setup_mlflow`, `mlflow.start_run`, autolog enable, manual `log_pipeline_config`/`log_dataset_config`/config-artifact-upload. Trainer becomes: HF wiring + ryotenkai event emission + (M5) model publish.

Delete:
- `packages/pod/src/ryotenkai_pod/trainer/managers/mlflow_manager/` (god-class facade + 4 mixin files)
- `packages/pod/src/ryotenkai_pod/trainer/mlflow/{primitives,autolog,_classifier_bootstrap,domain_logger,dataset_logger,model_registry,run_analytics,resilient_transport}.py`
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/system_metrics_callback.py` (native takes over)
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/cancellation_callback.py` (merged into `completion_callback.py` → `TerminalCallback`)
- `packages/pod/src/ryotenkai_pod/trainer/orchestrator/phase_executor/mlflow_logger.py` (third run-management surface; deleted)
- All Phase 7 no-op stubs ([domain_logger.py:325-401](packages/pod/src/ryotenkai_pod/trainer/mlflow/domain_logger.py:325))
- `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/environment.py` (URI stamping moves to `MlflowTransport.__init__`; dead `force_unregister_atexit` deleted)
- `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/gateway.py` (logic absorbed into `MlflowTransport`)

Keep:
- `packages/pod/src/ryotenkai_pod/trainer/mlflow/metrics_buffer.py` (consumed by new `MetricSink`)
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/runner_event_callback.py` (typed envelopes — ADR-0009)
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/_flush_helper.py`

Update or delete `packages/pod/src/ryotenkai_pod/runner/mlflow_relay.py` based on M4 decision (current: dormant by default; if no use case remains, delete the file plus the conditional in `runner/main.py:394-419`).

### Phase M5 — Model registry → aliases *(blocks: M4)*

- `ModelPublisher.publish(run_id, model_uri)`:
  - `mlflow.transformers.log_model(..., save_pretrained=True)` (R-21)
  - `MlflowClient.set_registered_model_alias(name, "challenger", version)`
- CLI: `ryotenkai model promote --name <name> --version <v>` → sets `champion` alias (manual gate per Q-12).
- Delete all references to `Staging` / `Production` / `Archived` stages.

### Phase M6 — Infrastructure hardening (auth only) *(parallel)*

Per user direction: backup deferred; auth + critical security fixes in scope.

Modify `docker/mlflow/`:
- `docker-compose.mlflow.yml:54-61` — **remove** anonymous policy. Add explicit MinIO IAM for `mlflow-server` user.
- Add Caddy service:
  - Image: `caddy:2-alpine`
  - Listens on external `:5002`; reverse-proxies to `mlflow-server:5102` (internal).
  - Basic-auth via env: `CADDY_BASIC_AUTH_USER` / `CADDY_BASIC_AUTH_HASH` (bcrypt).
- Move MLflow internal port (current `:5002` → `:5102`).
- `expose-tailscale.sh` — target Caddy, not MLflow directly.
- `entrypoint.mlflow.sh` — fail-fast if `MLFLOW_FLASK_SERVER_SECRET_KEY` unset when `MLFLOW_SERVER_ALLOWED_HOSTS` set.

Client side: `MlflowAuthAdapter` consumes `MLflowAuthConfig` and injects `Authorization` header via `MlflowTransport`.

### Phase M7 — Lint enforcement & cleanup *(blocks: M1..M6)*

Add lint rules in `scripts/lint/mlflow_rules.py`:
1. Forbid `mlflow.autolog(` anywhere.
2. Forbid `mlflow.set_tracking_uri(` outside `packages/*/src/*/main.py` and `MlflowTransport.__init__`.
3. Forbid `mlflow.tracking.MlflowClient(` / `mlflow.MlflowClient(` outside `transport.py` and `read/client.py`.
4. Forbid `mlflow.start_run(` in `packages/pod/src/ryotenkai_pod/trainer/` (HF callback only).

Wire into pre-commit + CI.

Run `bash scripts/mutation/validate_agent_output.sh` against all new files.

---

## Risk register (top 10; full 32 in helixir memory `mem_5c41ff4b4249`)

| ID | Risk | Sev | Mitigation |
|---|---|---|---|
| R-01 | HF MLflowCallback misinterprets `MLFLOW_NESTED_RUN`, overwrites parent | High | Uppercase canonical `TRUE`; IT-test asserts child created with `tags.mlflow.parentRunId`; Finalizer verifies parent still RUNNING |
| R-03 | `MLFLOW_RUN_ID` lost over SSH non-interactive shell | High | Explicit `--export` / `SendEnv MLFLOW_*`; IT-test for env inheritance |
| R-04 | atexit `set_terminated` fails when MLflow offline → run stays RUNNING | High | Dead-letter `~/.ryotenkai/orphan_runs.jsonl` + reconciliation `ryotenkai runs reconcile` CLI |
| R-07 | Secret leak via `tracking_uri` userinfo | High | Pydantic validator rejects userinfo; credentials only via env-var-by-name |
| R-11 | HF Trainer 5.x API drift | High | Pin HF version in `pyproject.toml`; CI smoke test "nested child created" |
| R-17 | Race: SIGTERM handler vs normal finalize | High | `threading.Lock` + idempotency via `ryotenkai.lifecycle.finalized` tag |
| R-18 | HF callback flushes metrics async after `Trainer.train()` returns → metrics land on closed run | Crit | Wiring forces blocking `on_train_end` flush; control waits N seconds after trainer exit |
| R-19 | Pod-runner and pod-trainer concurrent journal writers | High | Single-writer invariant (trainer only); lockfile sentinel |
| R-20 | Half-migration (M2 done, M4 not) → orphan top-level runs | High | M2+M4 atomic PR; `RYOTENKAI_MLFLOW_PATTERN_A=1` flag controls both sides |
| R-26 | Caddy auth token expiry mid-run | High | Long-lived service token; transport detects 401 → fail-fast (no retry); rotation outside training |

---

## Open questions (with recommendations)

| ID | Question | Recommendation |
|---|---|---|
| Q-02 | Resume strategy | Adopt only if `prev_status == RUNNING`; else new run + `ryotenkai.lineage.resumes_from` tag |
| Q-09 | OAuth2-proxy in Phase 1? | Basic-auth via Caddy now; OAuth2 post-MVP |
| Q-11 | `set_experiment_tag` for cross-run lineage? | Yes — cheap UI value |
| Q-12 | Secondary HF MLflow callback for custom events? | No — journal SSOT; `report_to="mlflow"` only |
| Q-13 (new) | `SystemPromptLoader` cache TTL? | 300 s default; per-call override via `InferenceLLMConfig.system_prompt_cache_ttl_s` |
| Q-14 (new) | `MLflowConnectionConfig` validation: should both URIs reject same value (dev-only setup)? | No — same URI for both fields is legitimate when control + pod are on same host (e.g. local dev) |

---

## Files summary

### Created

`packages/shared/src/ryotenkai_shared/infrastructure/mlflow/`: `protocols.py`, `config.py`, `auth.py`, `uri.py` (rename), `taxonomy.py`, `transport.py`, `dataset.py`, `journal_uploader.py`, `metric_sink.py`, `dead_letter.py`, `run_handle.py`, `errors.py`

`packages/shared/src/ryotenkai_shared/config/integrations/mlflow_project.py`

`packages/shared/src/ryotenkai_shared/inference/prompts/system_prompt_loader.py`

`packages/control/src/ryotenkai_control/pipeline/mlflow/`: `__init__.py`, `lifecycle/{preflight,opener,coord,finalizer}.py`, `read/{client,tree_walker}.py`

`packages/control/src/ryotenkai_control/reports/composer.py`, `reports/slices/{header,metrics,loss_curve,eval,inference,artifacts,lineage}.py`

`packages/pod/src/ryotenkai_pod/trainer/mlflow/`: `hf_wiring.py`, `model_publisher.py`

`tests/_fakes/`: `tracking_client.py`, `metric_sink.py`, `artifact_sink.py`, `run_query.py`, `model_registry.py`, `journal_uploader.py`, `prompt_registry.py`

`scripts/lint/mlflow_rules.py`

`docker/mlflow/Caddyfile`

### Modified

- [orchestrator.py](packages/control/src/ryotenkai_control/pipeline/orchestrator.py) — remove `_mlflow_manager` shim; integrate `RunLifecycleCoord`
- [training_launcher.py:566-576](packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment/training_launcher.py:566) — Pattern A env propagation
- [run_lifecycle_coordinator.py](packages/control/src/ryotenkai_control/pipeline/run_lifecycle_coordinator.py) — delegate finalize
- [cli/_signals.py](packages/control/src/ryotenkai_control/cli/_signals.py) — remove duplicate atexit-unregister
- [api/services/connection_test.py](packages/control/src/ryotenkai_control/api/services/connection_test.py) — replace with `PreflightConnectivityCheck` call
- [run_training.py](packages/pod/src/ryotenkai_pod/trainer/run_training.py) — replace `_setup_mlflow` + manual run open with `HFMlflowWiring`
- [trainers/factory.py](packages/pod/src/ryotenkai_pod/trainer/trainers/factory.py) — remove `SystemMetricsCallback` registration (native takes over)
- [model_evaluator.py:316-344](packages/control/src/ryotenkai_control/pipeline/stages/model_evaluator.py:316), [runpod/inference/pods/provider.py:1013-1019](packages/providers/src/ryotenkai_providers/runpod/inference/pods/provider.py:1013), [single_node/inference/provider.py:620-627](packages/providers/src/ryotenkai_providers/single_node/inference/provider.py:620) — update `SystemPromptLoader` import path (relocated)
- [docker/mlflow/docker-compose.mlflow.yml](docker/mlflow/docker-compose.mlflow.yml), [entrypoint.mlflow.sh](docker/mlflow/entrypoint.mlflow.sh), [expose-tailscale.sh](docker/mlflow/expose-tailscale.sh)
- `pyproject.toml` — pin `mlflow>=3.0,<3.6`, pin `transformers`
- `.importlinter` (or `[tool.importlinter]`) — add 4 new contracts

### Deleted

- `packages/control/src/ryotenkai_control/pipeline/mlflow_attempt/` (closes layering breach)
- `packages/control/src/ryotenkai_control/events/mlflow_finalizer.py`
- `packages/pod/src/ryotenkai_pod/trainer/managers/mlflow_manager/`
- `packages/pod/src/ryotenkai_pod/trainer/mlflow/primitives.py`, `autolog.py`, `_classifier_bootstrap.py`, `domain_logger.py`, `dataset_logger.py`, `model_registry.py`, `run_analytics.py`, `resilient_transport.py`
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/system_metrics_callback.py`, `cancellation_callback.py` (merged)
- `packages/pod/src/ryotenkai_pod/trainer/orchestrator/phase_executor/mlflow_logger.py`
- `packages/control/src/ryotenkai_control/reports/core/builder.py`
- `packages/shared/src/ryotenkai_shared/config/integrations/mlflow_integration.py` (replaced by `MLflowConnectionConfig`)
- `packages/shared/src/ryotenkai_shared/config/integrations/mlflow.py` (replaced by `MLflowProjectConfig`)
- `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/system_prompt.py` (relocated)
- `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/environment.py` (URI stamping → `MlflowTransport`; dead `force_unregister_atexit` removed)
- `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/gateway.py` (logic → `MlflowTransport`)

### Preserved (strong assets)

- `packages/pod/src/ryotenkai_pod/trainer/mlflow/metrics_buffer.py`
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/_flush_helper.py`
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/runner_event_callback.py`
- `packages/control/src/ryotenkai_control/events/` (ADR-0009 — entire journal subsystem)

---

## Verification

### Per-phase smoke gates (DoD per CLAUDE.md)

```bash
.venv/bin/python -m pytest tests/_lint -q                       # sentinel suite green
.venv/bin/python -m pytest tests/unit -q                        # 0 failed on touched lane
bash scripts/mutation/validate_agent_output.sh                  # mutation kill-rate ≥ threshold
.venv/bin/python -m lint_imports                                # importlinter contracts pass
```

### Pattern A integration test (M2+M4)

```bash
.venv/bin/python -m pytest tests/integration/mlflow/test_pattern_a_lifecycle.py -q
```

Asserts:
1. `ParentRunOpener.open()` creates root run with all required `ryotenkai.lineage.*` + `ryotenkai.lifecycle.*` tags.
2. Pod-trainer subprocess receives `MLFLOW_RUN_ID` + `MLFLOW_NESTED_RUN=TRUE`.
3. HF MLflowCallback creates nested child with `tags.mlflow.parentRunId == root_run_id`.
4. After `Trainer.train()` returns, **parent run is still RUNNING** (NOT closed by callback).
5. `log_pipeline_config` / `log_dataset_config` / config artifact appear **once** on parent run (no duplicates on trainer's child run).
6. `MlflowFinalizer.finalize(FINISHED)` closes parent; `set_terminated` is idempotent on second call (tag-guard).
7. SIGTERM triggers `Finalizer.finalize(KILLED)`; tag `ryotenkai.lifecycle.status == "KILLED"`.

### Dual-URI resolution test (M1)

```bash
.venv/bin/python -m pytest tests/unit/shared/infrastructure/mlflow/test_runtime_uri_resolver.py -q
```

Asserts:
1. `for_control_plane(cfg)` ignores `MLFLOW_TRACKING_URI` env (uses `effective_local`).
2. `for_training(cfg, env_override=X)` returns X over config.
3. Both URIs same value → both roles resolve to same URI (Q-14).
4. Both URIs None → `MLflowConnectionConfig` validation rejects at construct time.
5. `tracking_uri` with userinfo → rejected (R-07).

### system_prompt cache test (M1)

```bash
.venv/bin/python -m pytest tests/unit/shared/inference/prompts/test_system_prompt_loader.py -q
```

Asserts:
1. Second `load()` within TTL → no `IPromptRegistry.load` call.
2. `system_prompt_on_mlflow_failure="fail"` → raises on MLflow error.
3. `system_prompt_on_mlflow_failure="fallback_to_file"` + file configured + MLflow down → returns file content.

### Report-path benchmark (M3)

```bash
.venv/bin/python -m pytest tests/integration/reports/test_report_round_trips.py -q
```

Asserts: ≤20 HTTP round-trips per report (via `FakeRunQuery` counter).

### Infrastructure smoke (M6)

```bash
curl -i http://localhost:5002/api/2.0/mlflow/experiments/list                        # → 401
curl -i -u user:pass http://localhost:5002/api/2.0/mlflow/experiments/list           # → 200
curl -i http://localhost:9000/mlflow/some/artifact.bin                               # → 403 (anonymous closed)
```

### Lint enforcement (M7)

```bash
.venv/bin/python scripts/lint/mlflow_rules.py packages/   # → exit 0
```

Rejects: any `mlflow.autolog`, `mlflow.set_tracking_uri` outside whitelist, `MlflowClient(` outside whitelist, `mlflow.start_run` in `packages/pod/src/ryotenkai_pod/trainer/`.

### Feature-flag rollout (M2+M4 atomic)

```bash
# Old path (no flag) — keeps existing behaviour until both sides ready
RYOTENKAI_MLFLOW_PATTERN_A=0 ryotenkai run examples/configs/sft_smoke.yaml

# New path (flag on) — atomic enable on control + pod
RYOTENKAI_MLFLOW_PATTERN_A=1 ryotenkai run examples/configs/sft_smoke.yaml
```

After M7 lands, flag is removed; Pattern A is the only path.

---

## Out of scope (deferred per user direction)

- PostgreSQL backup (pg_dump cron + replication)
- MinIO bucket versioning + cross-region replication
- `mlflow gc` cron + retention policy
- OAuth2-proxy (basic-auth via Caddy is sufficient now)
- High-availability MLflow deployment
- Restore-test job

Revisit if the workspace moves beyond local-only.
