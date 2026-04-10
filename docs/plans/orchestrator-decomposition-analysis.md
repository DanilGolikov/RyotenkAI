# Pipeline Orchestrator Decomposition Analysis

**Thoroughness Level**: Very Thorough  
**Date**: 2026-04-10  
**File Under Analysis**: `/Users/daniil/MyProjects/RyotenkAI/src/pipeline/orchestrator.py`

---

## Executive Summary

**File Metrics:**
- **Line Count**: 2,266 lines
- **Class Count**: 1 (PipelineOrchestrator) + 1 exception class (LaunchPreparationError)
- **Method Count**: 56 methods (56 total, 2 public, 54 private)
- **Complexity**: Bug-prone (hotspot score 95%), +2,339 lines in 90 days vs only 34 deleted
- **Status**: HEAVILY DECOMPOSABLE — clear logical groupings exist that can be extracted with minimal API changes

---

## 1. Structure: Classes and Methods by Responsibility

### 1.1 PipelineOrchestrator Class Overview

#### Public Interface (2 methods)
1. **`run()`** → Main entry point, delegates to `_run_stateful()`
2. **`notify_signal(signal_name)`** → External shutdown signal handler

#### Query/List Methods (3 methods)
- `list_stages()` — Return list of stage names
- `list_restart_points(run_dir)` — Query restart point information
- `get_stage_by_name(name)` — Retrieve stage instance by name

#### Constructor (1 method)
- `__init__()` → 118 lines of configuration loading, secret validation, stage initialization

---

### 1.2 Method Grouping by Logical Concern (7 Categories)

#### A. **STATE MANAGEMENT** (7 methods, ~20 lines each)
**Concern**: Mark stage status changes in `PipelineAttemptState`

```
_mark_stage_running()         # Set status=RUNNING
_mark_stage_completed()       # Set status=COMPLETED + copy outputs
_mark_stage_failed()          # Set status=FAILED + error message
_mark_stage_skipped()         # Set status=SKIPPED + reason
_mark_stage_interrupted()     # Set status=INTERRUPTED + timestamp
_finalize_attempt_state()     # Mark attempt finished
_save_state()                 # Persist state to disk
```

**Extraction Candidate**: STRONG — Self-contained, no orchestration logic
**Estimated Lines**: ~140 lines total
**Public API Impact**: LOW (all called internally from `_run_stateful`)

---

#### B. **MLFLOW INTEGRATION** (6 methods, ~40-50 lines each)
**Concern**: MLflow parent/nested run lifecycle, preflight checks, event logging

```
_setup_mlflow()               # Initialize MLflow manager (36 lines)
_setup_mlflow_for_attempt()   # Create attempt-level nested run (54 lines)
_ensure_mlflow_preflight()    # Validate MLflow connectivity (53 lines)
_teardown_mlflow_attempt()    # End nested run with status (41 lines)
_get_mlflow_run_id()          # Query root run ID (20 lines)
_open_existing_root_run()     # Reopen root MLflow run (10 lines)
```

**Extraction Candidate**: STRONG — Distinct responsibility, minimal cross-dependencies
**Estimated Lines**: ~220 lines total
**Public API Impact**: LOW (all called from `_run_stateful` or `__init__`)

---

#### C. **VALIDATION ARTIFACT HANDLING** (9 methods, ~30-50 lines each)
**Concern**: Capture dataset validation plugin results via callbacks; build state outputs

```
_on_dataset_scheduled()                    # Initialize per-dataset accumulator (11 lines)
_on_dataset_loaded()                       # Update sample count/critical failures (7 lines)
_on_validation_completed()                 # Mark dataset as passed (7 lines)
_on_validation_failed()                    # Mark dataset as failed (7 lines)
_on_plugin_start()                         # Cache plugin description (5 lines)
_on_plugin_complete()                      # Append passed plugin result (27 lines)
_on_plugin_failed()                        # Append failed plugin result (30 lines)
_flush_validation_artifact()               # Write validation_artifact_ref.json (31 lines)
_build_dataset_validation_state_outputs()  # Extract state outputs dict (42 lines)
```

**Extraction Candidate**: STRONG — Clean event callback pattern
**Estimated Lines**: ~170 lines total
**Accumulated Data Structures**:
- `self._validation_accumulator` (dict[str, ValidationDatasetData])
- `self._validation_plugin_descriptions` (dict[tuple[str, str], str])

**Public API Impact**: MEDIUM (callbacks injected into DatasetValidator; public outputs used in state)

---

#### D. **REPORTING & METRICS** (4 methods, ~40-60 lines each)
**Concern**: Aggregate training metrics from stages; generate summary/experiment report

```
_aggregate_training_metrics()      # Collect metrics from MLflow (59 lines)
_collect_descendant_metrics()      # Recursive metric descendant collection (61 lines)
_generate_experiment_report()      # Create report via ExperimentReportGenerator (40 lines)
_print_summary()                   # Format + print summary table (151 lines)
```

**Extraction Candidate**: STRONG — Isolated from main execution flow
**Estimated Lines**: ~310 lines total
**Called**: End of pipeline (success case), not in main loop

**Public API Impact**: LOW (only called post-execution for reporting)

---

#### E. **CONFIGURATION & VALIDATION** (2 methods, ~45 lines each)
**Concern**: Build config hashes for drift detection; validate config changes across restarts

```
_build_config_hashes()        # Hash training_critical, late_stage, model_dataset (22 lines)
_validate_config_drift()      # Detect config changes; emit ConfigDriftError (45 lines)
```

**Extraction Candidate**: MEDIUM — Straightforward logic but tightly coupled to state model
**Estimated Lines**: ~70 lines total
**Public API Impact**: LOW (called only during bootstrap)

---

#### F. **STAGE ORCHESTRATION & LINEAGE** (8 methods, ~20-50 lines each)
**Concern**: Stage ordering, restart logic, lineage tracking, prerequisite validation

```
_get_stage_index()                  # Find stage by name in self.stages (6 lines)
_compute_enabled_stage_names()      # Filter stages by config (11 lines)
_normalize_stage_ref()              # Convert stage ref to canonical name (24 lines)
_derive_resume_stage()              # Find next incomplete stage (18 lines)
_forced_stage_names()               # Get stages forced to run from start (9 lines)
_invalidate_lineage_from()          # Clear downstream artifacts on restart (11 lines)
_restore_reused_context()           # Load context from previous restart points (34 lines)
_validate_stage_prerequisites()     # Check inference health, model availability (30 lines)
```

**Extraction Candidate**: STRONG — Cohesive stage-ordering logic
**Estimated Lines**: ~140 lines total
**Public API Impact**: LOW (all internal to `_run_stateful`)

---

#### G. **LOGGING, CONTEXT & DETAILS** (5 methods, ~30-120 lines each)
**Concern**: Per-stage logging, context extraction from stages, restart output handling

```
_log_stage_specific_info()          # Log validation/training/model details (118 lines)
_sync_root_context_from_stage()     # Update context from stage outputs (7 lines)
_extract_restart_outputs()          # Load stage outputs from previous attempt (49 lines)
_fill_from_context()                # Populate collector from context dict (47 lines)
_get_stage_skip_reason()            # Query skip_reason from state (10 lines)
```

**Extraction Candidate**: MEDIUM — Strongly coupled to context structure
**Estimated Lines**: ~230 lines total
**Public API Impact**: MEDIUM (reads/writes shared context dict)

---

#### H. **INITIALIZATION & CLEANUP** (11 methods, various sizes)
**Concern**: Stage/collector setup, error handling, GPU release, artifact flushing

```
_init_stages()                      # Instantiate all stages + callbacks (26 lines)
_init_collectors()                  # Create artifact collectors for each stage (31 lines)
_record_launch_rejection_attempt()  # Log rejection before execution (41 lines)
_flush_pending_collectors()         # Flush unflushed collectors to disk (27 lines)
_cleanup_resources()                # Close MLflow, release GPU, clean logs (59 lines)
_maybe_early_release_gpu()          # GPU early release for GPU deployer (22 lines)
_is_inference_runtime_healthy()     # HTTP health check for inference (16 lines)
list_stages()                       # Public: return stage names (41 lines)
list_restart_points()               # Public: query restart points (68 lines)
get_stage_by_name()                 # Public: lookup stage by name (6 lines)
notify_signal()                     # Public: set shutdown signal flag (4 lines)
```

**Extraction Candidate**: MIXED
- **_init_stages, _init_collectors**: STRONG candidates
- **_cleanup_resources, _maybe_early_release_gpu**: MEDIUM candidates
- **list_*, get_*, notify_signal**: Keep in main class (public API)

**Estimated Lines**: ~340 lines total

---

#### I. **MAIN EXECUTION LOOP** (2 methods, 371 + 13 lines)
**Concern**: Core stage-by-stage execution, error handling, state transitions

```
run()                               # Public entry, call _run_stateful (13 lines)
_run_stateful()                     # Main loop: bootstrap → iterate stages → finalize (371 lines)
```

**Extraction Candidate**: STRONG → Extract parts but keep core orchestration
- Extract: bootstrap logic, state transitions to separate handlers
- Keep in main: stage execution loop itself

**Estimated Lines**: ~384 lines total

---

## 2. Current Decomposition: Existing Pipeline Modules

### 2.1 src/pipeline/stages/ — 5 Stage Implementations (3,298 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `dataset_validator.py` | 991 | Validate datasets with plugins, emit structured events |
| `training_monitor.py` | 833 | Monitor GPU/training progress via polling |
| `gpu_deployer.py` | 475 | Deploy GPU trainer (multi-provider support) |
| `inference_deployer.py` | 425 | Deploy inference endpoint |
| `model_evaluator.py` | 277 | Evaluate model metrics (accuracy, perplexity, etc) |
| `base.py` | 203 | Abstract PipelineStage base class |
| `__init__.py` | 55 | Exports + StageNames enum |

**Observations**:
- Stages already properly decomposed (one per file)
- Each implements `PipelineStage.run(context) → Result`
- DatasetValidator has its own callback event system (`DatasetValidatorEventCallbacks`)

---

### 2.2 src/pipeline/state/ — State Management (471 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `models.py` | 229 | `PipelineState`, `PipelineAttemptState`, `StageRunState` dataclasses |
| `store.py` | 202 | `PipelineStateStore` for persist/load operations |
| `__init__.py` | 40 | Exports + utility functions (`build_attempt_state`, `hash_payload`, etc) |

**Observations**:
- State model is separate and clean
- Store handles disk I/O
- `__init__.py` re-exports state utilities (could be expanded)

---

### 2.3 src/pipeline/artifacts/ — Artifact Collection (425 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `base.py` | 252 | `StageArtifactCollector` + `StageArtifactEnvelope` (Typed dict pattern) |
| `schemas.py` | 133 | TypedDict schemas for each stage's `data` field |
| `__init__.py` | 40 | Exports all schemas + base classes |

**Observations**:
- Clean separation: `StageArtifactCollector` handles single stage
- Orchestrator manages dict of collectors (one per stage)
- Validation-specific schema: `ValidationArtifactData`, `ValidationDatasetData`, `ValidationPluginData`

---

### 2.4 src/pipeline/restart_points.py (131 lines)

Utility for querying restart point metadata from run directory.

---

### 2.5 src/pipeline/run_inspector.py (27 lines)

Stub for future inspection utilities.

---

## 3. Direct Dependencies: Import Analysis

### 3.1 Orchestrator Import Summary

**Total import statements**: 18  
**Distinct src.* modules imported**: 9

```
src.constants                          → PROVIDER_RUNPOD
src.config.datasets.constants          → SOURCE_TYPE_HUGGINGFACE
src.config.runtime                     → RuntimeSettings, load_runtime_settings
src.pipeline.artifacts                 → StageArtifactCollector, ValidationArtifactData, etc
src.pipeline.artifacts.base            → utc_now_iso
src.pipeline.constants                 → CTX_*, MLFLOW_*, EXIT_CODE_*, SEPARATOR_*
src.pipeline.domain                    → RunContext
src.pipeline.stages                    → DatasetValidator, GPUDeployer, ... (5 stages)
src.pipeline.stages.gpu_deployer       → IEarlyReleasable (for GPU early release)
src.pipeline.state                     → PipelineState, PipelineAttemptState, etc
src.reports                            → ExperimentReportGenerator
src.training.managers.mlflow_manager   → MLflowManager
src.utils.config                       → PipelineConfig, Secrets, etc
src.utils.logger                       → logger, console, init_run_logging, etc
src.utils.result                       → Result, Ok, Err, AppError, etc
```

**Analysis**:
- **High coupling**: Directly imports from 9 modules in src.*
- **But mostly for types/utilities**: Not logic-heavy external dependencies
- **Encapsulation good**: Stages are loosely coupled via context dict + Result types

---

## 4. Training Orchestrator Pattern: Reference Architecture

### 4.1 Training Orchestrator Structure (3,007 lines distributed)

```
src/training/orchestrator/
├── __init__.py                    (40 lines) — Facade exports
├── strategy_orchestrator.py        (386 lines) — FACADE: coordinates all components
├── chain_runner.py                (216 lines) — Iterate phases, pass model between phases
├── phase_executor/
│   ├── executor.py               (315 lines) — Execute single phase (train + eval)
│   ├── training_runner.py         (460 lines) — Trainer setup + checkpoint save
│   ├── mlflow_logger.py           (254 lines) — Log phase start/completion/error
│   ├── adapter_cache.py           (288 lines) — Adapter LoRA caching logic
│   └── __init__.py               (12 lines)
├── dataset_loader.py              (264 lines) — Load + validate datasets
├── metrics_collector.py            (172 lines) — Extract metrics from checkpoints
├── resume_manager.py              (273 lines) — Resume/checkpoint logic
└── shutdown_handler.py            (327 lines) — SIGINT/SIGTERM graceful shutdown
```

**Key Pattern Insights**:

1. **Facade Pattern** (`strategy_orchestrator.py` ~120 lines actual code after extraction)
   - Single public entry: `StrategyOrchestrator(model, tokenizer, config)`
   - Main method: `run_chain() → Result[Model, Error]`
   - Delegates to specialized components: ChainRunner, ResumeManager, MetricsCollector

2. **Single Responsibility Decomposition**
   - `ChainRunner`: Phase iteration only (~60 lines of logic)
   - `PhaseExecutor`: Single phase execution
   - `DatasetLoader`: Dataset prep (isolated from orchestration)
   - `MetricsCollector`: Metrics aggregation (isolated)
   - `ResumeManager`: Resume/checkpoint logic (isolated)
   - `ShutdownHandler`: Signal handling (singleton pattern)

3. **Callback/Event Pattern**
   - `DataBuffer` emits phase lifecycle events
   - PhaseExecutor subscribes via `DataBufferEventCallbacks`
   - StrategyOrchestrator composes these components

4. **Public API is SMALL**
   - Only `StrategyOrchestrator.run_chain()` is public
   - All component classes private/internal
   - __init__.py exports only Facade + ShutdownHandler utilities

---

## 5. Coupling Signals & Extraction Opportunities

### 5.1 Self-Contained Extraction Groups (HIGH PRIORITY)

#### GROUP A: Validation Artifact Manager (9 methods, ~170 lines)
**Files affected**: `orchestrator.py` only (no cross-module calls)
**Potential location**: `src/pipeline/validation/artifact_manager.py`

**Methods to extract**:
```python
class ValidationArtifactManager:
    """Manage dataset validation callback events and artifact flushing."""
    
    def __init__(self):
        self._accumulator: dict[str, ValidationDatasetData] = {}
        self._plugin_descriptions: dict[tuple[str, str], str] = {}
    
    # Callbacks (from DatasetValidator)
    def on_dataset_scheduled(self, name, path, mode) → None
    def on_dataset_loaded(self, name, path, count, critical) → None
    def on_validation_completed(self, name, path, metrics, warnings) → None
    def on_validation_failed(self, name, path, errors) → None
    def on_plugin_start(self, name, path, plugin_id, description) → None
    def on_plugin_complete(self, name, path, plugin_id, ...) → None
    def on_plugin_failed(self, name, path, plugin_id, ...) → None
    
    # Output generation
    def flush_artifact(self, collector, started_at, duration_s) → None
    def build_state_outputs(self, stage_ctx=None, error=None) → dict
    def get_callbacks_for_validator(self) → DatasetValidatorEventCallbacks
```

**Benefits**:
- Reduces orchestrator by 170 lines
- Single responsibility: validation artifact lifecycle
- Reusable for other code needing validation callbacks
- Testable in isolation

---

#### GROUP B: Stage Execution Orchestration (8 methods, ~140 lines)
**Files affected**: `orchestrator.py` + `src/pipeline/state/` (state model)
**Potential location**: `src/pipeline/executor/stage_executor.py`

**Methods to extract**:
```python
class StageExecutionOrchestrator:
    """Manage stage ordering, lineage, prerequisites."""
    
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
    
    def get_stage_index(self, name: str) → int
    def compute_enabled_stages(self, start_stage: str) → list[str]
    def normalize_stage_ref(self, ref: str|int|None) → str
    def derive_resume_stage(self, state: PipelineState) → str|None
    def forced_stage_names(self, start_stage: str) → set[str]
    def invalidate_lineage_from(self, lineage, start_stage) → dict
    def restore_reused_context(self, attempt, lineage, start_stage, enabled) → None
    def validate_stage_prerequisites(self, stage_name, start_stage) → AppError|None
```

**Benefits**:
- Clear separation: orchestration logic vs execution
- Reduces orchestrator complexity in main loop
- Enables reuse in validation/testing
- Single responsibility: stage sequencing

---

#### GROUP C: MLflow Integration Handler (6 methods, ~220 lines)
**Files affected**: `orchestrator.py` only
**Potential location**: `src/pipeline/mlflow/attempt_manager.py`

**Methods to extract**:
```python
class MLflowAttemptManager:
    """Manage MLflow parent/nested run lifecycle for pipeline attempts."""
    
    def __init__(self, config: PipelineConfig, secrets: Secrets):
        self._manager: MLflowManager | None = None
        self._root_run: Any = None
        self._attempt_run: Any = None
    
    def initialize(self) → MLflowManager | None
    def setup_for_attempt(self, state, attempt, start_idx) → None
    def ensure_preflight(self, state) → None
    def end_attempt(self, success: bool) → None
    def get_run_id(self) → str|None
    def reopen_root_run(self, run_id: str) → Any
```

**Benefits**:
- Isolates external MLflow API calls
- Reduces orchestrator by 220 lines
- Easier to mock/test MLflow interactions
- Single responsibility: MLflow attempt lifecycle

---

#### GROUP D: State Marking Operations (7 methods, ~140 lines)
**Files affected**: `orchestrator.py` + `src/pipeline/state/models.py`
**Potential location**: Keep in `src/pipeline/state/models.py` as static methods or new `StateTransitioner` class

**Methods to extract**:
```python
class StateTransitioner:
    """Apply state transitions to PipelineAttemptState."""
    
    @staticmethod
    def mark_running(attempt, stage_name, started_at) → None
    @staticmethod
    def mark_completed(attempt, stage_name, outputs) → None
    @staticmethod
    def mark_failed(attempt, stage_name, error, failure_kind, outputs=None) → None
    @staticmethod
    def mark_skipped(attempt, stage_name, reason, outputs=None) → None
    @staticmethod
    def mark_interrupted(attempt, stage_name, started_at) → None
    @staticmethod
    def finalize(state, attempt, status) → None
    @staticmethod
    def save(state_store, state) → None
```

**Benefits**:
- State mutations grouped in one place
- Easier to reason about state transitions
- Testable state machine logic
- Reduces orchestrator by 140 lines

---

#### GROUP E: Reporting & Metrics (4 methods, ~310 lines)
**Files affected**: `orchestrator.py` + imports from `src.reports`
**Potential location**: `src/pipeline/reporting/metrics_reporter.py`

**Methods to extract**:
```python
class MetricsReporter:
    """Aggregate and report training metrics."""
    
    def __init__(self, mlflow_manager: MLflowManager|None, config: PipelineConfig):
        self._mlflow = mlflow_manager
        self._config = config
    
    def aggregate_training_metrics(self, context) → None
    def collect_descendant_metrics(self, max_depth=2) → list[dict]
    def generate_experiment_report(self, run_id=None) → None
    def print_summary(self, context, state, attempt) → None
```

**Benefits**:
- Reporting is orthogonal to execution
- Reusable for other analysis/export scenarios
- Reduces orchestrator by 310 lines
- Single responsibility: metrics aggregation & reporting

---

### 5.2 Medium-Complexity Extractions (MEDIUM PRIORITY)

#### GROUP F: Configuration & Validation (2 methods, ~70 lines)
**Methods**: `_build_config_hashes()`, `_validate_config_drift()`
**Potential location**: `src/pipeline/config/drift_validator.py`

---

#### GROUP G: Logging & Context Details (5 methods, ~230 lines)
**Methods**: `_log_stage_specific_info()`, `_sync_root_context_*`, `_extract_restart_outputs()`, `_fill_from_context()`
**Potential location**: `src/pipeline/context/context_manager.py`
**Complexity**: Tightly coupled to context dict structure; requires clear interface

---

### 5.3 Initialization & Setup (LOWER PRIORITY)

#### GROUP H: Initialization (2 methods, ~60 lines)
**Methods**: `_init_stages()`, `_init_collectors()`
**Location**: Could stay in `__init__()` or move to factory class
**Trade-off**: Small methods, low complexity; extraction gains are modest

---

## 6. Recommended Decomposition Plan

### Phase 1: High-Impact, Low-Risk (Highest ROI)

1. **Validation Artifact Manager** → 170 lines recovered
   - New file: `src/pipeline/validation/artifact_manager.py`
   - Reduced orchestrator complexity: 7 private methods → 1 dependency
   - Zero impact on public API

2. **MLflow Attempt Manager** → 220 lines recovered
   - New file: `src/pipeline/mlflow/attempt_manager.py`
   - Isolated external dependency
   - Testable in isolation

3. **State Transitioner** → 140 lines recovered
   - Extend: `src/pipeline/state/models.py` (add StateTransitioner class)
   - Single responsibility: state transitions
   - Easier state machine testing

**Total Lines Recovered**: ~530 lines (23% of orchestrator)

---

### Phase 2: Medium-Complexity, Good Gains (Medium ROI)

4. **Stage Execution Orchestrator** → 140 lines recovered
   - New file: `src/pipeline/executor/stage_executor.py`
   - Enables reuse in testing/validation
   - Clear stage-sequencing logic

5. **Metrics Reporter** → 310 lines recovered
   - New file: `src/pipeline/reporting/metrics_reporter.py`
   - Reporting is orthogonal to execution
   - Reusable for other analysis tools

**Total Lines Recovered**: ~450 lines (additional; 45% total with Phase 1)

---

### Phase 3: Architectural Refactor (Lower Priority)

6. **Context Manager** → 230 lines recovered
   - Requires clear interface for context access
   - More disruptive; defer for separate initiative

7. **Configuration Validator** → 70 lines recovered
   - Could be extracted but small win

**Total Lines Recovered**: ~300 lines (additional; 58% total)

---

## 7. Extraction Candidates: Quick Wins (No Public API Changes)

### 7.1 Methods Extractable WITHOUT Changing Public API

✅ **VALIDATION ARTIFACT GROUP** (9 methods)
- Currently: Private callbacks + helpers
- Extraction: Same interface, just moved to separate class
- Impact: Zero on `PipelineOrchestrator.run()` callers

✅ **STATE MARKING GROUP** (7 methods)
- Currently: Private state mutations
- Extraction: Move to `StateTransitioner` helper
- Impact: Zero on public API

✅ **MLFLOW INTEGRATION GROUP** (6 methods)
- Currently: Private MLflow setup/teardown
- Extraction: Move to `MLflowAttemptManager`
- Impact: Zero on public API (only internal initialization changes)

✅ **STAGE ORCHESTRATION GROUP** (8 methods)
- Currently: Private stage ordering helpers
- Extraction: Move to `StageExecutionOrchestrator`
- Impact: Zero on public API (all calls internal to `_run_stateful`)

✅ **REPORTING & METRICS GROUP** (4 methods)
- Currently: Private post-execution reporting
- Extraction: Move to `MetricsReporter`
- Impact: Zero on public API (called post-execution only)

---

### 7.2 Keep in Main Class (Public API)

Must stay in `PipelineOrchestrator`:
- `run()` — Public entry point
- `list_stages()` — Public query
- `list_restart_points()` — Public query
- `get_stage_by_name()` — Public query
- `notify_signal()` — Public signal handler

**Refactored orchestrator would be ~500-600 lines** (down from 2,266):
```
__init__()                      ~118 lines (config loading, stage setup)
run()                           ~13 lines (entry point)
_run_stateful()                 ~280 lines (core loop, simplified)
list_stages()                   ~41 lines
list_restart_points()           ~68 lines
get_stage_by_name()             ~6 lines
notify_signal()                 ~4 lines
_cleanup_resources()            ~59 lines
_maybe_early_release_gpu()      ~22 lines
_is_inference_runtime_healthy() ~16 lines
_bootstrap_pipeline_state()     ~76 lines
_record_launch_rejection_attempt() ~41 lines (or move to error handler)
_flush_pending_collectors()     ~27 lines
+ dependency injection of component managers
```

---

## 8. Proposed Directory Structure (Post-Decomposition)

```
src/pipeline/
├── orchestrator.py                    # 600 lines (down from 2,266)
├── domain.py                          # (existing)
├── constants.py                       # (existing)
├── stages/                            # (existing)
│   ├── *.py
│   └── __init__.py
├── state/                             # (existing, + extended)
│   ├── models.py                      # + StateTransitioner class
│   ├── store.py
│   └── __init__.py
├── artifacts/                         # (existing)
│   ├── base.py
│   ├── schemas.py
│   └── __init__.py
├── validation/                        # NEW
│   ├── artifact_manager.py            # ValidationArtifactManager
│   └── __init__.py
├── mlflow/                            # NEW
│   ├── attempt_manager.py             # MLflowAttemptManager
│   └── __init__.py
├── executor/                          # NEW
│   ├── stage_executor.py              # StageExecutionOrchestrator
│   └── __init__.py
├── reporting/                         # NEW
│   ├── metrics_reporter.py            # MetricsReporter
│   └── __init__.py
├── context/                           # NEW (Phase 3)
│   ├── context_manager.py             # Context + lineage helpers
│   └── __init__.py
├── restart_points.py                  # (existing)
├── run_inspector.py                   # (existing)
└── __init__.py                        # Exports main entry point
```

---

## 9. Training Orchestrator Comparison Matrix

| Aspect | Training Orchestrator | Pipeline Orchestrator (Current) | Pipeline Orchestrator (Target) |
|--------|----------------------|--------------------------------|--------------------------------|
| **Main Facade** | StrategyOrchestrator (120 LOC) | PipelineOrchestrator (2,266 LOC) | PipelineOrchestrator (600 LOC) |
| **Component Classes** | ChainRunner, PhaseExecutor, DatasetLoader, MetricsCollector, ResumeManager, ShutdownHandler | (all monolithic) | ValidationArtifactManager, MLflowAttemptManager, StageExecutionOrchestrator, MetricsReporter, StateTransitioner |
| **Lifecycle Managers** | 6 separate managers | 1 monolithic class | 6+ specialized managers |
| **Testing Surface** | Small facade + testable components | Large monolithic class | Small facade + testable components |
| **Extensibility** | Easy (add new component) | Hard (modify monolith) | Easy (add new component) |
| **MLflow Integration** | Via PhaseExecutor + MLflowLogger | Directly in orchestrator | Via MLflowAttemptManager |
| **Metrics Collection** | MetricsCollector (172 LOC) | _aggregate_* + _collect_* (120 LOC mixed in) | MetricsReporter (310 LOC extracted) |
| **Artifact Management** | Each stage handles own artifacts | Central orchestrator + validation callbacks | Orchestrator delegates to ValidationArtifactManager |

---

## 10. Key Findings Summary

### Decomposability Assessment: HIGH ✓

1. **Natural Groupings Exist**: 56 methods naturally cluster into 8 logical concerns
2. **Low Cross-Group Dependencies**: Methods within a group rarely call methods in other groups
3. **Clear Responsibilities**: Each group can be described in 1-2 sentences
4. **Minimal Public API Impact**: 73% of methods are private; extraction doesn't break callers

### Complexity Drivers

| Aspect | Impact | Root Cause |
|--------|--------|-----------|
| **File Size** | +2,266 lines | Single class accumulates all orchestration logic |
| **Method Count** | 56 methods | No decomposition → all methods in one class |
| **Cyclomatic Complexity** | High in `_run_stateful` | 371 lines, ~15 conditional branches, loop + exception handling |
| **Testing Difficulty** | Hard to unit-test | Setup/teardown scattered; state mutations mixed with orchestration |
| **Code Reuse** | Low | Helpers (e.g., validation callbacks) tied to orchestrator |

### Training Orchestrator Lessons (Applicable Pattern)

✓ **Facade Pattern**: StrategyOrchestrator delegates to 6 specialized managers  
✓ **Single Responsibility**: Each manager handles one aspect (phases, metrics, resume)  
✓ **Dependency Injection**: Testable via injected factories/managers  
✓ **Callback Events**: Components communicate via event callbacks, not direct calls  
✓ **Public API Minimal**: Only facade exposed; managers are internal  

**These patterns apply directly to pipeline orchestrator decomposition.**

---

## 11. Extraction Implementation Guide

### Template for Group Extraction

1. **Create new file** in appropriate subdirectory
2. **Move methods** (copy from orchestrator, don't delete yet)
3. **Update imports** in new class
4. **Add __init__.py** export if needed
5. **Inject dependency** into PipelineOrchestrator
   ```python
   def __init__(self, ...):
       self._validation_mgr = ValidationArtifactManager()
       self._mlflow_mgr_container = MLflowAttemptManager(config, secrets)
   ```
6. **Update method calls** in orchestrator to delegate
   ```python
   # Before:
   self._on_dataset_scheduled(dataset_name, path, mode)
   
   # After:
   self._validation_mgr.on_dataset_scheduled(dataset_name, path, mode)
   ```
7. **Run tests** to verify no behavior changes
8. **Delete old methods** from orchestrator

### Testing Strategy Post-Extraction

- Unit test each extracted component in isolation
- Integration test PipelineOrchestrator with mocked components
- E2E test full pipeline (should pass unchanged)
- No changes to public API (`run()`, `list_stages()`, etc.)

---

## 12. Risk Assessment

### Extraction Risks: LOW

- **Behavior Preservation**: All public methods unchanged; only internal structure shifts
- **Backward Compatibility**: No external code depends on private method names
- **Testing Coverage**: Existing tests should still pass with refactored code
- **Rollback**: Easy to revert if issues arise (extract, not rewrite)

### Implementation Risks: MEDIUM

- **Dependency Injection**: Must correctly thread dependencies into components
- **Shared State**: Validation accumulator, plugin descriptions must be managed correctly
- **MLflow Lifecycle**: Ensure parent/nested run creation/closure still works
- **Error Handling**: Ensure error paths still propagate correctly

---

## Conclusion

**The pipeline orchestrator is highly suitable for decomposition using the training orchestrator as a reference pattern.** A multi-phase extraction can reduce the file from 2,266 to ~600 lines, with each extracted component handling a single responsibility:

1. **Phase 1** (HIGH ROI, LOW RISK): 530 lines extracted (23%)
2. **Phase 2** (MEDIUM ROI, MEDIUM RISK): 450 additional lines (45% total)
3. **Phase 3** (LOWER ROI, HIGHER RISK): 300 additional lines (58% total)

Each phase can be completed independently without breaking the public API or requiring downstream changes. Following the training orchestrator's Facade + Component pattern enables better testability, reuse, and maintainability.
