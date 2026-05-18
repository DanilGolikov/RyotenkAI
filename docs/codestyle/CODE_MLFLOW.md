# MLflow integration in this project

Guidelines for working with MLflow tracking, runs, model registry, and reports.

---

## Contents

- [Instructions for LLMs](#instructions-for-llms)
- [Architecture overview](#architecture-overview)
- [Run hierarchy (Pattern A)](#run-hierarchy-pattern-a)
- [Taxonomy (tags / params / metrics)](#taxonomy-tags--params--metrics)
- [Configuration](#configuration)
- [Write-path: training time](#write-path-training-time)
- [Read-path: reports + API](#read-path-reports--api)
- [Lifecycle: open, finalize, signals](#lifecycle-open-finalize-signals)
- [Model registry: aliases over stages](#model-registry-aliases-over-stages)
- [System prompt loader](#system-prompt-loader)
- [Adding a new MLflow integration point](#adding-a-new-mlflow-integration-point)
- [Anti-patterns](#anti-patterns)
- [Common LLM mistakes](#common-llm-mistakes)
- [Tests](#tests)

---

## Instructions for LLMs

> This document is for developers and LLM agents. When generating or refactoring code that touches MLflow in this project — read and apply the rules below.

**Quick reference:** narrow Protocols (≤7 methods each) injected via DI | NEVER ad-hoc `MlflowClient()` | NEVER `mlflow.autolog()` | NEVER `mlflow.set_tracking_uri()` outside `MlflowTransport.__init__` | NEVER `mlflow.start_run()` in pod-trainer (HF MLflowCallback adopts `MLFLOW_RUN_ID` from env) | one dotted namespace `ryotenkai.<domain>.<area>` for tags/params/metrics | aliases (`@champion`, `@challenger`) — not deprecated stages.

**When editing code in this project, follow these rules.**

### Mandatory rules (MUST)

1. **Never call `mlflow.start_run()` / `mlflow.end_run()` directly** outside of the canonical `MlflowTransport` and `ParentRunOpener` / `MlflowFinalizer`. The pod-trainer must adopt the parent run via `MLFLOW_RUN_ID` + `MLFLOW_NESTED_RUN=TRUE` env vars (Pattern A). Sentinel: `scripts/lint/mlflow_rules.py::NO_START_RUN_IN_TRAINER`.
2. **Never call `mlflow.autolog()` or `mlflow.transformers.autolog()`.** They conflict with HF Trainer's `report_to=["mlflow"]` and produce double-logged metrics + duplicate runs. Use `HFMlflowWiring.configure_training_args(...)` in trainers. Sentinel: `NO_AUTOLOG`.
3. **Never construct `MlflowClient()` ad-hoc.** Depend on `ITrackingClient` (writes) or `IRunQuery` (reads) injected via DI. Only `MlflowTransport`, `MlflowReadClient`, and `MlflowModelRegistry` are allowed to instantiate the SDK client. Sentinel: `NO_AD_HOC_MLFLOW_CLIENT`.
4. **Never call `mlflow.set_tracking_uri(...)` anywhere except `MlflowTransport.__init__`.** Mutating the process-wide singleton breaks tests and produces sticky URIs across runs. Sentinel: `NO_SET_TRACKING_URI_GLOBAL`.
5. **Never invent new `ryotenkai.*` tag/param/metric keys** outside `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/taxonomy.py`. Add the key to `TagKey` / `ParamKey` / `MetricKey` enum, then use the enum value at the call site. Every write path runs `ReservedPrefixGuard.assert_safe(key)`.
6. **Never set `mlflow.*` tags** except the whitelisted ones (`mlflow.note.content`, `mlflow.runName`, `mlflow.parentRunId`, `mlflow.source.*`, `mlflow.user`). `ReservedPrefixGuard` rejects everything else.
7. **Never mock the MLflow Protocols** (`ITrackingClient`, `IRunQuery`, `IMetricSink`, `IArtifactSink`, `IModelRegistry`, `IJournalUploader`, `IPromptRegistry`) via `unittest.mock`. Use the canonical fakes in `tests/_fakes/mlflow_*.py`. Sentinel: `tests/_lint/test_no_protocol_mocking.py`.
8. **Never call `register_model()` with deprecated stages** (`Staging` / `Production`). Use aliases — `MlflowModelRegistry.set_alias(name, "challenger", version)`. Promotion `challenger → champion` is a manual CLI gate (`ryotenkai model promote`).
9. **Never mutate the global `mlflow` module** (no `mlflow.set_experiment(...)`, no env-var rewrites mid-process). MLflow runtime state is owned by `MlflowTransport`.

### Quick choice tree

| Situation | Action |
|---|---|
| Need to open / close an MLflow run | Use `RunLifecycleCoord` (control). Trainer **never** opens runs. |
| Need to log metrics in training | Already covered: HF `MLflowCallback` does it via `report_to=["mlflow"]` (configured by `HFMlflowWiring`). For custom metrics use `mlflow.log_metric()` on the active run (= HF nested child) — no client construction needed. |
| Need to read run / search runs / list children | Inject `IRunQuery` (concrete: `MlflowReadClient`) via DI. Use `RunTreeWalker` for BFS over a run subtree. |
| Need to log a model artifact at end of training | `mlflow.transformers.log_model(..., save_pretrained=True)` followed by `ModelPublisher.publish(...)` (alias-based). See `pod/trainer/run_training.py::_publish_trained_model`. |
| Need to add an MLflow tag for lineage | Add the enum value to `TagKey` in `taxonomy.py`, then `client.set_tags(run_id, {TagKey.NEW_TAG.value: ...})`. |
| Need to add a new MLflow read site in reports | Inject `MlflowReadClient` into the composer slice via DI. Never instantiate. |
| Need to register a model under a new alias | `MlflowModelRegistry.set_alias(name, alias, version)`. Never use `transition_model_version_stage`. |
| Need to read a system prompt for inference | Inject `SystemPromptLoader` from `ryotenkai_shared.inference.prompts`. Pass `IPromptRegistry` via DI. |
| Need to add MLflow connectivity check | Use `PreflightConnectivityCheck.run()`. Never duplicate the probe logic. |

---

## Architecture overview

```
[packages/shared/.../infrastructure/mlflow/]      ← Protocols + low-level
  protocols.py        : 7 narrow Protocols (≤7 methods each, no Any)
  config.py           : MLflowConnectionConfig (Pydantic, base)
  auth.py             : MLflowAuthConfig discriminated union + MlflowAuthAdapter
  uri.py              : RuntimeUriResolver + RuntimeUri (frozen value)
  taxonomy.py         : TagKey / ParamKey / MetricKey enums + ReservedPrefixGuard
  transport.py        : MlflowTransport (implements ITrackingClient; tenacity retry)
  dataset.py          : HFDatasetLogger (mlflow.data.huggingface_dataset facade)
  journal_uploader.py : JournalUploader (implements IJournalUploader; idempotent)
  metric_sink.py      : MetricSink (implements IMetricSink; log_batch async)
  dead_letter.py      : DeadLetterBuffer (bounded on-disk JSONL)
  run_handle.py       : RunHandle frozen value object
  registry.py         : MlflowModelRegistry (implements IModelRegistry)

[packages/shared/.../config/integrations/]
  mlflow_project.py   : MLflowProjectConfig(MLflowConnectionConfig) ← pipeline YAML

[packages/shared/.../inference/prompts/]
  system_prompt_loader.py : SystemPromptLoader (relocated; +TTL cache +failure modes)

[packages/control/.../pipeline/mlflow/]           ← lifecycle + read
  lifecycle/
    preflight.py      : PreflightConnectivityCheck
    opener.py         : ParentRunOpener (open root, open_attempt, adopt_root)
    coord.py          : RunLifecycleCoord (atexit + SIGTERM/SIGINT, mutex)
    finalizer.py      : MlflowFinalizer (idempotent via lifecycle.finalized tag)
  read/
    client.py         : MlflowReadClient (implements IRunQuery; LRU cache)
    tree_walker.py    : RunTreeWalker (single BFS for run tree)

[packages/control/.../reports/]
  composer.py         : ReportComposer (orchestrates slices)
  slices/             : header / metrics / loss_curve / eval / inference / artifacts / lineage

[packages/pod/.../trainer/mlflow/]                ← thin wiring only
  hf_wiring.py        : HFMlflowWiring (configures HF TrainingArguments; never opens runs)
  model_publisher.py  : ModelPublisher (log_model + set_alias)
  metrics_buffer.py   : MetricsBuffer (legacy preserved — used by MetricSink)
```

### Importlinter contracts

Add via `pyproject.toml` `[[tool.importlinter.contracts]]`:

- `shared.infrastructure.mlflow` — **leaf**, no outbound to control/pod/providers/community
- `shared.inference.prompts` — **leaf**, same rule
- `control.pipeline.mlflow` — **must not import pod or providers**
- `reports.composer` / `reports.slices` — must not import legacy `core.builder`

---

## Run hierarchy (Pattern A)

This is the canonical mental model. **Memorize it.**

```
Process: control-plane (Mac orchestrator)
=========================================
[#1] Root run                                    ParentRunOpener.open()
     run_name = state.logical_run_id
     tags: ryotenkai.lineage.* + ryotenkai.lifecycle.opened_by
     params: pipeline config (ONCE)
     artifacts: config.yaml (ONCE)
       │
       └─ NATIVE NESTED (mlflow.start_run(nested=True))
       │
   [#2] Attempt run                              ParentRunOpener.open_attempt()
        run_name = {logical_run_id}_attempt_{N}
        tags: ryotenkai.attempt.id, ryotenkai.attempt.no
             │
             ┊ Process boundary — env exported by training_launcher:
             ┊   MLFLOW_TRACKING_URI / MLFLOW_RUN_ID = <attempt_id>
             ┊   MLFLOW_NESTED_RUN = "TRUE"   ← uppercase canonical
             ┊   MLFLOW_EXPERIMENT_NAME
             ┊   MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = "true"
             ┊
        Process: pod-trainer subprocess
        ================================
        ⛔  NO mlflow.start_run() in trainer.
        ⛔  NO mlflow.autolog().
        ⛔  NO duplicate log_pipeline_config / log_dataset_config / config artifact.
        ⛔  NO trainer-side top-level run.
        
        HFMlflowWiring.configure_training_args(training_args):
          - sets training_args.report_to = ["mlflow"]
          - calls mlflow.set_system_metrics_node_id(rank or 0)
        
        HF MLflowCallback adopts MLFLOW_RUN_ID + MLFLOW_NESTED_RUN=TRUE
        and creates a nested child of Attempt #2 AUTOMATICALLY.
             │
             └─ [#3] HF training run (one per Trainer.train() invocation)
                  metrics: train/loss, eval/*, system metrics
                  → all native, no manual code on our side
        
        End of training → ModelPublisher.publish(run_id=os.environ["MLFLOW_RUN_ID"], ...)
          - mlflow.transformers.log_model(save_pretrained=True)
          - MlflowModelRegistry.set_alias(name, "challenger", version)

Reconciliation: MlflowFinalizer.finalize() is the SINGLE close path.
  - idempotent via ryotenkai.lifecycle.finalized tag
  - registered as atexit + SIGTERM/SIGINT handler in RunLifecycleCoord
  - closes attempt #2 then root #1 in order
```

### What lives where

| Data | Run target | Writer |
|---|---|---|
| Pipeline config params (`ryotenkai.config.*`) | **Root #1** | `ParentRunOpener.open()` (ONCE) |
| Dataset config | **Root #1** | `ParentRunOpener.open()` (ONCE) |
| Config YAML artifact | **Root #1** | `ParentRunOpener.open()` (ONCE) |
| Lineage tags (`ryotenkai.lineage.*`) | **Root #1** + **Attempt #2** | `ParentRunOpener` (both) |
| Attempt metadata (`ryotenkai.attempt.*`) | **Attempt #2** | `ParentRunOpener.open_attempt()` |
| HF `train/loss`, `train/learning_rate`, `eval/*` | **HF training run #3** | HF MLflowCallback (native) |
| GPU/CPU/RAM metrics | **HF training run #3** | Native MLflow system-metrics sampler |
| Per-phase effective hyperparams | **HF training run #3** | `HFMlflowWiring` (one set_params at start) |
| Dataset linkage (`mlflow.data.Dataset` log_input) | **HF training run #3** | `HFDatasetLogger` |
| `events.jsonl` SSOT journal artifact | **Attempt #2** | `JournalUploader.upload()` via `MlflowFinalizer` |
| Final pipeline state.json | **Attempt #2** | `MlflowFinalizer.finalize()` |
| Registered model + `@challenger` alias | (run_id from HF #3) | `ModelPublisher.publish()` |
| `ryotenkai.lifecycle.finalized=true` + status | **all closing runs** | `MlflowFinalizer.finalize()` (idempotent tag-guarded) |

**If you can't find your metric in the UI — it's almost certainly on HF training run #3.** Per-step metrics, system metrics, eval/* all go there. The parent runs hold lineage and aggregate-at-finalize data.

---

## Taxonomy (tags / params / metrics)

Single convention: **lowercase dotted** under the `ryotenkai.*` namespace. Matches the `kind` format from ADR-0009 unified events.

Defined in `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/taxonomy.py`.

### Required tags (set by Opener)

```
ryotenkai.lineage.pipeline_id
ryotenkai.lineage.run_id              ← our state.json logical_run_id
ryotenkai.lineage.config_sha256
ryotenkai.lineage.code_commit
ryotenkai.lifecycle.opened_by         ← host:user
ryotenkai.engine.kind                 ← sft | dpo | grpo | sapo
ryotenkai.provider.kind
ryotenkai.provider.gpu
```

### Required tags (set by Finalizer)

```
ryotenkai.lifecycle.finalized = "true"
ryotenkai.lifecycle.status            ← RunStatus.{RUNNING|FINISHED|FAILED|KILLED}
ryotenkai.journal.sha256              ← integrity check for uploaded journal
ryotenkai.exit.reason                 ← only if status != FINISHED
```

### Reserved prefixes

| Prefix | Status | Why |
|---|---|---|
| `ryotenkai.*` | Ours | Add to `TagKey` enum first; lint enforces |
| `mlflow.*` | **Forbidden** except whitelist (`mlflow.note.content`, `mlflow.runName`, `mlflow.parentRunId`, `mlflow.user`, `mlflow.source.*`) | MLflow itself sets these; collision = silent UI breakage |
| `hf.*` | Accepted | HF Trainer auto-emits these |

### Metrics naming

- HF Trainer emits `train/loss`, `train/learning_rate`, `eval/*` — accepted as-is.
- Custom domain metrics: `ryotenkai.metric.<area>.<name>` (e.g. `ryotenkai.metric.summary.final_train_loss`).
- Native system metrics: `gpu/0/utilization`, `cpu/percent`, etc. — emitted by MLflow's sampler, accepted.

---

## Configuration

### `MLflowConnectionConfig` — base (settings UI form)

In `packages/shared/.../infrastructure/mlflow/config.py`. Fields:

```python
tracking_uri: str | None              # public/Tailscale-funnel URI (pod uses this)
local_tracking_uri: str | None        # loopback URI (control uses this)
ca_bundle_path: str | None
connect_timeout_s: float = 5.0
request_timeout_s: float = 30.0
retry_total_budget_s: float = 30.0
auth: MLflowAuthConfig                # discriminated: none | basic | bearer
```

**Validators:**
- At least one of `tracking_uri` / `local_tracking_uri` is required.
- Userinfo in URIs (`http://user:pass@host`) is rejected (R-07 — credentials must go through `auth.*`).

### `MLflowProjectConfig(MLflowConnectionConfig)` — pipeline YAML

In `packages/shared/.../config/integrations/mlflow_project.py`. Adds:

```python
experiment_name: str                  # required, pattern: env__team__purpose
run_description_file: str | None
system_metrics: SystemMetricsConfig
model_registry_name_template: str = "ryotenkai/{experiment}/{model_family}"
alias_on_success: str = "challenger"
```

**Why two configs:** the settings UI cares about reachability + auth only. The pipeline YAML adds experiment-scoped knobs. Both share `MLflowConnectionConfig` as the base — no asymmetric duplicate schemas.

### URI resolution policy

| Where called | URI used |
|---|---|
| Control-plane (Mac) | `RuntimeUriResolver.for_control_plane(cfg)` — picks `local_tracking_uri` first, falls back to `tracking_uri`. Env var **ignored**. |
| Pod-trainer subprocess | `RuntimeUriResolver.for_training(cfg, env_override)` — picks `env_override` first, then `MLFLOW_TRACKING_URI` env, then `tracking_uri`, then `local_tracking_uri`. |

**Both URIs point at one MLflow server**, addressed by two different network paths.

---

## Write-path: training time

The canonical path. **Do not deviate.**

```python
# 1. Control plane (RunLifecycleCoord enter)
PreflightConnectivityCheck(client, timeout_s=5).run()         # fail-fast ping

# 2. Open root + attempt runs
root_run = opener.open(
    experiment=cfg.experiment_name,
    logical_run_id=state.logical_run_id,
    config_sha256=...,
    code_commit=...,
    engine_kind="sft",
    provider_kind="runpod",
    provider_gpu="A100-80GB",
)
attempt_run = opener.open_attempt(
    root_run=root_run,
    logical_run_id=state.logical_run_id,
    attempt_id=attempt.attempt_id,
    attempt_no=attempt.attempt_no,
)

# 3. Export env to pod subprocess (training_launcher._build_job_env)
env["MLFLOW_TRACKING_URI"] = RuntimeUriResolver.for_training(cfg).uri
env["MLFLOW_RUN_ID"] = attempt_run.run_id                      # canonical Pattern A
env["MLFLOW_NESTED_RUN"] = "TRUE"                              # uppercase
env["MLFLOW_EXPERIMENT_NAME"] = cfg.experiment_name
env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
# auth env if cfg.auth.kind != "none"

# 4. Trainer side (pod, run_training.py + TrainerFactory)
HFMlflowWiring.validate_env()                                  # fail-fast guard
HFMlflowWiring.configure_training_args(training_args, local_rank=local_rank)
# HF MLflowCallback auto-attaches via env

trainer.train()                                                # active run = nested child

# 5. End of training: publish model
ModelPublisher.publish(
    run_id=os.environ["MLFLOW_RUN_ID"],
    artifact_path="model",
    registered_name="ryotenkai/{experiment}/{model_family}",
    alias_on_success="challenger",
)

# 6. Finalize on control side (RunLifecycleCoord.finalize OR atexit/SIGTERM)
finalizer.finalize(
    run=attempt_run,
    status=RunStatus.FINISHED,
    journal_path=workspace / "events.jsonl",
    journal_sha256=sha256(workspace / "events.jsonl"),
)
# Then same for root_run.
```

### Strong assets preserved (do not refactor without reason)

- `MetricsBuffer` + `MetricsDecimator` — durability of metrics through offline windows
- `RunnerEventCallback` — async typed envelopes (ADR-0009 SSOT journal)
- `_flush_helper.py` — exemplary DRY-extracted shared helper

---

## Read-path: reports + API

### Single client via DI

**Never construct `MlflowClient()`.** Inject `MlflowReadClient` (implements `IRunQuery`) into the consumer at the composition root (control's `main.py` or `cli/app.py`).

```python
# Composition root
read_client = MlflowReadClient(tracking_uri=RuntimeUriResolver.for_control_plane(cfg).uri)
adapter = MLflowAdapter(run_query=read_client)                 # NOT tracking_uri!
report_gen = ExperimentReportGenerator(run_query=read_client)
retriever = ModelRetriever(run_query=read_client, ...)
deleter = RunDeleter(run_query_factory=lambda uri: MlflowReadClient(tracking_uri=uri))
```

### Single tree walker

For BFS over a run subtree (children of a parent, descendants for aggregation, deletion sweep) — use `RunTreeWalker`:

```python
walker = RunTreeWalker(run_query=read_client, cache_maxsize=64)
tree: RunNode = walker.walk(parent_run_id, max_depth=10)        # cached if parent terminal
flat: list[RunHandle] = walker.flat_descendants(parent_run_id)
```

**Never re-implement BFS** over `search_runs(parentRunId)`. Sentinel: `mlflow_rules.py` doesn't enforce this directly, but PR review will catch it.

### Decomposed reports

The old god-class `reports/core/builder.py` (884 LOC) is replaced by `ReportComposer` + 7 slice modules. Adding a new section = new slice file (~100 LOC).

```python
class CustomSlice(ReportSliceBuilder):
    name = "custom"
    def build(self, ctx: ReportContext) -> SliceOutput:
        # use ctx.run_query, ctx.tree_walker, ctx.artifact_sink
        return SliceOutput(title="Custom", markdown="...", warnings=[])

composer = ReportComposer([HeaderSlice(), MetricsSlice(), CustomSlice(), ...])
output = composer.compose(ctx)
markdown = output.to_markdown()
```

### Performance budget

Target: **≤20 HTTP round-trips per report** (down from baseline 120). Achieved via:
- `MlflowReadClient`'s LRU cache for terminal-status runs
- `RunTreeWalker`'s tree cache
- `concurrent.futures.ThreadPoolExecutor` for parallel `list_artifacts` / `get_run` (scoped per `ReportGenerator`)
- `IRunQuery.search` with `max_results>=200` instead of unbounded loops

---

## Lifecycle: open, finalize, signals

### Single boundary: `RunLifecycleCoord`

The **only** component that:
- Calls `atexit.register(...)`
- Installs `SIGTERM` / `SIGINT` handlers
- Holds the mutex around `finalize`

```python
from ryotenkai_control.pipeline.mlflow.lifecycle import (
    PreflightConnectivityCheck,
    ParentRunOpener,
    MlflowFinalizer,
    RunLifecycleCoord,
)

with RunLifecycleCoord(opener=opener, finalizer=finalizer, preflight=preflight) as coord:
    preflight.run()
    coord.bind_root_run(opener.open(...))
    coord.bind_attempt_run(opener.open_attempt(...))
    try:
        run_pipeline_attempt()
    finally:
        coord.finalize(
            status=RunStatus.FINISHED,
            journal_path=workspace / "events.jsonl",
            journal_sha256=sha,
        )
```

### Finalizer idempotency contract

```python
# MlflowFinalizer.finalize() guarantees:
# 1. If ryotenkai.lifecycle.finalized == "true" on the run → no-op (return).
# 2. Else: upload journal → set tags (finalized, status, exit_reason) → set_terminated.
# 3. NEVER raises. All errors swallowed + logged.
# 4. Safe to call from atexit, SIGTERM handler, normal finally, multiple times.
```

### Status mapping

| Trigger | Status |
|---|---|
| Normal completion | `FINISHED` |
| Stage exception | `FAILED` |
| User cancel / SIGTERM / SIGINT | `KILLED` (+ `ryotenkai.exit.reason=signal:<n>`) |
| Pod cloud-killed (OOM / spot preemption) | `KILLED` (+ `ryotenkai.exit.reason=pod_oom`/`spot_preemption`) |

### Restart / resume

Default: **new run + `ryotenkai.lineage.resumes_from=<prev_run_id>` tag**.

Adopt existing: **only if `prev_status == RunStatus.RUNNING`** (i.e. previous pipeline crashed before finalize). Use `ParentRunOpener.adopt_root(prev_run_id)`.

---

## Model registry: aliases over stages

### Why aliases

- MLflow **stages** (`Staging` / `Production`) are **deprecated since 2.9** and will be removed.
- **Aliases** (`@champion`, `@challenger`) are the supported mechanism.
- Multiple aliases per version → enables A/B tests.

### Publish flow (auto)

`ModelPublisher.publish()` is called at end of training (see `run_training.py::_publish_trained_model`):

```python
mlflow.transformers.log_model(
    transformers_model=trainer.model,
    artifact_path="model",
    save_pretrained=True,            # critical — embed weights for hub-independence
)
publisher.publish(
    run_id=os.environ["MLFLOW_RUN_ID"],
    artifact_path="model",
    registered_name="ryotenkai/dev__alignment__sft/llama3-8b",
    alias_on_success="challenger",  # default; env override MLFLOW_ALIAS_ON_SUCCESS
)
```

### Promote flow (manual, gated by operator)

```bash
# Promote challenger to champion after manual review
ryotenkai model promote --name ryotenkai/dev__alignment__sft/llama3-8b --version 3
# → sets alias=champion on version 3
```

The CLI lives at `packages/control/.../cli/commands/model.py`. **Aliases are movable** — re-running the command against a different version transparently reassigns the pointer.

### Naming convention

`model_registry_name_template = "ryotenkai/{experiment}/{model_family}"` (from `MLflowProjectConfig`).

`{experiment}` = `MLflowProjectConfig.experiment_name` (`env__team__purpose`).
`{model_family}` = derived from trainer config (e.g. `llama3-8b`, `qwen2-7b`).

---

## System prompt loader

Lives in `packages/shared/.../inference/prompts/system_prompt_loader.py` (relocated from `infrastructure/mlflow/` — it's a **domain loader**, not infrastructure).

### Usage

```python
from ryotenkai_shared.inference.prompts import SystemPromptLoader, OnMlflowFailure
from ryotenkai_shared.infrastructure.mlflow.protocols import IPromptRegistry

# Composition root: inject IPromptRegistry (concrete: MlflowTransport or MlflowModelRegistry)
loader = SystemPromptLoader(registry=registry, cache_ttl_s=300.0, cache_maxsize=64)

result = loader.load(
    llm_cfg,                          # InferenceLLMConfig
    on_mlflow_failure="warn",         # "fail" | "warn" | "fallback_to_file"
)
if result is not None:
    system_prompt = result.text       # for the LLM
    audit_source = result.source      # {"type": "mlflow", "name": "...", "version": "3"}
```

### Failure modes

| Mode | Behavior on MLflow Prompt Registry error |
|---|---|
| `fail` | Raise — pipeline aborts |
| `warn` (default) | Log warning, return None (caller continues without prompt) |
| `fallback_to_file` | Try `llm_cfg.system_prompt_path` as fallback if set |

### Cache

Bounded in-memory dict, keyed by `(name_or_uri,)`, TTL 300s, FIFO eviction at `cache_maxsize`. Thread-safe.

---

## Adding a new MLflow integration point

### Adding a tag

1. Add the key to `TagKey` enum in `taxonomy.py`:
   ```python
   class TagKey(StrEnum):
       ...
       NEW_FIELD = "ryotenkai.<area>.new_field"
   ```
2. Use the enum at the call site:
   ```python
   tracking_client.set_tags(run_id, {TagKey.NEW_FIELD.value: "value"})
   ```
3. `ReservedPrefixGuard.assert_safe()` runs automatically inside `MlflowTransport.set_tags`.

### Adding a metric

Standard: emit through HF Trainer (`trainer.log({...})`) for training metrics. For custom domain metrics from control-plane:

```python
metric_sink.log(
    run_id=attempt_run.run_id,
    metrics={"ryotenkai.metric.summary.custom": value},
    step=step,
)
metric_sink.flush(run_id, blocking=True)
```

### Adding a new read query (reports)

1. Add the method to `IRunQuery` Protocol (`protocols.py`) — ≤7 methods total, keep narrow.
2. Implement in `MlflowReadClient`.
3. Add the matching method to `FakeRunQuery` in `tests/_fakes/mlflow_run_query.py`.
4. Use in a new `ReportSliceBuilder` or extend an existing slice.

### Adding a new MLflow-touching CLI command

1. Inject `MlflowReadClient` / `MlflowModelRegistry` at the composition root (`cli/app.py`).
2. Pass via `CLIContext` or as direct kwarg.
3. **Never** instantiate `MlflowClient` directly in the command body.

---

## Anti-patterns

### 🚫 `mlflow.start_run()` in trainer code

```python
# BAD — old Pattern (trainer opens its own top-level run)
mlflow_mgr.start_run(run_name=f"{model}_{strategy}_{ts}")
mlflow_mgr.set_tags({"mlflow.parentRunId": parent_run_id})  # tag-only cosmetic nesting
trainer.train()
```

```python
# GOOD — Pattern A (HF callback adopts MLFLOW_RUN_ID + MLFLOW_NESTED_RUN=TRUE)
HFMlflowWiring.validate_env()
HFMlflowWiring.configure_training_args(training_args)
trainer.train()                                              # active run = HF nested child
```

### 🚫 `mlflow.autolog()` on top of `report_to="mlflow"`

Produces **double-logged metrics + duplicate runs**. HF MLflowCallback already logs everything autolog would. Sentinel `NO_AUTOLOG` enforces.

### 🚫 Ad-hoc `MlflowClient()` construction

```python
# BAD
import mlflow
mlflow.set_tracking_uri(uri)
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
```

```python
# GOOD
# Composition root: read_client = MlflowReadClient(tracking_uri=uri)
# Consumer: takes IRunQuery via __init__
def __init__(self, run_query: IRunQuery): self._run_query = run_query
run: RunHandle = self._run_query.get_run(run_id)
```

### 🚫 Custom system-metrics callback

```python
# BAD — custom SystemMetricsCallback emitting gpu/cpu/ram metrics
class SystemMetricsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        mlflow.log_metrics({"gpu/0/utilization": ...})
```

```python
# GOOD — native MLflow sampler (BP #4)
# In training_launcher env block:
env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
# That's it. Native sampler runs in background, rank-zero gated.
```

### 🚫 Walk run-tree multiple times

```python
# BAD — three independent BFS implementations in mlflow_adapter, deletion, summary_reporter
def _get_sorted_children(self, parent_id):
    return client.search_runs(filter_string=f"tags.mlflow.parentRunId = '{parent_id}'")
```

```python
# GOOD — one canonical walker
walker = RunTreeWalker(run_query=read_client)
descendants = walker.flat_descendants(parent_run_id)
```

### 🚫 Mutate global `mlflow` module mid-process

```python
# BAD
mlflow.set_tracking_uri(uri)        # called from a constructor — sticky URI
mlflow.set_experiment(name)
```

```python
# GOOD
# Only MlflowTransport.__init__ calls mlflow.set_tracking_uri ONCE.
# Experiment is passed per-call to start_run/start_nested_run.
```

### 🚫 Custom resilient transport monkey-patching MLflow SDK

The old `ResilientMLflowTransport` monkey-patched `mlflow.log_metric` / `log_metrics` / `set_tag` etc. globally. This is brittle (breaks on MLflow upgrades) and leaks state (`.runner/buffer.flush_offset` files in workspace).

```python
# GOOD — explicit retry at the transport level via tenacity
class MlflowTransport:
    def __init__(self, ..., retry_total_budget_s: float = 30.0):
        self._retry = tenacity.Retrying(
            stop=tenacity.stop_after_delay(retry_total_budget_s),
            wait=tenacity.wait_exponential(multiplier=0.5, max=8),
            retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
        )
```

### 🚫 Userinfo in tracking URI

```python
# BAD — credentials in URI string (leaks to logs, MLflow tags)
tracking_uri = "https://user:pass@mlflow.example.com"
```

```python
# GOOD — env-var-referenced secrets via auth config
auth = _AuthBasic(kind="basic", username="mlflow_user", password_env_var="MLFLOW_PASSWORD")
```

### 🚫 Deprecated model stages

```python
# BAD
client.transition_model_version_stage(name, version, stage="Production")
```

```python
# GOOD
client.set_registered_model_alias(name, "champion", version)
```

---

## Common LLM mistakes

LLM agents working on this codebase frequently make these mistakes — **don't.**

### 1. Putting `mlflow.set_tracking_uri()` in a constructor

LLMs love to write defensive `set_tracking_uri()` calls "just in case". Don't. It's a process-wide mutation that breaks parallel tests. Only `MlflowTransport.__init__` may call it.

### 2. Calling `mlflow.autolog()` "for convenience"

LLMs see `mlflow.autolog()` in tutorials and add it to "improve metric coverage". Don't — HF Trainer's `report_to=["mlflow"]` already handles everything autolog does, and combining them produces duplicate runs.

### 3. Inventing new `ryotenkai.*` tags ad-hoc

LLMs write `client.set_tags(run_id, {"ryotenkai.custom.thing": "val"})` without adding to the `TagKey` enum first. Add the enum value, then use it.

### 4. Constructing `MlflowClient()` because "DI is overkill for one call"

It isn't. The lint rule `NO_AD_HOC_MLFLOW_CLIENT` rejects this with a clear message. Inject `IRunQuery` via DI.

### 5. Opening a "scratch" run in trainer for debugging

```python
# BAD
with mlflow.start_run(run_name="debug"):
    mlflow.log_metric("debug_value", 42)
```

This breaks Pattern A — HF callback's nested-child semantics rely on `MLFLOW_RUN_ID` being the parent. Use `logger.debug(...)` for debugging, or write to the active run (`mlflow.log_metric(...)` without start_run uses HF's nested child).

### 6. Adding `mlflow.parentRunId` tag manually

```python
# BAD — tag-only cosmetic nesting (old anti-pattern)
client.set_tag(my_run_id, "mlflow.parentRunId", parent_run_id)
```

Don't. Use `start_nested_run()` (or let HF callback do it via `MLFLOW_NESTED_RUN=TRUE` env). MLflow handles the tag automatically when you create a structurally-nested run.

### 7. Reading run data via `mlflow.get_run()` from the global module

```python
# BAD
import mlflow
run = mlflow.get_run(run_id)
```

Use `IRunQuery.get_run(run_id)` via injected `MlflowReadClient`.

### 8. Mocking the Protocols

```python
# BAD
from unittest.mock import MagicMock
client = MagicMock(spec=ITrackingClient)
```

The sentinel `test_no_protocol_mocking` rejects this. Use:

```python
from tests._fakes.mlflow_tracking_client import FakeTrackingClient
client = FakeTrackingClient()
```

### 9. Adding `report_to=["mlflow", "tensorboard"]` and assuming both work

Verify that `MLFLOW_RUN_ID` + `MLFLOW_NESTED_RUN=TRUE` are in env. HF's MLflowCallback respects these; other callbacks may not. If you need TensorBoard alongside, that's fine — but **never** call `mlflow.autolog()`.

### 10. Forgetting `save_pretrained=True` on `mlflow.transformers.log_model`

```python
# BAD — defaults to save_pretrained=False on some versions
mlflow.transformers.log_model(model, artifact_path="model")
```

Always pass `save_pretrained=True` for own-trained models. Without it, MLflow stores only a HF Hub reference; if the upstream model is later deleted/moved, the artifact breaks.

---

## Tests

### Canonical fakes

For each Protocol, use the canonical fake from `tests/_fakes/`:

| Protocol | Fake |
|---|---|
| `ITrackingClient` | `mlflow_tracking_client.FakeTrackingClient` |
| `IMetricSink` | `mlflow_metric_sink.FakeMetricSink` |
| `IArtifactSink` | `mlflow_artifact_sink.FakeArtifactSink` |
| `IRunQuery` | `mlflow_run_query.FakeRunQuery` |
| `IModelRegistry` | `mlflow_model_registry.FakeModelRegistry` |
| `IJournalUploader` | `mlflow_journal_uploader.FakeJournalUploader` |
| `IPromptRegistry` | `mlflow_prompt_registry.FakePromptRegistry` |

### 7-class structure for production methods

Per CLAUDE.md mandate. Each production method (in components dealing with non-trivial behaviour like `MlflowFinalizer.finalize`, `ParentRunOpener.open`) needs:

- `TestPositive` — happy path
- `TestNegative` — error paths
- `TestBoundary` — empty/zero/max values
- `TestInvariants` — constant pins (timeouts, run statuses)
- `TestDependencyErrors` — when `ITrackingClient.ping` raises
- `TestRegressions` — specific bug references
- `TestLogicSpecific` — truth tables for dispatch logic

### Mutation testing gate

```bash
bash scripts/mutation/validate_agent_output.sh
```

Required for any PR touching `packages/*/src/*`. Exit 1 = surviving mutations.

### Lint gates

```bash
# 1. Importlinter
.venv/bin/lint-imports

# 2. MLflow integration rules (AST checker)
.venv/bin/python scripts/lint/mlflow_rules.py packages

# 3. Every-module-has-tests sentinel
.venv/bin/python -m pytest tests/_lint/test_every_module_has_tests.py
```

All three must be green for any PR touching the MLflow integration surface.

---

## Reference files

| Concern | Primary file |
|---|---|
| Protocols (write/read/registry) | `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/protocols.py` |
| Configuration | `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/config.py` + `packages/shared/src/ryotenkai_shared/config/integrations/mlflow_project.py` |
| Taxonomy | `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/taxonomy.py` |
| Transport (write client) | `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/transport.py` |
| Read client | `packages/control/src/ryotenkai_control/pipeline/mlflow/read/client.py` |
| Tree walker | `packages/control/src/ryotenkai_control/pipeline/mlflow/read/tree_walker.py` |
| Lifecycle | `packages/control/src/ryotenkai_control/pipeline/mlflow/lifecycle/{preflight,opener,coord,finalizer}.py` |
| HF wiring | `packages/pod/src/ryotenkai_pod/trainer/mlflow/hf_wiring.py` |
| Model publisher | `packages/pod/src/ryotenkai_pod/trainer/mlflow/model_publisher.py` |
| Model registry | `packages/shared/src/ryotenkai_shared/infrastructure/mlflow/registry.py` |
| System prompt loader | `packages/shared/src/ryotenkai_shared/inference/prompts/system_prompt_loader.py` |
| Reports composer + slices | `packages/control/src/ryotenkai_control/reports/composer.py` + `packages/control/src/ryotenkai_control/reports/slices/` |
| Lint rules | `scripts/lint/mlflow_rules.py` |
| Plan / context | `docs/plans/vectorized-fluttering-mist.md` |
