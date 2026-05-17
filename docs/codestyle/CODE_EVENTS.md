# Event system in this project

Guidelines for working with typed events for observability, replay, and reporting.

---

## Contents

- [Instructions for LLMs](#instructions-for-llms)
- [Architecture overview](#architecture-overview)
- [Naming convention](#naming-convention)
- [Envelope structure](#envelope-structure)
- [When to emit (and when NOT to)](#when-to-emit-and-when-not-to)
- [How to add a new event type](#how-to-add-a-new-event-type)
- [DI / wiring](#di--wiring)
- [Anti-patterns](#anti-patterns)
- [Common LLM mistakes](#common-llm-mistakes)
- [Tests](#tests)

---

## Instructions for LLMs

> This document is for developers and LLM agents. When generating or refactoring event-emitting code in this project — read and apply the rules below.

**Quick reference:** typed Pydantic event with `kind: Literal[...]` discriminator | emit via `IEventEmitter`, never raw bus.publish | `extra="forbid"` + `frozen=True` on every event AND payload | `kind` field name (not `type` — sentinel `tests/_lint/test_discriminator_uniformity.py` enforces) | UNKNOWN_OFFSET=-1 sentinel when caller doesn't know offset

**When editing code in this project, follow these rules.**

### Mandatory rules (MUST)

1. **Never `bus.publish(kind: str, payload: dict)`.** Always build a typed Pydantic event and emit through `IEventEmitter.emit(envelope)`. The raw `bus.publish(BaseEvent)` exists for the bus's internal contract; callers go through the emitter.
2. **Never re-use an existing `kind` for a semantically different event.** If shape changes — new dotted `kind`, never reuse. Schema evolution rule is fixed by ADR-0009.
3. **Never put `traceback.format_exc()` output in `context` / `payload` fields meant for structured data.** Traceback goes only into a dedicated `traceback_excerpt: str` field, truncated to 2048 bytes.
4. **Never mock `IEventEmitter` via `unittest.mock`.** Use `tests/_fakes/event_emitter.FakeEventEmitter`. Sentinel `tests/_lint/test_no_protocol_mocking.py` enforces.
5. **Never emit logs `logger.info("training started")` as the sole signal.** If the event is operator-visible OR consumed by reports / UI / CLI — emit a typed event. Logs are complementary, not the source of truth.
6. **Never edit `events.jsonl` directly.** Append-only journal. Truncation happens only via `JournalReader.truncate_torn_tail` (atomic tmp+rename), never `open(path, "w")`.

### Quick choice tree

| Situation | Action |
|-----------|--------|
| Adding lifecycle event for a stage / pipeline phase | Add typed event to `packages/shared/src/ryotenkai_shared/events/types/<domain>.py` |
| Recording a one-off operator message that nobody else consumes | `logger.info(...)` (NOT event) |
| Code on pod needs to tell control "something happened" | Emit via `PodEventEmitter` → flows over HTTP loopback → WS → control journal |
| Stage finished, report needs to show it | Stage emits `*StartedEvent` / `*CompletedEvent` / `*FailedEvent` via injected `IEventEmitter` |
| Need real-time UI on frontend | Don't add new transport — events.jsonl + SSE `/api/v1/runs/{id}/events/stream` already covers it |
| Need persistent record for cross-machine report | Already covered — `MlflowFinalizer` uploads `events.jsonl` as MLflow artifact on run end |

---

## Architecture overview

```
┌───────────────────────────────────┐                ┌──────────────────────────────────┐
│ Pod-side (RunPod / single_node)   │                │ Control-side (Mac)               │
│                                   │                │                                  │
│  trainer subprocess               │                │  orchestrator                    │
│   │ HF Trainer callbacks fire     │                │   │ stage.run()                  │
│   ▼                               │                │   ▼                              │
│  RunnerEventCallback              │                │  ControlEventEmitter             │
│   │ build TypedEvent              │                │   │ emit(typed_event)            │
│   ▼ async queue + HTTP loopback   │                │   ├─► JournalWriter (events.jsonl)│
│  PodEventEmitter                  │                │   │   length-prefix JSONL, fsync │
│   │ assign offset (threading.Lock)│                │   └─► InMemoryBus (Ray-style)    │
│   ├─► EventBus (ring buffer 10k)  │                │       │                          │
│   └─► EventJournal (disk JSONL)   │                │       ▼                          │
│       │                           │                │  ┌─ SSE subscribers              │
│       ▼                           │                │  │  (frontend, CLI --follow)     │
│  WebSocket → /api/v1/jobs/{id}/   │ ─────WS────► │  ├─ EventEmitterRegistry          │
│  events  (incremental)            │                │  │  (process-wide singleton)     │
│                                   │ ◄────HTTP─── │  └─ MlflowFinalizer               │
│  GET /events/replay (NDJSON)      │   on gap       │     uploads journal + manifest   │
│  fallback for long Mac sleeps     │                │     to MLflow on run end         │
└───────────────────────────────────┘                └──────────────────────────────────┘
```

**Sources of truth:**
- Live: `workspace/runs/<run_id>/events.jsonl` on Mac (control SSOT).
- Pod-only buffer: pod-local journal for recovery if WS gap exceeds bus capacity.
- Long-term: MLflow artifact `events/events.jsonl` + `events_manifest.json` uploaded at run end.

**Packages:**
- `packages/shared/src/ryotenkai_shared/events/` — envelope, codec, discriminator, IEventEmitter Protocol, all 57+ typed event classes, upcasters. Imported by both pod and control.
- `packages/pod/src/ryotenkai_pod/runner/` — `event_bus`, `event_journal`, `event_emitter` (PodEventEmitter), API routers.
- `packages/control/src/ryotenkai_control/events/` — JournalWriter, JournalReader, InMemoryBus, EventDedup, ControlEventEmitter, EventEmitterRegistry, MlflowFinalizer, DedupTTLSweeper.

---

## Naming convention

Format: `ryotenkai.<area>.<domain>.<verb_past_or_noun>`

- **area**: `pod` | `control`
- **domain**: stage / subsystem (`training`, `memory`, `health`, `stage`, `run`, `gpu`, `inference`, `evaluation`, `dataset`, `model`, `lifecycle`, `journal`)
- **verb_past**: for facts (`started`, `completed`, `failed`, `cancelled`, `skipped`, `interrupted`, `deployed`, `deactivated`, `provisioned`, `synced`, `preempted`, `cleared`, `detected`)
- **noun**: for snapshots (`snapshot`, `step`, `log`)

Examples:
- `ryotenkai.pod.training.started`
- `ryotenkai.pod.training.checkpoint_saved`
- `ryotenkai.pod.memory.oom_detected`
- `ryotenkai.control.stage.completed`
- `ryotenkai.control.gpu.cleanup_failed`

**Anti-examples:**
- `pod.training.starting` (present participle — not allowed)
- `ryotenkai.pod.training.start_event` (redundant suffix)
- `ryotenkai.training.started` (missing area prefix)
- `ryotenkai.pod.training_started` (underscore instead of dot for domain boundary)

---

## Envelope structure

Every event extends `BaseEvent` (`shared/events/envelope.py`):

```python
class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_id: UUID = Field(default_factory=new_uuid7)
    kind: str                                 # discriminator (Literal in subclass)
    source: str                               # "pod://{run_id}/trainer" | "control://orchestrator"
    time: datetime = Field(default_factory=utc_now)
    run_id: str
    stage_id: str | None = None               # auto-filled from ContextVar
    offset: int                               # monotonic per (run_id, source)
    schema_version: int = 1
    severity: Severity                        # pinned via Literal in subclass
```

Every concrete event class:

```python
class TrainingStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_steps: int
    num_train_epochs: int
    per_device_batch_size: int
    learning_rate: float
    algorithm: Literal["sft", "cpt", "dpo", "grpo", "sapo"]


class TrainingStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.training.started"] = "ryotenkai.pod.training.started"
    severity: Literal["info"] = "info"
    payload: TrainingStartedPayload
```

**Rules:**
- `kind` Literal — pinned default = name itself.
- `severity` Literal — pinned default, one of `debug` / `info` / `warning` / `error` / `critical`.
- `payload` — separate Pydantic class, also `frozen + extra=forbid`.
- Payload field types: only `str`, `int`, `float`, `bool`, `None`, `datetime`, `UUID`, `Literal[...]`, or nested frozen Pydantic models. No `Any`, no untyped dicts (except `UnknownEvent.raw_payload` — special catch-all only).

**Severity defaults:**
- `started`/`completed` → `info`
- `failed` / `oom_detected` / `preempted` → `error`
- `warning` / `pressure_warning` / `threshold_reached` → `warning`
- `step` / `log` / `snapshot` → `debug`
- `cancelled` / `interrupted` → `warning`

---

## When to emit (and when NOT to)

### EMIT a typed event when

- A stage starts, completes, or fails (lifecycle facts that reports / UI must show).
- A user-visible milestone happens (checkpoint saved, model deployed, evaluation finished).
- A resource transitions state (GPU provisioned, pod stopped, container removed).
- An error happens that operators may want to filter or alert on.
- A metric snapshot is taken at a meaningful boundary (epoch end, eval, checkpoint).

### DO NOT emit an event when

- It's a per-tick debug log nobody consumes (`logger.debug(f"checking {i}")`).
- It would duplicate a `mlflow.log_metric()` call exactly. Metrics stay in MLflow; events carry state transitions.
- The "event" is just intermediate state inside a single function (no cross-component consumer).
- It's a private internal counter (use the existing observability counters in `ControlEventEmitter` / `InMemoryBus` / etc.).

**Rule of thumb:** if no consumer (report builder / UI / CLI / external tool) will ever read it — it's a log line, not an event.

---

## How to add a new event type

1. **Pick a `kind` name** following the naming convention. Check it doesn't exist (`grep -rn "Literal\[.\"ryotenkai\." packages/shared/src/ryotenkai_shared/events/types/`).

2. **Add to the right domain file** under `packages/shared/src/ryotenkai_shared/events/types/`:

   ```python
   # types/control_evaluation.py (or wherever it belongs)
   class EvaluationCheckpointVerifiedPayload(BaseModel):
       model_config = ConfigDict(frozen=True, extra="forbid")
       checkpoint_path: str
       checksum: str
       size_bytes: int


   class EvaluationCheckpointVerifiedEvent(BaseEvent):
       kind: Literal["ryotenkai.control.evaluation.checkpoint_verified"] = "ryotenkai.control.evaluation.checkpoint_verified"
       severity: Literal["info"] = "info"
       payload: EvaluationCheckpointVerifiedPayload
   ```

3. **Register in discriminator** — add to `_ALL_EVENTS` tuple in `packages/shared/src/ryotenkai_shared/events/discriminator.py`, grouped by domain comment.

4. **Re-export** — add to `types/__init__.py` `__all__`.

5. **Add unit tests** in `tests/unit/shared/events/types/test_<domain>.py`:
   - `TestPositive`: round-trip serialize → deserialize via `EVENT_ADAPTER`.
   - `TestNegative`: missing required payload field → ValidationError.
   - `TestInvariants`: `kind` Literal pinned, `severity` default, `extra="forbid"` rejects unknown.

6. **Wire emit** in the producer code:

   ```python
   from ryotenkai_shared.events import UNKNOWN_OFFSET
   from ryotenkai_shared.events.types.control_evaluation import (
       EvaluationCheckpointVerifiedEvent,
       EvaluationCheckpointVerifiedPayload,
   )

   with self._emitter.stage_scope("evaluator"):
       self._emitter.emit(
           EvaluationCheckpointVerifiedEvent(
               source="control://orchestrator/evaluator",
               run_id=self._run_id,
               offset=UNKNOWN_OFFSET,                # emitter assigns
               payload=EvaluationCheckpointVerifiedPayload(
                   checkpoint_path=str(path),
                   checksum=sha,
                   size_bytes=size,
               ),
           )
       )
   ```

7. **Run sentinels** before committing:
   ```bash
   .venv/bin/python -m pytest tests/_lint -q
   ```

---

## DI / wiring

`IEventEmitter` is injected via DI through `PipelineBootstrap.build()` (`packages/control/src/ryotenkai_control/pipeline/bootstrap/pipeline_bootstrap.py`).

Stage constructors accept `emitter: IEventEmitter` (NOT optional in production):

```python
class MyStage(PipelineStage):
    def __init__(self, *, emitter: IEventEmitter, run_id: str, ...):
        self._emitter = emitter
        self._run_id = run_id
```

For providers (`SingleNodeProvider`, `RunpodProvider`) — emitter is bound lazily via `set_emitter()` after gpu_deployer creates them (provider construction predates emitter availability in current flow).

For tests, inject `FakeEventEmitter` from `tests/_fakes/event_emitter.py`. The fake is `IEventEmitter`-compatible at runtime (`@runtime_checkable Protocol`).

---

## Anti-patterns

### ❌ Open-dict payload

```python
# WRONG — re-introduces Phase 1-10's removed string-sniffing bug
bus.publish("training_step", {"step": 100, "loss": 0.42})
```

```python
# RIGHT
emitter.emit(TrainingStepEvent(
    source="pod://run-abc/trainer",
    run_id=run_id,
    offset=UNKNOWN_OFFSET,
    payload=TrainingStepPayload(step=100, loss=0.42, learning_rate=2e-5, grad_norm=0.5),
))
```

### ❌ Reusing `kind` for different shapes

```python
# WRONG — breaks consumers expecting v1 payload
class TrainingStartedPayloadV2(BaseModel):
    total_steps: int        # renamed from max_steps
    # ...

class TrainingStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.training.started"] = "ryotenkai.pod.training.started"
    payload: TrainingStartedPayloadV2          # ← silent break
```

```python
# RIGHT — new dotted kind, OR additive optional field
class TrainingStartedPayload(BaseModel):
    max_steps: int                              # keep
    total_steps: int | None = None              # additive, optional, default
```

For renames or removals: register an upcaster in `shared/events/upcasters/` and bump `schema_version`. Never silently mutate an existing `kind`'s payload.

### ❌ Putting traceback into `context`

```python
# WRONG
emitter.emit(StageFailedEvent(
    payload=StageFailedPayload(
        stage_name="dataset_validator",
        error_type="ValueError",
        message="bad input",
        traceback_excerpt=traceback.format_exc(),   # ← OK here only
        # ⚠ never duplicate into a "context" payload field
    )
))
```

Traceback lives ONLY in `traceback_excerpt: str`, truncated to 2048 bytes. Never embed in arbitrary string fields.

### ❌ Logging instead of emitting (for operator-visible events)

```python
# WRONG — invisible to reports, UI, CLI
logger.info("inference deployed at https://example/v1")
```

```python
# RIGHT
emitter.emit(InferenceDeployedEvent(
    payload=InferenceDeployedPayload(
        endpoint="https://example/v1",
        model_id="qwen-2.5-0.5b-tuned",
    )
))
# logger.info still allowed as complement, but event is the source of truth
```

### ❌ Calling `bus.publish_legacy` for new code

`publish_legacy(kind, payload)` exists ONLY as a backward-compat shim for 7 pre-existing pod telemetry kinds (`cancellation_started`, `watchdog_timeout`, etc.). Never use for new emissions. Add a typed event class instead.

### ❌ Mocking `IEventEmitter`

```python
# WRONG — sentinel `test_no_protocol_mocking.py` blocks this
emitter = MagicMock(spec=IEventEmitter)
```

```python
# RIGHT
from tests._fakes.event_emitter import FakeEventEmitter
emitter = FakeEventEmitter()
stage = MyStage(emitter=emitter, run_id="test")
stage.run()
events = [e for e in emitter.emitted if isinstance(e, MyExpectedEvent)]
```

---

## Common LLM mistakes

When generating event-emitting code, LLMs frequently produce these — DO NOT:

1. **Inventing a new `kind` field name (`type`, `event_type`, `name`).** Always `kind`. Sentinel enforces.
2. **Omitting `extra="forbid"`** on the payload model. Always include `model_config = ConfigDict(frozen=True, extra="forbid")`.
3. **Putting `Any` / `dict[str, Any]` / open kwargs in payload fields.** Pin scalar types or nested frozen models. The only exception is `UnknownEvent.raw_payload`.
4. **Forgetting `offset=UNKNOWN_OFFSET`.** Caller must pass `UNKNOWN_OFFSET` (= -1) — emitter rewrites it under its lock.
5. **Forgetting `source`.** Required field — caller supplies `"pod://<run_id>/<role>"` or `"control://orchestrator/<stage>"`. Emitter never overwrites it.
6. **Forgetting `stage_scope`.** Stage code should wrap emissions in `with self._emitter.stage_scope("stage_name"):` so `stage_id` is auto-populated via ContextVar.
7. **Re-using existing `kind` for semantically different payload.** Adds silent breaking change. New kind required.
8. **Writing to `events.jsonl` directly with `open(path, "a").write(json)`.** Always go through `JournalWriter.append(event)` — preserves length-prefix framing.
9. **Skipping the `IEventEmitter` Protocol — calling `bus.publish` directly from a stage.** Stages do not own the bus. They own the emitter.
10. **Generating a new event without registering in `discriminator.py`.** The `_ALL_EVENTS` tuple must include the new class; otherwise `EVENT_ADAPTER` validation falls through to `UnknownEvent`.

---

## Tests

### Required test coverage for a new event

- Round-trip test: `EVENT_ADAPTER.validate_python(json.loads(event.model_dump_json())) == event`.
- Negative: payload missing required field → `ValidationError`.
- Negative: extra payload field → `ValidationError` (proves `extra="forbid"`).
- Invariant: `kind` Literal default is the dotted name; `severity` default matches spec.

For critical event modules (`codec.py`, `journal_writer.py`, `in_memory_bus.py`, `emitter.py`, `dedup.py`) — apply the project's 7-class test template (TestPositive / Negative / Boundary / Invariants / DependencyErrors / Regressions / LogicSpecific) and aim for ≥95% mutation kill rate.

### Required canonical fake

Production `IEventEmitter` is `@runtime_checkable Protocol`. Tests use `tests/_fakes/event_emitter.FakeEventEmitter`. It auto-fills `offset` per source, supports `stage_scope` via ContextVar, exposes `emitted` and `received_remote` lists, supports `inject_emit_failure(n)` and `inject_validation_failure(n)` for chaos injection.

---

## See also

- `docs/adrs/2026-05-17-unified-event-system.md` — architectural decision record (rationale, risk ledger, rejected alternatives).
- `packages/shared/src/ryotenkai_shared/events/README.md` — shared/events developer guide.
- `packages/control/src/ryotenkai_control/events/README.md` — control-side specifics.
- `docs/codestyle/CODE_ERRORS.md` — typed error hierarchy (events carry error info via `error_type` / `message` / `traceback_excerpt` from those exceptions).
