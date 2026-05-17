# Unified Event System for RyotenkAI

**Status:** Executed in 1 day (2026-05-17). All 10 phases complete. See [ADR-0009](../adrs/2026-05-17-unified-event-system.md).

## Context

**Зачем.** Сейчас в проекте сосуществуют **три параллельные event-подсистемы**:
1. **Pod EventBus + EventJournal** (`packages/pod/src/ryotenkai_pod/runner/event_bus.py`, `event_journal.py`) — in-memory ring buffer 10k events + ротирующийся JSONL на диск (5×100MiB). Live-стриминг через WebSocket. Формат: `{offset, ts, kind, payload}` (open dict).
2. **Pod MLflowEventLog** (`packages/pod/src/ryotenkai_pod/trainer/mlflow/event_log.py`) — отдельный in-memory list, в конце run flush'ится как MLflow artifact `training_events.json`. OTEL-like формат `{timestamp, event_type, severity, severity_number, message, category, source, attributes}`.
3. **Control StageArtifactEnvelope** (`packages/control/src/ryotenkai_control/pipeline/artifacts/base.py`) — per-stage `*_results.json` в MLflow. `TimelineEvent` синтезируется в момент генерации отчёта string-парсингом.

**Проблемы (КРИТ).**
- **Dual-path logging:** `training_started`, `checkpoint_saved`, `epoch_*` публикуются и в EventBus (через `RunnerEventCallback` HTTP loopback), и в MLflowEventLog (через `TrainingEventsCallback`). При отказе одного канала события расходятся.
- **String sniffing:** `mlflow_adapter.py:712-715` — `if "cache cleared" in msg: etype="cache_clear"` для классификации memory events.
- **HTTP backpressure:** `RunnerEventCallback._flush()` после 3 фейлов self-disable — все события до конца сессии теряются, trainer продолжает работу слепо.
- **Coverage gaps:** `inference_deployer.py`, `deployment_manager.py`, pre-flight checks эмитят только в `logger.info()` — невидимы для отчётов и frontend.

**Цель.** Заменить три подсистемы на **единую** с одним SSOT, одной типизированной schema, едиными API для consumers (отчёты, frontend, CLI, MLflow). На перспективу — стабильная база под фронт, расширение plugin-событий, observability.

**Что НЕ в скоупе этого PR (out of scope).**
- Изменения во frontend (`web/`). Существующий polling `GET /job/events` остаётся рабочим; новый SSE-эндпоинт добавляется параллельно — фронт переключится в отдельном PR.
- Replay для старых runs до merge: `training_events.json` artifacts от старых runs не будут читаться новым report adapter (обратная совместимость явно отброшена согласно требованиям).

---

## Архитектурные решения (зафиксированы пользователем)

1. **Полная переработка** — один большой PR, закрываем все 3 КРИТ-бага, выкатываем единую систему. ~11-12 рабочих дней.
2. **Гибрид storage**: local JSONL-журнал на Mac как hot-path SSOT (`workspace/runs/<run_id>/events.jsonl`) + копия в MLflow артефакт (`events.jsonl`) на завершении run для долговременного хранения.
3. **SSE для frontend** (одностороний server→client, нативный reconnect через `Last-Event-ID`). WebSocket остаётся для CLI (legacy, pod-side уже WS).
4. **Closed discriminated union** — каждый event-тип отдельный Pydantic v2 класс с `type: Literal[...]`. `extra="forbid"` на payload. Закрытое множество известных типов; `UnknownEvent` catch-all для forward-compat.

---

## Event taxonomy (55 типов + UnknownEvent)

**Конвенции наименования.** Dotted reverse-DNS: `ryotenkai.<area>.<domain>.<verb_past_or_noun>`. Area ∈ `{pod, control}`. Verb — past tense для фактов (`started`, `completed`, `failed`), noun — для снапшотов (`snapshot`, `step`).

**Severity defaults:** `started`/`completed` → `info`; `failed`/`oom_detected`/`preempted` → `error`; `warning`/`pressure_warning` → `warning`; `step`/`log`/`snapshot` → `debug`; `cancelled`/`interrupted` → `warning`.

### Pod domain (24 типа)

| Type | Payload | Producer | Severity |
|------|---------|----------|----------|
| `ryotenkai.pod.lifecycle.runner_started` | `version`, `git_sha`, `gpu_count` | runner | info |
| `ryotenkai.pod.lifecycle.runner_shutdown` | `reason`, `graceful: bool` | runner | info |
| `ryotenkai.pod.lifecycle.job_submitted` | `job_id`, `config_hash`, `image_tag` | runner | info |
| `ryotenkai.pod.lifecycle.trainer_spawned` | `pid`, `cmdline`, `cwd` | runner | info |
| `ryotenkai.pod.lifecycle.trainer_spawn_failed` | `reason`, `exit_code` | runner | error |
| `ryotenkai.pod.lifecycle.trainer_exited` | `exit_code`, `signal`, `duration_s` | runner | info |
| `ryotenkai.pod.training.started` | `max_steps`, `num_train_epochs`, `per_device_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `algorithm` | trainer | info |
| `ryotenkai.pod.training.epoch_started` | `epoch`, `global_step` | trainer | info |
| `ryotenkai.pod.training.epoch_completed` | `epoch`, `global_step`, `mean_loss`, `duration_s` | trainer | info |
| `ryotenkai.pod.training.step` | `step`, `loss`, `learning_rate`, `grad_norm`, `tokens_per_sec`, `samples_per_sec` | trainer | debug |
| `ryotenkai.pod.training.log` | `step`, `metrics: dict[str, float]` | trainer | debug |
| `ryotenkai.pod.training.eval_metrics` | `step`, `metrics`, `dataset_name` | trainer | info |
| `ryotenkai.pod.training.checkpoint_saved` | `step`, `local_path`, `size_bytes`, `is_best` | trainer | info |
| `ryotenkai.pod.training.completed` | `final_step`, `mean_loss`, `duration_s`, `tokens_processed` | trainer | info |
| `ryotenkai.pod.training.failed` | `error_type`, `message`, `traceback_excerpt`, `step` | trainer | error |
| `ryotenkai.pod.memory.cache_cleared` | `device`, `before_bytes`, `after_bytes`, `trigger ∈ {scheduled,threshold,manual}` | trainer | info |
| `ryotenkai.pod.memory.oom_detected` | `device`, `allocated_bytes`, `reserved_bytes`, `step` | trainer | error |
| `ryotenkai.pod.memory.pressure_warning` | `device`, `utilization_pct`, `threshold_pct` | trainer | warning |
| `ryotenkai.pod.memory.threshold_reached` | `device`, `metric`, `value`, `threshold`, `action_taken` | trainer | warning |
| `ryotenkai.pod.health.snapshot` | `cpu_pct`, `ram_bytes`, `gpu[]`, `disk_free_bytes` | runner heartbeat | debug |
| `ryotenkai.pod.health.idle_detected` | `idle_duration_s`, `last_activity_at` | runner | warning |
| `ryotenkai.pod.health.max_lifetime_reached` | `started_at`, `max_lifetime_s` | runner | warning |
| `ryotenkai.pod.io.trainer_stdout` | `line`, `stream` (debug-only, gated) | runner | debug |
| `ryotenkai.pod.io.trainer_stderr` | `line`, `stream` (debug-only, gated) | runner | debug |

### Control domain (31 тип)

| Type | Payload | Producer | Severity |
|------|---------|----------|----------|
| `ryotenkai.control.run.started` | `run_name`, `algorithm`, `model_id`, `dataset_id`, `config_hash` | orchestrator | info |
| `ryotenkai.control.run.completed` | `duration_s`, `final_status`, `mlflow_run_id` | orchestrator | info |
| `ryotenkai.control.run.failed` | `failing_stage`, `error_type`, `message`, `traceback_excerpt` | orchestrator | error |
| `ryotenkai.control.run.cancelled` | `reason`, `cancelled_at_stage` | orchestrator | warning |
| `ryotenkai.control.stage.started` | `stage_name`, `stage_index`, `total_stages`, `inputs_summary` | each stage entry | info |
| `ryotenkai.control.stage.completed` | `stage_name`, `duration_s`, `outputs_summary` | each stage exit | info |
| `ryotenkai.control.stage.failed` | `stage_name`, `error_type`, `message`, `traceback_excerpt`, `retry_count` | each stage on exception | error |
| `ryotenkai.control.stage.skipped` | `stage_name`, `reason` | orchestrator | info |
| `ryotenkai.control.stage.interrupted` | `stage_name`, `signal`, `cleanup_completed` | signal handler | warning |
| `ryotenkai.control.dataset.validation_started` | `dataset_path`, `validator_chain` | dataset stage | info |
| `ryotenkai.control.dataset.validation_completed` | `num_samples`, `num_rejected`, `schema_version`, `checks_passed[]` | dataset stage | info |
| `ryotenkai.control.dataset.validation_failed` | `failed_check`, `details` | dataset stage | error |
| `ryotenkai.control.gpu.deployment_started` | `provider`, `gpu_type`, `gpu_count`, `region` | gpu_deployer | info |
| `ryotenkai.control.gpu.deployment_completed` | `instance_id`, `endpoint`, `provision_duration_s`, `cost_per_hour_usd` | gpu_deployer | info |
| `ryotenkai.control.gpu.deployment_failed` | `reason`, `provider_error_code` | gpu_deployer | error |
| `ryotenkai.control.gpu.preempted` | `instance_id`, `preemption_reason` | provider watcher | error |
| `ryotenkai.control.gpu.ssh_provisioned` | `host`, `key_fingerprint` | single_node provider | info |
| `ryotenkai.control.gpu.code_synced` | `local_sha`, `remote_sha`, `bytes_transferred` | provider | info |
| `ryotenkai.control.training.monitor_started` | `pod_endpoint`, `poll_interval_s` | monitor stage | info |
| `ryotenkai.control.training.monitor_timeout` | `last_event_at`, `timeout_s` | monitor stage | error |
| `ryotenkai.control.model.retrieval_started` | `source_path`, `target_path` | retriever stage | info |
| `ryotenkai.control.model.retrieval_completed` | `bytes_transferred`, `duration_s`, `checksum` | retriever stage | info |
| `ryotenkai.control.evaluation.started` | `plugin_names[]`, `model_path` | evaluation stage | info |
| `ryotenkai.control.evaluation.plugin_started` | `plugin_name`, `plugin_version` | evaluator | info |
| `ryotenkai.control.evaluation.plugin_completed` | `plugin_name`, `metrics`, `duration_s` | evaluator | info |
| `ryotenkai.control.evaluation.plugin_failed` | `plugin_name`, `error_type`, `message` | evaluator | error |
| `ryotenkai.control.evaluation.completed` | `aggregated_metrics`, `total_duration_s` | evaluation stage | info |
| `ryotenkai.control.inference.deployment_started` | `target ∈ {vllm,sglang,hf_endpoint}`, `model_path` | inference_deployer | info |
| `ryotenkai.control.inference.health_check_completed` | `endpoint`, `latency_ms`, `model_loaded: bool` | inference_deployer | info |
| `ryotenkai.control.inference.deployed` | `endpoint`, `api_key_ref`, `model_id` | inference_deployer | info |
| `ryotenkai.control.inference.deactivated` | `endpoint`, `reason` | deployment_manager | info |

### Pydantic envelope (общий для всех)

```python
class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_id: UUID                                   # UUIDv7 (natural ordering)
    type: str                                        # discriminator (Literal in subclasses)
    source: str                                      # "pod://{run_id}/trainer" | "control://orchestrator/{stage}"
    time: datetime                                   # UTC microseconds
    run_id: str
    stage_id: str | None = None                      # related-resource (ContextVar-populated)
    offset: int                                      # monotonic per (run_id, source)
    schema_version: int = 1
    severity: Literal["debug","info","warning","error","critical"]

class TrainingStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_steps: int
    num_train_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    algorithm: Literal["sft","cpt","dpo","grpo","sapo"]

class TrainingStartedEvent(BaseEvent):
    type: Literal["ryotenkai.pod.training.started"] = "ryotenkai.pod.training.started"
    severity: Literal["info"] = "info"
    payload: TrainingStartedPayload

# Catch-all для forward-compat (см. R-11 в Risk Ledger)
class UnknownEvent(BaseEvent):
    type: Literal["ryotenkai.unknown"] = "ryotenkai.unknown"
    original_type: str
    raw_payload: dict[str, Any]

Event = Annotated[
    TrainingStartedEvent | TrainingStepEvent | ... | UnknownEvent,
    Discriminator("type"),
]
```

---

## Module layout

```
packages/shared/src/ryotenkai_shared/events/
├── __init__.py                # re-exports: BaseEvent, Event, EventAdapter, IEventEmitter, UnknownEvent
├── envelope.py                # BaseEvent + severity Literal + UUIDv7 helper
├── codec.py                   # to_jsonl(event)->str; from_jsonl(line, strict=False)->Event|UnknownEvent
├── discriminator.py           # Event union assembly + TypeAdapter[Event] (cached singleton)
├── emitter.py                 # IEventEmitter Protocol (interface only)
├── upcasters/
│   ├── __init__.py            # registry + chain runner
│   ├── _types.py              # Upcaster = Callable[[dict, int, int], dict]
│   └── README.md              # convention docs for adding upcasters
└── types/
    ├── __init__.py            # re-exports + assembles Event union
    ├── pod_lifecycle.py
    ├── pod_training.py
    ├── pod_memory.py
    ├── pod_health.py
    ├── pod_io.py
    ├── control_run.py
    ├── control_stage.py
    ├── control_dataset.py
    ├── control_gpu.py
    ├── control_training.py
    ├── control_model.py
    ├── control_evaluation.py
    └── control_inference.py

packages/pod/src/ryotenkai_pod/runner/
├── event_bus.py               # REWIRE: bounded deque + multi-consumer cursors + envelope schema
├── event_journal.py           # FORMAT: envelope-per-line JSONL; truncate-invalid-on-resume
└── event_emitter.py           # NEW: PodEventEmitter (IEventEmitter impl)

packages/pod/src/ryotenkai_pod/trainer/
├── mlflow/event_log.py        # DELETE (second SSOT eliminated)
└── callbacks/
    ├── training_events_callback.py   # DELETE (dual-path eliminated)
    └── runner_event_callback.py      # REWIRE: envelope, async queue (no self-disable)

packages/control/src/ryotenkai_control/events/
├── __init__.py
├── emitter.py                 # ControlEventEmitter (IEventEmitter impl)
├── journal_writer.py          # append+batched-fsync; length-prefixed lines
├── journal_reader.py          # tail-able; truncate partial last line on init
├── in_memory_bus.py           # Ray-style MultiConsumerEventBuffer
├── mlflow_finalizer.py        # upload events.jsonl + events_manifest.json on run end
├── dedup.py                   # (run_id, source, offset) dedup with reconstruction-on-restart
└── replay.py                  # offset-range slice for HTTP replay

packages/control/src/ryotenkai_control/api/routers/
└── events.py                  # NEW: GET /runs/{id}/events ; GET /runs/{id}/events/stream (SSE)

packages/control/src/ryotenkai_control/reports/adapters/
├── mlflow_adapter.py          # SIMPLIFY: no event_log artifact parsing
└── journal_adapter.py         # NEW: reads workspace journal + MLflow artifact fallback

packages/control/src/ryotenkai_control/pipeline/
├── orchestrator.py            # injects ControlEventEmitter + ContextVar stage_scope
├── stages/                    # все callback dataclasses удалены, emitter.emit() везде
└── artifacts/base.py          # StageArtifactEnvelope: timeline_events удалено
```

**Import-lint compliance (CLAUDE.md):** все Protocols и DTO в `shared/events/`. Concrete implementations — в `pod/` или `control/`. Control никогда не импортирует pod-concrete.

---

## Data flow

### 1. Pod trainer → control SSOT

```
trainer subprocess
  │ HF Trainer callback fires on_step_end
  ▼
RunnerEventCallback._on_step_end()
  │ builds TrainingStepEvent (offset=None, time=now)
  │ enqueue to background async-queue (bounded 10k, drop-oldest, NON-blocking)
  ▼
async background task: POST /internal/events to pod runner (loopback 127.0.0.1)
  │
  ▼
PodEventEmitter.emit(event)
  │ 1. assigns offset (monotonic counter per (run_id,source), locked)
  │ 2. PodEventJournal.append(envelope_json) ─► pod local JSONL (recovery)
  │ 3. EventBus.publish(event)                ─► in-memory ring (multi-consumer)
  ▼
WebSocket: control JobClient receives frame
  │
  ▼
control listener task → ControlEventEmitter.emit_remote(event)
  │ 1. validate envelope (TypeAdapter[Event])
  │ 2. dedup by (run_id, source, offset)      ─► drop if seen
  │ 3. JournalWriter.append(line)             ─► workspace/runs/<run_id>/events.jsonl (SSOT)
  │ 4. InMemoryBus.publish(event)             ─► for SSE subscribers
```

### 2. Control orchestrator → SSOT

```
orchestrator.run_pipeline()
  │ ContextVar current_stage_id.set("gpu_deployer")
  ▼
emitter.emit(StageStartedEvent(payload=...))
  │
  ▼
ControlEventEmitter.emit(event)
  │ 1. fill event_id (UUIDv7), time, source ("control://orchestrator/{stage}")
  │ 2. fill stage_id from ContextVar
  │ 3. assign offset (monotonic per (run_id,"control://orchestrator"))
  │ 4. JournalWriter.append (immediate fsync for severity ≥ error)
  │ 5. InMemoryBus.publish
```

### 3. Frontend SSE subscribe

```
Browser opens EventSource("/api/runs/{id}/events/stream?after_offset=42")
  │
  ▼
FastAPI GET /runs/{id}/events/stream
  │ 1. Validate run_id exists
  │ 2. SUBSCRIBE to InMemoryBus FIRST (track current_max_offset = M)
  │ 3. JournalReader.replay_from(after=42, until=M)  ── historical events
  │ 4. Drain bus from cursor=M, filter by predicates, forward
  │ 5. Loop: yield "id: {offset}\nevent: {type}\ndata: {json}\n\n"
  │    : keepalive every 15s
  ▼
Browser EventSource
  │ on disconnect: auto-reconnect with Last-Event-ID header → step 1
```

Subscribe-first-then-replay (шаг 2 перед шагом 3) закрывает race "event между catchup и live phase".

### 4. End of run → MLflow

```
orchestrator finally block:
  │
  ▼
emitter.emit(RunCompletedEvent | RunFailedEvent | RunCancelledEvent)
  │
  ▼
JournalWriter.close()
  │ fsync final batch
  ▼
MLflowFinalizer.upload(run_id)
  │ 1. compute events_manifest.json:
  │    {schema_version, total_events, first_offset, last_offset,
  │     type_histogram, schema_versions_present, events_sha256,
  │     journal_complete: bool}
  │ 2. mlflow.log_artifact("events.jsonl", "events/")    (with retry policy)
  │ 3. mlflow.log_artifact("events_manifest.json", "events/")
  │
  ▼
[local journal stays until workspace retention policy expires]
```

---

## EventEmitter API

```python
class IEventEmitter(Protocol):
    def emit(self, event: BaseEvent) -> None:
        """Append event to journal and publish to bus.

        Contracts:
          - sync, returns after enqueue (publish async internally)
          - thread-safe (multiple stages may emit concurrently)
          - NEVER raises to caller; on internal error logs + increments
            events_emit_failed_total{reason}
          - if event.offset is None — assigns next monotonic offset
          - if event.event_id is None — fills UUIDv7
          - if event.time is None — fills UTC now
          - if event.stage_id is None and ContextVar is set — fills it
          - caller MUST supply source; emitter never overwrites it
        """

    def emit_remote(self, event: BaseEvent) -> None:
        """Receive event from remote source (pod via WS).

        Contracts:
          - same as emit() but event_id/offset/time/source NEVER overwritten
          - dedups by (run_id, source, offset) — silent drop if seen
          - validation errors → increments events_remote_invalid_total + drop
        """

    @contextmanager
    def stage_scope(self, stage_id: str) -> Iterator[None]:
        """ContextVar scope: all emit() calls within auto-fill stage_id.
        Nested scopes override outer; on exit token restored."""
```

### Замена callbacks

Сейчас в `pipeline/stages/`:
```python
@dataclass
class TrainingMonitorEventCallbacks:
    on_started: Callable[[...], None]
    on_resource_check: Callable[[...], None]
    on_timeout: Callable[[...], None]
```

После — **удаляем**, в stage:
```python
class TrainingMonitor:
    def __init__(self, emitter: IEventEmitter, ...):
        self._emitter = emitter

    def run(self):
        with self._emitter.stage_scope("training_monitor"):
            self._emitter.emit(MonitorStartedEvent(...))
            ...
            self._emitter.emit(MonitorResourceCheckEvent(...))
```

Удаляются: `TrainingMonitorEventCallbacks`, `DatasetValidatorEventCallbacks`, `EvaluatorEventCallbacks`, `GPUDeployerEventCallbacks`, `ModelRetrieverEventCallbacks`.

Добавляются (coverage gaps): `InferenceDeployer`, `DeploymentManager`, providers (ssh_provisioned, code_synced) — все через тот же emitter.

---

## Storage layer

### Pod-side journal (recovery only)

- Path: `/workspace/pod_events/<run_id>/events.NNN.jsonl` (внутри Docker)
- Формат: envelope JSON один на строку (без legacy wrapper)
- Rotation: 5×100MiB как сейчас
- Используется **только** если control недоступен длительно (>5 мин WS off) — после реконнекта control запрашивает `GET /pod/events?after_offset=N`

### Control-side journal (SSOT)

- Path: `workspace/runs/<run_id>/events.jsonl`
- Append-only, `open(path, "a", buffering=1)` line-buffered
- **Length-prefixed lines** (см. R-04): `<length>\t<envelope_json>\n` — позволяет детектировать torn writes
- **Batched fsync**: 50 events ИЛИ 1 сек ИЛИ severity ≥ error (immediate) ИЛИ close
- **Offset counter**: per `(run_id, source)` пара. In-memory `AtomicInt` + `threading.Lock`. На resume — восстанавливается из последней валидной строки journal'а.
- **Crash recovery**: на `JournalReader.resume()` — сканируем с конца, partial last line trunc'ится atomic'ом (write to tmp + rename); detect через mismatch length prefix.

### MLflow artifact (on finalize)

При `RunCompleted/RunFailed/RunCancelled`:

```json
events_manifest.json:
{
  "schema_version": 1,
  "total_events": 8421,
  "first_offset_per_source": {"pod://abc/trainer": 0, "control://orchestrator": 0},
  "last_offset_per_source": {...},
  "first_time": "2026-05-16T10:00:00.123456Z",
  "last_time": "2026-05-16T11:32:14.654321Z",
  "type_histogram": {"ryotenkai.pod.training.step": 7800, ...},
  "schema_versions_present": [1],
  "events_sha256": "abc123...",
  "journal_complete": true,
  "cancellation_reason": null
}
```

`journal_complete: false` + `cancellation_reason` устанавливаются если run был cancelled до flush.

**Retry policy:** exponential backoff (3 attempts, 1s/5s/30s), на финальный failure — alert + journal остаётся локально с manifest meta-флагом `mlflow_uploaded: false` для последующего ручного upload.

---

## Transport layer

### HTTP replay

```
GET /api/runs/{run_id}/events
  ?after_offset=<int>        # exclusive; default 0
  &limit=<int>               # default 1000, max 10000
  &type_prefix=<str>         # e.g. "ryotenkai.pod.training."
  &severity=<csv>            # "warning,error,critical"
  &stage_id=<str>
  &source=<str>              # filter by source URI

Response: application/x-ndjson
Headers: X-Next-Offset: <int>
```

Server-side filtering — linear scan по journal с predicate (acceptable для <100MB).

### SSE streaming

```
GET /api/runs/{run_id}/events/stream
  ?after_offset=<int>  | Last-Event-ID header (header takes precedence)
  &type_prefix=<str>
  &severity=<csv>
  &stage_id=<str>

Response: text/event-stream
  id: <offset>
  event: ryotenkai.pod.training.step
  data: <envelope_json>

  : keepalive

Close codes (sent as final SSE error event):
  4410: journal exhausted (requested offset older than oldest available)
  4422: invalid query params
```

**Backpressure:** per-consumer bounded queue (capacity 1000); медленный consumer → drop-oldest + increment `events_dropped_total{consumer="sse:<conn_id>"}`. Дропнутые события всё ещё в journal — клиент при reconnect получит их через replay HTTP.

### CLI

`ryotenkai job events <run_id>`:
- Default: ходит на control SSE (если run завершён или control имеет journal)
- Fallback: WS на pod (legacy для активных runs если control unreachable)

### Pod fallback HTTP

```
GET /pod/events?run_id=<id>&after_offset=<int>&source=<src>
```

Используется control при cold reconnect к pod (e.g. WS dropped >5 мин). Возвращает события из pod-local journal. Закрывает R-02 (pod journal lost) на window pod alive.

---

## Schema evolution

- `schema_version: int = 1` на envelope
- Каждый event-класс имеет `SCHEMA_VERSION: ClassVar[int]`
- Codec deserialization:
  1. Parse `schema_version` и `type` из raw dict
  2. Run upcaster chain (pure functions) `(raw_dict, from, to) → new_raw_dict`
  3. Validate через `TypeAdapter[Event]`
  4. На validation error в `strict=False` режиме → wrap as `UnknownEvent(original_type=..., raw_payload=...)`

**Правила:**
- Дополнения backward-compat: новое optional поле + default
- Семантическое изменение → **новый dotted type**, никогда reuse
- Rename — через computed alias ИЛИ новый type
- Удаление поля — через миграцию upcaster + новый type если это load-bearing field

Пример pure upcaster:
```python
def v1_to_v2_training_started(raw: dict, _from: int, _to: int) -> dict:
    payload = raw["payload"]
    payload["total_steps"] = payload.pop("max_steps")  # rename
    return raw

upcasters: dict[str, list[Upcaster]] = {
    "ryotenkai.pod.training.started": [v1_to_v2_training_started],
}
```

---

## Risk Ledger (top 20 после 3 итераций deep-think)

| # | Risk | Sev | Scenario | Mitigation |
|---|------|-----|----------|-----------|
| R-01 | **Pod journal lost on pod crash; data not in control SSOT** | КРИТ | WS down 10 мин, pod emits 5000 events в local journal, pod crashes → данные потеряны навсегда. | (a) `GET /pod/events?after_offset=N` HTTP fallback endpoint обязателен; (b) control периодически polls `last_persisted_offset` pod при WS disconnect; (c) control alert если control's last_offset_for_source отстаёт от pod's > N. |
| R-02 | **Pod WS publish fails silently; event in pod journal but not control SSOT** | КРИТ | Pod успел запихнуть в local journal но WS publish дропнут; control никогда не узнает. | Pod posts batched events HTTP /events когда WS down (fire-and-forget с retry queue до 10000 в pod memory). Control reconnect → ACK последний offset → pod очищает буфер. |
| R-03 | **Big-bang PR rollback complexity** | КРИТ | После merge обнаружена race condition. 11+ дней изменений, revert болезненный. | (a) staged commits внутри PR (shared/events first, потом pod, потом control), каждый commit зелёный CI — позволяет частичный revert до конкретного коммита; (b) load-test before merge (1000 events/s × 1 час). Single-user разработка → `git revert HEAD` остаётся валидным fallback'ом. |
| R-04 | **JSONL torn write на kill -9; partial last line silently lost** | КРИТ | Pod emit'нул 200-byte event, write() flushed 150 bytes, kill -9. Журнал имеет invalid JSON → reader json.loads fails, line skipped silently. | Length-prefixed lines `<length>\t<json>\n`. На read проверяем match. На init: scan с конца; partial line (length mismatch или incomplete write) → truncate via tmp+rename. Документировать invariant. |
| R-05 | **Offset collision при concurrent emit на pod** | КРИТ | Два callback'а (TrainerCallback + HealthReporter) одновременно вызывают publish() → offset incremented dual-thread → коллизия. | `threading.Lock` вокруг offset counter + assert single event loop. Test concurrent emitters под нагрузкой. Metric `offset_collisions_total{source}`. |
| R-06 | **Emitter silently drops events; no alerts** | КРИТ | Контракт "never raises" + disk full → emit fails silently → reports incomplete. | Mandatory metrics: `events_emitted_total`, `events_dropped_total{consumer,reason}`, `events_emit_failed_total{error}`. Alert: `events_emit_failed_rate > 0.1%` over 5 min. Health endpoint `/api/v1/health/events`. |
| R-07 | **Run cancelled до finalize; partial journal uploaded as "complete"** | ВЫСОК | SIGINT → orchestrator exits до `MlflowFinalizer.upload()` → manifest incomplete но загружен. | `journal_complete: false` + `cancellation_reason` в manifest. Emit `RunCancelledEvent` ДО upload в finally блоке. Upload non-best-effort (если cancelled — упрощённый manifest без sha256 over full file). |
| R-08 | **MLflow upload fails, no retry, journal lost from workspace eventually** | ВЫСОК | MLflow off, upload throws, exception swallowed, workspace cleanup через неделю удаляет journal. | Retry policy: 3 attempts (1s/5s/30s). На финальный fail — `manifest.mlflow_uploaded=false`, alert операторам. Workspace retention не удаляет журналы у которых `mlflow_uploaded=false`. |
| R-09 | **Pydantic TypeAdapter validation cost в hot path** | ВЫСОК | 50 union members × 50000 events × discriminator lookup → bottleneck. | Benchmark обязателен перед merge (`bench/event_throughput.py`). Cache TypeAdapter singleton. Если бенчмарк показывает >100µs/event при p99 — фасе фикс: nested unions by area (`{pod,control}.*` → tag-prefix dispatch). |
| R-10 | **fsync blocks event loop** | ВЫСОК | fsync per batch (50 events) — 5-10ms на HDD/NFS → event loop тормозит. | fsync в `asyncio.to_thread()` / ThreadPoolExecutor. Документировать: writes до fsync видны readers (page cache) но не durable. Monitor `journal_fsync_latency_seconds` p99. |
| R-11 | **Unknown event type на старом consumer** | ВЫСОК | Новый pod emit'нул event которого нет в union у старого control → ValidationError → drop. | `UnknownEvent` catch-all variant в union (см. taxonomy). Codec.from_jsonl с `strict=False` (default для read paths). Strict mode (`strict=True`) только при emit на producer side. |
| R-12 | **Resume semantics undefined** | ВЫСОК | User `ryotenkai run resume <id>` — append в существующий events.jsonl или новый файл? | Decision: `resume` использует **тот же run_id** и append'ит в existing events.jsonl. Offset counter resume'ится из last valid offset journal'а. Emit `RunResumedEvent` с `previous_attempt_at`, `events_before_resume: int` для report attribution. Корректно при ContextVar разделении. |
| R-13 | **Dedup set memory growth** | ВЫСОК | Stuck run → (run_id, source, offset) set растёт неограниченно. | (a) Cleanup на terminal state (completed/failed/cancelled). (b) TTL eviction `> 24h` для stuck runs. (c) Monitor `dedup_set_size{run_id}` per run. (d) При control restart — reconstruct set из journal (последние 10k offsets per source). |
| R-14 | **WS frame size limit для large payloads** | СРЕД | Event с 100KB payload (e.g. большой metrics dict) → WS frame reject 64KB default. | Validate `len(json.dumps(payload)) < 16KB` в emit() (pre-publish). Reject с `PayloadTooLargeError`. Документировать: max payload size = 16KB. |
| R-15 | **ContextVar stage_id loss в subprocess / asyncio.create_task** | СРЕД | Trainer subprocess не наследует ContextVar; `asyncio.create_task` без `copy_context()` теряет. | Trainer subprocess получает `stage_id` через `--stage-id` CLI arg (явно). `asyncio.create_task(coro, context=copy_context())` обязательно в helper'е wrapper. Sentinel test: grep `create_task` в emitter pathway → должен иметь context. |
| R-16 | **Frontend coordination — openapi.json changes** | СРЕД | Новый SSE endpoint → openapi.json меняется → frontend types регенерируются → большой diff. | Существующий `/job/events` polling endpoint остаётся (не удаляется в этом PR). Новый `/runs/{id}/events/stream` добавляется параллельно. Frontend переключается в отдельном PR. Diff openapi.json минимален. |
| R-17 | **Sentinel `test_every_module_has_tests` блокирует PR** | СРЕД | 20+ новых файлов в packages/*/src/ → sentinel block. | Pre-create stub test files для всех новых модулей. Создать `FakeEventEmitter`, `FakeJournalWriter`, `FakeInMemoryBus` в `tests/_fakes/` ДО PR submission. CLAUDE.md compliance проверяется локально перед push. |
| R-18 | **Mutation gate fails on codec** | СРЕД | `scripts/mutation/validate_agent_output.sh` exit 1 на critical modules (codec, journal_writer, in_memory_bus). | Минимум 95% kill rate на critical модули. Hypothesis-based fuzz testing для codec (random payloads). Reference template из CLAUDE.md: `tests/unit/providers/single_node/training/test_preempt_inference_container.py`. |
| R-19 | **SSE catchup → live race (event между фазами теряется)** | СРЕД | Catchup читает до offset N, между этим и subscribe new event N+1 произведён → не доставлен. | **Subscribe-FIRST architecture**: subscribe to InMemoryBus, capture current_max=M; replay journal `after=client_offset until=M`; затем drain bus from cursor=M. Dedup по offset на client side не нужен (no overlap). |

---

## Implementation phases

| # | Описание | Days |
|---|----------|------|
| 0 | **Smoke test**: Pydantic v2 discriminated union + UUIDv7 + TypeAdapter; JSON round-trip; codec timing baseline | 0.5 |
| 1 | **Shared events package**: envelope, IEventEmitter Protocol, codec (length-prefix), discriminator, 55 event classes (24 pod + 31 control + UnknownEvent), upcaster scaffolding. **Sentinel-compliant: создание `tests/_fakes/event_emitter.py`, `event_journal.py`, `in_memory_bus.py` ДО любых тестов** | 1.5 |
| 2 | **Pod-side rewire**: переписать `event_bus.py` (envelope schema + multi-consumer cursors); нормализовать `event_journal.py` (length-prefix lines, truncate-invalid-on-init); **удалить** `mlflow/event_log.py` и `training_events_callback.py`; переписать `runner_event_callback.py` (async queue, no self-disable); создать `PodEventEmitter`. Pod HTTP fallback endpoint `GET /pod/events?after_offset=N&source=...` | 2.0 |
| 3 | **Control foundation**: `events/journal_writer.py` (length-prefix + batched fsync via to_thread), `journal_reader.py` (truncate-invalid-on-init), `in_memory_bus.py` (Ray-style MultiConsumerEventBuffer), `dedup.py` (reconstruct from journal), `ControlEventEmitter` impl с ContextVar `stage_scope`. DI wiring в orchestrator. | 1.5 |
| 4 | **Stage migration**: убрать `*EventCallbacks` dataclass'ы во всех stages (dataset_validator, gpu_deployer, training_monitor, model_retriever, evaluator); заменить на `emitter.emit(...)`; удалить `timeline_events` поле из StageArtifactEnvelope. ContextVar stage_scope в orchestrator. | 1.5 |
| 5 | **Coverage gaps**: добавить events в `inference_deployer.py` (deployment_started, health_check_*, deployed, deactivated); `deployment_manager.py` (полная новая coverage); pre-flight events (ssh_provisioned, code_synced) в providers. | 1.0 |
| 6 | **API + transport**: новый router `api/routers/events.py` — `GET /runs/{id}/events` + `GET /runs/{id}/events/stream` (SSE) с subscribe-first ordering; Last-Event-ID reconnect; server-side filtering. `MlflowFinalizer.upload` с events.jsonl + manifest + retry policy. | 1.0 |
| 7 | **Report adapter rewrite**: новый `reports/adapters/journal_adapter.py` (читает workspace journal или MLflow artifact fallback); упростить `mlflow_adapter.py` (удалить парсинг event_log artifact и string sniffing); report generator берёт events из journal_adapter. | 1.0 |
| 8 | **Observability**: метрики Prometheus-style (`events_emitted_total`, `events_dropped_total{consumer,reason}`, `events_emit_failed_total{error}`, `journal_fsync_latency_seconds`, `bus_depth_current`, `dedup_set_size{run_id}`, `offset_collisions_total{source}`); health endpoint `/api/v1/health/events`. | 0.5 |
| 9 | **Tests**: 7 классов на каждый production-метод (CLAUDE.md), упор на critical modules (codec, emitter, journal_writer, in_memory_bus, dedup); hypothesis fuzz для codec; integration test full pipeline; resume test; SSE smoke; mutation gate ≥95% на critical. | 1.5 |
| 10 | **Docs + ADR**: ADR-0009 unified events; README в `shared/events/` с конвенциями; обновление CLAUDE.md разделов про event flow. | 0.5 |

**Total: ~12 дней.** Phases 1, 2, 3 могут идти параллельно после Phase 0. Integration в Phase 4-7.

---

## Critical files to modify

### Удалить
- `packages/pod/src/ryotenkai_pod/trainer/mlflow/event_log.py`
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/training_events_callback.py`
- string sniffing блок в `packages/control/src/ryotenkai_control/reports/adapters/mlflow_adapter.py:712-715` (и связанная classification logic)
- `timeline_events` поле в `packages/control/src/ryotenkai_control/pipeline/artifacts/base.py`
- Все `*EventCallbacks` dataclass'ы в `packages/control/.../pipeline/stages/` (внутри файлов)

### Создать
- `packages/shared/src/ryotenkai_shared/events/` — полный пакет (см. module layout)
- `packages/pod/src/ryotenkai_pod/runner/event_emitter.py`
- `packages/control/src/ryotenkai_control/events/` — полный пакет (8 файлов)
- `packages/control/src/ryotenkai_control/api/routers/events.py`
- `packages/control/src/ryotenkai_control/reports/adapters/journal_adapter.py`
- `tests/_fakes/event_emitter.py`, `event_journal.py`, `in_memory_bus.py` — canonical fakes
- `docs/adrs/2026-05-XX-unified-event-system.md` — ADR-0009

### Существенно изменить
- `packages/pod/src/ryotenkai_pod/runner/event_bus.py` — schema на envelope; multi-consumer cursors; drop-oldest metrics
- `packages/pod/src/ryotenkai_pod/runner/event_journal.py` — length-prefix; truncate-invalid-on-init; envelope-per-line
- `packages/pod/src/ryotenkai_pod/runner/api/internal.py` — endpoint receives envelope (was open dict)
- `packages/pod/src/ryotenkai_pod/runner/api/events.py` — добавить fallback HTTP `/pod/events`
- `packages/pod/src/ryotenkai_pod/trainer/callbacks/runner_event_callback.py` — async queue with drop-oldest, no self-disable
- `packages/control/src/ryotenkai_control/api/routers/jobs.py` — удалить старый events endpoint, перенести в `events.py` router
- `packages/control/src/ryotenkai_control/reports/adapters/mlflow_adapter.py` — упростить: events читаются из journal_adapter
- `packages/control/src/ryotenkai_control/pipeline/orchestrator.py` — DI emitter, ContextVar stage_scope, finalize in finally
- `packages/control/src/ryotenkai_control/pipeline/stages/dataset_validator.py` — emitter-based
- `packages/control/src/ryotenkai_control/pipeline/stages/gpu_deployer.py` — emitter-based
- `packages/control/src/ryotenkai_control/pipeline/stages/training_monitor.py` — emitter-based + emit_remote bridge для WS events
- `packages/control/src/ryotenkai_control/pipeline/stages/model_retriever.py` — emitter-based
- `packages/control/src/ryotenkai_control/pipeline/stages/model_evaluator.py` — emitter-based (callbacks уже работают, унифицировать)
- `packages/control/src/ryotenkai_control/pipeline/stages/inference_deployer.py` — добавить emit (был только logger.info)
- `packages/control/src/ryotenkai_control/pipeline/stages/managers/deployment_manager.py` — добавить emit (нет events)
- `packages/providers/src/ryotenkai_providers/single_node/training/provider.py` — добавить ssh_provisioned + code_synced events
- `packages/providers/src/ryotenkai_providers/runpod/training/` — аналогично

---

## Verification

### Unit tests (`tests/unit/.../events/`)

Per CLAUDE.md — 7 классов на каждый production-метод. Reference template: `tests/unit/providers/single_node/training/test_preempt_inference_container.py`.

Per event-type (~50):
- TestPositive — round-trip serialize → deserialize
- TestNegative — missing required, extra field (extra=forbid), type mismatch
- TestBoundary — empty strings, max int, datetime extremes
- TestInvariants — `type` Literal pinned, severity default, schema_version pinned
- TestDependencyErrors — UUIDv7 generator error, datetime now error
- TestRegressions — конкретные баги
- TestLogicSpecific — discriminator dispatch correctness

Critical modules с full mutation testing (≥95% kill rate):
- `shared/events/codec.py` — to_jsonl, from_jsonl, length-prefix framing, truncate detection
- `control/events/journal_writer.py` — append, fsync batching, atomic-temp-rename для recovery
- `control/events/in_memory_bus.py` — multi-consumer cursors, drop-oldest, metrics
- `control/events/emitter.py` — emit, emit_remote, stage_scope ContextVar
- `control/events/dedup.py` — reconstruct from journal, TTL eviction

Hypothesis fuzz: random payload generation, malformed JSONL lines, partial writes.

### Integration

`tests/integration/test_event_pipeline_e2e.py`:
1. Spin up fake pod runner (in-process), fake stages
2. Run orchestrator с 5 stages
3. Assertions: все expected types в `events.jsonl`; offsets monotonic per source; `events_manifest.json` valid + sha256 match
4. MLflow upload mock — manifest корректен с `journal_complete=true`

`tests/integration/test_resume.py`:
1. Start run, emit 100 events
2. SIGKILL (no graceful close)
3. Assert: journal has partial last line possibly
4. Restart with same run_id
5. JournalWriter.resume() — truncates partial via tmp+rename
6. Continue from offset 100 (counter resumes)
7. Final: 100+N events, all valid, all length-prefix verified

`tests/integration/test_pod_backpressure.py`:
1. PodEventEmitter с заглушенным HTTP loopback (HTTP 500 на каждый)
2. Trainer-side emit 10000 events sync
3. Assert: queue caps at 10000, `events_dropped_total{consumer="loopback"}` increments, **trainer emit p99 < 1ms** (no blocking)
4. Loopback recovers → queue drains → journal contains events from after recovery (older dropped)

`tests/integration/test_sse_stream.py`:
1. Start API server
2. Emit 50 events to journal
3. SSE client с `after_offset=10`
4. Receive 40 historical events monotonic
5. Emit 10 more live (concurrent producer)
6. Client receives those too (no race lost между catchup и live)
7. Disconnect, reconnect с `Last-Event-ID: <last>` — no duplicates, no missing

`tests/integration/test_dedup_resume.py`:
1. Pod emits offset 0..100 → control journal has them
2. Control restart → dedup set reconstructed из last 10k journal entries
3. Pod re-sends offset 95..100 (network jitter scenario)
4. Assert: dedup detected; offset 95..100 dropped silently; journal не grew

### Performance baseline

`bench/event_throughput.py`:
- Emit 100k events sync → throughput ≥ 50k/s on M-series Mac (journal + bus, no SSE)
- Validation cost: p99 TypeAdapter.validate < 100µs/event

`bench/sse_fanout.py`:
- 1 producer @ 1k events/s × 50 SSE consumers; drop rate < 0.1%
- Event-loop latency p99 < 50ms (fsync в thread pool)

### Smoke (manual)

Documented в `docs/dev/event_system_smoke.md`:
- frontend open + start run → DevTools Network → SSE connection visible
- kill API artificially → EventSource auto-reconnects с Last-Event-ID
- inspect MLflow UI после run → events/events.jsonl artifact + manifest

### Mutation gate

Per CLAUDE.md `scripts/mutation/validate_agent_output.sh`:
- Target ≥95% kill rate на: `codec.py`, `journal_writer.py`, `journal_reader.py`, `in_memory_bus.py`, `emitter.py`, `dedup.py`
- Surviving mutations → дорабатывать тесты до зелёного gate перед "готово"

---

## Definition of Done

- [ ] Все sentinels из CLAUDE.md зелёные: `test_no_protocol_mocking`, `test_every_module_has_tests`, `test_xfail_debt_completeness`, custom event sentinels (subscribe-first race, length-prefix invariant)
- [ ] `uv run lint-imports` zero violations (shared не зависит от control/pod; control не импортирует pod-concrete)
- [ ] `mypy .` zero errors на новых пакетах (Pydantic v2 stubs OK)
- [ ] `pytest tests/unit -q` zero failed
- [ ] `pytest tests/integration -q` zero failed (включая resume, backpressure, SSE)
- [ ] `bash scripts/mutation/validate_agent_output.sh` exit 0
- [ ] `bench/event_throughput.py` показывает no regression (≥ baseline или +5% улучшение)
- [ ] ADR-0009 написан и rev'ed
- [ ] CLAUDE.md обновлён (event flow секция)
- [ ] Все 50 event-классов имеют docstring с примером payload
- [ ] Health endpoint `/api/v1/health/events` отвечает
- [ ] Metrics exported (см. R-06)
- [ ] CHANGELOG.md обновлён с warning о backward compat break для старых runs

---

## Tradeoffs и rejected alternatives

| Alternative | Rejection rationale |
|-------------|---------------------|
| **gRPC** для pod↔control | Существующий transport (HTTP loopback + WS) handles bidirectional. gRPC требует code-gen, новую зависимость. Pydantic discriminated union даёт ту же type-safety. |
| **Kafka/NATS/Redis Streams** | Single-tenant workspace, single pod на run. Distributed broker overkill: 50-200ms latency + 3+ HA pods для одной машины. |
| **CloudEvents full envelope** | Берём pattern (event_id, source URI, time, type, dotted naming), но не букву spec. Binding mode (binary/structured) не нужен — мы не interop с external CloudEvents consumers. |
| **SQLite вместо JSONL** | JSONL append-only + line-buffered + length-prefix = crash-safe by design, читается стандартным tail/grep, переносим. SQLite fsync slower; artefact в MLflow должен быть human-readable. Если journal вырастет до GB — добавим sidecar SQLite index. |
| **WebSocket для frontend** | SSE проще: одностороний server→client, native browser EventSource, automatic Last-Event-ID reconnect, no upgrade handshake. WS overkill для read-only стрима. |
| **Open dict payload (текущая as-is)** | Источник всех 3 КРИТ-багов: string sniffing, расходящиеся schema, dual-path drift. Closed union ловит drift в compile-time (mypy) + runtime (Pydantic). |
| **Backward compat с старыми runs** | Пользователь явно отказался. Стоимость двух parser'ов выше пользы. Документируем в migration note. |
| **OTel GenAI semconv** | Spec — inference-only по состоянию 2026-Q1. Для training conventions нет стандарта. Local namespace `ryotenkai.*` остаётся. |
| **MLMD lineage tracking** | gRPC + Postgres + Envoy overkill для monorepo trainer. Если в будущем нужна lineage — можно построить поверх journal. |

---

## Open вопросы (закрыты после 3 итераций deep-think)

Все 95+ выявленных рисков адресованы либо в Risk Ledger top-20 (наиболее критичные), либо в design decisions (subscribe-first SSE, length-prefix journal, dedup reconstruction, feature flag, retry policies). Не осталось открытых вопросов требующих архитектурной развилки.

Возможные расширения OUT OF SCOPE этого PR:
- Frontend переключение на SSE (отдельный PR после стабилизации backend)
- Plugin event registration API (если community plugins будут эмитить — будет отдельный design)
- Long-term journal compression / cold storage policy
- Cross-run event aggregation views (e.g., "all OOMs across last 30 days")
- Event-driven automation rules à la Prefect (`if event=oom_detected then notify_slack`)
