# ryotenkai_control.events

Control-side event infrastructure: `ControlEventEmitter` +
`JournalWriter`/`JournalReader` + `InMemoryBus` + `EventDedup` +
`MlflowFinalizer` + `EventEmitterRegistry` + observability metrics.

This package satisfies the
[`IEventEmitter`](../../../../../shared/src/ryotenkai_shared/events/protocol.py)
Protocol with control-plane semantics: durable JSONL journal on disk as
the SSOT, in-memory bus for live SSE/WS fan-out, dedup for
remote-replay correctness, and an MLflow finalizer for long-term
artifact storage.

See also: [ADR-0009](../../../../../../docs/adrs/2026-05-17-unified-event-system.md),
the [shared/events README](../../../../../shared/src/ryotenkai_shared/events/README.md),
and the [plan file](../../../../../../docs/plans/ethereal-tumbling-patterson.md).

## Architecture

```
   orchestrator / stages                pod (remote, via WS frame)
        │ emitter.emit(event)              │ emitter.emit_remote(event)
        │                                  │
        ▼                                  ▼
+----------------------------------+----------------------------+
|              ControlEventEmitter (single instance per run)    |
|  - assigns offset (monotonic per (run_id, source))            |
|  - fills stage_id from ContextVar (stage_scope)               |
|  - validates envelope via EVENT_ADAPTER                       |
|  - never raises; on internal error logs + increments counter  |
+--------------------+-------------------------------------+----+
                     │                                     │
                     ▼                                     ▼
              JournalWriter                          InMemoryBus
              (length-prefix JSONL,                  (deque + per-consumer
               batched fsync 50ev/1s,                 cursors, drop-oldest
               immediate for severity≥err)            on overflow)
                     │                                     │
        +------------+------------+         +--------------+--------------+
        │                         │         │                             │
        ▼                         ▼         ▼                             ▼
events.jsonl                JournalReader  SSE subscribers          HTTP replay
(workspace/runs/             (tail + offset slice via              filtered
 <run_id>/)                   reconstruction)                       slice_journal()

                     ↓ (run end)
              MlflowFinalizer
              uploads events.jsonl + events_manifest.json (sha256)
              under the events/ artifact prefix, with retry policy
```

`EventDedup` sits between `emit_remote` and the journal: it tracks
`(run_id, source, offset)` tuples seen and silently drops duplicates
(R-13). On process restart it reconstructs the set from the last 10k
journal entries per source so a pod resend after reconnect is still
deduplicated.

`EventEmitterRegistry` is a process-wide `dict[str, ControlEventEmitter]`
guarded by a lock. The orchestrator registers the live emitter under
`run_id` so the API events router can locate the in-memory bus for SSE
fan-out without holding a reference across request boundaries.

## Components

| Class | Module | Responsibility |
|---|---|---|
| `ControlEventEmitter` | `emitter.py` | `IEventEmitter` impl; composes journal/bus/dedup. ContextVar `stage_scope`. Never raises. |
| `JournalWriter` | `journal_writer.py` | Append-only JSONL with length-prefix framing. Batched fsync (50 events OR 1 s OR severity ≥ error). |
| `JournalReader` | `journal_reader.py` | Tail-able reader. Truncates partial last line on init via tmp+rename (R-04). Per-source offset reconstruction. |
| `InMemoryBus` | `in_memory_bus.py` | Bounded ring buffer (`deque(maxlen=10000)`) + per-consumer cursors. Slow consumers see drops as their own counter; producers stay fast. Ray-style `MultiConsumerEventBuffer`. |
| `EventDedup` | `dedup.py` | `(run_id, source, offset)` dedup set + TTL eviction + `reconstruct_from_journal()` on restart. |
| `MlflowFinalizer` | `mlflow_finalizer.py` | Uploads `events.jsonl` + `events_manifest.json` (with `events_sha256`, `type_histogram`, `journal_complete`) at run end. Retry 3× (1s/5s/30s); `manifest.mlflow_uploaded=false` on terminal failure. |
| `EventEmitterRegistry` | `registry.py` | Process-wide singleton: `run_id → ControlEventEmitter`. Allows the SSE router to find the live bus per run. |
| `slice_journal` | `replay.py` | Pure offset-range + predicate slice helper used by HTTP replay. |
| `EventSubsystemMetrics` + `collect_metrics` | `metrics.py` | Snapshot aggregator for the `/api/v1/health/events` endpoint (Risk Ledger R-06). |

## How the orchestrator wires the emitter

The orchestrator builds the emitter via the `for_run` convenience
constructor (which assembles the journal / bus / dedup from a
`run_directory`), registers it under the registry, and unwinds in a
`try/finally`:

```python
# packages/control/.../pipeline/orchestrator.py (simplified)

from ryotenkai_control.events import (
    ControlEventEmitter,
    EventEmitterRegistry,
)

def run_pipeline(self, run_id: str, run_directory: Path) -> None:
    emitter = ControlEventEmitter.for_run(
        run_id=run_id,
        run_directory=run_directory,
    )
    registry = EventEmitterRegistry.instance()
    registry.register(run_id, emitter)
    try:
        emitter.emit(RunStartedEvent(...))
        for stage in self._stages:
            with emitter.stage_scope(stage.name):
                stage.run(emitter=emitter)
        emitter.emit(RunCompletedEvent(...))
    except Exception as exc:
        emitter.emit(RunFailedEvent(...))
        raise
    finally:
        registry.deregister(run_id)
        emitter.close()  # flushes journal, finalises MLflow
```

The pattern guarantees:

* The registry never leaks a slot for a crashed run.
* The journal is flushed and the MLflow upload is attempted regardless
  of how the pipeline exited.
* `RunFailedEvent` / `RunCancelledEvent` lands in the journal *before*
  finalize, so the manifest's `journal_complete` flag is accurate.

## How stages emit events

Stages take the emitter as a constructor argument (or as an `emit`
keyword on `run`) and use `stage_scope` to auto-fill `stage_id`:

```python
class TrainingMonitor:
    def __init__(self, *, emitter: IEventEmitter, ...):
        self._emitter = emitter

    def run(self) -> None:
        with self._emitter.stage_scope("training_monitor"):
            self._emitter.emit(TrainingMonitorStartedEvent(
                source="control://orchestrator/training_monitor",
                run_id=self._run_id,
                offset=0,  # emitter overrides with monotonic counter
                payload=TrainingMonitorStartedPayload(
                    pod_endpoint=self._endpoint,
                    poll_interval_s=self._interval,
                ),
            ))
            ...
            self._emitter.emit(TrainingMonitorTimeoutEvent(...))
```

The `*EventCallbacks` dataclass pattern that previously wrapped these
calls (`TrainingMonitorEventCallbacks`, `DatasetValidatorEventCallbacks`,
etc.) was removed in Phase 4 — see the deletion list in
[ADR-0009](../../../../../../docs/adrs/2026-05-17-unified-event-system.md).

## Subscribe-first SSE invariant

The Risk Ledger entry R-19 in the plan describes a race the SSE handler
must avoid: between the catchup phase (read journal up to current tail)
and the live phase (drain the in-memory bus from the cursor), a
producer can emit a new event that's neither in the catchup range nor
visible to the cursor. The fix is **subscribe first, snapshot the
current_max, then replay**:

```python
# packages/control/.../api/routers/events.py (algorithm sketch)

async def sse_stream(run_id: str, after_offset: int):
    emitter = EventEmitterRegistry.instance().get(run_id)
    bus = emitter.bus

    # 1. SUBSCRIBE FIRST — captures the cursor at "now"
    async with bus.subscribe(after_offset=None) as cursor:
        current_max = bus.current_max_offset()

        # 2. CATCHUP — replay journal from after_offset up to current_max
        async for event in journal_reader.replay(
            run_id=run_id, after=after_offset, until=current_max,
        ):
            yield sse_format(event)

        # 3. LIVE — drain the bus from the cursor (no overlap, no gap)
        async for event in cursor:
            yield sse_format(event)
```

Phase 9's hypothesis fuzz **found a production bug** in step 1: the
original `InMemoryBus.subscribe` had the cursor-comparison branch
inverted, causing missed events when the consumer's cursor was ahead
of `deque[0].offset`. The fuzz reproducer landed as a regression test
in the same phase.

## Counter observability

Each collaborator exposes raw counters; the aggregator builds a single
snapshot for the health endpoint:

```python
from ryotenkai_control.events import collect_metrics

snapshot: EventSubsystemMetrics = collect_metrics(
    emitter=emitter,
    bus=emitter.bus,
    journal_writer=emitter.journal_writer,
    dedup=emitter.dedup,
)
# snapshot.events_emitted_total, .events_dropped_total{consumer,reason},
# .events_emit_failed_total{error}, .bus_depth_current,
# .dedup_set_size, .journal_fsync_latency_seconds (histogram), ...
```

`GET /api/v1/health/events` (registered in `api/routers/health.py`)
returns the snapshot per-run. Operators use it to alert on
`events_emit_failed_rate > 0.1%` over 5 minutes.

## Module layout

```
events/
├── __init__.py             # flat re-exports of the public surface
├── emitter.py              # ControlEventEmitter (IEventEmitter impl)
├── journal_writer.py       # append + batched fsync; length-prefix
├── journal_reader.py       # tail + truncate-invalid-on-init; replay
├── in_memory_bus.py        # MultiConsumerEventBuffer
├── dedup.py                # (run_id, source, offset) set; TTL evict
├── mlflow_finalizer.py     # upload events.jsonl + manifest; retry
├── registry.py             # EventEmitterRegistry singleton
├── replay.py               # slice_journal pure helper for HTTP replay
└── metrics.py              # EventSubsystemMetrics aggregator
```

## Test fakes

The canonical fakes for `IEventEmitter` (and the journal / bus
collaborators) live under `tests/_fakes/` per the sentinel rule in
`CLAUDE.md`. Do NOT mock the Protocol — the `tests/_lint/test_no_protocol_mocking.py`
sentinel blocks the PR. Extend the existing fake or add a new one if
the test scenario isn't covered.
