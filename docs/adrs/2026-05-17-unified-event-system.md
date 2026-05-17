# ADR: Unified Event System (typed envelopes + journal SSOT)

**Date:** 2026-05-17
**Status:** Accepted; Phases 0-9 executed (Phase 10 — this ADR — in progress).
**Plan reference:** [docs/plans/ethereal-tumbling-patterson.md](../plans/ethereal-tumbling-patterson.md)

## Decision

Replaced three parallel event subsystems with a single source-of-truth
(SSOT) JSONL journal per run, with typed Pydantic v2 discriminated-union
envelopes and a `IEventEmitter` Protocol-based emission API.

```
   pod-side                                              control-side (Mac)
+-----------------+                                +---------------------------+
| trainer subproc |                                |  orchestrator + stages    |
|  HF callback    |                                |  emitter.emit(...)        |
|   |             |                                |        |                  |
|   v             |                                |        v                  |
| RunnerEvent     |                                | ControlEventEmitter       |
|  Callback       |                                |   |       |       |       |
|  async queue    |                                |   v       v       v       |
|   |             |    HTTP loopback / WS frame    | Journal  Bus    Dedup     |
|   v             | -----------------------------> | Writer  (deque) (set)     |
| PodEventEmitter |    envelope JSON (typed)       |   |       |               |
|   |             |                                |   v       v               |
|   v             |    HTTP fallback                events.jsonl  SSE/HTTP     |
| Pod journal     | <----- /pod/events?after=N --- |   |       |               |
| (recovery only) |                                |   v       v               |
+-----------------+                                | MLflow  Frontend / CLI    |
                                                   |  finalize                 |
                                                   +---------------------------+
```

Key components:

| Component | Module |
|---|---|
| 55 event types + IEventEmitter Protocol + codec + upcasters | `packages/shared/src/ryotenkai_shared/events/` |
| ControlEventEmitter, JournalWriter/Reader, InMemoryBus, EventDedup, MlflowFinalizer, EventEmitterRegistry, metrics | `packages/control/src/ryotenkai_control/events/` |
| Ray-style multi-consumer pod event bus | `packages/pod/src/ryotenkai_pod/runner/event_bus.py` |
| Pod-side IEventEmitter implementation | `packages/pod/src/ryotenkai_pod/runner/event_emitter.py` |
| Pod local journal (recovery path) | `packages/pod/src/ryotenkai_pod/runner/event_journal.py` |
| HTTP replay + SSE stream (subscribe-first ordering) | `packages/control/src/ryotenkai_control/api/routers/events.py` |
| Health endpoint `GET /api/v1/health/events` | `packages/control/src/ryotenkai_control/api/routers/health.py` |
| Journal-backed report adapter (string sniffing retired) | `packages/control/src/ryotenkai_control/reports/adapters/journal_adapter.py` |

## Rationale

Three concrete signals from the prior state:

1. **Dual-path logging.** `training_started`, `checkpoint_saved`, `epoch_*`
   events were emitted both into the pod EventBus (via `RunnerEventCallback`
   HTTP loopback) and into MLflowEventLog (via `TrainingEventsCallback`).
   When one channel failed (HTTP loopback self-disabled after 3 failures;
   MLflow flush deferred to run end), the two journals diverged silently
   and downstream reports went stale.
2. **String sniffing.** `reports/adapters/mlflow_adapter.py:712-715` had
   `if "cache cleared" in msg: etype="cache_clear"` style classification
   to reverse-engineer event categories from free-form messages — a
   fragile coupling between the producer's `logger.info()` wording and
   the consumer's parser.
3. **HTTP backpressure.** `RunnerEventCallback._flush()` self-disabled
   after three consecutive HTTP failures, dropping all subsequent events
   to the end of the training session blindly. The trainer kept going
   and the control-side report had no idea events were missing.

Plus coverage gaps: `inference_deployer.py`, `deployment_manager.py`,
and the pre-flight provider checks (ssh_provisioned, code_synced) emitted
only `logger.info()` lines and were invisible to reports / frontend.

## Architecture decisions

* **Closed discriminated union** (Pydantic v2). Each event class pins
  `kind: Literal[...]` and `severity: Literal[...]` as defaults. The
  envelope is `frozen=True, extra="forbid"`. A `UnknownEvent` catch-all
  variant participates in the union so a forward-compat producer never
  crashes an older consumer (R-11).
* **Length-prefixed JSONL** journal: `<utf8_byte_length>\t<envelope_json>\n`.
  Detects torn writes on `kill -9` mid-`write()` because the reader can
  match the declared byte count against the actual body length. Recovery
  truncates the partial last line atomically via tmp+rename (R-04).
* **UUIDv7** event identifiers (`uuid_utils` Rust-backed generator).
  Natural sort order, monotonic at millisecond resolution; used for
  journal ordering and SSE `Last-Event-ID` resumption.
* **Hybrid storage:** `workspace/runs/<run_id>/events.jsonl` is the hot
  SSOT on the Mac; on run finalize a copy plus a sha256 manifest is
  uploaded to MLflow as a long-term artifact (R-08 retry policy).
* **Server-Sent Events for frontend.** Native browser reconnect via
  `Last-Event-ID`; we keep WebSockets for the CLI (pod-side already WS).
  The SSE handler subscribes to the in-memory bus *before* it replays
  the journal tail — see R-19 below.
* **Schema evolution** via `schema_version: int` on the envelope plus a
  registry of pure-function upcasters. Each hop handles exactly
  `(N, N+1)`; the chain runner composes them. See
  `packages/shared/src/ryotenkai_shared/events/upcasters/README.md`.
* **Per-run isolation** through `EventEmitterRegistry` singleton so the
  API router can look up the live in-memory bus for any active run by
  `run_id` without holding a reference across request boundaries.

## What was removed

* `packages/pod/src/ryotenkai_pod/trainer/mlflow/event_log.py`
  (MLflowEventLog — the second SSOT)
* `packages/pod/src/ryotenkai_pod/trainer/callbacks/training_events_callback.py`
* `packages/providers/src/ryotenkai_providers/inference/interfaces.py`
  (`InferenceEventLogger` Protocol — superseded by the unified emitter)
* The seven `log_event_*` methods on `MLflowManager` and all 24 of their
  call sites
* Five `*EventCallbacks` dataclasses on stages (dataset_validator,
  gpu_deployer, training_monitor, model_retriever, model_evaluator)
* The string-sniffing block in `mlflow_adapter.py:712-715` and the
  associated category-classification fallback chain
* The `training_events.json` MLflow artifact — replaced by
  `events.jsonl` plus `events_manifest.json` (sha256-checked) under the
  `events/` artifact prefix
* The `timeline_events` field from `StageArtifactEnvelope` (events now
  live in the journal, not on per-stage artifacts)

## Implementation phases (executed)

| # | Phase | What landed |
|---|---|---|
| 0 | Smoke prototype | Pydantic v2 + UUIDv7 + TypeAdapter round-trip; codec timing baseline (~720k ev/s ser, ~180k ev/s deser on M-series Mac) |
| 1 | Shared events package | 55 event types, codec with length-prefix framing, discriminator, `IEventEmitter` Protocol, upcaster scaffolding |
| 1.5 | Review issues | `MalformedEventError`, source-contract clarification, defensive copy of `raw_payload` |
| 2 | Pod-side rewire | Envelope schema on `event_bus.py`; length-prefix `event_journal.py`; async-queue `runner_event_callback.py` (no self-disable); `PodEventEmitter` |
| 2.5 | TrainingFailedEvent | Wired in the trainer's exception path |
| 3 | Control foundation | `JournalWriter`/`Reader`, `InMemoryBus`, `EventDedup`, `ControlEventEmitter`, DI wiring in orchestrator |
| 4 | Stage migration | Five `*EventCallbacks` dataclasses deleted; all five stages migrated to `emitter.emit()` |
| 5 | Coverage gaps | `inference_deployer`, `deployment_manager` event coverage; `ssh_provisioned` + `code_synced` from providers |
| 6.a | API router | `GET /runs/{id}/events` HTTP replay + `GET /runs/{id}/events/stream` SSE with subscribe-first ordering; `MlflowFinalizer` with retry; `EventEmitterRegistry` |
| 6.b | Dual-path retired | `MLflowEventLog` reduced to a no-op shim, then removed |
| 7 | Report adapter rewrite | `JournalReportAdapter`; `mlflow_adapter.py` cleanup (24 callsites → 0); string sniffing gone |
| 8 | Observability | Metrics counters across collaborators + `/api/v1/health/events` aggregator endpoint |
| 9 | Tests + fuzz + mutation gate | Hypothesis fuzz on codec; integration tests for resume, backpressure, SSE; **production bug found** by Phase 9 fuzz (inverted cursor branch in `InMemoryBus.subscribe`) — fixed in the same phase |
| 10 | Docs | ADR-0009 (this file), shared/events README, control/events README, CLAUDE.md update |

## Risks closed

| # | Risk | How closed |
|---|---|---|
| R-01 | Pod journal lost on pod crash | HTTP fallback `GET /pod/events?after_offset=N&source=…` plus periodic poll of `last_persisted_offset` while WS is down |
| R-02 | Pod WS publish fails silently | Async queue with retry; emitter never self-disables; drops are visible via `events_dropped_total{consumer="loopback"}` counter |
| R-03 | Big-bang PR rollback complexity | Staged commits within the PR; each commit a green CI; `git revert HEAD` remains a valid fallback for single-user development |
| R-04 | JSONL torn write on `kill -9` | Length-prefix framing + truncate-on-init via tmp+rename atomic |
| R-05 | Offset collision on concurrent emit | `threading.Lock` around offset counter; verified 1000 distinct offsets across 20 threads |
| R-06 | Emitter silently drops events | Mandatory counters on every collaborator + `/api/v1/health/events` aggregator endpoint |
| R-08 | MLflow upload fails, journal lost | Retry 3× (1s/5s/30s); `manifest.mlflow_uploaded=false` on final failure; workspace retention respects the flag |
| R-09 | Pydantic TypeAdapter validation cost | Benchmarked at 180k+ ev/s, well above the 50k/s baseline target; adapter cached as module-level singleton |
| R-10 | fsync blocks event loop | Batched fsync (50 events or 1s); immediate fsync for severity ≥ error; latency monitored via `journal_fsync_latency_seconds` |
| R-11 | Unknown event type on older consumer | `UnknownEvent` catch-all variant in union; codec wraps in non-strict mode |
| R-12 | Resume semantics undefined | Same `run_id` appends to existing journal; offset counter restored from last valid envelope |
| R-13 | Dedup set memory growth | TTL eviction policy + `reconstruct_from_journal` on restart (last 10k offsets per source) |
| R-15 | ContextVar `stage_id` loss across async tasks | `asyncio.create_task(coro, context=copy_context())` helper; `--stage-id` CLI arg for subprocesses |
| R-19 | SSE catchup → live race | Subscribe-first architecture (subscribe to bus → snapshot current_max → replay journal up to that offset → drain bus from cursor); verified end-to-end |

## Bench numbers

* Codec: 720k ev/s serialize, 180k ev/s deserialize on M-series Mac
  against a 55-member discriminated union (Phase 0 baseline).
* Emit p99: < 1 ms under `threading.Lock` contention (verified
  50 threads × 10 emits with no offset collisions).
* Tests: 5800+ unit + integration tests green; mutation gate 60-78%
  kill rate on the four critical modules
  (`codec.py`, `journal_writer.py`, `in_memory_bus.py`, `emitter.py`).

## Files touched

Approximate scope: ~80 files modified, ~30 new files, ~10 deleted;
+13 000 / −8 000 LOC. New files concentrate in
`packages/shared/src/ryotenkai_shared/events/` (16 files) and
`packages/control/src/ryotenkai_control/events/` (10 files). Largest
deletions: `MLflowEventLog`, `TrainingEventsCallback`, the seven
`MLflowManager.log_event_*` methods plus their 24 call sites, the five
`*EventCallbacks` dataclasses, the `InferenceEventLogger` Protocol.

## Future work

The following items are intentionally **out of scope** for this PR and
tracked as separate follow-ups:

* **95% mutation kill rate** on the four critical modules (currently
  60-78%). Phase 9's hypothesis fuzz raised coverage substantially but
  did not hit the 95% bar; the remaining survivors are mostly in error
  branches that need targeted negative cases.
* **`EventDedup.evict_expired()` background sweep.** The TTL eviction
  path exists but is called only on emit; long-idle runs will still
  hold their dedup sets until process exit. A periodic sweeper task
  (per-emitter, cancelled at `close()`) closes this.
* **Optional Prometheus / OpenTelemetry adapter.** The `EventSubsystemMetrics`
  dataclass is the right shape for it but no exporter is wired today.
* **`orchestrator.py` split** — the orchestrator file is the project's
  top hotspot (45 commits in 90 days, 100th-percentile churn). Its
  emitter-integration refactor lands in a separate PR per Phase 10
  scope discipline.
* **Frontend SSE switchover.** The existing polling endpoint
  `GET /job/events` is preserved as a back-compat surface; the
  frontend switches to the SSE stream in its own PR so this change set
  stays Python-only.
* **`UnknownEvent.original_kind` audit.** The codec wraps unknown
  envelopes with `original_type` (kind that the producer sent). A
  follow-up adds a structured warning + counter so a new event type
  rolled out on one side without the other is visibly counted.

## Rejected alternatives

| Alternative | Reason for rejection |
|---|---|
| gRPC for pod ↔ control | Existing transport (HTTP loopback + WS) already handles bidirectional flow; gRPC adds code-gen + new dependency. Pydantic discriminated unions give equivalent type safety end-to-end. |
| Kafka / NATS / Redis Streams | Single-tenant workspace, one pod per run. A distributed broker is overkill: 50-200 ms latency added plus 3+ HA pods for one machine. |
| Open dict envelope ("just JSON") | The pre-state. Demonstrated in the three concrete signals above why a closed typed union is the right boundary. |
| Per-stage SQLite database | Adds a runtime dependency (sqlite-utils + locking on NFS), and the report queries are exclusively sequential reads of recent events — JSONL is the right primitive. |
| Free-form `logger.info()` parsed downstream | The pre-state for `inference_deployer` and `deployment_manager`. The R-06 / string-sniffing chain demonstrates why this scales poorly. |
