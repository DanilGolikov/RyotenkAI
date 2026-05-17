# Event System Research: 10 Orchestration Tools + Patterns

**Context.** RyotenkAI has three parallel event subsystems (in-memory bus + JSONL journal on
pod; MLflow `training_events.json` artifact; stage-level snapshots on control). Goals:
single source of truth, no brittle string parsing, replay-on-reconnect, multiple consumers
(reports / WebSocket UI / CLI / MLflow). This document is purely research — no
code/architecture changes proposed for the project yet (that's a follow-up plan).

---

## 1. Comparative Matrix

| Tool | Schema model | Storage / SSOT | Streaming transport | Replay/Catchup | Filtering | Failure modes | Versioning |
|---|---|---|---|---|---|---|---|
| **Dagster** | Closed discriminated union: `DagsterEventType` enum (RUN_*, STEP_*, ASSET_*, ENGINE_EVENT, RESOURCE_INIT_*) wrapped in `EventLogEntry`; structured payload + raw `compute_logs` separately | SQL event log (`SqlEventLogStorage`: Postgres/MySQL/SQLite, run-sharded by default); append-only; numeric `id` cursor per shard + timestamp run cursor | GraphQL **subscriptions over WebSocket** (Apollo client `useSubscription`); server polls DB and pushes batches | `logs_after(run_id, after_cursor)` — id-based cursor. Key lesson: **don't reactively change cursor variable** — long-lived subscription only, server does batching internally | `EventLogRecordsFilter` (asset_key, partitions, event type, run_id, after_cursor) | DB layer = SSOT; if WS dies, client reconnects with stored last cursor; sensors use independent cursor state | Closed Python enum, additive; new event types added behind feature flags |
| **Prefect 3** | Open schema with structured fields: `id`, `event` (dotted name), `occurred`, `resource` (primary, dot-delimited dotted ID), `related` (list), `payload` (free-form), `account`/`workspace` | Backend store (Cloud or self-hosted Postgres); event feed is the SSOT, audit logs derive from it | WS for UI feed; **webhook ingestion** for external (CloudEvents-compatible); push to automations | Event feed is queryable historical store; no formal "replay API" — historical query by time/resource | `match` / `match_related` with wildcards (`prefect.flow-run.*`), AND across labels, OR across values; proactive triggers (absence-of-event) | Webhook 503 on backpressure; automations decoupled from emission | Open key-value resource labels; "custom state names" became first-class in v3 → new dotted event names |
| **Airflow** | No structured event union — uses **listener hooks** (pluggy) called inline: `on_task_instance_running/success/failed`, `on_dag_run_*`. State machine on `TaskInstance.state` (SUCCESS/FAILED/SKIPPED/...) | RDBMS metadata DB is SSOT (task_instance, dag_run tables); XCom for inter-task value passing (not events) | No native stream; UI polls REST API. External: ship logs to OTel/StatsD via providers | State is row in DB; "replay" = re-query DB by `dag_run_id` | SQL queries on metadata DB; REST API filters | Listeners run inline → can block scheduler; Airflow 3 added "listener fired via API" distinction. Known bugs (issue #53162) | Hook signatures change between major versions (2→3); compatibility through `pluggy` versioning |
| **Kubeflow Pipelines (MLMD)** | Typed graph: **Executions**, **Artifacts**, **Events** (linking executions ↔ artifacts), **Contexts**. Artifact schemas declared (`system.Dataset`, `system.Model`, ...) — typed properties | MLMD gRPC service over Postgres/MySQL (SSOT for lineage). KFP backend RDBMS for runs; object storage for artifacts | gRPC (MLMD client); KFP UI polls API | Lineage graph traversal: `GetEventsByArtifactIDs`, `GetExecutionsByContext` — content-addressable, not offset-based | By artifact ID, execution context, type | MLMD must be reachable to launch step; driver/launcher publish synchronously | Schema is extensible via custom artifact types; "system.*" types stable |
| **MLflow** | **No event API** per se. "Events" are: metric history (timestamped append), system tags (git commit, user, source), run inputs (datasets), artifacts, run state transitions (`RUNNING`→`FINISHED`/`FAILED`/`KILLED`) | Backend store (file/SQL); artifact store (S3/GCS/local) | None native — clients poll REST API | Replay via REST: `GetMetricHistory(run_id, key)` returns full time series | REST filters: by run_id, experiment, tag, metric key/range; SQL-like `MlflowClient.search_runs` | If backend down → SDK retries with backoff; can lose in-flight metrics | Schema additions are backwards-compatible (params immutable, tags overwritable, metrics append-only) |
| **W&B** | Out-of-process agent (`wandb-core`) with internal **file_stream** protocol over HTTP; run states: `running`/`finished`/`crashed`/`failed`/`killed` (state machine in backend). Events implicit in metric/log/system-stats streams | W&B cloud SaaS / on-prem server; SSOT in their backend | Server-push for live updates in UI; agent → backend over HTTPS (multipart streams) | Resume by run ID (`wandb.init(id=..., resume=True)`); backend reconstructs from streams | UI filters by tag, sweep, group; programmatic API for `runs()` with MongoDB-style filters | Internet outage → agent retries, eventually marks `crashed`. Separate process so training never blocked | SaaS controls schema; client SDK stable across versions |
| **Ray** | **Protobuf-defined event types** (one .proto per type under `src/ray/protobuf/public/events_*_event.proto`). Common base + event-specific fields. `RayEventInterface` C++ class hierarchy | Per-node `AggregatorAgent` with bounded **MultiConsumerEventBuffer** (deque, drops oldest on overflow); export to user HTTP endpoints. State APIs query GCS | gRPC from workers → aggregator → HTTP POST (batched JSON) to external. Dashboard polls state API | Per-consumer cursor in buffer; **drop oldest on overflow** with per-consumer eviction counters; no durable replay without external sink | `RAY_DASHBOARD_AGGREGATOR_AGENT_EXPOSABLE_EVENT_TYPES` env var allowlist; CLI `ray list cluster-events --filters` | Bounded buffer → backpressure visible via metrics; external HTTP sink optional. Head-node ephemerality was original motivation for redesign | Add new event type = new .proto file; additive |
| **Argo Workflows** | Two layers: (a) **k8s native events** (workflow status via watch API on CRDs); (b) **Argo Events** = full event-driven framework, **CloudEvents-compliant**. Sensors filter on JSON body via expressions | etcd (k8s state) + EventBus (NATS/Kafka/JetStream) for Argo Events | k8s watch API streams ADDED/MODIFIED/DELETED; EventBus topics for cross-component | k8s resource version (`resourceVersion`) is the watch cursor; reconnect with last version | CEL/jq-style expressions on event body in Sensors; resource event source uses Informer | EventBus = 3-pod HA cluster; argo-server `--event-operation-queue-size`, `--event-worker-count` for backpressure; 64 concurrent before 503; sub-second latency tricky (50–200ms EventBus overhead) | CloudEvents `type` for routing; resource versioning via CRD `apiVersion` |
| **OpenTelemetry GenAI** | **SemConv-defined event names** (currently 2 stable-ish): `gen_ai.client.inference.operation.details`, `gen_ai.evaluation.result`. Plus spans (Inference/Embeddings/Retrieval/Tool) and metrics (`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`, time-to-first-token, etc.). Attribute keys namespaced `gen_ai.*`. **No training-specific conventions** (inference-only as of 2026-Q1) | OTel exporters (OTLP) → vendor backend (Datadog, Grafana, Jaeger, ...) | OTLP over gRPC/HTTP; OTel Collector pipelines | Backend-dependent (no native replay; backend handles retention) | Standard OTel attribute filtering; trace context propagation | Collector with retry/queue/batch; sampling | `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` env var lets users opt into newer convention versions while default stays pinned |
| **CloudEvents v1.0** | Spec-level envelope: **required** `id`, `source`, `specversion`, `type`; **optional** `subject`, `time`, `datacontenttype`, `dataschema`, `data`. Two modes: structured (whole event in body) and binary (attrs in transport headers) | None — it's just a wire format / data model | Transport-agnostic: HTTP, Kafka, AMQP, MQTT, NATS, ... | Spec doesn't dictate; transport-dependent | `type` is canonical discriminator; `subject` for fine-grained routing; `source` + `id` is unique key | None at spec level | `specversion` field; `dataschema` URI per event |

---

## 2. Per-Tool Deep Notes

### Dagster
- `DagsterEvent` is a **closed Python class** with `DagsterEventType` enum (~40 types). Three big families:
  - Run lifecycle: `RUN_START`, `RUN_SUCCESS`, `RUN_FAILURE`, `RUN_CANCELED`, `ENGINE_EVENT`.
  - Step lifecycle: `STEP_START/SUCCESS/FAILURE/SKIPPED/UP_FOR_RETRY/RESTARTED`, `STEP_WORKER_*`, `RESOURCE_INIT_*`.
  - Asset events: `ASSET_MATERIALIZATION`, `ASSET_OBSERVATION`, `ASSET_CHECK_EVALUATION`, ...
- **Storage:** `SqlEventLogStorage` is the abstract base — `dagster_postgres`/`dagster_mysql`/sqlite implement it. Run-sharded SQLite means each run gets its own DB file (cheap, no central bottleneck for OSS).
- **Cursor pattern lesson (load-bearing).** The Dagit log viewer originally re-subscribed every time the cursor advanced — Apollo tore down WS each time, fetching the same logs repeatedly. The fix was a **single long-lived subscription** with the cursor managed server-side via `logs_after()`. → If you build WS streaming, the **cursor must not be a subscription parameter that mutates**.

### Prefect 3
- Schema: every event has a **resource** (the thing the event is about) as a dotted ID, e.g. `prefect.flow-run.<uuid>`, and a list of **related** resources (deployment, work-pool, tags). This is the discriminator surface — `match` filters wildcard against `prefect.resource.id`, `match_related` filters the related list.
- **CloudEvents-compatible** ingress (webhooks). Custom Python emit via `prefect.events.emit_event(...)`.
- "Custom state names" (v3) emit `prefect.flow-run.<CustomName>` — schema is open/extensible.
- **Proactive triggers**: "if expected event hasn't occurred in 30 min, fire". Pattern is highly useful for "stuck training" detection.

### Airflow
- Listeners run **inline in the worker process** via `pluggy`. Risk: a slow listener slows the scheduler. Airflow 3 added a distinction between `RuntimeTaskInstance` (real execution) and `TaskInstance` (DB-only state mark via API).
- No streaming. UI polls REST. Producers/consumers tightly coupled to DB schema.
- For an ML pipeline this is the **anti-model**: rich row state in DB but no good stream out.

### Kubeflow Pipelines / MLMD
- Lineage-first thinking: events are **edges in a graph** linking executions and artifacts, not a flat log. `EventType.{INPUT, OUTPUT, DECLARED_INPUT, DECLARED_OUTPUT}`.
- Artifact has typed schema (`system.Model`, `system.Dataset`, `system.Metrics`, ...). Reproducibility & caching key off MLMD fingerprints.
- This is **excellent for "what produced this artifact"** queries but heavy infra (gRPC + Postgres + Envoy).

### MLflow
- The closest analog to what RyotenkAI already does: append-only metric history is effectively a typed event stream. **Params are immutable, tags overwritable, metrics append-only** — these write semantics are the audit story.
- No event API. If you want one you build it externally (as RyotenkAI's `training_events.json` artifact already attempts).
- Auto-system-tags (`mlflow.source.git.commit`, `mlflow.user`, `mlflow.source.name`) are an implicit audit trail.

### Weights & Biases
- Best-in-class **resilience**: out-of-process agent, retries, run states (`running` → `crashed` on connectivity loss → resumable). Worth emulating for the WS journal.
- Closed SaaS schema; no public event spec.

### Ray
- The Ray Event Export redesign (2.55+) is a **modern reference architecture** to study:
  - One `AggregatorAgent` **per node**, bounded `MultiConsumerEventBuffer` (deque + asyncio.Lock + asyncio.Condition).
  - Multi-consumer cursors so multiple sinks can drain at different rates.
  - **Drop oldest on overflow** with per-consumer eviction counters → graceful degradation.
  - Adding new event type = drop a new `.proto` file in `src/ray/protobuf/public/events_*_event.proto`.
- For pod-side journal, this is closest in spirit to RyotenkAI's needs (single host, multiple consumers, bounded memory).

### Argo Workflows
- Two strategies are present:
  1. Use k8s `watch` semantics directly (resourceVersion cursor) on CRDs — cheap, no extra infra.
  2. Argo Events: full CloudEvents-compliant framework with EventBus (NATS by default), Sensors, Triggers.
- The watch-API pattern (long-poll with monotonic version cursor) is a strong primitive **even outside k8s** — adopt the *idea* without the infra.

### OpenTelemetry GenAI
- As of 2026-Q1, GenAI semconv covers **inference only** — no training conventions (no `gen_ai.training.loss`, `gen_ai.training.step`). Standard attributes like `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.finish_reasons`.
- Useful for inference/eval consumers but won't directly solve training-loop event modeling.
- **Versioning lever**: `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` — pin default + opt-in to newer. Worth borrowing.

### CloudEvents v1.0
- Vendor-neutral envelope: `specversion`, `type`, `source`, `id`, `time`, `subject`, `datacontenttype`, `dataschema`, `data`. `source` + `id` MUST be unique.
- **Pros** for adopting in RyotenkAI: cross-system interop (Kubeflow/KServe/Knative/Argo all speak it), free schema-evolution discipline via `dataschema` URI, transport-agnostic.
- **Cons**: ceremony for a single in-process app — 6 mandatory attrs per event when you only need 3.
- **Verdict**: adopt the **discriminator pattern** (`type`, `source`, `id`, `time`) and the dotted-name convention (`com.ryotenkai.training.step`) without the full envelope. Add a wire-format `to_cloudevent()` adapter at the egress boundary if/when external interop is needed.

---

## 3. Patterns Worth Adopting (recommendations)

### A. Schema design
1. **Closed discriminated union of event types** (Dagster/Ray model), not open key-value (Prefect model). Better static checks, better consumer ergonomics, cheaper to evolve safely when producers & consumers are in the same monorepo (which RyotenkAI is).
2. **Dotted reverse-DNS type names**: `ryotenkai.pod.training.step`, `ryotenkai.pod.training.checkpoint`, `ryotenkai.control.stage.transition`. Matches CloudEvents `type`, Prefect resource convention, OTel attribute namespacing.
3. **Common envelope** for every event: `event_id` (UUIDv7 for natural ordering), `type`, `source` (e.g. `pod://{run_id}/trainer`), `time`, `run_id`, `schema_version` (small int), plus a typed `payload`. UUIDv7 gives you `source+id` uniqueness *and* natural sort, replacing the need for a separate monotonic offset on the producer side.
4. **`schema_version` field on every event** (cheap, self-describing — sources: Marten, Axon best practices). Keep upcasters as **pure functions one-hop at a time** (v1→v2, v2→v3) registered in a chain.
5. **Versioning rules** (event-sourcing canon): additive non-breaking changes are default; new optional fields with defaults; **semantic change → new event type, never reuse**; stream migration is nuclear option. Codify in a doc + linter.

### B. Storage / SSOT
6. **JSONL append-only journal as SSOT on pod-side**, MLflow artifact derived from journal at finalize (not the other way around). One file per `run_id`. Matches Dagster's append-only philosophy with simpler infra.
7. **Numeric monotonic offset per run** as primary cursor (Dagster's `id` pattern), wall-clock `time` as secondary. UUIDv7 in the envelope is the global unique ID.
8. **Bounded in-memory buffer with multi-consumer cursors** (Ray's `MultiConsumerEventBuffer` pattern): `deque(maxlen=N)`, per-consumer cursor, drop-oldest-on-overflow with eviction counters exposed as metrics.

### C. Streaming
9. **One long-lived WebSocket subscription**, cursor managed server-side, not as a subscription variable (Dagster lesson). On client reconnect: client sends "last seen offset"; server replies with everything after, then continues live.
10. **SSE is a valid alternative** to WS when you only need server→client (simpler, plays nicer with HTTP middleware, easy reconnect via `Last-Event-ID` header). Worth considering instead of WS if no client→server messaging is needed.
11. **Heartbeat events** every N seconds — Prefect's "absence of expected event" trigger relies on this; also doubles as WS liveness probe.

### D. Replay / catchup
12. **HTTP endpoint `/events?run_id=X&after_offset=N&limit=M`** for cold replay. Same cursor model as live stream so client logic is identical (read from store until caught up, switch to WS, no gaps).
13. **Per-consumer cursor persistence is the consumer's job**, not the bus's (Ray model). The bus exposes offsets; consumers (UI/reports/MLflow) checkpoint them.

### E. Query / filter
14. **Server-side filter by type prefix** (e.g. `type.startswith("ryotenkai.pod.training.")`) and `run_id`. Keep filter language minimal — avoid Argo-style expression evaluation; if you need fuzzy filtering it's a sign the consumers should denormalize into their own store.
15. **Related-resources pattern from Prefect** is useful: every event carries `{run_id, stage_id, task_id?}` so a single index covers stage/task drilldown.

### F. Failure modes
16. **Producer never blocks on consumers**: append to journal synchronously (it's local disk, microseconds), publish to in-memory bus best-effort. If bus drops, journal still has it; consumer catches up via replay endpoint.
17. **Bounded buffers with drop-oldest + visible metrics** (Ray pattern). Expose `events_dropped_total{consumer="..."}` so backpressure is observable.
18. **Out-of-process agent like W&B** is overkill for RyotenkAI's scale — the trainer is already a subprocess. Just don't let the WS server block the trainer.

### G. Versioning
19. **`schema_version: int` on envelope + chain of upcasters at deserialize time**. New consumers always see latest schema; old data is upcast lazily.
20. **Borrow OTel's opt-in env-var pattern** if you ever ship breaking changes externally: `RYOTENKAI_EVENT_SCHEMA_OPT_IN=v2` keeps old default while letting integrators move.

---

## 4. What NOT to Copy (anti-patterns / overkill)

- **Don't adopt full CloudEvents envelope** as your internal model. 9 attrs + structured/binary mode + dataschema URIs is wire-format ceremony you don't need until you export to Knative/Argo/Kubeflow. Adopt the *discipline* (discriminator + unique id), not the *format*.
- **Don't deploy MLMD** for lineage. Excellent design but gRPC + Postgres + Envoy is wildly disproportionate for a monorepo trainer.
- **Don't follow Airflow's listener model**. Inline pluggy hooks in the trainer hot loop = blocking risk. Use an out-of-band journal + bus instead.
- **Don't introduce an EventBus cluster** (NATS/Kafka/Argo EventBus). 50–200ms latency, ops cost, and 3+ pods of HA for a single-host pod is overkill.
- **Don't build a CEP/expression-filter language** (Argo Sensors, Prefect Jinja templating in triggers). Simple type-prefix + run_id filter is enough; if a consumer needs more, it can read everything and filter client-side.
- **Don't expose the cursor as a mutable WebSocket subscription variable** (Dagster's painful lesson). Cursor lives in server state; client just gets the firehose for the subscription's lifetime.
- **Don't replace metrics with events**. OTel metrics (counter/histogram/gauge) and events serve different purposes — token/loss time series are metrics; state transitions and structured records are events. Don't conflate. MLflow's `metric history` is technically an event stream but it's typed as a metric.
- **Don't skip `schema_version`** because "we're a monorepo and ship them together." That's true until it isn't — log replay from a past run breaks. The cost (1 int per event) is zero.
- **Don't over-mirror the Prefect "open schema with arbitrary labels" model**. Looks flexible, but the bug rate of brittle string parsing in consumers is exactly what RyotenkAI's current design is suffering from.
- **Don't try to standardize on OTel GenAI semconv for training events** — the spec is inference-only as of 2026-Q1, and the SIG isn't taking training conventions yet. Keep a local namespace (`ryotenkai.*`) and adopt OTel attributes *only* for the inference/eval consumers that already match.

---

## 5. Recommended Direction (one-paragraph synthesis)

Treat the **pod-side append-only JSONL journal as the single source of truth**. Define events as a **closed discriminated union** (`type: Literal[...]` per variant) with a common envelope (`event_id` UUIDv7, `type` dotted name, `source`, `time`, `run_id`, `offset`, `schema_version`, `payload`). Producer writes to journal (durable) and a bounded in-memory **MultiConsumerEventBuffer** (Ray-style) in the same call. Two egress paths share one cursor model: a `/events?after_offset=N` HTTP endpoint for replay, and a single long-lived WS/SSE subscription that pushes new events as they're appended (Dagster's lesson: cursor is server-side state, not a subscription variable). On client reconnect: read from `/events` until caught up, then upgrade to WS. MLflow `training_events.json` becomes a *derived view* materialized at finalize, not a parallel source. Schema evolution is governed by `schema_version` + a chain of pure-function upcasters; semantic changes get a new dotted name, never reuse. Adopt CloudEvents as a wire format **only** at egress boundaries to external systems (none today), not internally. Skip MLMD, EventBus, CEP filtering, and full OTel GenAI semconv — they're solutions to problems an order of magnitude bigger than RyotenkAI's. Borrow Prefect's resource/related convention so a single index by `run_id`+`stage_id` covers UI drilldown.

---

## 6. Primary Sources

### Dagster
- [Dagster Internals API](https://docs.dagster.io/api/dagster/internals)
- [Dagster Logging Guide](https://docs.dagster.io/guides/monitor/logging)
- [DeepWiki: Dagster Storage and Persistence](https://deepwiki.com/dagster-io/dagster/5-graphql-api)
- [DeepWiki: Dagster GraphQL API](https://deepwiki.com/dagster-io/dagster/6-graphql-api)
- [dagster._core.events source](https://docs.dagster.io/_modules/dagster/_core/events)
- [Dagster blog — Web Workers Fall Short for Data UIs (cursor lesson)](https://dagster.io/blog/web-workers-performance-issue)
- [dagster_postgres.event_log source](https://docs.dagster.io/_modules/dagster_postgres/event_log/event_log)

### Prefect
- [Prefect Events concepts](https://docs.prefect.io/v3/concepts/events)
- [Prefect Automations](https://docs.prefect.io/v3/concepts/automations)
- [Track activity through events](https://docs.prefect.io/v3/automate/events/events)
- [Define custom event triggers](https://docs.prefect.io/v3/automate/events/custom-triggers)
- [Real-Time Event Workflows with Debezium + Prefect (2025)](https://www.prefect.io/blog/change-data-capture-tutorial-real-time-event-workflows-with-debezium-and-prefect)

### Airflow
- [Airflow Listeners (3.x)](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/listeners.html)
- [Listener Plugin howto](https://airflow.apache.org/docs/apache-airflow/stable/howto/listener-plugin.html)
- [event_listener API reference](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/example_dags/plugins/event_listener/index.html)

### Kubeflow Pipelines / MLMD
- [ML Metadata in Kubeflow](https://www.kubeflow.org/docs/components/pipelines/concepts/metadata/)
- [DeepWiki: KFP Storage and Metadata](https://deepwiki.com/kubeflow/pipelines/4.4-storage-and-metadata)
- [DeepWiki: KFP MLMD and Caching](https://deepwiki.com/kubeflow/pipelines/5.4-ml-metadata-and-caching)
- [KFP Artifacts user guide](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)

### MLflow
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow REST API](https://mlflow.org/docs/latest/rest-api.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [MlflowClient](https://mlflow.org/docs/latest/python_api/mlflow.client.html)

### Weights & Biases
- [W&B Runs Overview](https://docs.wandb.ai/models/runs)
- [W&B Troubleshooting (crashed-state semantics)](https://docs.wandb.ai/guides/technical-faq/troubleshooting)

### Ray
- [Ray Event Export user guide](https://docs.ray.io/en/latest/ray-observability/user-guides/ray-event-export.html)
- [Ray Event Exporter Infrastructure (internals)](https://docs.ray.io/en/latest/ray-core/internals/ray-event-exporter.html)
- [Ray Dashboard Observability](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
- [Ray issue #53073 — head-node ephemerality motivation](https://github.com/ray-project/ray/issues/53073)

### Argo Workflows / Events
- [Argo Workflows Events](https://argoproj.github.io/argo-workflows/events/)
- [Argo Events project](https://github.com/argoproj/argo-events)
- [Argo Events docs](https://argoproj.github.io/argo-events/)
- [Argo Events APIs](https://argoproj.github.io/argo-events/APIs/)

### OpenTelemetry
- [Semantic conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Semantic conventions for GenAI events](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/)
- [Semantic conventions for GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [Semantic conventions for GenAI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [Datadog native OTel GenAI support](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)

### CloudEvents
- [CloudEvents spec on GitHub](https://github.com/cloudevents/spec)
- [CloudEvents v1.0 core spec](https://github.com/cloudevents/spec/blob/main/cloudevents/spec.md)
- [CloudEvents primer (discriminator concept)](https://github.com/cloudevents/spec/blob/main/cloudevents/primer.md)
- [CloudEvents CNCF project page](https://www.cncf.io/projects/cloudevents/)

### Patterns / industry
- [Martin Fowler — What do you mean by "Event-Driven"?](https://martinfowler.com/articles/201701-event-driven.html)
- [ThoughtWorks — Event-Driven Architecture](https://www.thoughtworks.com/en-us/insights/decoder/e/event-driven-architecture)
- [Event-Driven.io — Simple patterns for events schema versioning](https://event-driven.io/en/simple_events_versioning_patterns/)
- [Marten — Events Versioning](https://martendb.io/events/versioning)
- [Axon — Event Versioning](https://docs.axoniq.io/axon-framework-reference/4.11/events/event-versioning/)
- [Artium — Upcasting deep dive](https://artium.ai/insights/event-sourcing-what-is-upcasting-a-deep-dive)
- [Greg Young — Versioning in an Event Sourced System (free book)](https://leanpub.com/esversioning/read) *(canonical reference)*
