# Runner architecture

The Job Server (or "runner") is a single-process FastAPI service that
runs **inside the training pod** and supervises one training job at a
time. Together with the Mac control plane it replaces the older
marker-file + `watchdog.sh` IPC with push-based events over an
HTTP/WebSocket transport tunnelled through SSH.

This document is the field manual: where each piece lives, how they
talk, what files persist where, and where to look when something
breaks.

> See [`docs/plans/harmonic-rolling-crayon.md`](plans/harmonic-rolling-crayon.md)
> for the full rationale and migration history. This is the post-cutover
> reference.

## Topology

```
┌──────────────── Mac (control plane) ────────────────────┐
│ FastAPI (src/api)                                       │
│  ├ routers/launch     start a run                       │
│  ├ routers/runs       project/run/attempt CRUD          │
│  ├ routers/datasets   preview + validation              │
│  └ routers/jobs       proxy to in-pod runner            │
│                                                         │
│ CLI (src/cli)                                           │
│  └ ryotenkai job <verb>   talk to the runner directly   │
│                                                         │
│ Pipeline subprocess (src/pipeline/orchestrator.py)      │
│  └ TrainingLauncher → SSHTunnelManager + JobClient      │
└─────────────────────────────────────────────────────────┘
          │     HTTP + WebSocket
          │     over `ssh -L 18080:127.0.0.1:8080`
          ▼
┌──────────────── Pod (training container) ───────────────┐
│ ENTRYPOINT: dumb-init                                   │
│   ├ /usr/sbin/sshd                                      │
│   └ uvicorn src.runner.main:app  127.0.0.1:8080         │
│                                                         │
│ Job Server  (src/runner/)                               │
│  ├ JobLifecycleFSM     state.jsonl persisted            │
│  ├ EventBus            offset-based ring buffer         │
│  ├ Supervisor          subprocess.Popen + signals       │
│  ├ MLflowRelay         async + circuit breaker          │
│  ├ IdleDetector        pynvml + nvidia-smi fallback     │
│  ├ HealthReporter      GPU/RAM snapshots                │
│  └ PluginUnpacker      reward plugins payload           │
│        │                                                │
│        ▼ subprocess.Popen (one process group)           │
│  python -m src.training.run_training                    │
│   └ RunnerEventCallback → loopback POST /internal/events│
└─────────────────────────────────────────────────────────┘
```

## Components

### Mac side

| Component | Path | Responsibility |
|---|---|---|
| `TrainingLauncher` | `src/pipeline/stages/managers/deployment/training_launcher.py` | Pack reward plugins, open SSH tunnel, submit job, stash handles on pipeline context, persist `JobSubmission`. |
| `TrainingMonitor` | `src/pipeline/stages/training_monitor.py` | Subscribe to runner events via WebSocket, dispatch progress / health / terminal callbacks. |
| `SSHTunnelManager` | `src/api/services/tunnel_service.py` | Open `ssh -fN -L`, isolated ControlMaster socket per run, readiness probe, idempotent close. |
| `JobClient` | `src/api/clients/job_client.py` | HTTP + WS facade with auto-reconnect and offset tracking. |
| `PluginPacker` | `src/pipeline/stages/managers/deployment/plugin_packer.py` | Walk strategy chain, dedupe reward plugins, build a single deterministic ZIP. |
| `JobSubmission` | `src/pipeline/state/job_submission.py` | Immutable on-disk record (`attempts/<n>/job_submission.json`) used by CLI/Web to dial the runner from a fresh process. |

### Pod side

| Component | Path | Responsibility |
|---|---|---|
| FastAPI app | `src/runner/main.py` | Lifespan: `restore_or_init`, dumb-init-friendly shutdown. |
| `JobLifecycleFSM` | `src/runner/state.py` | Transitions, persistence (`state.jsonl` append-only + `state.json` snapshot). |
| `EventBus` | `src/runner/event_bus.py` | Pub/sub + bounded ring buffer (default 10k events). |
| `Supervisor` | `src/runner/supervisor.py` | Spawns trainer in its own process group, two-phase shutdown (SIGTERM → wait → SIGKILL on `pgid`), exit-code parsing for native crashes. |
| `IdleDetector` | `src/runner/idle_detector.py` | Python replacement for `watchdog.sh`. GPU util via pynvml, fallback to `nvidia-smi`. Triggers FSM stop, never `kill -9`. |
| `HealthReporter` | `src/runner/health_reporter.py` | Periodic GPU/RAM/CPU snapshots → events. |
| `MLflowRelay` | `src/runner/mlflow_relay.py` | Async forward via `asyncio.Queue`; reuses `MLflowTransportCircuitBreaker`. |
| `PluginUnpacker` | `src/runner/plugin_unpacker.py` | Self-contained ZIP extractor (no `src.community` dep) under `/workspace/community/reward/<id>/`. |
| `PodStopper` | `src/runner/pod_stopper.py` | RunPod GraphQL self-stop on terminal FSM transition. |
| Trainer callback | `src/training/callbacks/runner_event_callback.py` | Buffered POST to `127.0.0.1:8080/internal/events`; degrades gracefully on 3 consecutive failures. |

## Endpoints

The runner serves three HTTP and one WebSocket endpoint, all bound to
loopback so the SSH tunnel is the only ingress path. The control-plane
FastAPI mirrors them as a proxy at `/api/v1/runs/{run_id}/job/...` so
the Web UI (which cannot open SSH tunnels) can poll without holding
its own connection.

| Method | Path (in-pod) | Mac-side proxy | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/jobs` | n/a (launcher submits direct) | Multipart submit: JSON `job_spec` + ZIP `plugins_payload`. |
| `GET` | `/api/v1/jobs/{id}` | `GET /api/v1/runs/{run_id}/job/status` | Current FSM snapshot. |
| `POST` | `/api/v1/jobs/{id}/stop` | `POST /api/v1/runs/{run_id}/job/stop` | Graceful stop request. |
| `WS` | `/api/v1/jobs/{id}/events?since=N` | `GET /api/v1/runs/{run_id}/job/events?since=N&limit=N` | Live event stream. The Mac proxy buffers a slice; the CLI / pipeline use the WS directly via `JobClient.subscribe_events`. |
| `POST` | `/internal/events` | n/a (loopback only) | Trainer callback ingestion. |

## Persistence

| Where | What | Format |
|---|---|---|
| Pod, `/workspace/.ryotenkai/state.jsonl` | Append-only FSM transition log | One JSON record per transition. |
| Pod, `/workspace/.ryotenkai/state.json` | Last known state | Atomic write via `atomic_write_json`. |
| Pod, `/workspace/.runner/events/events.NNN.jsonl` | Phase 12.B durable event journal (rotated, 5 × 100 MB) | One JSON record per event (`v`, `offset`, `ts`, `kind`, `payload`). |
| Mac, `<run>/attempts/<n>/job_submission.json` | SSH endpoint + `job_id` to dial back into the pod | `JobSubmission` schema v1. |
| Mac, `<run>/attempts/<n>/job_events.jsonl` | Caught-up runner events | One JSON record per event (offset / kind / payload). |
| Mac, `<run>/pipeline_state.json` | High-level run status | Existing `PipelineStateStore`. |

The Mac control plane is the source of truth for **where the pod
lives**; the runner is the source of truth for **what the job is
doing**. Restart-safe by construction: re-reading `state.jsonl` lets
the runner resume a `preparing`/`stopping` state into `failed` on
container restart.

## Lifecycles

### FSM transitions

```
                              POST /jobs
                                  │
                                  ▼
                          [preparing]   ← unpack plugins, validate
                              │   │
                       prep ok│   │ prep failed
                              ▼   ▼
                          [running] ──────► [failed]
                              │
                ┌─────────────┼──────────────┐
        natural │             │ stop request  │ idle / crash
        finish  │             │               │
                ▼             ▼               ▼
         [completed]     [stopping] ──► [completed | cancelled | failed]
```

Terminal states are `completed`, `failed`, `cancelled`. They trigger
`PodStopper` on RunPod (best-effort, logged via `pod_stop_attempt`
event). Once terminal the FSM accepts no more transitions; the
proxy keeps serving the snapshot until the pod is reaped.

### Detach / reattach

1. Mac launches the run. `TrainingLauncher` opens an SSH tunnel and
   POSTs the multipart job. The tunnel + JobClient are stashed on the
   pipeline context, and `JobSubmission` is persisted to the attempt
   directory.
2. Mac goes to sleep. The SSH tunnel collapses. The runner keeps
   training and keeps appending to its in-memory ring buffer + the
   on-disk `state.jsonl`.
3. Mac wakes. The user runs `ryotenkai job events <run> --follow`
   (or opens the Web UI's Live tab). The CLI reads
   `job_submission.json`, opens a new tunnel, and resumes the WS
   subscription with the highest local offset. Anything that happened
   while the Mac was asleep streams in as backlog; live events
   continue from the same socket.

## Wire formats

### Job submission (`POST /api/v1/jobs`)

```
multipart/form-data
  └─ job_spec: application/json
       {
         "job_id":   "run-foo:attempt:1",
         "command":  ["python", "-m", "src.training.run_training", "--config", "..."],
         "env":      { "HF_TOKEN": "...", "MLFLOW_TRACKING_URI": "...", ... }
       }
  └─ plugins_payload: application/zip   (optional — empty for SFT-only runs)
       reward/<plugin_id>/
         ├─ manifest.toml
         └─ plugin.py | plugin/
```

### Event envelope

Every event the runner emits — whether internal or trainer-pushed —
shares the same shape:

```json
{
  "offset":   42,
  "kind":     "step",
  "ts":       "2026-04-26T12:00:00.123+00:00",
  "payload":  { "loss": 0.42, "step": 100, "epoch": 1 }
}
```

`offset` is monotonic per job and used as the cursor for resume. The
ring buffer drops the oldest events first; clients receive
`ReplayTruncatedError` if they ask for an offset that has rolled off.

## Stop semantics (Phase 9)

`ryotenkai job stop` is **irreversible**. Pod is terminated (RunPod
`podTerminate`, single_node `docker rm -f`), the run is closed in
MLflow with `RunStatus.KILLED`, and the trainer's last checkpoint
(at the most recent `save_steps` boundary) stays on disk in
`attempts/<n>/`.

### Layer ownership at stop

| Layer | Stop responsibility |
|---|---|
| User CLI / Web UI | Send stop intent via `JobClient.request_stop` |
| Mac control plane FastAPI | Proxy stop request; on terminal event invoke `provider.cleanup_pod()` (PRIMARY pod removal) + read `cancelled.marker` for MLflow reconciliation |
| Runner Supervisor (in-pod) | SIGTERM trainer pgid; SIGKILL escalation after `--grace`; FSM transitions; emit `cancellation_started` / `cancellation_completed` |
| Trainer subprocess | `ShutdownHandler` flag → `CancellationCallback` cooperative loop exit → `flush_buffer` (5s budget) → emit `cancellation_finalized`; HF Trainer's MLflow callback closes the run with `KILLED` |
| In-pod `PodStopper` | Safety-net: at FSM terminal → `podTerminate` (delete, not sleep). User-stop ignores `RUNPOD_KEEP_ON_ERROR` |
| Single-node provider | `cleanup_after_run(container_name)` — `docker rm -f` via SSH (10s timeout, fail-soft) |

### Cancellation event chain

Six event kinds telegraph the stop chain end-to-end. All carry
`latency_ms` measured against the `requested_at_ms` anchor stamped
on `cancellation_started` so dashboards can build SLO histograms
without correlating across processes.

| Event kind | Emitted by | Carries |
|---|---|---|
| `stop_requested` | Supervisor (kept for backwards-compat) | `grace_seconds` |
| `cancellation_started` | Supervisor | `requested_at_ms`, `grace_seconds`, `reason` |
| `cancellation_finalized` | CancellationCallback (trainer) | `flushed_count`, `flush_timed_out`, `marker_written`, `flush_budget_seconds` |
| `trainer_exited` | Supervisor (existing) | `exit_code`, `signal`, `cancellation_requested` |
| `cancellation_completed` | Supervisor | `total_latency_ms`, `terminal_state`, `exit_code`, `signal`, `requested_at_ms` |
| `pod_stop_attempt` | PodStopper (existing) | `terminal_state`, `outcome` |
| `mlflow_reconciled_post_sigkill` | Mac TrainingMonitor | `run_id`, `marker_path` |
| `cleanup_pod_failed` | Mac orchestrator (reserved) | `provider`, `pod_id`, `error_code`, `message` |

Constants live in `src/runner/cancellation_telemetry.py` — single
source of truth so dashboards / SLO alerts grep for stable strings.

### SIGKILL fallback: `cancelled.marker`

When the trainer's MLflow flush exceeds its 5-second budget the
process may be SIGKILLed by the supervisor's grace timer before
HF's MLflow callback runs. To rescue the operator-visible state:

1. `CancellationCallback.on_train_end` writes
   `<workspace>/cancelled.marker` containing
   `{run_id, flushed_count, ts_ms, reason}` via
   `atomic_write_text`.
2. Mac `TrainingMonitor` after the WS subscription returns:
   - reads `attempts/<n>/cancelled.marker` (atomic — partial files
     never seen),
   - calls `MlflowClient.get_run(run_id).info.status` — if `RUNNING`,
   - calls `MlflowClient.set_terminated(run_id, status="KILLED")`,
   - emits `mlflow_reconciled_post_sigkill` for forensics.
3. If the marker is absent (normal flow — flush completed in time)
   reconciliation is a no-op.

The marker is workspace-scoped (one per attempt) so reconciliation
can't apply the wrong run_id. Best-effort throughout: any failure
logs at WARNING and the rest of cleanup proceeds — the operator
can still manually fix the MLflow status if everything fails.

### Pause: explicitly out of scope

Pause / Resume is a separate plan (placeholder in
`harmonic-rolling-crayon.md` Phase 10). Today: Stop = terminate;
Resume creates a new attempt + new pod from the prior checkpoint
(`ryotenkai run resume <run>`). The `CancellationCallback`
flag-driven cooperative exit is the same mechanism Pause would
reuse with a different `ShutdownReason` enum value — adding pause
later won't require architectural rework, just additional UI/CLI
surface + a config knob.

## Sleep & resume semantics (Phase 11)

Phase 11 changed the natural-completion path from "always
terminate" to "stop the pod (sleep) so the user can resume on Mac
wake". User-stop is unchanged from Phase 9 — explicit user action
still calls `podTerminate`.

### Decision matrix

The runner's `PodTerminator` (`src/runner/pod_terminator.py`) picks
between four outcomes on every FSM terminal transition:

| terminal | `mac_alive` | `volume_kind` | `KEEP_ON_ERROR` | Outcome | RunPod call |
|---|---|---|---|---|---|
| `cancelled` | * | * | * | `terminated_user_stop` | `podTerminate` |
| `failed` | * | * | true | `kept_alive_for_debug` | (none) |
| `failed` | true | persistent | false | `terminated_safety` | `podTerminate` |
| `failed` | false | persistent | false | `stopped_for_resume` | `podStop` |
| `failed` | * | network | false | `terminated_safety` | `podTerminate` |
| `completed` | true | persistent | * | `stopped_for_resume_short_grace` | wait → `podStop` |
| `completed` | false | persistent | * | `stopped_for_resume` | `podStop` |
| `completed` | * | network | * | `terminated_safety` | `podTerminate` |

`mac_alive` comes from `MacHeartbeat` (`src/runner/heartbeat.py`)
which the WebSocket handler and REST `GET /jobs/{id}` handler
both refresh via `mark_active()`. TTL = 60s. ModelRetriever's
periodic GET keeps the heartbeat alive while it pulls adapters
off the pod, so the SHORT_GRACE window effectively extends to
cover the whole download (capped at 10 min).

`volume_kind` is read from `RUNPOD_VOLUME_KIND` env (set by
`TrainingLauncher._build_job_env`). RunPod constraint: pods with
network volumes can't be stopped, only terminated — the matrix
short-circuits to `terminated_safety` for `network`. Default
`persistent` matches today's training-pod config.

### Marker symmetry: `cancelled.marker` vs `completion.marker`

Two marker files live in the attempt directory and drive Mac-side
MLflow reconciliation when the trainer's MLflow callback couldn't
reach upstream (Mac was asleep when `end_run` fired):

| Marker | Written when | Reconciles to |
|---|---|---|
| `cancelled.marker` (Phase 9.C) | Cancellation flush exceeded 5s budget | MLflow `RUNNING` → `KILLED` |
| `completion.marker` (Phase 11.A) | **Always** on natural completion | MLflow `RUNNING` → `FINISHED` |

`completion.marker` is always-written so Mac on wake can tell
"trainer finished naturally" from "trainer is still running, no
recent events". The marker payload carries `flush_timed_out:
bool` so the operator log warns about partial data when
appropriate.

When both markers exist (rare race — cancellation hit at the very
end of natural completion), cancellation wins. Mac's
`TrainingMonitor._reconcile_terminal_marker_if_present` walks
`_TERMINAL_MARKER_PRIORITY` and forces the right MLflow status.

### Resume flow on Mac wake

Phase 11.C-1 ships the data-model + decision logic; the
operator-facing surface (CLI hint, REST endpoint, Web UI button)
lands in Phase 11.C-2.

What's available today:

* `PipelineAttemptState.pod_metadata: PodMetadata | None` —
  persisted on every attempt that recorded a pod_id. Legacy
  attempts (pre-Phase-11.C) deserialize to `None` and the probe
  treats them as RUNNING.
* `PodAvailabilityProbe`
  (`src/pipeline/launch/pod_availability.py`) — queries the
  provider's pod status, maps to one of:
  * `RUNNING` — no action; SSH connect proceeds.
  * `SLEEPING_RESUMABLE` — call `resume_pod_with_retry`.
  * `SLEEPING_RESUME_FAILED` — capacity exhausted; surface to
    user.
  * `GONE` — terminated; user must `run restart` from a
    checkpoint.
  * `PROBE_FAILED` — RunPod outage; retry later.
* `resume_pod_with_retry` — capacity-aware backoff (10s / 30s /
  60s / 120s, total 5min budget) wrapping
  `RunPodAPIClient.resume_pod`.

### SLO targets (Phase 11)

| Metric | Target |
|---|---|
| Natural completion → pod stopped (Mac asleep) | p95 < 5s |
| Mac alive → grace + retriever done → pod stopped | p95 < 90s |
| Resume from EXITED (capacity available) | p95 < 60s |
| Resume capacity-exhausted error rate | < 5% |
| `completion.marker` write success rate | > 99.9% |
| MLflow run `FINISHED` reconciliation success | > 99% |

### Operator quick reference

```bash
# Watch a running cancellation chain in real time (text mode renderer):
ryotenkai job events <run_dir> --follow | grep -E "cancellation|trainer_exited|pod_stop"

# After-the-fact: reconstruct the chain from the runner's event store
# and pull total latency.
grep cancellation_completed pipeline.log | head

# Did SIGKILL fire before flush?
test -f attempts/<n>/cancelled.marker && echo "yes"
```

## Failure modes & where to look

| Symptom | Likely cause | First place to check |
|---|---|---|
| `ryotenkai job status` → 502 `runner_unreachable` | SSH tunnel can't reach `127.0.0.1:8080` (sshd or runner not up) | Pod's `dumb-init` logs, `runner_unreachable` detail message |
| `ReplayTruncatedError` mid-resume | Mac was asleep longer than 10k events worth of training | Bump `RYOTENKAI_EVENT_BUFFER_SIZE` env on the pod |
| Trainer crashes with `rc>128` | Native segfault (bitsandbytes / flash-attn) | `training.faulthandler.log` on the pod, parsed exit code in the `failed` event payload |
| Pod billed after run completes | `PodStopper` skipped (missing creds) or failed | `pod_stop_attempt` event payload — `decision` field tells you which branch fired |
| `LaunchAbortedError` before submit | Preflight or stale-plugin check refused the launch | `run_preflight` output in the launcher logs |
| `RunnerEventCallback` silent | 3 consecutive 5xx → callback self-disabled for the session | Trainer keeps running; check pod-side `internal/events` access log |

## Durability semantics (Phase 12)

The runner persists two streams to pod disk so a Mac sleep window
longer than the ring buffer's capacity (~5.5 min at typical event
rate) doesn't lose data.

### EventBus journal — `<workspace>/.runner/events/`

Append-only JSONL files, rotated at 100 MB, capped at 5 files
(500 MB total). Schema versioned per record (`v=1` today). On crash
recovery, the runner picks up the highest-numbered file and continues
appending. fsync batched: every 50 events OR 1 s, whichever fires
first. Per-write `flush()` (cheap) keeps records visible to concurrent
readers without paying fsync latency on the publisher path.

WebSocket subscribers transparently get disk replay when their
`since=N` cursor is older than the in-memory ring's oldest offset
but still present on disk. The 4410 close code now fires only when
the requested offset is older than the oldest persisted record (a
truly impossible cursor — `DiskJournalExhausted`). Legacy
`BufferTruncatedError` keeps the same close code for journal-less
buses (test fixtures, fall-back boot mode).

### MLflow metrics buffer — `/workspace/metrics_buffer.jsonl`

The trainer's `ResilientMLflowTransport` buffers metric calls when
the MLflow upstream is unreachable. On natural completion the
`CompletionCallback` drains the buffer (5 s budget). If MLflow is
still asleep at that point — which is the entire reason we buffered
in the first place — the buffer file remains.

After Mac wake + pod resume, `ModelRetriever` fetches the buffer file
alongside the model artefacts and replays each record into the
**same MLflow run_id** that was active during training. The buffer
file only contains records the trainer never managed to flush, so
replay is safe-by-construction (no dedup required).

By default, `training.metrics_buffer.keep_all=true` — every metric
is preserved losslessly. See **Metrics buffer** below for the opt-in
decimation policy.

### Storage layout summary

| Path | Owner | Lifecycle | Retrieved by |
|---|---|---|---|
| `/workspace/.ryotenkai/state.jsonl` | FSM | append-only, never GC'd | n/a (Mac doesn't read) |
| `/workspace/.runner/events/events.NNN.jsonl` | EventBus journal | rotation cap 500 MB | n/a (replayed via WS only) |
| `/workspace/metrics_buffer.jsonl` | Resilient MLflow transport | drained on flush | ModelRetriever (Phase 12.A.1) |
| `/workspace/.runner/buffer.flush_offset` | Resilient transport | last successful drain marker | (informational only) |
| `/workspace/output/` | Trainer | grows during training | ModelRetriever (existing) |
| `/workspace/completion.marker` | CompletionCallback | one-shot per attempt | TrainingMonitor (Phase 11.A) |
| `/workspace/cancelled.marker` | CancellationCallback | one-shot per attempt | TrainingMonitor (Phase 9.C) |

### Telemetry events (Phase 12.C)

| Kind | Source | Payload | Use |
|---|---|---|---|
| `events_disk_pressure` | EventBus / health-check | `total_bytes`, `file_count`, `threshold_bytes` | Alert on sustained 90%+ journal footprint. |
| `events_rotated` | EventJournal on_rotate | `from_seq`, `to_seq`, `file_size_bytes`, `oldest_remaining_seq` | Track rotation cadence. |
| `events_gc_ran` | EventJournal init | `deleted_seqs`, `deleted_bytes` | Audit crash-recovery cleanup. |
| `metrics_buffer_retrieved` | ModelRetriever (Mac) | `replayed`, `line_count`, `size_bytes`, `missing`, `oversized` | Audit replay outcomes per attempt. |

The full set is exposed in `src.runner.cancellation_telemetry` as
`DURABILITY_EVENT_KINDS` (disjoint from `CANCELLATION_EVENT_KINDS`)
and in `TERMINAL_EVENT_KINDS` (superset for "any flow signal" views).

## Metrics buffer (Phase 12.A.2 — config-driven decimation)

When the MLflow upstream is unreachable (typically because the Mac is
asleep), the trainer's :class:`ResilientMLflowTransport` buffers
``mlflow.log_metric`` calls to ``<workspace>/metrics_buffer.jsonl``.
Phase 12.A.1 retrieves and replays this file on Mac wake. Phase 12.A.2
exposes the **decimation policy** as user config so long runs can
trade per-step granularity for bounded disk + replay overhead — or,
by default, keep every metric losslessly.

### Default behaviour: lossless

Old configs without a ``training.metrics_buffer`` block — and any new
config that omits the block — inherit ``keep_all=true``:

```yaml
training:
  metrics_buffer:
    keep_all: true   # default; every metric preserved
```

This is **strictly more permissive** than the Phase 9 hard-coded
3-tier policy. Per the user mandate: data fidelity is the default;
decimation is opt-in.

### Opt-in decimation for very long runs

Flip ``keep_all=false`` to enable the three-window decimator. Each
window keeps every Nth step within a time band:

```yaml
training:
  metrics_buffer:
    keep_all: false
    decimation:
      window_first_minutes: 10        # 0–10 min
      window_first_keep_every: 1        # keep every step
      window_mid_minutes: 30          # 10–40 min
      window_mid_keep_every: 2          # keep every other step
      window_late_keep_every: 5         # 40+ min: keep every 5th step
```

The defaults shown above mirror the legacy Phase 9 hard-coded
behaviour, so flipping ``keep_all=false`` without tuning the windows
reproduces the pre-12.A.2 buffer shape. Tuning the windows is for
operators who:

* Run for many hours and don't need per-step granularity in the late
  bulk of training.
* Care more about replay throughput than fine-grained loss curves.

### Schema constraints

* All five window numbers must be ``>= 1``.
* Unknown keys under ``metrics_buffer`` or ``decimation`` are
  rejected at config-load time (``StrictBaseModel`` ``extra=forbid``).
* Schema source: :class:`MetricsBufferConfig` /
  :class:`DecimationWindowConfig` in
  ``src.config.training.metrics_buffer``.

## Configuration knobs

| Env var | Where | What |
|---|---|---|
| `RYOTENKAI_RUNTIME_IMAGE_OVERRIDE` | Mac launcher | Override the pinned training image (CI / dev). |
| `RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_VLLM` | Mac launcher | Override the pinned vLLM inference image (CI / dev). |
| `RYOTENKAI_EVENT_BUFFER_SIZE` | Pod | Ring-buffer capacity (default 10000). |
| `RUNPOD_AUTO_STOP` | Pod | Set `false` to keep the pod alive after terminal FSM (debug). |
| `RUNPOD_KEEP_ON_ERROR` | Pod | Set `true` to keep the pod alive on `failed` only. |
| `COMMUNITY_STRICT` | Pod | Forces fail-fast plugin loading. The runner image sets this to `1` after the first `PluginUnpacker` call. |
| `WORKSPACE_PATH` | Pod | Workspace root for the trainer; default `/workspace`. Drives the `MetricsBuffer` location (Phase 12.A) and the EventJournal path (Phase 12.B). |

## Cross-references

- [`docs/plans/harmonic-rolling-crayon.md`](plans/harmonic-rolling-crayon.md) — full plan and migration history (Phases 0 → 8).
- [`docs/web-ui.md`](web-ui.md) — control-plane API and Web UI architecture.
- [`community/README.md`](../community/README.md) — plugin authoring guide; see "Where each kind runs" for the cloud-training topology split.
