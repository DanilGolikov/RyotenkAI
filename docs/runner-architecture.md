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

## Failure modes & where to look

| Symptom | Likely cause | First place to check |
|---|---|---|
| `ryotenkai job status` → 502 `runner_unreachable` | SSH tunnel can't reach `127.0.0.1:8080` (sshd or runner not up) | Pod's `dumb-init` logs, `runner_unreachable` detail message |
| `ReplayTruncatedError` mid-resume | Mac was asleep longer than 10k events worth of training | Bump `RYOTENKAI_EVENT_BUFFER_SIZE` env on the pod |
| Trainer crashes with `rc>128` | Native segfault (bitsandbytes / flash-attn) | `training.faulthandler.log` on the pod, parsed exit code in the `failed` event payload |
| Pod billed after run completes | `PodStopper` skipped (missing creds) or failed | `pod_stop_attempt` event payload — `decision` field tells you which branch fired |
| `LaunchAbortedError` before submit | Preflight or stale-plugin check refused the launch | `run_preflight` output in the launcher logs |
| `RunnerEventCallback` silent | 3 consecutive 5xx → callback self-disabled for the session | Trainer keeps running; check pod-side `internal/events` access log |

## Configuration knobs

| Env var | Where | What |
|---|---|---|
| `RYOTENKAI_RUNTIME_IMAGE_OVERRIDE` | Mac launcher | Override the pinned training image (CI / dev). |
| `RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_VLLM` | Mac launcher | Override the pinned vLLM inference image (CI / dev). |
| `RYOTENKAI_EVENT_BUFFER_SIZE` | Pod | Ring-buffer capacity (default 10000). |
| `RUNPOD_AUTO_STOP` | Pod | Set `false` to keep the pod alive after terminal FSM (debug). |
| `RUNPOD_KEEP_ON_ERROR` | Pod | Set `true` to keep the pod alive on `failed` only. |
| `COMMUNITY_STRICT` | Pod | Forces fail-fast plugin loading. The runner image sets this to `1` after the first `PluginUnpacker` call. |

## Cross-references

- [`docs/plans/harmonic-rolling-crayon.md`](plans/harmonic-rolling-crayon.md) — full plan and migration history (Phases 0 → 8).
- [`docs/web-ui.md`](web-ui.md) — control-plane API and Web UI architecture.
- [`community/README.md`](../community/README.md) — plugin authoring guide; see "Where each kind runs" for the cloud-training topology split.
