# Event mirror — Mac-side cold replay of pod runner journal

> Status: implemented in PR2 of the trainer-log-file plan.
> See [`docs/plans/task-notification-task-id-b6y40vnmp-tas-majestic-stream.md`](../plans/task-notification-task-id-b6y40vnmp-tas-majestic-stream.md).

## TL;DR

The pod runner persists every event to
`/workspace/.runner/events/events.<seq>.jsonl`. Mac mirrors this file
into `runs/<id>/attempts/<n>/events/events_mirror.jsonl` while the
training monitor is alive, so:

* Frontend can replay the run after it ended (pod is gone, mirror
  isn't).
* CLI tools can `curl /api/v1/runs/<id>/attempts/<n>/events?since=N`
  for ad-hoc analysis.
* WebSocket reconnects use the mirror to fill the gap between
  `last_seen+1` and the live runner cursor.

## Three-channel architecture

```
                  POD                                    MAC

  trainer.logger  ────► training.log (FileHandler) ──► log_manager scp ──►  runs/<id>/.../logs/training.log
  (Channel A)                                                               (post-mortem artefact)

  trainer stdout  ────► Supervisor PIPE ──► event_bus ──► events.000.jsonl  (pod-side journal)
  (Channel B)                                                  ↓ WS
                                                          JobClient.subscribe_events
                                                                 ↓
                                          ┌────────────────────────────────┐
                                          │  TrainingMonitor                │
                                          │  - terminal-state dispatch      │
                                          │  - logger.info("[TRAINER:…]")   │  ← visible in
                                          │  - mirror.write(event)          │    training_monitor.log
                                          └────────────────────────────────┘
                                                                 ↓
                                                  events_mirror.jsonl     ← cold-replay source
                                                  (Mac-side)

  Frontend live-tail (Channel C):
    WS /api/v1/runs/<id>/attempts/<n>/events/stream?since=N
       │
       ├── catchup ──► reads events_mirror.jsonl from since onward
       │
       └── live ─────► if run still active: opens its own SSH tunnel
                       + JobClient.subscribe_events from last_seen+1
                       and relays each frame to the browser.
```

Each channel survives independently: a crashed Supervisor leaves the
file artefact intact; a torn-down WebSocket leaves the mirror+pod
journal complete; a missing mirror still has the pod journal as
authoritative.

## Components

### Pipeline-side (writer)

* [`src/pipeline/stages/managers/event_mirror.py`](../../src/pipeline/stages/managers/event_mirror.py)
  — `EventMirrorWriter` opens
  `<attempt>/events/events_mirror.jsonl` in append mode (lazy on
  first write), serialises each event as one JSONL line, fsync's
  every N writes (default 50) for crash safety.
* [`src/pipeline/stages/training_monitor.py`](../../src/pipeline/stages/training_monitor.py)
  — `TrainingMonitor.execute` instantiates the writer for the
  duration of the stage. `_watch` calls `mirror.write(event)`
  before dispatching callbacks, so the on-disk record is complete
  even if a callback raises.

### API-side (readers)

* [`src/api/routers/run_events.py`](../../src/api/routers/run_events.py)
  — `GET /api/v1/runs/<id>/attempts/<n>/events?since=N&limit=K&kind=…`
  reads the mirror file, applies filters, returns events + cursor.
  Used by frontend for poll-style fallbacks and by CLI tools.
* [`src/api/ws/run_events.py`](../../src/api/ws/run_events.py)
  — `WS /api/v1/runs/<id>/attempts/<n>/events/stream?since=N` does
  catchup-from-mirror then (if the run is still active) opens its
  own SSH tunnel + JobClient and relays the live runner stream to
  the browser. Two `init` frames signal the phase change.

## Frame protocol (WS)

Server → client:

```jsonc
// Catchup phase begins
{"type": "init", "since": 0, "phase": "catchup"}

// Each event from mirror or live runner
{"type": "event", "event": {"v":1, "offset":42, "ts":"…", "kind":"trainer_log",
                            "payload": {"kind":"stdout", "line":"…"}}}

// Live phase begins (after catchup, run still active)
{"type": "init", "since": 43, "phase": "live"}

// Server-initiated end of stream
{"type": "eof", "reason": "trainer_exited"|"runner_eof"|"no_live_source"|"terminal_in_mirror"}
```

Close codes:

| Code | Meaning |
|---|---|
| 1000 | Clean close (run reached terminal state) |
| 1011 | Internal error (mirror read failed, runner client error) |
| 4404 | Run / attempt dir not found, or runner reports unknown job |
| 4410 | Runner reported `ReplayTruncatedError` (offset gone) |
| 4503 | SSH tunnel could not be opened |

Client never sends frames; server ignores any input.

## Reconnect semantics

Frontend stores `last_offset` from the last `event` frame it saw.
On `close`, after exponential-backoff sleep, it reconnects with
`?since=<last_offset+1>`. The Mac WS endpoint:

1. Reads the mirror file, sends every event with `offset >= since`.
2. If the run is still active, opens SSH tunnel + JobClient,
   subscribes from `since=last_seen+1`.

Because the mirror is append-only and offsets are monotonic from the
runner's journal, there are no gaps and no duplicates across
reconnects.

## File-size bounds

Pod-side journal has hard caps: 100 MiB per segment × 5 segments =
500 MiB. The Mac mirror grows linearly with the same data, so the
upper bound for `events_mirror.jsonl` is also ≈ 500 MiB. For
typical training runs (≤ 10k stdout lines per epoch × 10 epochs)
the mirror sits around 5–20 MiB.

If a run exhausts the cap, the mirror just keeps growing — pod
journal rotation drops oldest segments but the Mac mirror has no
rotation logic. This is **deliberately deferred** until production
measurements show it's a real problem; per the YAGNI principle,
adding rotation now would solve a hypothetical issue.

## Failure modes

| Symptom | Likely cause | Mitigation |
|---|---|---|
| Mirror file empty / missing after run | `attempt_directory` not in pipeline context | Older callers / direct unit tests skip mirror writer with debug log; fix is to populate `PipelineContextKeys.ATTEMPT_DIRECTORY` |
| WS endpoint closes 4503 immediately | SSH tunnel could not open (pod terminated, key path wrong) | Frontend treats this as terminal — don't auto-reconnect |
| WS endpoint closes 4410 | Runner's in-memory event ring rolled past the requested offset | Frontend should refetch `/jobs/<id>/status` and start fresh from current offset |
| REST endpoint returns 503 | Mirror file vanished mid-read (e.g. cleanup race) | Retry; rare |

## When this matters in production

* **Crashed run, debug it offline.** The pod is gone but
  `events_mirror.jsonl` lives forever in the run's attempt dir.
  `jq -c 'select(.kind == "trainer_log") | .payload.line' events_mirror.jsonl`
  gives you a clean tail of trainer stdout.
* **Frontend reconnect during long training.** WS drops happen
  (laptop sleep, network blip). The mirror lets us replay missed
  events deterministically rather than show "?? events missing".
* **Multi-tab observation.** Two browser tabs on `/runs/<id>/live`
  open two WS endpoints; each opens its own SSH tunnel for the
  live phase. Pod sees 2 simultaneous JobClient subscribers — fine,
  the runner supports multi-consumer with independent cursors.
