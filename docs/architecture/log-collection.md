# Pod log collection — what writes where, and how it gets to Mac

## TL;DR

Two pod-side log files, one channel, no fallback layers:

```
Pod                                 SSH+rsync         Mac
─────────────────────────           ──────►          ───────────────────────────────────
/workspace/runner.log    (uvicorn / runner stdout)    runs/<id>/.../logs/runner.log
/workspace/training.log  (trainer subprocess)         runs/<id>/.../logs/training.log
```

Both files are pulled by the existing ``LogManager.download()`` chain
on the Mac. ``runner.log`` is the file added in this iteration to
capture the **uvicorn boot window**, where pre-import errors used to
disappear into a black hole.

## Why two files?

Different parts of the Python stack run in different lifecycles:

| Stage | What runs | Stdout goes where | Why we need a separate file |
|---|---|---|---|
| Container boot | `entrypoint.sh` (bash) | journald via PID-1 | Bash itself is trivial; nothing to log durably |
| Runner import | `python -m uvicorn` (the process that becomes PID 1's child) | `/workspace/runner.log` (NEW) | This is where ImportError / SyntaxError fire. They happen BEFORE Python's logging config is set up, so app-level logging cannot capture them. |
| Runner runtime | uvicorn + FastAPI + Supervisor | `/workspace/runner.log` (continued) | Lifecycle events: `/jobs/start`, `/healthz`, supervisor decisions |
| Trainer | `python -m src.training.run_training` (subprocess) | `/workspace/training.log` via Supervisor + EventJournal | Real training output: epochs, losses, MLflow events |

If the trainer never starts (uvicorn died at boot), only `runner.log`
has anything. If uvicorn boots fine and the trainer crashes at epoch 3,
`training.log` has the full trace and `runner.log` shows lifecycle
events around the crash.

## How `runner.log` actually gets written

`docker/training/entrypoint.sh` (the last lines of the script):

```bash
RUNNER_LOG="${RYOTENKAI_RUNNER_LOG:-/workspace/runner.log}"
mkdir -p "$(dirname "$RUNNER_LOG")" || true

if ! : >> "$RUNNER_LOG" 2>/dev/null; then
  echo "WARNING: $RUNNER_LOG not writable, falling back to /tmp/runner.log" >&2
  RUNNER_LOG="/tmp/runner.log"
fi

exec stdbuf -oL -eL "$PY_BIN" -m uvicorn src.runner.main:app \
  --host "$RYOTENKAI_RUNNER_HOST" \
  --port "$RYOTENKAI_RUNNER_PORT" \
  --no-access-log \
  >> "$RUNNER_LOG" 2>&1
```

Three load-bearing details:

1. **`exec` keeps dumb-init's child as uvicorn directly.** No bash
   subshell sits between PID 1 (dumb-init) and uvicorn. SIGTERM from
   `docker stop` propagates: `dumb-init → uvicorn → trainer`,
   triggering graceful shutdown of the asyncio loop and a clean
   subprocess SIGTERM to the trainer.

2. **Direct `>> file` redirect, not `tee`, not uvicorn `--log-config`.**
   See "What we considered and rejected" below.

3. **`stdbuf -oL -eL`** forces line-buffered stdout/stderr so that
   when uvicorn dies in 50 ms with `ModuleNotFoundError`, the
   traceback has actually been flushed to the file and is visible to
   the Mac via rsync. Without it, Python's default 4 KB block buffer
   means the file is empty when the process dies fast.

## How it lands on Mac

`src/pipeline/stages/managers/log_manager.py` is the single SSH-rsync
downloader. It now accepts a keyword-only `local_path` argument so
the same class can pull both files:

```python
# training.log (default path — kept identical for back-compat):
LogManager(ssh_client).download()

# runner.log (NEW):
layout = get_run_log_layout()
LogManager(
    ssh_client,
    remote_path="/workspace/runner.log",
    local_path=layout.remote_runner_log,
).download()
```

`gpu_deployer._download_remote_logs` calls both, in that order:
runner.log first (so it's available even if training.log download
fails), training.log second. Failures in either are best-effort
debug-level — they never raise out of the cleanup path because
terminating the pod afterwards is more important than perfect log
delivery.

## What we considered and rejected

### Why not `> >(tee -a "$RUNNER_LOG") 2>&1` (process substitution)

Process substitution puts a bash subshell between dumb-init and
uvicorn:

```
PID 1: dumb-init
   └── PID 7: bash subshell
          ├── PID 8: uvicorn
          └── PID 9: tee
```

`docker stop` → SIGTERM → dumb-init → bash subshell. Bash does NOT
forward signals to its children by default. After 10 s Docker
escalates to SIGKILL and uvicorn is killed hard, in-flight checkpoints
are lost. **Direct redirect avoids this entirely** — uvicorn is the
direct child of dumb-init.

### Why not uvicorn `--log-config` with a dual-handler config

`--log-config` is applied AFTER Python's logging system initializes,
which itself happens AFTER `import src.runner.main`. The exact
failure mode we just hit (`ModuleNotFoundError: src.utils` raised
during the `import` chain) fires BEFORE the log config is loaded —
the traceback would never reach the file. Direct redirect at the
shell level captures from the first byte.

### Why not provider-API stdout fetch (`docker logs` via RunPod GraphQL / SSH `docker logs`)

A second channel via the provider's native logs endpoint was
considered. It would protect against the case "SSH dead but provider
API still reachable". After analysis:

- RunPod's pod-logs endpoint historically caps at ~16 KB tail
  (versus a 50 MB rotated `runner.log`) — it gives less, not more.
- On `single_node`, the container is `docker rm -f`'ed by
  `cleanup_after_run()` BEFORE we'd typically fetch — would need a
  reordering of the cleanup chain.
- Maintenance cost: provider-Protocol extension, capability flag,
  per-provider implementation, factory-boot invariant assert,
  combinatorial test matrix — for a ~5 % edge case where direct file
  rsync also fails.

Decision: **drop layer 2.** When SSH is dead, we lose log delivery
regardless. This is acceptable because SSH-dead-AND-pod-still-running
is an exceptional sequence that does not match typical RunPod failure
modes.

### Why not a live SSH log streamer (`tail -F`)

`LogManager.download()` already does a delta-fetch via `tail -c`
during the monitoring loop. Calling it on a 15-second cadence
delivers ~95 % of "real-time tail" without a new module, new SSH
session pressure on the rsync ControlMaster, or new cancellation
edge cases. **YAGNI.**

## Test layers

Three structural test files lock this design in:

* `src/tests/unit/docker/test_entrypoint_runner_log.py` — invariants
  on `entrypoint.sh`. Catches regressions like "someone added `tee`
  back" or "someone removed `stdbuf`".
* `src/tests/unit/pipeline/stages/managers/test_log_manager_dual_logs.py`
  — `LogManager` `local_path` keyword-only behavior, two-instance
  isolation, default-path back-compat.
* `src/tests/unit/pipeline/test_gpu_deployer_runner_log.py` — chain
  invariants in the deployer: runner first, training second,
  exception in one does not skip the other, no SSH = silent skip.

Container-level smoke (real `docker run`) is gated to CI via the
existing build pipeline:

```bash
docker run --rm -d --name rk_smoke ryotenkai/ryotenkai-training-runtime:v1.0.4
sleep 8
docker exec rk_smoke test -s /workspace/runner.log
docker exec rk_smoke head /workspace/runner.log    # contains "Uvicorn running on"
docker stop rk_smoke   # must exit < 5 s (graceful shutdown intact)
```

## Trade-off accepted

After this change, `docker logs <container>` is **empty** for our
training pods. The Mac control plane does not call `docker logs`
through the provider control plane, so the loss is invisible to
production callers. Operators connecting via SSH can `tail -f
/workspace/runner.log` instead.

If a future operator workflow needs `docker logs` (e.g. ad-hoc
debugging without SSH access), revisit this trade-off — at that
point, adding a small `tee` with proper signal trap might be worth
the complexity. For today, KISS wins.
