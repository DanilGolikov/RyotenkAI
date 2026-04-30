# Thin training image: image = environment, code = run-scoped

> Status: implemented in image **v2.0.0**. Plan:
> [`docs/plans/task-notification-task-id-b6y40vnmp-tas-majestic-stream.md`](../plans/task-notification-task-id-b6y40vnmp-tas-majestic-stream.md).

## Decision

The training docker image (`ryotenkai/ryotenkai-training-runtime`)
ships **only the runtime environment**:

* CUDA + Python + PyTorch (base image)
* `apt` deps: sshd, rsync, git, curl, dumb-init
* `pip install -r requirements.runtime.txt`
* `/opt/helix/runtime_check.py` (dependency verifier)
* `/entrypoint.sh` (sshd + `PUBLIC_KEY` injection + `sleep infinity`)
* OCI labels + `/opt/ryotenkai/version.txt`

It does **not** ship any of `src/...`. The Mac control plane rsyncs
the required Python modules — including `src/runner` — into
`/workspace/runs/<run_id>` per run via
[`CodeSyncer`](../../src/pipeline/stages/managers/deployment/code_syncer.py),
then SSH-execs `python -m uvicorn src.runner.main:app` with
`PYTHONPATH=/workspace/runs/<run_id>:${PYTHONPATH}`.

## Rationale

The previous (v1.x) image baked the whole `src/` tree at
`/opt/ryotenkai/src` so the pod could self-bootstrap uvicorn from
`entrypoint.sh`. Two architectural shifts removed that need:

1. **Mac orchestrates uvicorn.** `entrypoint.sh` is now inert
   (`sleep infinity`); the Mac SSH-execs uvicorn after rsync via
   [`runner_launcher.launch_runner`](../../src/pipeline/stages/managers/deployment/runner_launcher.py).
   Code arrives BEFORE we need to execute it.
2. **Closure-walked rsync set already covers runner.** `CodeSyncer`
   already ships `src/{utils,community,workspace,inference,...}`
   (the trainer's deps); adding `src/runner` to the same list costs
   one entry and removes a coupling to the image.

With both in place, baking `src/` into the image bought nothing and
charged us a 10-minute Docker rebuild for every runner-code change.
v2.0.0 cuts that rebuild out of the iteration loop entirely.

## When the image rebuilds

After v2.0.0, an image rebuild is justified only by changes to:

* [`docker/training/Dockerfile.runtime`](../../docker/training/Dockerfile.runtime) — base image, sshd, system deps
* [`docker/training/requirements.runtime.txt`](../../docker/training/requirements.runtime.txt) — pip libs
* [`docker/training/entrypoint.sh`](../../docker/training/entrypoint.sh)
* [`docker/training/runtime_check.py`](../../docker/training/runtime_check.py)

Changes anywhere under `src/**` — runner code, wire schemas, FSM,
event-bus, etc. — propagate via rsync on the next run. No rebuild.

## Failure modes

| Symptom on pod | Cause | Where to look |
|---|---|---|
| `ModuleNotFoundError: No module named 'src.runner'` | `CodeSyncer` didn't run, or `src/runner` is missing on the Mac | Mac-side `pipeline.log` for `CodeSyncer` step; verify `src/runner/main.py` exists locally |
| `ModuleNotFoundError: No module named 'src.<other>'` | A new transitive import isn't in `CodeSyncer.REQUIRED_MODULES` | Add it to the list with a comment explaining the closure path |
| uvicorn binds but health probe times out | `RYOTENKAI_RUNTIME_PROVIDER` (or other lifespan-required env) missing | Check `provider.required_runtime_env_vars(...)` is forwarded by `training_launcher` |

## Rollback

Image v1.x tags remain on Docker Hub. Set the env var
`RYOTENKAI_RUNTIME_IMAGE_OVERRIDE=ryotenkai/ryotenkai-training-runtime:v1.0.7`
to pin a single Mac to a baked-in baseline image without touching
`src/runner/__about__.py`.

## Standalone debug

`docker run image bash` no longer has `src.runner` available. Mount
your local checkout:

```bash
docker run --rm -it \
  -v "$(pwd):/workspace/runs/dbg" \
  -e PYTHONPATH=/workspace/runs/dbg \
  ryotenkai/ryotenkai-training-runtime:v2.0.0 bash
```

## Regression guards

[`src/tests/unit/docker/test_dockerfile_thin.py`](../../src/tests/unit/docker/test_dockerfile_thin.py)
holds structural assertions against re-baking `src/` into the
Dockerfile (`COPY src` and `ENV PYTHONPATH=...opt/ryotenkai...`
are forbidden). If those tests fail, the migration has been
silently undone.
