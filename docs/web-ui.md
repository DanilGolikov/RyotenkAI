# Web UI

Kubernetes-way architecture: the FastAPI backend (`ryotenkai serve`) and the
React frontend (`web/`) are **sibling clients** to the same file-based pipeline
state store (`runs/<id>/pipeline_state.json` + `run.lock`) that the CLI already
uses. Nothing wraps the CLI through subprocess for read paths — the backend
imports `src.pipeline` directly.

See [docs/plans/jolly-baking-bird.md](plans/jolly-baking-bird.md) for the full
architecture rationale and reuse map.

## Quick start

```bash
# 1. Backend
ryotenkai serve --runs-dir runs --port 8000
# OpenAPI: http://127.0.0.1:8000/docs

# 2. Frontend (dev)
cd web && npm install && npm run dev
# UI on http://localhost:5173, proxies /api + WebSockets to :8000

# 3. Frontend (prod served by FastAPI)
cd web && npm run build
cd .. && ryotenkai serve        # mounts web/dist/ at /
```

## HTTP API

Base: `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + runs_dir readability |
| `GET` | `/runs` | list runs grouped by subfolder |
| `POST` | `/runs` | create empty run directory |
| `GET` | `/runs/{id}` | run detail with lock status + next attempt |
| `DELETE` | `/runs/{id}` | delete local + MLflow children |
| `GET` | `/runs/{id}/attempts/{n}` | attempt detail |
| `GET` | `/runs/{id}/attempts/{n}/stages` | ordered stage list |
| `GET` | `/runs/{id}/attempts/{n}/logs` | chunk tail (offset-based) |
| `WS` | `/runs/{id}/attempts/{n}/logs/stream` | live tail + state events |
| `GET` | `/runs/{id}/restart-points` | list restart options |
| `GET` | `/runs/{id}/default-launch-mode` | suggest `resume` or `restart` |
| `POST` | `/runs/{id}/launch` | detached subprocess launch |
| `POST` | `/runs/{id}/interrupt` | SIGINT by pid from `run.lock` |
| `POST` | `/config/validate` | static YAML checks |
| `GET` | `/config/default` | runs_dir + templates from `examples/` |
| `GET` | `/runs/{id}/report` | markdown report (generate if missing) |

## Launch / Interrupt semantics

- Launch spawns `python -m src.main train ...` via
  `src.pipeline.launch.spawn_launch_detached`. `start_new_session=True` — the
  pipeline keeps running after the API process dies.
- Source of truth for "is it running?" is the combination of `run.lock` (pid
  written by the orchestrator) + `os.kill(pid, 0)` liveness. Backend holds no
  in-memory registry between requests.
- Interrupt reads pid from `run.lock` and forwards SIGINT via
  `interrupt_launch_process`.
- Stale lock (pid not alive): `/interrupt` reports `process_not_found` and
  routes cleanup through `PipelineStateStore.remove_stale_lock` — this
  re-reads the pid inside the store module right before `unlink`, so if
  another process legitimately grabbed the lock in the meantime the file is
  left untouched.
- Launch rejects with `409 run_already_running` when an active lock exists.

## WebSocket log protocol

```json
{ "type": "init",   "file": "pipeline.log", "offset": 0 }
{ "type": "chunk",  "lines": ["..."], "offset": 1234 }
{ "type": "state",  "status": "running" }
{ "type": "eof" }
```

Server polls the log every `log_stream_poll_interval_ms` (default 500 ms) and
watches `pipeline_state.json` mtime to surface status transitions. Terminal
statuses (`completed`, `failed`, `interrupted`, `stale`, `skipped`) drain the
file and close with `eof`.

## Env

Prefix `RYOTENKAI_API_`:

| Var | Default |
|---|---|
| `HOST` | `127.0.0.1` |
| `PORT` | `8000` |
| `RUNS_DIR` | `runs` |
| `CORS_ORIGINS` | `http://localhost:5173` |
| `LOG_STREAM_POLL_INTERVAL_MS` | `500` |
| `MAX_LOG_CHUNK_BYTES` | `1048576` |
| `SERVE_SPA` | `true` |
| `WEB_DIST_DIR` | `web/dist` |

## Coexistence with CLI

Both the web UI and the CLI read and write the same state store. You can start
a run in the browser and inspect it with `ryotenkai inspect-run <run_dir>` or
vice versa. No feature flags — orthogonal clients.
