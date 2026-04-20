# RyotenkAI Web UI

React + Vite + TypeScript frontend for the RyotenkAI pipeline. Talks to the
FastAPI backend (`ryotenkai serve`) over `/api/v1` and `/api/v1/runs/.../logs/stream`
(WebSocket).

## Dev

```bash
# Backend (from repo root)
ryotenkai serve --runs-dir runs --cors-origins http://localhost:5173

# Frontend (from web/)
npm install
npm run dev           # Vite on :5173, proxies /api and /ws to :8000
```

## Prod

```bash
cd web && npm run build
cd .. && ryotenkai serve --runs-dir runs         # mounts web/dist at /
```

`ryotenkai serve` automatically mounts `web/dist/` when the directory exists.

## API types are generated, not hand-written

Pydantic on the backend is the single source of truth for request/response
shapes. The frontend consumes a generated TypeScript schema.

```bash
make gen-api              # regenerates web/src/api/{openapi.json,schema.d.ts}
make verify-api-sync      # gen-api + git diff --exit-code — CI runs this
```

- Any change to `src/api/schemas/` or a `response_model=` on a router
  needs a `make gen-api` + commit in the same changeset.
- `web/src/api/types.ts` is a thin layer of `type X = components['schemas']['X']`
  re-exports plus a few UI-only unions (Status, LaunchMode, PluginKind) —
  do not hand-author payload shapes there.
- The `ConfigBuilder` form still reads `GET /config/schema` at runtime,
  so schema-driven form rendering stays automatic.
