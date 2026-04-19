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
