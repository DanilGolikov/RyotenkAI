"""Regenerate ``web/src/api/openapi.json`` from the runner FastAPI app.

Phase 3 PR-3.4 of transport-unification-v2. Run from the repo
root:

    uv run python scripts/sync_openapi.py

The script imports ``ryotenkai_pod.runner.main:create_app`` (with
the env vars the lifespan needs to initialise mocked) and writes
the OpenAPI JSON to ``web/src/api/openapi.json``. The frontend's
TypeScript codegen consumes this file via its own Vite tooling.

CI integration is deliberately limited to a `git diff --exit-code`
check rather than auto-regen — any drift surfaces as a PR review
finding rather than a silently-mutating commit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "web" / "src" / "api" / "openapi.json"

    # Lifespan reads RYOTENKAI_RUNTIME_PROVIDER; provide a value that
    # resolves to the no-op pod lifecycle client so the app builds
    # without RunPod credentials.
    os.environ.setdefault("RYOTENKAI_RUNTIME_PROVIDER", "single_node")

    # Lazy import — keeps the script's startup cost low when invoked
    # for help/usage.
    from ryotenkai_pod.runner.main import create_app

    class _StubSupervisor:
        is_running = False

        async def shutdown(self) -> None:
            pass

    def _factory(fsm, bus, *, terminal_hook=None, stdio_log_path=None):  # type: ignore[no-untyped-def]
        return _StubSupervisor()

    app = create_app(supervisor_factory=_factory)

    schema = app.openapi()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(json.dumps(schema))} bytes to {out_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
