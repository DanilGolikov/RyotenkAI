from __future__ import annotations

import os
from pathlib import Path


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    runs_dir: Path = Path("runs"),
    cors_origins: list[str] | None = None,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """Run the RyotenkAI web backend via uvicorn."""
    import uvicorn

    # Pass settings via env so uvicorn reload workers pick them up.
    os.environ["RYOTENKAI_API_HOST"] = str(host)
    os.environ["RYOTENKAI_API_PORT"] = str(port)
    os.environ["RYOTENKAI_API_RUNS_DIR"] = str(runs_dir.expanduser().resolve())
    if cors_origins:
        # pydantic-settings parses comma-separated or JSON list from env
        os.environ["RYOTENKAI_API_CORS_ORIGINS"] = ",".join(cors_origins)

    uvicorn.run(
        "src.api.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


__all__ = ["run_server"]
