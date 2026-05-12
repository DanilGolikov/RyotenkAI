"""``fake-hf-hub`` HTTP sidecar — minimal canned-response surface (Phase 2 stub).

Implements two HF endpoints used by smoke tests:

* ``GET /api/models/{name}`` → stub model card
* ``GET /resolve/{name}/{file:path}`` → 256 bytes of zero — enough for a
  download-path probe without dragging in real model weights

Phase 4 will expand this to cover upload, revision listing, and rate-limit
chaos surfaces.

CLI:

    python -m tests._harness.stack.sidecars.hf_hub_server --port=N
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response

from tests._harness.stack.sidecars._base import (
    SidecarRuntime,
    add_control_routes,
    parse_args,
    parse_clock,
    run_uvicorn,
)

_STUB_BYTES = b"\x00" * 256


class _HFHubFake:
    def __init__(self) -> None:
        self._downloads: list[dict[str, Any]] = []

    def record_download(self, model: str, file: str) -> None:
        self._downloads.append({"model": model, "file": file})

    def reset(self) -> None:
        self._downloads.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"downloads": list(self._downloads)}


def make_app(*, fake: _HFHubFake, runtime: SidecarRuntime) -> FastAPI:
    app = FastAPI(title="fake-hf-hub-sidecar")

    add_control_routes(app, runtime=runtime, reset_fn=fake.reset)

    @app.get("/api/models/{model_name:path}")
    async def model_info(model_name: str) -> dict[str, Any]:
        return {
            "id": model_name,
            "modelId": model_name,
            "tags": ["fake"],
            "pipeline_tag": "text-generation",
            "siblings": [{"rfilename": "config.json"}],
        }

    @app.get("/resolve/{model_name}/{file_path:path}")
    async def resolve(model_name: str, file_path: str) -> Response:
        fake.record_download(model_name, file_path)
        return Response(content=_STUB_BYTES, media_type="application/octet-stream")

    return app


def main() -> None:
    args = parse_args("hf_hub")
    clock = parse_clock(args.clock)
    fake = _HFHubFake()
    runtime = SidecarRuntime(name="hf_hub", clock=clock, fake=fake, extra={"latency_ms": 0})
    app = make_app(fake=fake, runtime=runtime)
    if args.ready_file:
        from pathlib import Path
        Path(args.ready_file).write_text("port=" + str(args.port) + "\n")
    run_uvicorn(app, port=args.port)


if __name__ == "__main__":
    main()
