"""``fake-vllm`` HTTP sidecar — minimal canned-response surface (Phase 2 stub).

Implements just enough of the OpenAI ``/v1/completions`` shape to let the
control plane issue an inference smoke; full chat-completion / streaming /
chunk-shape work is deferred to Phase 4.

CLI:

    python -m tests._harness.stack.sidecars.vllm_server --port=N
"""

from __future__ import annotations

import itertools
from typing import Any

from fastapi import Body, FastAPI

from tests._harness.stack.sidecars._base import (
    SidecarRuntime,
    add_control_routes,
    parse_args,
    parse_clock,
    run_uvicorn,
)


class _VLLMFake:
    """Minimal "fake" with the same ``snapshot()`` shape as other fakes."""

    def __init__(self) -> None:
        self._counter = itertools.count(start=1)
        self._requests: list[dict[str, Any]] = []

    def record(self, request: dict[str, Any]) -> int:
        idx = next(self._counter)
        self._requests.append({"id": idx, **request})
        return idx

    def reset(self) -> None:
        self._counter = itertools.count(start=1)
        self._requests.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"requests": list(self._requests)}


def make_app(*, fake: _VLLMFake, runtime: SidecarRuntime) -> FastAPI:
    app = FastAPI(title="fake-vllm-sidecar")

    add_control_routes(app, runtime=runtime, reset_fn=fake.reset)

    @app.post("/v1/completions")
    async def completions(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        prompt = str(payload.get("prompt", ""))[:80]
        idx = fake.record(payload)
        return {
            "id": f"fake-cmpl-{idx}",
            "object": "text_completion",
            "model": payload.get("model", "fake-model"),
            "choices": [
                {
                    "index": 0,
                    "text": f"echo:{prompt}",
                    "logprobs": None,
                    "finish_reason": "stop",
                },
            ],
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "fake-model", "object": "model", "owned_by": "fake-vllm"}],
        }

    return app


def main() -> None:
    args = parse_args("vllm")
    clock = parse_clock(args.clock)
    fake = _VLLMFake()
    runtime = SidecarRuntime(name="vllm", clock=clock, fake=fake, extra={"latency_ms": 0})
    app = make_app(fake=fake, runtime=runtime)
    if args.ready_file:
        from pathlib import Path
        Path(args.ready_file).write_text("port=" + str(args.port) + "\n")
    run_uvicorn(app, port=args.port)


if __name__ == "__main__":
    main()
