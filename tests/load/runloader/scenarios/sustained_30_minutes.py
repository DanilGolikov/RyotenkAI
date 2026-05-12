"""Sustained scenario — N rps for "30 minutes".

Real-time (SCALE=60) actually runs for 30 minutes. CI (SCALE=1) runs
for 30 seconds compressed-time to keep the suite fast.
"""

from __future__ import annotations

import time

import httpx

from tests._harness.stack import Stack
from tests.load.runloader.framework import SLOSpec


class Sustained30Minutes:
    name = "sustained_30_minutes"
    tags = ["sustained"]
    concurrency = 4
    target_rps = 5.0
    duration_s = 0.5  # 0.5s compressed -> 30s with SCALE=60 -> 30m with SCALE=3600
    slo = [
        SLOSpec(
            name="sustained_30_minutes",
            p99_latency_ms=500.0,
            no_orphan_pods=True,
        ),
    ]

    async def precondition(self, stack: Stack) -> None:
        await stack._post(  # type: ignore[arg-type]
            stack.mlflow_url + "/api/setup",
        )

    async def run_step(self, stack: Stack, step_index: int) -> float:
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(stack.runpod_url + "/health")
            resp.raise_for_status()
        return (time.monotonic() - start) * 1000.0

    async def teardown(self, stack: Stack) -> None:
        await stack.reset()
