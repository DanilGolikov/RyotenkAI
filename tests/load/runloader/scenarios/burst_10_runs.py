"""Burst scenario — submit 10 attempts back-to-back.

Each "attempt" registers a pod, queries it, terminates it. SLOs:

* p99 launch latency ≤ 1500 ms (sidecar-local, so a generous budget
  catches gross regressions without flaking on cold loops).
* No orphan pods after teardown.
"""

from __future__ import annotations

import time

import httpx

from tests._harness.stack import Stack
from tests.load.runloader.framework import SLOSpec


class Burst10Runs:
    name = "burst_10_runs"
    tags = ["burst"]
    concurrency = 10
    target_rps = 20.0
    duration_s = 2.0  # short — burst is the whole point
    slo = [
        SLOSpec(
            name="burst_10_runs",
            p99_latency_ms=1500.0,
            no_orphan_pods=True,
            max_total_seconds=20.0,
        ),
    ]

    async def precondition(self, stack: Stack) -> None:
        # Pre-register a "template" pod that workers will clone from.
        # This avoids the burst overwhelming the sidecar with creates.
        await stack._post(  # type: ignore[arg-type]
            stack.runpod_url + "/control/register_pod",
            params={"pod_id": "template", "desired_status": "RUNNING"},
        )

    async def run_step(self, stack: Stack, step_index: int) -> float:
        pod_id = f"burst-{step_index:04d}"
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                stack.runpod_url + "/control/register_pod",
                params={"pod_id": pod_id, "desired_status": "RUNNING"},
            )
            await client.get(stack.runpod_url + f"/api/pods/{pod_id}")
            await client.post(stack.runpod_url + f"/api/pods/{pod_id}/terminate")
        return (time.monotonic() - start) * 1000.0

    async def teardown(self, stack: Stack) -> None:
        # Clean up the template pod registered in precondition.
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(stack.runpod_url + "/api/pods/template/terminate")
