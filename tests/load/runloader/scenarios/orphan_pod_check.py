"""Orphan-pod-check scenario.

Submit pods, cancel mid-flight, assert pods are TERMINATED and not
orphaned.
"""

from __future__ import annotations

import time

import httpx

from tests._harness.stack import Stack
from tests.load.runloader.framework import SLOSpec


class OrphanPodCheck:
    name = "orphan_pod_check"
    tags = ["orphan"]
    concurrency = 5
    target_rps = 10.0
    duration_s = 1.0
    slo = [
        SLOSpec(
            name="orphan_pod_check",
            p99_latency_ms=1500.0,
            no_orphan_pods=True,
        ),
    ]

    async def precondition(self, stack: Stack) -> None:
        return None

    async def run_step(self, stack: Stack, step_index: int) -> float:
        pod_id = f"orphan-{step_index:04d}"
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                stack.runpod_url + "/control/register_pod",
                params={"pod_id": pod_id, "desired_status": "RUNNING"},
            )
            # Cancel mid-flight.
            await client.post(stack.runpod_url + f"/api/pods/{pod_id}/terminate")
        return (time.monotonic() - start) * 1000.0

    async def teardown(self, stack: Stack) -> None:
        return None
