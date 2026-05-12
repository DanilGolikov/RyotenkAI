"""L6 smoke: Stack boots, all sidecars healthy, control axis works.

These tests exercise the orchestrator end-to-end without booting the
real control plane / runner — they're the cheapest possible "everything
is wired up" gate.
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack

pytestmark = [pytest.mark.asyncio, pytest.mark.stack]


_SIDECAR_NAMES = ("runpod", "mlflow", "vllm", "hf_hub")


async def test_stack_boots_and_all_sidecars_healthy(stack: Stack) -> None:
    # WHY: this is the L6 smoke gate — if any sidecar's /health is not 200
    # we'd rather fail here with a clear signal than have downstream tests
    # produce confusing protocol errors.
    assert stack.sidecars.keys() == set(_SIDECAR_NAMES)
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name in _SIDECAR_NAMES:
            handle = stack.sidecars[name]
            response = await client.get(handle.base_url + "/health")
            assert response.status_code == 200, f"{name} unhealthy"
            body = response.json()
            assert body["status"] == "ok"
            assert body["sidecar"] == name
            assert body["clock"] == "manual"


async def test_stack_advance_clock_broadcasts_to_all_sidecars(stack: Stack) -> None:
    # Sanity check: every sidecar started with t=0 (ManualClock default).
    initial = await stack.state_dump()
    assert set(initial.keys()) == set(_SIDECAR_NAMES)

    # advance_clock returns once /control/advance_clock has been ack'd by
    # all sidecars; the now/state probe afterwards must observe the bump.
    await stack.advance_clock(60.0)

    # Probe each sidecar's clock via the /control/advance_clock(0.0) path:
    # bumping by zero returns the current clock value. This avoids relying
    # on per-sidecar state shapes (some fakes don't surface "now").
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name in _SIDECAR_NAMES:
            handle = stack.sidecars[name]
            response = await client.post(
                handle.base_url + "/control/advance_clock",
                params={"seconds": 0.0},
            )
            assert response.status_code == 200
            assert response.json()["now"] == pytest.approx(60.0, abs=1e-6)


async def test_stack_reset_clears_chaos_on_all_sidecars(stack: Stack) -> None:
    # Arm two distinct chaos surfaces, reset, observe both clear.
    async with httpx.AsyncClient(timeout=5.0) as client:
        runpod = stack.sidecars["runpod"].base_url
        mlflow = stack.sidecars["mlflow"].base_url

        # runpod: 3 queued 429s.
        r1 = await client.post(runpod + "/control/inject_429", params={"count": 3})
        assert r1.status_code == 200
        # mlflow: queue an unavailable flag.
        r2 = await client.post(mlflow + "/control/set_unavailable", params={"value": "true"})
        assert r2.status_code == 200

        # Verify chaos is armed before reset.
        pre = await stack.state_dump()
        assert pre["runpod"]["chaos"]["rate_limit_remaining"] == 3

        await stack.reset()

        post = await stack.state_dump()
        assert post["runpod"]["chaos"]["rate_limit_remaining"] == 0
        # mlflow's snapshot shape doesn't surface _unavailable in the dump,
        # but reset() clears it — verify by hitting setup() which would
        # raise 503 if still set.
        resp = await client.post(mlflow + "/api/setup")
        assert resp.status_code == 200


async def test_stack_state_dump_aggregates_all_sidecars(stack: Stack) -> None:
    dump = await stack.state_dump()
    assert set(dump.keys()) == set(_SIDECAR_NAMES)
    # Every sidecar tags its own name into the snapshot — the orchestrator's
    # aggregator relies on this for debug bundles, so verify it.
    for name in _SIDECAR_NAMES:
        assert dump[name]["sidecar"] == name
        assert "latency_ms" in dump[name]


async def test_stack_shutdown_is_idempotent() -> None:
    # WHY explicit boot here: this test deliberately calls shutdown twice
    # — the fixture would shut down a third time which still must be safe.
    s = await Stack.boot(clock="manual")
    await s.shutdown()
    await s.shutdown()  # second call must be a no-op
