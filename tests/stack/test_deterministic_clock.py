"""L6 smoke: manual clock advances atomically; real clock rejects advance.

Stress-tests the contract that `clock="manual"` means *every* sidecar
moves together when ``Stack.advance_clock`` is called, and `clock="real"`
means the broadcast is refused with a 409 (RealClock fails fast).
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack

pytestmark = [pytest.mark.asyncio, pytest.mark.stack]


_SIDECAR_NAMES = ("runpod", "mlflow", "vllm", "hf_hub")


async def _now_for(client: httpx.AsyncClient, base_url: str) -> float:
    """Read the sidecar's current clock by advancing by zero."""
    # WHY: /control/state shapes differ per sidecar — they don't all
    # surface 'now'. /control/advance_clock with seconds=0 returns
    # {"now": ...} uniformly (see add_control_routes in _base.py).
    response = await client.post(
        base_url + "/control/advance_clock",
        params={"seconds": 0.0},
    )
    assert response.status_code == 200
    return float(response.json()["now"])


async def test_manual_clock_advances_atomically_across_sidecars(stack: Stack) -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        # All sidecars start at t=0 with ManualClock.
        for name in _SIDECAR_NAMES:
            t = await _now_for(client, stack.sidecars[name].base_url)
            assert t == pytest.approx(0.0, abs=1e-6), f"{name} not at t=0"

        # One broadcast — every sidecar advances by the same delta.
        await stack.advance_clock(60.0)
        for name in _SIDECAR_NAMES:
            t = await _now_for(client, stack.sidecars[name].base_url)
            assert t == pytest.approx(60.0, abs=1e-6), f"{name} drifted: t={t}"

        # Second broadcast composes.
        await stack.advance_clock(2.5)
        for name in _SIDECAR_NAMES:
            t = await _now_for(client, stack.sidecars[name].base_url)
            assert t == pytest.approx(62.5, abs=1e-6), f"{name} composition broken"


async def test_real_clock_rejects_advance(real_clock_stack: Stack) -> None:
    # Stack.advance_clock raises locally before issuing the broadcast.
    with pytest.raises(RuntimeError, match="manual"):
        await real_clock_stack.advance_clock(1.0)

    # And each sidecar rejects POST /control/advance_clock with 409.
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name in _SIDECAR_NAMES:
            response = await client.post(
                real_clock_stack.sidecars[name].base_url + "/control/advance_clock",
                params={"seconds": 1.0},
            )
            assert response.status_code == 409, f"{name} accepted advance on real clock"
            assert "RealClock" in response.json()["detail"]
