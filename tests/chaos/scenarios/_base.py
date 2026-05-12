"""Common helpers shared by every concrete :class:`ChaosScenario`.

Every concrete scenario lives in its own module and registers itself
via :func:`tests._harness.chaos.register_scenario`. They all share a
similar shape (precondition → inject → steady_state → cleanup); the
helpers below collapse the boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import httpx

from tests._harness.chaos import ScenarioContext


@dataclass
class ScenarioBase:
    """Mixin-style base for chaos scenarios.

    Subclasses set :attr:`name`, :attr:`tags`, :attr:`recovery_window`
    as class attributes and override the lifecycle hooks. Default
    implementations are no-ops so each scenario only overrides what it
    needs.
    """

    name: str = ""
    tags: list[str] = field(default_factory=list)
    recovery_window: timedelta = field(default_factory=lambda: timedelta(seconds=30))

    async def precondition(self, ctx: ScenarioContext) -> None:  # noqa: D401 -- override
        return None

    async def inject(self, ctx: ScenarioContext) -> None:
        return None

    async def steady_state(self, ctx: ScenarioContext) -> None:
        return None

    async def cleanup(self, ctx: ScenarioContext) -> None:
        # Best-effort sidecar reset so back-to-back scenario runs start
        # clean. Never raises.
        try:
            await ctx.stack.reset()
        except Exception:
            return None
        return None


async def sidecar_get(
    ctx: ScenarioContext,
    target: str,
    path: str,
    *,
    timeout: float = 5.0,
) -> httpx.Response:
    """GET against a sidecar's control axis or REST surface."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(ctx.stack.sidecars[target].base_url + path)


async def sidecar_post(
    ctx: ScenarioContext,
    target: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> httpx.Response:
    """POST against a sidecar's control axis or REST surface."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(
            ctx.stack.sidecars[target].base_url + path,
            params=params,
            json=json,
        )


__all__ = ["ScenarioBase", "sidecar_get", "sidecar_post"]
