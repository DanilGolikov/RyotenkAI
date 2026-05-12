"""Scenario 16 — ``openapi_drift_midflight``.

The frontend's cached OpenAPI schema drifts from the server's actual
response shape. The FE runtime guard (zod) should catch the
mismatch and surface a clear error rather than silently corrupting
state.

Note:
    Production FE runtime validation is wired for ONE endpoint only
    today (Phase 3 demo). The full guard rollout is Phase 6. This
    scenario therefore performs a precondition + injection (schema
    mutation) but ``steady_state`` is :func:`pytest.skip`-ed pending
    Phase 6 wiring.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests._harness.chaos import ScenarioContext, register_scenario
from tests.chaos.scenarios._base import ScenarioBase


@register_scenario
class OpenAPIDriftMidFlight(ScenarioBase):
    name = "openapi_drift_midflight"
    tags = ["schema"]
    recovery_window = timedelta(seconds=30)

    async def precondition(self, ctx: ScenarioContext) -> None:
        # Mark the scenario as documented but partially-implemented.
        ctx.extras["partial"] = True
        ctx.debug_recorder.record("precondition", "partial_scenario_marked")

    async def inject(self, ctx: ScenarioContext) -> None:
        # The actual mutation would happen in FE state; we record the
        # intent for the timeline but cannot drive it from the test
        # harness yet.
        ctx.debug_recorder.record("inject", "openapi_mutation_noted")

    async def steady_state(self, ctx: ScenarioContext) -> None:
        pytest.skip(
            "Phase 6: wire all FE endpoints to validateResponse before "
            "openapi_drift_midflight can verify FE runtime guard.",
        )


__all__: list[str] = []
