"""Tests for the ``openapi_drift_midflight`` scenario (partially implemented)."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.openapi_drift_midflight import OpenAPIDriftMidFlight

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_openapi_drift_partial_implementation() -> None:
    """Partial scenario — ``steady_state`` ``pytest.skip``s pending Phase 6 FE wiring."""
    s = await Stack.boot(clock="manual")
    try:
        scenario = OpenAPIDriftMidFlight()
        runner = ScenarioRunner()
        report = await runner.run(scenario, stack=s, seed=0)
        # The runner surfaces pytest.skip() as ``report.skipped`` rather
        # than treating it as a failure.
        assert report.skipped
        assert "Phase 6" in (report.skip_reason or "")
    finally:
        await s.shutdown()


async def test_openapi_drift_meta_partial_marked() -> None:
    s = await Stack.boot(clock="manual")
    try:
        scenario = OpenAPIDriftMidFlight()
        report = await ScenarioRunner().run(scenario, stack=s, seed=0)
        events = [e for e in report.timeline if e["event"] == "partial_scenario_marked"]
        assert events
    finally:
        await s.shutdown()
