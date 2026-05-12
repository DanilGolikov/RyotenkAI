"""Tests for the ``mac_sleep_during_run`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.mac_sleep_during_run import MacSleepDuringRun

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_mac_sleep_run_still_active() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(MacSleepDuringRun(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_mac_sleep_meta_clock_advanced() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(MacSleepDuringRun(), stack=s, seed=0)
        assert report.success
        injects = [e for e in report.timeline if e["event"] == "clock_advanced"]
        assert injects and injects[0]["payload"]["seconds"] == 600
    finally:
        await s.shutdown()
