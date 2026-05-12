"""Tests for the ``mac_sleep_plus_pod_crash`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.mac_sleep_plus_pod_crash import MacSleepPlusPodCrash

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_mac_sleep_plus_pod_crash_faithful() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(MacSleepPlusPodCrash(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_mac_sleep_plus_pod_crash_meta_terminated_recorded() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(MacSleepPlusPodCrash(), stack=s, seed=0)
        assert report.success
        pod = report.post_state["runpod"]["pods"]["p-crash"]
        assert pod["desired_status"] == "TERMINATED"
    finally:
        await s.shutdown()
