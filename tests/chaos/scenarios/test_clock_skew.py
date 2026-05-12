"""Tests for the ``clock_skew`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.clock_skew import ClockSkew

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_clock_skew_no_false_cancel() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(ClockSkew(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_clock_skew_meta_pod_running() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(ClockSkew(), stack=s, seed=0)
        assert report.success
        pod = report.post_state["runpod"]["pods"]["p-skew"]
        assert pod["desired_status"] == "RUNNING"
    finally:
        await s.shutdown()
