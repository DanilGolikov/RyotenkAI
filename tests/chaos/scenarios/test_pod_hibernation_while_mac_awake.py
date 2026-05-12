"""Tests for the ``pod_hibernation_while_mac_awake`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.pod_hibernation_while_mac_awake import PodHibernationWhileMacAwake

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_pod_hibernation_visible() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(PodHibernationWhileMacAwake(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_pod_hibernation_meta_state_changed() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(PodHibernationWhileMacAwake(), stack=s, seed=0)
        assert report.success
        pod = report.post_state["runpod"]["pods"]["p-hib"]
        assert pod["hibernated"] is True
        assert pod["desired_status"] == "HIBERNATED"
    finally:
        await s.shutdown()
