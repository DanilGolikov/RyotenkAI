"""Tests for the ``stale_ssh_control_master`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.stale_ssh_control_master import StaleSshControlMaster

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_stale_ssh_reconnect_transparent() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(StaleSshControlMaster(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_stale_ssh_meta_disconnect_armed() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(StaleSshControlMaster(), stack=s, seed=0)
        assert report.success
        injects = [e for e in report.timeline if e["event"] == "disconnect_after_2"]
        assert injects, "disconnect chaos not armed"
    finally:
        await s.shutdown()
