"""Tests for the ``ws_reconnect_storm`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.ws_reconnect_storm import WSReconnectStorm

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_ws_reconnect_storm_server_stable() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(WSReconnectStorm(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_ws_reconnect_storm_meta_reconnects_recorded() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(WSReconnectStorm(), stack=s, seed=0)
        assert report.success
        events = [e for e in report.timeline if e["event"] == "10_rapid_reconnects"]
        assert events
    finally:
        await s.shutdown()
