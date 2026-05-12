"""Tests for the ``disk_full_on_pod`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.disk_full_on_pod import DiskFullOnPod

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_disk_full_bounded_retries() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(DiskFullOnPod(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_disk_full_meta_attempts_bounded() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(DiskFullOnPod(), stack=s, seed=0)
        assert report.success
        events = [e for e in report.timeline if e["event"] == "bounded_retry"]
        assert events and events[0]["payload"]["attempts"] == 4
    finally:
        await s.shutdown()
