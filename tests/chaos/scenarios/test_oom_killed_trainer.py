"""Tests for the ``oom_killed_trainer`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.oom_killed_trainer import OomKilledTrainer

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_oom_classified_failed() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(OomKilledTrainer(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_oom_meta_event_recorded() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(OomKilledTrainer(), stack=s, seed=0)
        assert report.success
        events = [e for e in report.timeline if e["event"] == "oom_classified_failed"]
        assert events
    finally:
        await s.shutdown()
