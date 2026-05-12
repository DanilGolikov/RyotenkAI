"""Tests for the ``journal_rotation_race`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.journal_rotation_race import JournalRotationRace

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_journal_rotation_no_event_loss() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(JournalRotationRace(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_journal_rotation_meta_rotation_recorded() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(JournalRotationRace(), stack=s, seed=0)
        assert report.success
        events = [e for e in report.timeline if e["event"] == "rotation_simulated"]
        assert events, "rotation bookmark not recorded"
    finally:
        await s.shutdown()
