"""Tests for the ``trainer_callback_failure`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.trainer_callback_failure import TrainerCallbackFailure

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_trainer_callback_failures_surface() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(TrainerCallbackFailure(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_trainer_callback_meta_three_failures() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(TrainerCallbackFailure(), stack=s, seed=0)
        assert report.success
        surfaced = [e for e in report.timeline if e["event"] == "failures_surfaced"]
        assert surfaced and surfaced[0]["payload"]["count"] == 3
    finally:
        await s.shutdown()
