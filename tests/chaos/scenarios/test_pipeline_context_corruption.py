"""Tests for the ``pipeline_context_corruption`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.pipeline_context_corruption import PipelineContextCorruption

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_pipeline_corruption_detected() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(PipelineContextCorruption(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_pipeline_corruption_meta_event_recorded() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(PipelineContextCorruption(), stack=s, seed=0)
        assert report.success
        detected = [e for e in report.timeline if e["event"] == "corruption_detected"]
        assert detected, "detection not recorded"
    finally:
        await s.shutdown()
