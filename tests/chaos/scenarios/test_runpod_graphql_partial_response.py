"""Tests for the ``runpod_graphql_partial_response`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.runpod_graphql_partial_response import RunpodGraphqlPartialResponse

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_runpod_partial_classified_transient() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(RunpodGraphqlPartialResponse(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_runpod_partial_meta_502_observed() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(RunpodGraphqlPartialResponse(), stack=s, seed=0)
        assert report.success
        events = [e for e in report.timeline if e["event"] == "partial_classified_transient"]
        assert events and events[0]["payload"]["seen_502"] >= 1
    finally:
        await s.shutdown()
