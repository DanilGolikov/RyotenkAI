"""Tests for the ``provider_retry_storm`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.provider_retry_storm import ProviderRetryStorm

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_provider_retry_storm_bounded() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(ProviderRetryStorm(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_provider_retry_storm_meta_no_runaway() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(ProviderRetryStorm(), stack=s, seed=0)
        assert report.success
        calls = report.post_state["runpod"].get("call_history", [])
        assert len(calls) <= 10, f"runaway retries: {len(calls)}"
    finally:
        await s.shutdown()
