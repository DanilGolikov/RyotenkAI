"""Tests for the ``mlflow_double_finalization`` scenario."""

from __future__ import annotations

import pytest

from tests._harness.chaos import ScenarioRunner
from tests._harness.stack import Stack
from tests.chaos.scenarios.mlflow_double_finalization import MlflowDoubleFinalization

pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


async def test_mlflow_double_finalization_noop() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(MlflowDoubleFinalization(), stack=s, seed=0)
    finally:
        await s.shutdown()
    assert report.success, report.error


async def test_mlflow_double_finalization_meta_status() -> None:
    s = await Stack.boot(clock="manual")
    try:
        report = await ScenarioRunner().run(MlflowDoubleFinalization(), stack=s, seed=0)
        assert report.success
        runs = report.post_state["mlflow"]["runs"]
        # exactly one run, status FINISHED.
        statuses = {r["status"] for r in runs.values()}
        assert statuses == {"FINISHED"}, statuses
    finally:
        await s.shutdown()
