"""RunLoader catalog test — compressed-time runs in CI, real-time in nightly.

Each scenario is exercised once with :func:`scale_factor` honoured —
default ``SCALE=1`` for fast CI, ``SCALE=60`` (or higher) for nightly
real-time tests. All scenarios must finish within their per-scenario
budget and pass every declared SLO.
"""

from __future__ import annotations

import pytest

from tests._harness.stack import Stack
from tests.load.runloader.framework import RunLoaderScenario, run_scenario
from tests.load.runloader.scenarios.burst_10_runs import Burst10Runs
from tests.load.runloader.scenarios.orphan_pod_check import OrphanPodCheck
from tests.load.runloader.scenarios.sustained_30_minutes import Sustained30Minutes

pytestmark = [pytest.mark.load, pytest.mark.asyncio]


SCENARIOS: list[type[RunLoaderScenario]] = [
    Burst10Runs,
    Sustained30Minutes,
    OrphanPodCheck,
]


@pytest.mark.parametrize("scenario_cls", SCENARIOS, ids=lambda c: c.name)
async def test_runloader_scenario(scenario_cls: type[RunLoaderScenario]) -> None:
    s = await Stack.boot(clock="manual")
    try:
        scenario = scenario_cls()  # type: ignore[call-arg]
        report = await run_scenario(scenario, stack=s)
    finally:
        await s.shutdown()
    if not report.success:
        failures = [
            f"{r.name}: {', '.join(r.failures)}"
            for r in report.slo_results
            if not r.passed
        ]
        raise AssertionError(
            f"scenario {scenario_cls.name!r} failed SLOs: {failures}; "
            f"failed_steps={report.failed_steps}, total={report.total_steps}",
        )
    # Cheap sanity: we exercised SOMETHING.
    assert report.total_steps > 0
