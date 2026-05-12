"""Chaos-catalog discovery test.

Walks the :mod:`tests.chaos.scenarios` package, imports every module,
asserts the registry has exactly 18 scenarios with the canonical
names from the architecture plan.

Also parametrizes over each scenario and drives it end-to-end via a
:class:`ScenarioRunner`; the per-scenario tests under
``tests/chaos/scenarios/`` provide the deeper assertions. The catalog
test is intentionally light — it proves discovery, not behaviour.
"""

from __future__ import annotations

import pytest

from tests._harness.chaos import (
    ChaosScenario,
    ScenarioRunner,
    all_scenarios,
    load_catalog,
)
from tests._harness.stack import Stack

pytestmark = [
    pytest.mark.chaos,
]


EXPECTED_NAMES = frozenset({
    "runpod_429_storm",
    "mlflow_circuit_open",
    "mac_sleep_during_run",
    "mac_sleep_plus_pod_crash",
    "pod_hibernation_while_mac_awake",
    "journal_rotation_race",
    "stale_ssh_control_master",
    "concurrent_terminate",
    "pipeline_context_corruption",
    "trainer_callback_failure",
    "oom_killed_trainer",
    "disk_full_on_pod",
    "provider_retry_storm",
    "clock_skew",
    "ws_reconnect_storm",
    "openapi_drift_midflight",
    "runpod_graphql_partial_response",
    "mlflow_double_finalization",
})


def test_catalog_has_18_scenarios() -> None:
    """The registry must contain exactly 18 named scenarios."""
    catalog = load_catalog()
    assert len(catalog) == 18, (
        f"expected 18 scenarios, got {len(catalog)}: "
        f"{sorted(cls.name for cls in catalog)}"
    )


def test_catalog_names_match_plan() -> None:
    """Names must exactly match the catalog in the Phase 5 plan."""
    load_catalog()
    actual_names = {cls.name for cls in all_scenarios()}
    missing = EXPECTED_NAMES - actual_names
    extra = actual_names - EXPECTED_NAMES
    assert not missing, f"missing scenarios: {sorted(missing)}"
    assert not extra, f"unexpected scenarios: {sorted(extra)}"


def test_catalog_scenarios_have_required_attributes() -> None:
    """Each scenario must expose name / tags / recovery_window."""
    for cls in load_catalog():
        assert isinstance(cls.name, str) and cls.name, cls
        assert isinstance(cls.tags, list) and all(isinstance(t, str) for t in cls.tags), cls
        assert hasattr(cls, "recovery_window"), cls


def test_catalog_isinstance_chaos_scenario() -> None:
    """Each registered class must satisfy the runtime-checkable Protocol."""
    for cls in load_catalog():
        instance = cls()  # type: ignore[call-arg]
        assert isinstance(instance, ChaosScenario), cls


# ---------------------------------------------------------------------------
# Smoke-run each scenario against an isolated Stack via the runner.
# This is the "catalog drive-by" — the catalog test ensures every
# scenario can be precondition+inject+steady_state+cleanup-ed without
# leaving state behind. Per-scenario test files provide deeper checks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_cls",
    load_catalog(),
    ids=lambda cls: cls.name,
)
async def test_scenario_runs_against_stack(scenario_cls: type[ChaosScenario]) -> None:
    """Run each scenario end-to-end and assert success / clean cleanup."""
    s = await Stack.boot(clock="manual")
    try:
        runner = ScenarioRunner()
        scenario = scenario_cls()  # type: ignore[call-arg]
        report = await runner.run(scenario, stack=s, seed=0)
    finally:
        await s.shutdown()

    # Partial scenarios (e.g. ``openapi_drift_midflight``) legitimately
    # call ``pytest.skip`` inside ``steady_state`` — the runner captures
    # that and exposes ``report.skipped`` so the catalog can be honest.
    if report.skipped:
        pytest.skip(report.skip_reason or "partial scenario skipped")
    if not report.success:
        raise AssertionError(
            f"scenario {scenario_cls.name!r} did not recover: {report.error}",
        )


__all__: list[str] = []
