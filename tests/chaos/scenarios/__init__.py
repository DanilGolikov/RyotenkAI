"""Chaos scenario catalog.

Each module here implements one :class:`tests._harness.chaos.ChaosScenario`
and registers it via :func:`tests._harness.chaos.register_scenario`. The
test driver (``tests/chaos/scenarios/test_*.py``) either calls
``run_chaos_scenario(stack, scenario)`` (when a sidecar stack is needed)
or instantiates the scenario and drives it via in-process fakes (cheaper).
"""
