"""RunLoader — load-testing framework (Phase 5).

Scenarios are :class:`RunLoaderScenario` subclasses that drive a known
shape of synthetic load (burst / sustained / orphan-check) against the
hermetic stack and assert SLO compliance.
"""

from tests.load.runloader.framework import (
    RunLoaderScenario,
    SLOResult,
    SLOSpec,
    run_scenario,
)

__all__ = [
    "RunLoaderScenario",
    "SLOResult",
    "SLOSpec",
    "run_scenario",
]
