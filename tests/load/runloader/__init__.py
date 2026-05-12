"""RunLoader — load-testing framework (Phase 5).

Scenarios are :class:`RunLoaderScenario` subclasses that drive a known
shape of synthetic load (burst / sustained / orphan-check) against the
hermetic stack and assert SLO compliance.
"""

from tests.load.runloader.framework import (
    RunLoader,
    RunLoaderConfig,
    RunLoaderResult,
    RunLoaderScenario,
    SLOResult,
    SLOSpec,
    run_scenario,
)

__all__ = [
    "RunLoader",
    "RunLoaderConfig",
    "RunLoaderResult",
    "RunLoaderScenario",
    "SLOResult",
    "SLOSpec",
    "run_scenario",
]
