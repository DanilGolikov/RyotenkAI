"""Pure-python service layer shared by FastAPI routers and the CLI.

These modules are deliberately free of FastAPI types — no
``HTTPException``, no ``Depends``, no ``Request`` — so they can be
imported and called from any context (tests, CLI, future workers).
HTTP wiring lives in :mod:`src.api.routers`; the CLI imports submodules
from here directly.

Public surface — these are the modules the CLI commands consume:

- :mod:`config_service` — ``validate_config`` for ad-hoc YAML files.
- :mod:`connection_test` — provider / integration smoke tests.
- :mod:`delete_service` — destructive run cleanup (``runs rm``).
- :mod:`integration_service` — HuggingFace / MLflow credential CRUD.
- :mod:`launch_service` — start / resume / restart / interrupt /
  restart-points (the ``run`` noun).
- :mod:`log_service` — read / tail run logs.
- :mod:`plugin_service` — catalog list / show.
- :mod:`project_service` — workspace project CRUD (``project`` noun).
- :mod:`provider_service` — provider configuration CRUD.
- :mod:`report_service` — markdown report generation.
- :mod:`run_service` — runs / attempts / stages reads (``runs`` noun).

Import as submodules — the API surface is too rich to flatten without
naming collisions (e.g. ``validate_config`` exists in both
:mod:`config_service` and :mod:`project_service` with different shapes).
"""

from src.api.services import (
    config_service,
    connection_test,
    delete_service,
    integration_service,
    launch_service,
    log_service,
    plugin_service,
    project_service,
    provider_service,
    report_service,
    run_service,
)

__all__ = [
    "config_service",
    "connection_test",
    "delete_service",
    "integration_service",
    "launch_service",
    "log_service",
    "plugin_service",
    "project_service",
    "provider_service",
    "report_service",
    "run_service",
]
