"""Pure-function builders for test values (real instances, not mocks).

Each factory in this package constructs a *real* domain object with sensible
test defaults, accepting ``**overrides`` for the fields the caller cares about.
This is the third leg of the Phase 3B mock-elimination strategy:

* ``tests/_fakes/`` — stateful runtime substitutes (FakeMLflowManager, etc.)
* ``tests/_factories/`` — pure builders for value objects (PipelineConfig,
  RunData, …)
* ``tests/_helpers/`` — common test utilities

Use a factory whenever a test currently builds a ``MagicMock(spec=X)`` for
a value object that is cheap to construct end-to-end.
"""

from __future__ import annotations

from tests._factories.pipeline_config import make_pipeline_config
from tests._factories.run_data import make_run_data

__all__: list[str] = [
    "make_pipeline_config",
    "make_run_data",
]
