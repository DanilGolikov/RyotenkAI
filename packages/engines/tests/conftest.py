"""pytest configuration for ryotenkai_engines tests.

PR-1 stub — real fixtures (synthetic engine builder, manifest factory,
registry-with-tmp-path) land alongside their consuming tests in PR-2/PR-3.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_engine_registry_singleton() -> None:
    """Placeholder — once ``EngineRegistry`` is a lock-protected singleton
    (PR-2), this fixture clears its cache between tests so ``tmp_path``-based
    discovery works without leaking state."""
    # TODO(PR-2): import and reset the singleton here.
    yield
