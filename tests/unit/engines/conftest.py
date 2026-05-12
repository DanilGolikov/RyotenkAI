"""pytest configuration for ryotenkai_engines tests.

Ensures the lock-protected registry singleton + the cached union are
reset between tests so synthetic-manifest fixtures don't leak.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_engine_singletons() -> None:
    """Clear lock-protected singletons before AND after every test."""
    from ryotenkai_engines._config_union import reset_engine_config_union
    from ryotenkai_engines.registry import reset_registry

    reset_registry()
    reset_engine_config_union()
    yield
    reset_registry()
    reset_engine_config_union()
