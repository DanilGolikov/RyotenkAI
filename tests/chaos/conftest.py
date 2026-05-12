"""Chaos-test fixtures.

The chaos lane needs a per-scenario :class:`Stack` boot like
``tests/stack/conftest.py`` does — but anchored under a separate
log directory to keep artifact paths clear.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from tests._harness.stack import Stack

_LOG_ROOT = Path(__file__).resolve().parents[1] / ".chaos_logs"


@pytest_asyncio.fixture
async def chaos_stack(request: pytest.FixtureRequest) -> AsyncIterator[Stack]:
    log_dir = _LOG_ROOT / request.node.name.replace("/", "-").replace("::", "-")
    log_dir.mkdir(parents=True, exist_ok=True)
    s = await Stack.boot(clock="manual", log_dir=log_dir)
    try:
        yield s
    finally:
        await s.shutdown()
