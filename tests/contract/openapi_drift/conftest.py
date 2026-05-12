"""Schemathesis tests need the same per-test :class:`Stack` fixture as
``tests/stack/`` — replicated here because pytest scopes fixtures by
file ancestry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from tests._harness.stack import Stack

_LOG_ROOT = Path(__file__).resolve().parents[2] / ".stack_logs"


@pytest_asyncio.fixture
async def stack(request: pytest.FixtureRequest) -> AsyncIterator[Stack]:
    log_dir = _LOG_ROOT / (
        "contract-openapi-" + request.node.name.replace("/", "-").replace("::", "-")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    s = await Stack.boot(clock="manual", log_dir=log_dir)
    try:
        yield s
    finally:
        await s.shutdown()
