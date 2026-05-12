"""Shared fixtures for L6 stack smoke tests.

Stack API note: :class:`Stack` exposes ``Stack.boot()`` / ``await
stack.shutdown()`` rather than the spec's ``Stack.session()`` async-CM,
so we adapt with a thin async fixture here instead of changing the
orchestrator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import pytest_asyncio

from tests._harness.stack import Stack


@pytest_asyncio.fixture
async def stack() -> AsyncIterator[Stack]:
    """Per-test hermetic stack with all four sidecars on a manual clock."""
    s = await Stack.boot(clock="manual")
    try:
        yield s
    finally:
        await s.shutdown()


@pytest_asyncio.fixture
async def real_clock_stack() -> AsyncIterator[Stack]:
    """Per-test hermetic stack with all sidecars on a real clock."""
    s = await Stack.boot(clock="real")
    try:
        yield s
    finally:
        await s.shutdown()


def _stack_factory(clock: Literal["real", "manual"]) -> object:
    """Helper kept for symmetry; tests use the fixtures above."""
    return clock
