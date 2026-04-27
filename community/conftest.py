"""Pytest bootstrap for the ``community/`` tree.

Two side-effects, both at collection time:

1. **Preload** ``community_libs.<name>`` into :mod:`sys.modules`
   before per-plugin conftests rebind ``plugin.py``. That
   ``plugin.py`` does ``from community_libs.helixql import …`` at
   module level, which can only resolve once
   :func:`src.community.libs.preload_community_libs` has registered
   the namespace. Pytest's collection runs *before* the real catalog
   is touched, so we preload here unconditionally — once per session —
   to make those imports work.

2. **Reset** the shared :func:`community_libs.helixql.get_compiler`
   cache between tests via an autouse fixture. The factory keys by
   ``timeout_seconds`` and intentionally returns the same instance
   to multiple plugin instances (that's the dedup) — which means
   tests asserting on ``compiler._cache`` size leak across runs
   unless we explicitly clear it. Resetting at teardown is cheaper
   and less intrusive than rewriting every assertion.

Idempotent: re-preloading the same root is a no-op; resetting an
already-empty cache is a no-op.
"""

from __future__ import annotations

import pytest

from src.community.constants import COMMUNITY_ROOT
from src.community.libs import libs_root_for, preload_community_libs

preload_community_libs(libs_root_for(COMMUNITY_ROOT))


@pytest.fixture(autouse=True)
def _reset_helixql_compiler_cache():
    """Drop the shared compiler cache between tests.

    Tests that poke at ``compiler._cache`` (size, contents) used to
    rely on a fresh ``HelixCompiler`` per plugin instance — but the
    shared :func:`get_compiler` factory now returns one instance per
    ``timeout_seconds`` value across the whole process. Reset the
    cache here so per-test invariants survive.
    """
    yield
    try:
        from community_libs.helixql.compiler import reset_compiler_cache
    except ImportError:
        # Lib isn't preloaded yet (e.g. tests under community/libs/
        # itself, where the namespace is set up by their own conftest
        # later). Nothing to reset in that case.
        return
    reset_compiler_cache()
