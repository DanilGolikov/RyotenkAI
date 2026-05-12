"""Shared fixtures for ``tests/unit/providers/``.

Several legacy tests build provider instances via ``object.__new__`` to skip
the heavy pydantic-config validator, then call ``get_capabilities()``. Post
Phase B that method now reads manifest-derived ClassVars (``_manifest_*``)
that are stamped by :class:`ProviderRegistry` on first ``from_filesystem``
load. When the unit tests run in isolation (without the contract suite
having loaded first), the ClassVars are unset and ``get_capabilities`` raises.

The autouse session fixture below loads the registry once so the ClassVars
are populated before any provider construction happens. This mirrors what
the full-suite run does implicitly (contract collects before unit) and
makes the unit-tests-only command line robust.
"""

from __future__ import annotations

import contextlib
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def _provider_registry_loaded() -> None:
    """Stamp manifest-derived ClassVars on provider classes once per session.

    Side effect of ``ProviderRegistry.from_filesystem()``: it iterates every
    ``provider.toml`` and sets ``_manifest_capabilities`` / ``_manifest_provider_*``
    on each concrete provider class. Tests that bypass the registry still
    need those ClassVars to be present so ``get_capabilities`` does not
    raise.
    """
    # The runpod SDK isn't installed in slim CI venvs; stub it before
    # touching the registry (importing RunPodProvider triggers a top-level
    # ``import runpod`` in its module).
    if "runpod" not in sys.modules:
        stub = types.ModuleType("runpod")
        stub.api_key = ""
        stub.create_pod = MagicMock()
        stub.get_pod = MagicMock()
        stub.stop_pod = MagicMock()
        stub.resume_pod = MagicMock()
        stub.terminate_pod = MagicMock()
        sys.modules["runpod"] = stub

    from ryotenkai_providers.registry import ProviderRegistry, reset_registry

    reset_registry()
    registry = ProviderRegistry.from_filesystem()
    # ``from_filesystem`` only registers manifests; the ClassVars are
    # stamped lazily by ``_resolve_class``. Force a resolution of every
    # discovered provider's training class so the ClassVars land before
    # any test starts constructing providers.
    for pid in registry.list():
        manifest = registry.get_manifest(pid)
        for role in manifest.provider.roles:
            # Some roles (e.g. ``inference``) may not be resolvable in
            # slim CI venvs (missing optional deps); the unit tests that
            # care about those roles already skip themselves. Suppress
            # the error so we still stamp the others.
            with contextlib.suppress(Exception):
                registry._resolve_class(pid, role_key=role)
