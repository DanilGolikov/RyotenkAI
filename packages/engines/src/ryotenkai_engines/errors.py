"""Engine-system errors — re-exports from the typed hierarchy.

Phase A2 Batch 2 migration (sharded-stargazing-wigderson):
the legacy ``EngineRegistryError`` / ``EngineNotRegistered`` /
``EngineConfigError`` classes (plain ``Exception``/``ValueError`` subclasses)
have been removed in favour of the typed hierarchy rooted at
:class:`ryotenkai_shared.errors.RyotenkAIError`.

  * Registry-lookup misses and registry-plumbing failures now raise
    :class:`EngineNotRegisteredError` (DomainError, status 404,
    code ENGINE_NOT_REGISTERED). The legacy distinction between
    "lookup miss" and "registry-plumbing bug" collapses into a single
    domain error; callers branch on ``context["reason"]`` if they need
    a sub-category (e.g. ``locator_resolve_failed``,
    ``manifest_invalid``, ``runtime_id_drift``, …).

  * Engine-config invariant rejections now raise
    :class:`EngineConfigInvalidError` (DomainError, status 422,
    code ENGINE_CONFIG_INVALID). Subcodes ride in ``context["reason"]``
    (e.g. ``vllm_live_lora_not_supported``).

This module is kept as a thin re-export to preserve import paths for
in-package callers (``from ryotenkai_engines.errors import …``); the
concrete classes live in :mod:`ryotenkai_shared.errors`.
"""

from __future__ import annotations

from ryotenkai_shared.errors import (
    EngineConfigInvalidError,
    EngineNotRegisteredError,
)

__all__ = (
    "EngineConfigInvalidError",
    "EngineNotRegisteredError",
)
