"""Engine-system errors.

Three-level hierarchy:

  * :class:`EngineRegistryError` — registry / loader concerns (manifest
    parse, importlib resolution, missing entries). Configuration / plumbing
    bugs that the caller cannot retry away.
  * :class:`EngineConfigError` — engine-config validation concerns
    (``validate_config`` returned Err, capability/config mismatch). Surfaced
    when a user picks a config the engine refuses.
  * :class:`EngineNotRegistered` — specific lookup-miss subtype of
    :class:`EngineRegistryError`. Distinct because callers commonly handle
    "engine simply doesn't exist" cleanly.

All three carry the same ``code`` / ``details`` shape as
``ryotenkai_shared.utils.result.AppError`` so HTTP / CLI error renderers
stay uniform across the codebase.
"""

from __future__ import annotations

from typing import Any


class EngineRegistryError(Exception):
    """Raised when the registry itself can't satisfy a request.

    Mirrors :class:`ryotenkai_providers.registry.ProviderRegistryError` —
    same shape, separate hierarchy so callers can ``except`` precisely.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "ENGINE_REGISTRY_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details: dict[str, Any] = dict(details) if details else {}


class EngineNotRegistered(EngineRegistryError):
    """Specific lookup miss — the registry knows nothing about ``engine_id``."""

    def __init__(
        self,
        engine_id: str,
        *,
        known: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            message=(
                f"engine {engine_id!r} is not registered. "
                f"Known engines: {sorted(known)!r}."
            ),
            code="ENGINE_NOT_REGISTERED",
            details={"engine_id": engine_id, "known": list(known)},
        )


class EngineConfigError(ValueError):
    """Raised when an engine config fails an engine-side invariant.

    Distinct from Pydantic ``ValidationError`` (schema-level) — this fires
    after schema validation, when the engine's ``validate_config`` rejects
    the typed config (e.g. ``merge_before_deploy=False`` on vLLM today).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "ENGINE_CONFIG_INVALID",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details: dict[str, Any] = dict(details) if details else {}


__all__ = (
    "EngineRegistryError",
    "EngineNotRegistered",
    "EngineConfigError",
)
