"""ryotenkai_engines — inference engine plugin system.

Public API after PR-2:

  * :class:`IInferenceEngine` Protocol — engine plugin contract.
  * :class:`BaseEngineConfig`, :class:`LaunchSpec` — engine config base + structured launch description.
  * :class:`EngineCapabilities` — declarative engine capabilities.
  * :class:`EngineManifest` — Pydantic schema for ``engine.toml``.
  * :class:`EngineRegistry`, :func:`get_registry` — filesystem discovery.
  * :func:`resolve_image` — convention default + override chain.
  * :func:`get_engine_config_union` — Tag-based discriminated union, used by
    ``ryotenkai_shared.config.inference.schema``.
  * Errors: :class:`EngineRegistryError`, :class:`EngineNotRegistered`,
    :class:`EngineConfigError`.

PR-3 lands the first concrete engine (vLLM). See
``docs/plans/purring-sleeping-hartmanis.md`` for the full design.
"""

from __future__ import annotations

from ryotenkai_engines._config_union import (
    DISCRIMINATOR_FIELD,
    build_engine_config_union,
    get_engine_config_union,
    reset_engine_config_union,
)
from ryotenkai_engines.capabilities import ApiDialect, EngineCapabilities
from ryotenkai_engines.errors import (
    EngineConfigError,
    EngineNotRegistered,
    EngineRegistryError,
)
from ryotenkai_engines.images import (
    DEFAULT_IMAGE_REGISTRY,
    ENV_IMAGE_OVERRIDE_PATTERN,
    ENV_IMAGE_REGISTRY,
    resolve_image,
)
from ryotenkai_engines.interfaces import (
    BaseEngineConfig,
    IInferenceEngine,
    LaunchSpec,
)
from ryotenkai_engines.manifest import (
    LATEST_ENGINE_SCHEMA_VERSION,
    EngineManifest,
)
from ryotenkai_engines.registry import (
    EngineRegistry,
    LoadFailure,
    get_registry,
    reset_registry,
)

__version__ = "1.0.0"

__all__ = (
    # Protocol + types
    "IInferenceEngine",
    "BaseEngineConfig",
    "LaunchSpec",
    # Capabilities
    "ApiDialect",
    "EngineCapabilities",
    # Manifest schema
    "EngineManifest",
    "LATEST_ENGINE_SCHEMA_VERSION",
    # Registry
    "EngineRegistry",
    "LoadFailure",
    "get_registry",
    "reset_registry",
    # Image resolution
    "resolve_image",
    "ENV_IMAGE_REGISTRY",
    "DEFAULT_IMAGE_REGISTRY",
    "ENV_IMAGE_OVERRIDE_PATTERN",
    # Discriminated config
    "build_engine_config_union",
    "get_engine_config_union",
    "reset_engine_config_union",
    "DISCRIMINATOR_FIELD",
    # Errors
    "EngineRegistryError",
    "EngineNotRegistered",
    "EngineConfigError",
)
