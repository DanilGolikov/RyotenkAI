"""``EngineRegistry`` — filesystem discovery of engine plugins.

PR-1 stub — landed in PR-2. Verbatim structural copy of
``ProviderRegistry`` (``packages/providers/src/ryotenkai_providers/registry.py``):

  * ``from_filesystem(root=None)`` walks
    ``packages/engines/src/ryotenkai_engines/*/engine.toml`` and validates
    each manifest. Failures collected in ``LoadFailure`` tuples — never
    raises during discovery.

  * ``list()``, ``get_manifest(id)``, ``get_runtime(id)``,
    ``get_config_class(id)``, ``get_image(id, *, provider_overrides, env)``,
    ``failures()``.

  * Lock-protected module-level singleton; first call pays import cost,
    subsequent calls O(1).
"""

from __future__ import annotations

# TODO(PR-2): EngineRegistry, LoadFailure, _resolve_class.
__all__: tuple[str, ...] = ()
