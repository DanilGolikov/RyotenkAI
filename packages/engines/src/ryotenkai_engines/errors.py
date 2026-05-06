"""Engine-system errors.

PR-1 stub — landed in PR-2. Mirrors the error hierarchy in
``packages/providers/src/ryotenkai_providers/registry.py`` (``LoadFailure``,
``ProviderRegistryError``).
"""

from __future__ import annotations

# TODO(PR-2):
# class EngineRegistryError(RuntimeError): ...     # registry-level (manifest, importlib)
# class EngineConfigError(ValueError): ...         # config-level (validation)
# class EngineNotRegistered(EngineRegistryError):  ...  # specific lookup miss

__all__: tuple[str, ...] = ()
