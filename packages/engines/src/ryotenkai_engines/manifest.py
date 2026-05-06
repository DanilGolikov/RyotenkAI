"""``EngineManifest`` Pydantic schema for ``engine.toml`` files.

PR-1 stub — landed in PR-2. Mirrors ``ProviderManifest`` (in
``packages/providers/src/ryotenkai_providers/manifest.py``) — same
``model_validator``-driven invariants, same ``extra="forbid"`` strictness,
same ``LATEST_*_SCHEMA_VERSION`` upgrade gate.

Top-level blocks:
  * ``[engine]``        — id, name, version, upstream_version (info), stability, homepage
  * ``[capabilities]``  — api_dialect, supports_*, supported_*, default_port
  * ``[image]``         — OPTIONAL; default derived by convention if absent
  * ``[entry_points.runtime]``       — IInferenceEngine class locator
  * ``[entry_points.config_schema]`` — BaseEngineConfig subclass locator
"""

from __future__ import annotations

# TODO(PR-2): EngineManifest, EngineSpec, ImageSpec, EntryPoints sub-models,
#             LATEST_ENGINE_SCHEMA_VERSION, _ENGINE_ID_RE, capability invariants.
LATEST_ENGINE_SCHEMA_VERSION = 1

__all__: tuple[str, ...] = ("LATEST_ENGINE_SCHEMA_VERSION",)
