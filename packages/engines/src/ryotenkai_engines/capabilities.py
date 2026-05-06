"""``EngineCapabilities`` Pydantic model — mirrors ``engine.toml [capabilities]``.

PR-1 stub — landed in PR-2. Cross-field invariants enforced via
``@model_validator``:

  * ``supports_quantization == False`` ⇒ ``supported_quantizations`` MUST be empty.
  * ``api_dialect == "custom"`` triggers a registry-time warning until/unless
    ``ModelClientFactory`` gains a non-OpenAI builder.

Drift detector ``scripts/check_engine_manifests.py`` (PR-10) cross-checks
that every shipped engine's runtime ``get_capabilities()`` exactly matches
its ``engine.toml [capabilities]`` block.
"""

from __future__ import annotations

# TODO(PR-2): EngineCapabilities pydantic model + invariants.
__all__: tuple[str, ...] = ()
