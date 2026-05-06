"""Image name resolution — convention default + override chain.

PR-1 stub — landed in PR-2. Resolution order (first match wins):

  1. ENV override:       ``RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_<ENGINE_UPPER>``
  2. Provider override:  ``provider.toml [capabilities.inference.engine_overrides.<id>].image``
  3. Manifest explicit:  ``engine.toml [image].default``
  4. CONVENTION fallback: ``f"{prefix}/inference-{id}:{version}"``
                          where ``prefix`` is env
                          ``RYOTENKAI_INFERENCE_IMAGE_REGISTRY`` or ``"ryotenkai"``,
                          ``version`` is ``engine.toml [engine].version``.

The convention lets engine authors drop a ``Dockerfile`` + ``engine.toml``
and have the image name auto-derive — no central map to edit.
"""

from __future__ import annotations

#: Env var that overrides the default image registry prefix.
ENV_IMAGE_REGISTRY = "RYOTENKAI_INFERENCE_IMAGE_REGISTRY"

#: Default registry prefix when ``ENV_IMAGE_REGISTRY`` is unset.
DEFAULT_IMAGE_REGISTRY = "ryotenkai"

#: Per-engine env override pattern (formatted with engine_id.upper()).
ENV_IMAGE_OVERRIDE_PATTERN = "RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_{engine_upper}"


# TODO(PR-2): resolve_image(engine_id, *, provider_overrides, env).

__all__ = (
    "ENV_IMAGE_REGISTRY",
    "DEFAULT_IMAGE_REGISTRY",
    "ENV_IMAGE_OVERRIDE_PATTERN",
)
