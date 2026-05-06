"""Image name resolution — convention default + override chain.

Resolution order (first match wins):

  1. **ENV override.** ``RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_<ENGINE_UPPER>``.
     For CI swaps, dev overrides, A/B tests. Operators control this
     without touching code or manifests.

  2. **Provider override.** Per-engine override declared in
     ``provider.toml [capabilities.inference.engine_overrides.<id>].image``.
     Used when a particular provider needs a custom build (e.g. RunPod
     pre-built CUDA-12.4 image).

  3. **Manifest explicit.** ``engine.toml [image].default``. Author opt-out
     from convention — for forks, custom registries, multi-arch tags.

  4. **CONVENTION fallback.** ``f"{prefix}/inference-{id}:{version}"``
     where ``prefix`` is env ``RYOTENKAI_INFERENCE_IMAGE_REGISTRY`` or
     ``"ryotenkai"``, ``version`` is ``engine.toml [engine].version``.

The convention is what makes the plugin system extensible without
touching a central image map. Author drops ``Dockerfile`` + ``engine.toml``;
the image name auto-derives.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from ryotenkai_engines.manifest import EngineManifest


#: Env var that overrides the default image registry prefix.
ENV_IMAGE_REGISTRY = "RYOTENKAI_INFERENCE_IMAGE_REGISTRY"

#: Default registry prefix when ``ENV_IMAGE_REGISTRY`` is unset.
DEFAULT_IMAGE_REGISTRY = "ryotenkai"

#: Per-engine env override pattern. Format with ``engine_upper=engine_id.upper()``.
ENV_IMAGE_OVERRIDE_PATTERN = "RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_{engine_upper}"


def _convention_image_name(*, engine_id: str, engine_version: str, env: Mapping[str, str]) -> str:
    """Pure function: build the convention default image name.

    Separated from :func:`resolve_image` so unit tests can pin exactly the
    one piece of behaviour without monkeypatching the manifest registry.
    """
    prefix = env.get(ENV_IMAGE_REGISTRY, DEFAULT_IMAGE_REGISTRY)
    return f"{prefix}/inference-{engine_id}:{engine_version}"


def resolve_image(
    *,
    engine_id: str,
    manifest: EngineManifest,
    provider_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the image name for ``engine_id`` using the override chain.

    Args:
        engine_id: Engine id (must match ``manifest.engine.id``).
        manifest: The :class:`EngineManifest` for this engine. Used for the
            ``[image].default`` lookup AND for the ``engine.version`` that
            seeds the convention fallback.
        provider_overrides: Per-engine override mapping from the provider
            manifest. Shape: ``{<engine_id>: <something with ``.image``
            attribute or ``image`` key>}``. Pass ``None`` when the caller
            doesn't have provider-side overrides (e.g. unit tests).
        env: Mapping that env-override lookups consult. Defaults to
            ``os.environ``. Tests pass a custom dict.

    Returns:
        The resolved image name (e.g. ``"ryotenkai/inference-vllm:1.0.0"``).
    """
    if engine_id != manifest.engine.id:
        raise ValueError(
            f"engine_id {engine_id!r} does not match manifest.engine.id "
            f"{manifest.engine.id!r} — caller bug.",
        )

    env = env if env is not None else os.environ

    # 1. Env override.
    env_key = ENV_IMAGE_OVERRIDE_PATTERN.format(engine_upper=engine_id.upper())
    override = env.get(env_key)
    if override:
        return override

    # 2. Provider override. We accept either an attribute access (Pydantic
    # model with ``.image``) or a dict-like with ``"image"`` key — both
    # shapes show up depending on whether the provider manifest has been
    # parsed yet at the call site.
    if provider_overrides:
        per_engine = provider_overrides.get(engine_id) if hasattr(provider_overrides, "get") else None
        if per_engine is not None:
            image_attr = getattr(per_engine, "image", None)
            if image_attr is None and isinstance(per_engine, Mapping):
                image_attr = per_engine.get("image")
            if image_attr:
                return image_attr

    # 3. Manifest explicit.
    if manifest.image is not None and manifest.image.default:
        return manifest.image.default

    # 4. Convention fallback.
    return _convention_image_name(
        engine_id=engine_id,
        engine_version=manifest.engine.version,
        env=env,
    )


__all__ = (
    "ENV_IMAGE_REGISTRY",
    "DEFAULT_IMAGE_REGISTRY",
    "ENV_IMAGE_OVERRIDE_PATTERN",
    "resolve_image",
)
