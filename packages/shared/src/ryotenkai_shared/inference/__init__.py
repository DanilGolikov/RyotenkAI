"""DEPRECATED — superseded by ``ryotenkai_engines``.

The legacy single-engine setup pinned image names + supported-engines
list in this package. After the discriminated-union refactor, image
resolution lives in ``ryotenkai_engines.get_registry().get_image()``
and the supported-engines list is ``EngineRegistry.list()``.

The exports below are KEPT as deprecated shims for any external scripts
that still reference them. New code MUST go through ``ryotenkai_engines``.
"""

from __future__ import annotations

from ryotenkai_shared.inference.__about__ import (
    INFERENCE_IMAGE_OVERRIDE_ENV_PREFIX,
    INFERENCE_IMAGES,
    SUPPORTED_INFERENCE_ENGINES,
    resolve_inference_image,
)

__all__ = [
    "INFERENCE_IMAGES",
    "INFERENCE_IMAGE_OVERRIDE_ENV_PREFIX",
    "SUPPORTED_INFERENCE_ENGINES",
    "resolve_inference_image",
]
