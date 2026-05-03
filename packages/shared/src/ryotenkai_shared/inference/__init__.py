"""Pinned constants and helpers for inference engines.

Mirrors the :mod:`src.runner` package's ``__about__`` pattern: the
single source of truth for which docker image is used per inference
engine lives in code, not in user YAML. Adding a new engine (TGI,
Triton, etc.) is one new entry in :data:`INFERENCE_IMAGES` plus the
engine-specific config class.
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
