"""Pinned inference docker images — single source of truth.

The user picks ``inference.engine`` (currently only ``vllm``;
future engines like ``tgi`` / ``triton`` will land as new entries
here) and the pipeline auto-resolves the image from
:data:`INFERENCE_IMAGES`. Image versions are bumped in lock-step
with the code that ships them — no user YAML field, no version
drift between config and runtime.

The Mac control-plane reads this module from
:mod:`src.providers.single_node.inference.provider` (single_node
docker run) and from
:mod:`src.providers.runpod.inference.pods.provider` (RunPod pod
imageName). Both call :func:`resolve_inference_image` rather than
indexing the dict directly so the env-override logic is centralised.

Override (CI / dev only):
    ``RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_<ENGINE_UPPER>``
    e.g. ``RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_VLLM=registry.local/inference-vllm:dev``

Production override is intentionally not user-facing — production
users get the version baked into the release.
"""

from __future__ import annotations

import os
from typing import Final

# Bumped in lock-step with the docker image published by
# ``packages/engines/scripts/build_and_push_engines.sh``. The unified
# vLLM image covers both the LoRA merge step and the vLLM serve step
# (per packages/engines/src/ryotenkai_engines/vllm/IMAGE_README.md) —
# there is no longer a two-container strategy.
_DEFAULT_INFERENCE_IMAGES: Final[dict[str, str]] = {
    "vllm": "ryotenkai/inference-vllm:v1.0.0",
}


# Env-var override prefix. The full var name is
# ``<PREFIX><ENGINE_UPPER>`` — kept as a constant so tests and the
# CI smoke-test scripts can construct it without hardcoding the
# string in two places.
INFERENCE_IMAGE_OVERRIDE_ENV_PREFIX: Final[str] = (
    "RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_"
)


def _resolve_one(engine: str, default: str) -> str:
    """Read ``RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_<engine>``; fall
    back to the pinned default. Empty / whitespace-only values are
    treated as "no override" so a stray ``=`` in a CI .env file
    doesn't accidentally clear the image."""
    env_name = INFERENCE_IMAGE_OVERRIDE_ENV_PREFIX + engine.upper()
    override = os.environ.get(env_name, "").strip()
    return override or default


def _resolve_all() -> dict[str, str]:
    return {
        engine: _resolve_one(engine, default)
        for engine, default in _DEFAULT_INFERENCE_IMAGES.items()
    }


# Resolved at import time — same pattern as :data:`RUNTIME_IMAGE`.
# A test that wants to flip the value at runtime should use
# ``monkeypatch.setattr(src.inference.__about__, "INFERENCE_IMAGES", ...)``
# rather than touching env vars after import.
INFERENCE_IMAGES: Final[dict[str, str]] = _resolve_all()


# Engines we ship images for. Validators read this to fail fast on
# an unknown engine name in user YAML — e.g. typo ``vlllm``.
SUPPORTED_INFERENCE_ENGINES: Final[tuple[str, ...]] = tuple(_DEFAULT_INFERENCE_IMAGES)


def resolve_inference_image(engine: str) -> str:
    """Look up the docker image for ``engine``.

    Raises:
        KeyError: ``engine`` is not in :data:`INFERENCE_IMAGES` —
            indicates either a typo in user config or a new engine
            that hasn't been added to this module yet.
    """
    try:
        return INFERENCE_IMAGES[engine]
    except KeyError as exc:
        raise KeyError(
            f"unknown inference engine {engine!r}; "
            f"supported: {sorted(INFERENCE_IMAGES)}",
        ) from exc


__all__ = [
    "INFERENCE_IMAGES",
    "INFERENCE_IMAGE_OVERRIDE_ENV_PREFIX",
    "SUPPORTED_INFERENCE_ENGINES",
    "resolve_inference_image",
]
