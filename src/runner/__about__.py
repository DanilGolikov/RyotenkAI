"""Pinned constants for the runner package.

``RUNTIME_IMAGE`` is the **single source of truth** for which docker
image the Mac control plane provisions on RunPod / single_node hosts.
Provider configs read this constant directly. Phase 6.6 removed the
user-facing ``image_name`` / ``docker_image`` Pydantic fields —
versions are tied to the release, not to user YAML, eliminating the
config-vs-code drift problem.

Override is supported only via ``RYOTENKAI_RUNTIME_IMAGE_OVERRIDE``
environment variable, intended for CI smoke tests and dev iteration —
*not* a user-facing config.
"""

from __future__ import annotations

import os
from typing import Final

# Bumped in lock-step with the docker image published by
# ``docker/training/build_and_push.sh``. The publisher names the
# repo ``${DOCKER_USERNAME}/ryotenkai-training-runtime``; with our
# Docker Hub user ``ryotenkai`` that resolves to the doubled-prefix
# path below — kept as-is to match the publish script and avoid a
# rename round-trip on Docker Hub.
_DEFAULT_RUNTIME_IMAGE: Final[str] = (
    "ryotenkai/ryotenkai-training-runtime:v1.0.4"
)


def _resolve_runtime_image() -> str:
    override = os.environ.get("RYOTENKAI_RUNTIME_IMAGE_OVERRIDE", "").strip()
    return override or _DEFAULT_RUNTIME_IMAGE


RUNTIME_IMAGE: Final[str] = _resolve_runtime_image()


__all__ = ["RUNTIME_IMAGE"]
