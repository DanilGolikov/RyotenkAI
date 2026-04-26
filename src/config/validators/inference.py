from __future__ import annotations

from typing import TYPE_CHECKING

from src.constants import (
    SUPPORTED_INFERENCE_ENGINES,
    SUPPORTED_INFERENCE_PROVIDERS,
)
from src.inference import SUPPORTED_INFERENCE_ENGINES as _ENGINES_WITH_IMAGES

if TYPE_CHECKING:
    from ..inference.schema import InferenceConfig


def validate_inference_enabled_is_supported(cfg: InferenceConfig) -> None:
    """
    Fail-fast guard: inference stage is feature-flagged, but not all combinations
    are implemented in the runtime code yet.
    """

    if not cfg.enabled:
        return

    # Current supported providers:
    # - single_node (MVP)
    # - runpod (Pods + Network Volume: persistent HF cache + stop/resume)
    if cfg.provider not in SUPPORTED_INFERENCE_PROVIDERS:
        raise ValueError(
            f"inference.enabled=true but inference.provider='{cfg.provider}' is not supported yet. "
            f"Supported: {', '.join(repr(p) for p in SUPPORTED_INFERENCE_PROVIDERS)}."
        )
    if cfg.engine not in SUPPORTED_INFERENCE_ENGINES:
        raise ValueError(
            f"inference.enabled=true but inference.engine='{cfg.engine}' is not supported yet. "
            f"Supported engine for now: {', '.join(repr(e) for e in SUPPORTED_INFERENCE_ENGINES)}."
        )
    # Belt-and-braces: an engine listed in SUPPORTED_INFERENCE_ENGINES
    # but missing from :data:`INFERENCE_IMAGES` would crash later
    # at provision time. Surface the gap here instead.
    if cfg.engine not in _ENGINES_WITH_IMAGES:
        raise ValueError(
            f"inference.engine='{cfg.engine}' has no pinned docker image "
            f"in src.inference.__about__.INFERENCE_IMAGES — "
            f"add it there before enabling. Available: {sorted(_ENGINES_WITH_IMAGES)}."
        )


__all__ = [
    "validate_inference_enabled_is_supported",
]
