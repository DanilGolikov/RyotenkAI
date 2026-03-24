from __future__ import annotations

from typing import TYPE_CHECKING

from src.constants import (
    PROVIDER_SINGLE_NODE,
    SUPPORTED_INFERENCE_ENGINES,
    SUPPORTED_INFERENCE_PROVIDERS,
)

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


def validate_inference_images_required_for_provider(cfg: InferenceConfig) -> None:
    """
    Enforce provider-specific requirements for inference engine images.

    Today:
    - single_node uses a two-container Docker strategy and requires both `merge_image` and `serve_image`
    - runpod (Pods) uses a unified image configured under `providers.runpod.inference.pod.image_name`
    """

    if not cfg.enabled:
        return

    # Only single_node runtime consumes these fields today.
    if cfg.provider != PROVIDER_SINGLE_NODE:
        return

    vllm = cfg.engines.vllm
    merge_image = (vllm.merge_image or "").strip()
    serve_image = (vllm.serve_image or "").strip()

    missing: list[str] = []
    if not merge_image:
        missing.append("inference.engines.vllm.merge_image")
    if not serve_image:
        missing.append("inference.engines.vllm.serve_image")
    if missing:
        raise ValueError(
            f"inference.enabled=true and inference.provider={PROVIDER_SINGLE_NODE!r} requires: {', '.join(missing)}"
        )


__all__ = [
    "validate_inference_enabled_is_supported",
    "validate_inference_images_required_for_provider",
]
