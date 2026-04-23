from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from src.config.integrations.mlflow import MLflowConfig, MLflowTrackingRef

MLflowRuntimeRole = Literal["control_plane", "training"]


@dataclass(frozen=True)
class ResolvedMLflowUris:
    tracking_uri: str | None
    local_tracking_uri: str | None
    effective_local_tracking_uri: str
    effective_remote_tracking_uri: str
    runtime_tracking_uri: str
    runtime_role: MLflowRuntimeRole


def _clean_uri(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def resolve_mlflow_uris(
    config: MLflowConfig | MLflowTrackingRef,
    *,
    runtime_role: MLflowRuntimeRole,
    env_tracking_uri: str | None = None,
) -> ResolvedMLflowUris:
    # A bare ``MLflowTrackingRef`` means the project references an integration
    # but the runtime hasn't merged in the Settings → Integrations payload yet.
    # Return empty URIs so callers can skip MLflow setup without crashing on
    # missing attributes; a proper resolver should replace the ref with a
    # ``MLflowConfig`` before we get here in real pipelines.
    if isinstance(config, MLflowTrackingRef):
        return ResolvedMLflowUris(
            tracking_uri=None,
            local_tracking_uri=None,
            effective_local_tracking_uri="",
            effective_remote_tracking_uri="",
            runtime_tracking_uri=_clean_uri(env_tracking_uri) or _clean_uri(os.getenv("MLFLOW_TRACKING_URI")) or "",
            runtime_role=runtime_role,
        )

    tracking_uri = _clean_uri(config.tracking_uri)
    local_tracking_uri = _clean_uri(config.local_tracking_uri)
    effective_local_tracking_uri = local_tracking_uri or tracking_uri or ""
    effective_remote_tracking_uri = tracking_uri or local_tracking_uri or ""

    runtime_tracking_uri = effective_local_tracking_uri
    if runtime_role == "training":
        runtime_tracking_uri = (
            _clean_uri(env_tracking_uri)
            or _clean_uri(os.getenv("MLFLOW_TRACKING_URI"))
            or effective_remote_tracking_uri
        )

    return ResolvedMLflowUris(
        tracking_uri=tracking_uri,
        local_tracking_uri=local_tracking_uri,
        effective_local_tracking_uri=effective_local_tracking_uri,
        effective_remote_tracking_uri=effective_remote_tracking_uri,
        runtime_tracking_uri=runtime_tracking_uri,
        runtime_role=runtime_role,
    )


__all__ = [
    "MLflowRuntimeRole",
    "ResolvedMLflowUris",
    "resolve_mlflow_uris",
]
