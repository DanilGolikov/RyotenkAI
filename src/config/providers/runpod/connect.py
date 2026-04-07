from __future__ import annotations

from pydantic import Field

from ...base import StrictBaseModel


class RunPodSSHConfig(StrictBaseModel):
    """SSH settings for RunPod pods (minimal, non-interactive)."""

    key_path: str = Field(
        ...,
        description=(
            "REQUIRED: Path to SSH private key used for RunPod pod access. "
            "RyotenkAI uses the RunPod SDK for pod lifecycle, "
            "then uses direct SSH over exposed TCP for runtime operations inside the pod."
        ),
    )


class RunPodConnectConfig(StrictBaseModel):
    """Connection config for RunPod provider."""

    ssh: RunPodSSHConfig = Field(..., description="SSH key config for pod access")


__all__ = [
    "RunPodConnectConfig",
    "RunPodSSHConfig",
]
