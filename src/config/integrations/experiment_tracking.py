from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel

# NOTE: Runtime imports are required for Pydantic field types.
from .huggingface import HuggingFaceHubConfig  # noqa: TC001
from .mlflow import MLflowConfig  # noqa: TC001


class ExperimentTrackingConfig(StrictBaseModel):
    """
    All integrations in one place.

    If a block is not specified, the integration is disabled.
    If specified, all fields are required.
    """

    mlflow: MLflowConfig | None = Field(None)
    huggingface: HuggingFaceHubConfig | None = Field(None)

    def get_report_to(self) -> list[str]:
        """Get list of trackers for HuggingFace Trainer."""
        trackers = []
        if self.mlflow and self.mlflow.enabled:
            trackers.append("mlflow")
        return trackers if trackers else ["none"]


__all__ = [
    "ExperimentTrackingConfig",
]
