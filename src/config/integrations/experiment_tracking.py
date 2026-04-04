from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel

# NOTE: Runtime imports are required for Pydantic field types.
from .huggingface import HuggingFaceHubConfig  # noqa: TC001
from .mlflow import MLflowConfig  # noqa: TC001


class ExperimentTrackingConfig(StrictBaseModel):
    """
    All integrations in one place.

    MLflow is optional — when absent the pipeline runs without experiment tracking.
    When provided, all MLflow fields are required.
    """

    mlflow: MLflowConfig | None = Field(None)
    huggingface: HuggingFaceHubConfig | None = Field(None)

    def get_report_to(self) -> list[str]:
        """Get list of trackers for HuggingFace Trainer."""
        if self.mlflow is None:
            return ["none"]
        return ["mlflow"]


__all__ = [
    "ExperimentTrackingConfig",
]
