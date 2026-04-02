from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel

# NOTE: Runtime imports are required for Pydantic field types.
from .huggingface import HuggingFaceHubConfig  # noqa: TC001
from .mlflow import MLflowConfig  # noqa: TC001


class ExperimentTrackingConfig(StrictBaseModel):
    """
    All integrations in one place.

    MLflow is mandatory for pipeline observability and fail-fast launch checks.
    If specified, all fields are required.
    """

    mlflow: MLflowConfig = Field(...)
    huggingface: HuggingFaceHubConfig | None = Field(None)

    def get_report_to(self) -> list[str]:
        """Get list of trackers for HuggingFace Trainer."""
        return ["mlflow"]


__all__ = [
    "ExperimentTrackingConfig",
]
