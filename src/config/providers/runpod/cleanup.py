from __future__ import annotations

from pydantic import Field

from ...base import StrictBaseModel


class RunPodCleanupConfig(StrictBaseModel):
    """Cleanup policy for RunPod pods."""

    auto_delete_pod: bool = Field(True, description="Delete pod after pipeline completes")
    keep_pod_on_error: bool = Field(False, description="Keep pod alive on error for debugging")
    on_interrupt: bool = Field(
        True,
        description="Cleanup cloud resources (terminate pod) when pipeline is interrupted via Ctrl+C (SIGINT).",
    )
    terminate_after_retrieval: bool = Field(
        False,
        description=(
            "Terminate training pod immediately after ModelRetriever completes "
            "(before InferenceDeployer / ModelEvaluator). "
            "Saves GPU cost when inference or evaluation stages are enabled."
        ),
    )


__all__ = [
    "RunPodCleanupConfig",
]
