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
    auto_stop_after_training: bool = Field(
        True,
        description=(
            "Automatically stop the pod (not terminate) after training completes. "
            "The pod calls RunPod API to stop itself, releasing the GPU and stopping billing. "
            "Container disk and volume data are preserved for ModelRetriever. "
            "Requires RUNPOD_API_KEY to be available. "
            "When keep_pod_on_error=true and training fails, the pod is NOT stopped."
        ),
    )


__all__ = [
    "RunPodCleanupConfig",
]
