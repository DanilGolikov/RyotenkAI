"""Inference-deployer slice (placeholder for M3.A).

The legacy report builder pulls inference deployment metadata
(endpoint URL, vLLM pod id, deploy duration) from
``inference_deployer_results.json``. Equivalent loading requires the
same artifact-Loader Protocol slated for M3.B.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_control.reports.composer import SliceOutput

if TYPE_CHECKING:
    from ryotenkai_control.reports.composer import ReportContext


class InferenceSlice:
    """Placeholder for the inference-deployment section.

    Will report endpoint URL, pod id, and deploy duration in M3.B.
    """

    name = "inference"

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build a placeholder inference slice for ``ctx``."""
        warnings: list[str] = [
            "inference: placeholder — IInferenceResultsLoader not wired (M3.B)",
        ]
        md = (
            "_TODO (M3.B)_: load ``inference_deployer_results.json`` and "
            "render endpoint metadata.\n"
            f"\nTarget run: `{ctx.run_id}`\n"
        )
        return SliceOutput(
            title="Inference Deployment",
            markdown=md,
            warnings=tuple(warnings),
        )


__all__ = ["InferenceSlice"]
