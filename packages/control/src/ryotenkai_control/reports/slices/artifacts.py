"""Artifacts slice: list MLflow artifacts attached to the root run.

Acknowledges the existence of an :class:`IArtifactSink` for symmetry
with the rest of the read-path, even though listing is the inverse
operation. In M3.A this slice emits a placeholder note about which
sink (if any) is wired; M3.B will introduce a typed
``IArtifactList`` Protocol so the slice can enumerate logical
artifact paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_control.reports.composer import SliceOutput

if TYPE_CHECKING:
    from ryotenkai_control.reports.composer import ReportContext


class ArtifactsSlice:
    """Render the artifact-listing section.

    M3.A scope: report whether an :class:`IArtifactSink` is wired and
    print the run id; placeholder for full enumeration in M3.B.
    """

    name = "artifacts"

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build the artifacts slice for ``ctx``."""
        warnings: list[str] = []
        if ctx.artifact_sink is None:
            warnings.append("artifacts: no IArtifactSink wired in ctx")
            sink_state = "_(no artifact sink wired)_"
        else:
            sink_state = f"_(artifact sink: `{type(ctx.artifact_sink).__name__}`)_"

        md = (
            f"Artifact sink: {sink_state}\n"
            f"\n_TODO (M3.B)_: enumerate artifact paths via "
            "``IArtifactList`` Protocol.\n"
            f"\nTarget run: `{ctx.run_id}`\n"
        )
        return SliceOutput(
            title="Artifacts",
            markdown=md,
            warnings=tuple(warnings),
        )


__all__ = ["ArtifactsSlice"]
