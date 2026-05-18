"""Loss-curve slice (placeholder for M3.A).

Rendering an actual loss curve requires step-indexed metric history,
which is not part of :class:`IRunQuery`. The required
``IMetricHistoryQuery`` Protocol lands in M3.B (see ``vectorized-fluttering-mist.md``
§Read path).

For now this slice advertises the section, validates that the run
resolves, and emits a clearly-labelled ``TODO``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_control.reports.composer import SliceOutput

if TYPE_CHECKING:
    from ryotenkai_control.reports.composer import ReportContext


class LossCurveSlice:
    """Placeholder for the loss-curve section.

    Will be populated in M3.B once metric history is available via a
    dedicated Protocol.
    """

    name = "loss_curve"

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build a placeholder loss-curve slice for ``ctx``."""
        warnings: list[str] = [
            "loss_curve: placeholder — IMetricHistoryQuery not yet wired (M3.B)",
        ]
        md = (
            "_TODO (M3.B)_: render loss curves from per-step metric history "
            "via ``IMetricHistoryQuery``.\n"
            f"\nTarget run: `{ctx.run_id}`\n"
        )
        return SliceOutput(
            title="Loss Curves",
            markdown=md,
            warnings=tuple(warnings),
        )


__all__ = ["LossCurveSlice"]
