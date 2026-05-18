"""Evaluation-results slice (placeholder for M3.A).

The legacy report builder loads ``evaluation_results.json`` envelopes
from MLflow artifacts. Doing the same here requires either:

1. a typed ``IEvalResultsLoader`` Protocol, or
2. a generalised artifact-download Protocol on the read path.

Both are in scope for M3.B. M3.A surfaces the section as a TODO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_control.reports.composer import SliceOutput

if TYPE_CHECKING:
    from ryotenkai_control.reports.composer import ReportContext


class EvalSlice:
    """Placeholder for the evaluation-results section.

    Will load ``evaluation_results.json`` envelopes in M3.B via a
    typed loader Protocol.
    """

    name = "eval"

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build a placeholder eval slice for ``ctx``."""
        warnings: list[str] = [
            "eval: placeholder — IEvalResultsLoader not yet wired (M3.B)",
        ]
        md = (
            "_TODO (M3.B)_: load ``evaluation_results.json`` envelope and "
            "render per-judge / per-prompt aggregates.\n"
            f"\nTarget run: `{ctx.run_id}`\n"
        )
        return SliceOutput(
            title="Evaluation Results",
            markdown=md,
            warnings=tuple(warnings),
        )


__all__ = ["EvalSlice"]
