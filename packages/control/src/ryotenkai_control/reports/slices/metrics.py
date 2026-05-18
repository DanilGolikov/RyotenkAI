"""Metrics slice: aggregated training/eval metrics across descendants.

Walks the run tree under ``ctx.run_id`` via :class:`RunTreeWalker`
and emits a count of descendants. Detailed metric aggregation (mean
loss, final eval accuracy, etc.) needs an ``IMetricHistoryQuery``
Protocol that lands in M3.B; until then this slice surfaces tree
shape so reports stay informative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_control.reports.composer import SliceOutput

if TYPE_CHECKING:
    from ryotenkai_control.reports.composer import ReportContext


class MetricsSlice:
    """Render an aggregated metrics summary.

    M3.A scope: tree topology (descendant count, per-status breakdown).
    M3.B will extend this to include actual metric history once
    ``IMetricHistoryQuery`` is available.
    """

    name = "metrics"

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build the metrics slice for ``ctx``."""
        warnings: list[str] = []
        try:
            descendants = ctx.tree_walker.flat_descendants(ctx.run_id)
        except Exception as exc:  # noqa: BLE001 — soft-fail with warning
            warnings.append(f"failed to walk run tree: {exc}")
            return SliceOutput(
                title="Metrics",
                markdown="_No descendants resolved._\n",
                warnings=tuple(warnings),
            )

        status_counts: dict[str, int] = {}
        for handle in descendants:
            status_counts[handle.status.value] = (
                status_counts.get(handle.status.value, 0) + 1
            )

        # Root is included in flat_descendants; show total runs (incl. root)
        # and the status histogram.
        lines = [
            f"- **Total runs (root + descendants)**: {len(descendants)}",
        ]
        for status, count in sorted(status_counts.items()):
            lines.append(f"- **{status}**: {count}")
        lines.append(
            "\n_TODO (M3.B)_: aggregate metric histories (loss, eval/*) "
            "via ``IMetricHistoryQuery``."
        )

        return SliceOutput(
            title="Metrics",
            markdown="\n".join(lines) + "\n",
            warnings=tuple(warnings),
        )


__all__ = ["MetricsSlice"]
