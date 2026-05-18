"""Lineage slice: surface ``ryotenkai.lineage.*`` info from the root handle.

Lineage tags are how the pipeline records provenance (parent
experiment id, source commit, base-model alias). The legacy builder
reads them directly from the MLflow run; in the slice world we limit
the data we surface to what's already on the :class:`RunHandle` plus
the result of a single :meth:`IRunQuery.get_run` call — slices must
not reach into MLflow's wide API.

M3.B will extend :class:`IRunQuery` (or introduce a sibling Protocol)
so the slice can read arbitrary tags. For now this slice is intentionally
narrow: it prints parent-run-id, experiment id, and status; deeper
lineage detail surfaces as a TODO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_control.reports.composer import SliceOutput

if TYPE_CHECKING:
    from ryotenkai_control.reports.composer import ReportContext


class LineageSlice:
    """Render the lineage section."""

    name = "lineage"

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build the lineage slice for ``ctx``."""
        warnings: list[str] = []
        try:
            handle = ctx.run_query.get_run(ctx.run_id)
        except Exception as exc:  # noqa: BLE001 — soft-fail with warning
            warnings.append(f"failed to resolve run for lineage: {exc}")
            return SliceOutput(
                title="Lineage",
                markdown="_Lineage unavailable: run could not be resolved._\n",
                warnings=tuple(warnings),
            )

        lines = [
            f"- **Run id**: `{handle.run_id}`",
            f"- **Experiment id**: `{handle.experiment_id}`",
        ]
        if handle.parent_run_id:
            lines.append(f"- **Parent run**: `{handle.parent_run_id}`")
        else:
            lines.append("- **Parent run**: _(top-level)_")
        lines.append(
            "\n_TODO (M3.B)_: render ``ryotenkai.lineage.*`` tags via "
            "a tag-aware Protocol extension."
        )
        return SliceOutput(
            title="Lineage",
            markdown="\n".join(lines) + "\n",
            warnings=tuple(warnings),
        )


__all__ = ["LineageSlice"]
