"""Header slice: experiment / run identity + lineage summary.

Emits the leading section of an experiment report — run id,
experiment id, status, and a compact summary of ``ryotenkai.lineage.*``
tags pulled from the root run's :class:`RunHandle`. Per-tag detail is
deferred to :class:`~.lineage.LineageSlice`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_control.reports.composer import SliceOutput

if TYPE_CHECKING:
    from ryotenkai_control.reports.composer import ReportContext


class HeaderSlice:
    """Render the run-identity preamble.

    Pulls the root :class:`RunHandle` via
    :class:`~ryotenkai_shared.infrastructure.mlflow.protocols.IRunQuery`
    and prints the immutable identity fields. Always succeeds when the
    run id resolves; otherwise emits a warning and a minimal body.
    """

    name = "header"

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build the header slice for ``ctx``."""
        warnings: list[str] = []
        try:
            handle = ctx.run_query.get_run(ctx.run_id)
        except Exception as exc:  # noqa: BLE001 — surface to the report
            warnings.append(f"failed to resolve run: {exc}")
            md = f"Run id `{ctx.run_id}` could not be resolved.\n"
            return SliceOutput(
                title="Run", markdown=md, warnings=tuple(warnings)
            )

        lines = [
            f"- **Run id**: `{handle.run_id}`",
            f"- **Experiment id**: `{handle.experiment_id}`",
            f"- **Status**: `{handle.status.value}`",
            f"- **Tracking URI**: `{handle.tracking_uri}`",
        ]
        if handle.parent_run_id:
            lines.append(f"- **Parent run**: `{handle.parent_run_id}`")
        else:
            lines.append("- **Parent run**: _(top-level)_")
        md = "\n".join(lines) + "\n"
        return SliceOutput(
            title="Run",
            markdown=md,
            warnings=tuple(warnings),
        )


__all__ = ["HeaderSlice"]
