"""Slice-oriented report composition (replacement for ``core/builder.py``).

The legacy :mod:`ryotenkai_control.reports.core.builder` is an 884-LOC
god-class that aggregates header, metrics, loss curves, eval results,
inference info, artifacts, and lineage tags into a single markdown
report. Phase M3.A introduces a slice-oriented decomposition that
exists ALONGSIDE the legacy builder:

* Each report section is a :class:`ReportSliceBuilder` Protocol impl
  in :mod:`ryotenkai_control.reports.slices`.
* :class:`ReportComposer` is the orchestrator — it knows nothing
  about specific sections; it just walks the slice list, calls each
  one's ``build(ctx)``, and assembles the outputs.
* Slices receive a :class:`ReportContext` carrying the DI'd
  dependencies (``IRunQuery``, :class:`RunTreeWalker`, etc.). This
  flattens the dependency graph so individual slices can be tested in
  isolation against fakes.

The legacy ``builder.py`` is NOT deleted in M3.A. The composer is
ready to be wired up in M3.B (alongside the migration of the six
ad-hoc ``MlflowClient()`` callsites).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ryotenkai_shared.infrastructure.mlflow.protocols import (
        IArtifactSink,
        IRunQuery,
    )

    from ryotenkai_control.pipeline.mlflow.read.tree_walker import RunTreeWalker


@dataclass(frozen=True)
class SliceOutput:
    """Output of a single :class:`ReportSliceBuilder`.

    :param title: Human-readable section title; rendered as a markdown
        heading by :class:`ReportComposer`.
    :param markdown: Section body (markdown). May be empty for placeholder
        slices.
    :param warnings: Non-fatal issues encountered while building this
        slice. Surfaced in the final report under a "Warnings"
        sub-section for the slice.
    """

    title: str
    markdown: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportOutput:
    """Aggregated output of :meth:`ReportComposer.compose`.

    :param slices: Ordered slice outputs, in the order they were built.
    """

    slices: tuple[SliceOutput, ...]

    def to_markdown(self) -> str:
        """Render the aggregated slices as a single markdown document.

        Each slice is emitted as a level-2 heading followed by its
        markdown body. Warnings (if any) are appended as a level-3
        heading inside the slice.
        """
        parts: list[str] = []
        for sl in self.slices:
            parts.append(f"## {sl.title}\n")
            if sl.markdown:
                parts.append(sl.markdown.rstrip() + "\n")
            if sl.warnings:
                parts.append("\n### Warnings\n")
                for w in sl.warnings:
                    parts.append(f"- {w}\n")
            parts.append("\n")
        return "".join(parts).rstrip() + "\n"


@dataclass
class ReportContext:
    """Bundle of DI'd dependencies passed to every slice builder.

    Slices read only the fields they need; the bundle is intentionally
    permissive so individual slices can be evolved without breaking
    others.

    :param run_id: Root run id the report targets.
    :param run_query: :class:`IRunQuery` for ad-hoc lookups.
    :param tree_walker: Pre-constructed :class:`RunTreeWalker` rooted
        at the run query. Slices that need the descendant set should
        prefer this over manual BFS.
    :param artifact_sink: Optional :class:`IArtifactSink`. Slices that
        need to upload generated artifacts (e.g. PNGs of loss curves)
        consult this; ``None`` disables artifact-writing slices.
    :param extra: Free-form bag for additional dependencies that
        haven't earned a typed field yet. Slices reading this MUST
        document the keys they expect.
    """

    run_id: str
    run_query: IRunQuery
    tree_walker: RunTreeWalker
    artifact_sink: IArtifactSink | None = None
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class ReportSliceBuilder(Protocol):
    """Protocol for a single report section.

    Implementations:

    * Declare a stable :attr:`name` (used in logging / debugging).
    * Implement :meth:`build` purely in terms of the supplied
      :class:`ReportContext` — slices must not reach into globals or
      construct their own MLflow clients.
    """

    name: str

    def build(self, ctx: ReportContext) -> SliceOutput:
        """Build the slice output for ``ctx``."""
        ...


class ReportComposer:
    """Sequential orchestrator over :class:`ReportSliceBuilder`s.

    :param slices: Ordered iterable of slice builders. The order in
        which the composer visits them is the order in which their
        outputs appear in the final report.
    """

    def __init__(self, slices: Sequence[ReportSliceBuilder]) -> None:
        self._slices: tuple[ReportSliceBuilder, ...] = tuple(slices)

    @property
    def slices(self) -> tuple[ReportSliceBuilder, ...]:
        """Read-only view of the configured slice list."""
        return self._slices

    def compose(self, ctx: ReportContext) -> ReportOutput:
        """Build every slice in order and aggregate the outputs.

        Each slice is called once; exceptions propagate (the composer
        does NOT swallow per-slice failures — that would mask
        regressions). Slices that want soft-failure semantics should
        catch their own errors and surface them via
        :attr:`SliceOutput.warnings`.
        """
        outputs: list[SliceOutput] = []
        for sl in self._slices:
            outputs.append(sl.build(ctx))
        return ReportOutput(slices=tuple(outputs))


__all__ = [
    "ReportComposer",
    "ReportContext",
    "ReportOutput",
    "ReportSliceBuilder",
    "SliceOutput",
]
