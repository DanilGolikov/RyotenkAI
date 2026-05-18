"""Tests for :class:`ReportComposer` and its slice protocol.

The composer is intentionally tiny — its only job is to walk a list of
slice builders, call ``build(ctx)`` on each, and aggregate the outputs.
These tests use minimal fake slices to verify ordering, aggregation,
and error propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from tests._fakes.mlflow_run_query import FakeRunQuery

from ryotenkai_control.pipeline.mlflow.read.tree_walker import RunTreeWalker
from ryotenkai_control.reports.composer import (
    ReportComposer,
    ReportContext,
    ReportOutput,
    ReportSliceBuilder,
    SliceOutput,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class _RecordingSlice:
    """Fake slice that records every call."""

    name: str
    title: str = "section"
    body: str = "body"
    warnings: tuple[str, ...] = ()
    calls: list[str] = field(default_factory=list)

    def build(self, ctx: ReportContext) -> SliceOutput:
        self.calls.append(ctx.run_id)
        return SliceOutput(title=self.title, markdown=self.body, warnings=self.warnings)


def _make_ctx(run_id: str = "r1") -> ReportContext:
    handle = RunHandle(
        run_id=run_id,
        experiment_id="e1",
        parent_run_id=None,
        tracking_uri="fake://",
        status=RunStatus.FINISHED,
    )
    rq = FakeRunQuery([handle])
    walker = RunTreeWalker(rq)
    return ReportContext(run_id=run_id, run_query=rq, tree_walker=walker)


class TestComposerOrchestration:
    def test_calls_each_slice_once(self) -> None:
        s1 = _RecordingSlice("a")
        s2 = _RecordingSlice("b")
        composer = ReportComposer([s1, s2])
        composer.compose(_make_ctx("r1"))
        assert s1.calls == ["r1"]
        assert s2.calls == ["r1"]

    def test_preserves_slice_order(self) -> None:
        s1 = _RecordingSlice("a", title="alpha")
        s2 = _RecordingSlice("b", title="beta")
        s3 = _RecordingSlice("c", title="gamma")
        composer = ReportComposer([s3, s1, s2])
        out = composer.compose(_make_ctx())
        assert [s.title for s in out.slices] == ["gamma", "alpha", "beta"]

    def test_propagates_warnings(self) -> None:
        s = _RecordingSlice("a", warnings=("w1", "w2"))
        composer = ReportComposer([s])
        out = composer.compose(_make_ctx())
        assert out.slices[0].warnings == ("w1", "w2")

    def test_slices_property_is_immutable_view(self) -> None:
        s1 = _RecordingSlice("a")
        composer = ReportComposer([s1])
        assert isinstance(composer.slices, tuple)
        assert composer.slices[0] is s1


class TestReportOutput:
    def test_to_markdown_emits_headings(self) -> None:
        out = ReportOutput(
            slices=(
                SliceOutput(title="One", markdown="body-one"),
                SliceOutput(title="Two", markdown="body-two"),
            )
        )
        rendered = out.to_markdown()
        assert "## One" in rendered
        assert "## Two" in rendered
        assert "body-one" in rendered
        assert "body-two" in rendered

    def test_to_markdown_includes_warnings_block(self) -> None:
        out = ReportOutput(
            slices=(
                SliceOutput(title="X", markdown="", warnings=("oops",)),
            )
        )
        rendered = out.to_markdown()
        assert "### Warnings" in rendered
        assert "- oops" in rendered

    def test_to_markdown_no_warnings_block_when_empty(self) -> None:
        out = ReportOutput(
            slices=(SliceOutput(title="X", markdown="body"),)
        )
        rendered = out.to_markdown()
        assert "### Warnings" not in rendered


class TestErrorPropagation:
    def test_slice_exception_propagates(self) -> None:
        class _Boom:
            name = "boom"

            def build(self, ctx: ReportContext) -> SliceOutput:
                raise RuntimeError("slice failed")

        composer = ReportComposer([_Boom()])
        with pytest.raises(RuntimeError, match="slice failed"):
            composer.compose(_make_ctx())


class TestProtocolConformance:
    def test_recording_slice_satisfies_protocol(self) -> None:
        s = _RecordingSlice("a")
        assert isinstance(s, ReportSliceBuilder)
