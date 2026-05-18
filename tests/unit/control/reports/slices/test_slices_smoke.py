"""Smoke tests for the seven slice builders.

Each test instantiates one slice and verifies that ``build(ctx)``
returns a :class:`SliceOutput` with a non-empty title and markdown,
using a minimal :class:`ReportContext` backed by
:class:`FakeRunQuery` and :class:`FakeArtifactSink`.
"""

from __future__ import annotations

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from tests._fakes.mlflow_artifact_sink import FakeArtifactSink
from tests._fakes.mlflow_run_query import FakeRunQuery

from ryotenkai_control.pipeline.mlflow.read.tree_walker import RunTreeWalker
from ryotenkai_control.reports.composer import ReportContext, SliceOutput

# Direct module imports so the ``test_every_module_has_tests`` sentinel
# sees the full dotted name of each slice module.
from ryotenkai_control.reports.slices.artifacts import ArtifactsSlice
from ryotenkai_control.reports.slices.eval import EvalSlice
from ryotenkai_control.reports.slices.header import HeaderSlice
from ryotenkai_control.reports.slices.inference import InferenceSlice
from ryotenkai_control.reports.slices.lineage import LineageSlice
from ryotenkai_control.reports.slices.loss_curve import LossCurveSlice
from ryotenkai_control.reports.slices.metrics import MetricsSlice


def _make_ctx(*, with_sink: bool = False) -> ReportContext:
    root = RunHandle(
        run_id="run-1",
        experiment_id="exp-1",
        parent_run_id=None,
        tracking_uri="fake://",
        status=RunStatus.FINISHED,
    )
    child = RunHandle(
        run_id="run-1-child",
        experiment_id="exp-1",
        parent_run_id="run-1",
        tracking_uri="fake://",
        status=RunStatus.FINISHED,
    )
    rq = FakeRunQuery([root, child])
    walker = RunTreeWalker(rq)
    sink = FakeArtifactSink(require_existing_file=False) if with_sink else None
    return ReportContext(
        run_id="run-1",
        run_query=rq,
        tree_walker=walker,
        artifact_sink=sink,
    )


class TestHeaderSlice:
    def test_renders_run_identity(self) -> None:
        out = HeaderSlice().build(_make_ctx())
        assert isinstance(out, SliceOutput)
        assert "run-1" in out.markdown
        assert "exp-1" in out.markdown
        assert "FINISHED" in out.markdown

    def test_handles_missing_run(self) -> None:
        rq = FakeRunQuery()
        walker = RunTreeWalker(rq)
        ctx = ReportContext(run_id="ghost", run_query=rq, tree_walker=walker)
        out = HeaderSlice().build(ctx)
        assert out.warnings  # surface failure as warning
        assert "ghost" in out.markdown


class TestMetricsSlice:
    def test_counts_descendants(self) -> None:
        out = MetricsSlice().build(_make_ctx())
        assert "Total runs" in out.markdown
        # root + 1 child
        assert "2" in out.markdown


class TestLossCurveSlice:
    def test_is_placeholder_with_warning(self) -> None:
        out = LossCurveSlice().build(_make_ctx())
        assert "TODO" in out.markdown
        assert any("M3.B" in w for w in out.warnings)


class TestEvalSlice:
    def test_is_placeholder_with_warning(self) -> None:
        out = EvalSlice().build(_make_ctx())
        assert "TODO" in out.markdown
        assert any("M3.B" in w for w in out.warnings)


class TestInferenceSlice:
    def test_is_placeholder_with_warning(self) -> None:
        out = InferenceSlice().build(_make_ctx())
        assert "TODO" in out.markdown
        assert any("M3.B" in w for w in out.warnings)


class TestArtifactsSlice:
    def test_no_sink_emits_warning(self) -> None:
        out = ArtifactsSlice().build(_make_ctx(with_sink=False))
        assert any("no IArtifactSink" in w for w in out.warnings)

    def test_with_sink_no_warning_about_missing_sink(self) -> None:
        out = ArtifactsSlice().build(_make_ctx(with_sink=True))
        assert not any("no IArtifactSink" in w for w in out.warnings)
        assert "FakeArtifactSink" in out.markdown


class TestLineageSlice:
    def test_emits_run_and_experiment(self) -> None:
        out = LineageSlice().build(_make_ctx())
        assert "run-1" in out.markdown
        assert "exp-1" in out.markdown
        assert "top-level" in out.markdown

    def test_handles_missing_run(self) -> None:
        rq = FakeRunQuery()
        walker = RunTreeWalker(rq)
        ctx = ReportContext(run_id="ghost", run_query=rq, tree_walker=walker)
        out = LineageSlice().build(ctx)
        assert out.warnings
        assert "unavailable" in out.markdown.lower()


class TestAllSlicesInstantiable:
    def test_each_slice_has_name_attribute(self) -> None:
        slices = [
            HeaderSlice(),
            MetricsSlice(),
            LossCurveSlice(),
            EvalSlice(),
            InferenceSlice(),
            ArtifactsSlice(),
            LineageSlice(),
        ]
        names = [s.name for s in slices]
        # All distinct.
        assert len(set(names)) == len(names)
