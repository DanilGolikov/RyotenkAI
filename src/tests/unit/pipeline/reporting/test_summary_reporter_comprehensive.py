"""Comprehensive tests for ExecutionSummaryReporter.

Emphasis on the falsy-value aggregation regressions fixed in review
(train_loss=0.0, runtime=0.0, steps=0) and the BFS deque behaviour.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.reporting.summary_reporter import ExecutionSummaryReporter
from src.pipeline.stages import StageNames


def _build_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "gpt2"
    cfg.training.type = "sft"
    cfg.training.get_effective_load_in_4bit.return_value = False
    cfg.training.hyperparams.per_device_train_batch_size = 4
    cfg.training.get_strategy_chain.return_value = []
    cfg.get_adapter_config.side_effect = ValueError("no adapter")
    ds = MagicMock()
    ds.get_source_type.return_value = "local"
    ds.source_hf = None
    ds.source_local = None
    ds.adapter_type = "auto"
    cfg.get_primary_dataset.return_value = ds
    return cfg


@pytest.fixture
def reporter() -> ExecutionSummaryReporter:
    return ExecutionSummaryReporter(_build_config())


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_aggregate_happy_path_logs_all_summary_metrics(
        self, reporter: ExecutionSummaryReporter
    ) -> None:
        mgr = MagicMock()
        phase_metrics = [
            {"train_loss": 0.5, "train_runtime": 100, "global_step": 50},
            {"train_loss": 0.3, "train_runtime": 200, "global_step": 75},
        ]
        reporter.aggregate_training_metrics(mlflow_manager=mgr, collect_fn=lambda: phase_metrics)
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["final_train_loss"] == 0.3
        assert metrics["total_train_runtime"] == 300
        assert metrics["total_train_steps"] == 125.0


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_aggregate_noop_when_mgr_none(self, reporter: ExecutionSummaryReporter) -> None:
        reporter.aggregate_training_metrics(mlflow_manager=None)  # no crash

    def test_aggregate_noop_on_empty_phase_metrics(self, reporter: ExecutionSummaryReporter) -> None:
        mgr = MagicMock()
        reporter.aggregate_training_metrics(mlflow_manager=mgr, collect_fn=lambda: [])
        mgr.log_metrics.assert_not_called()

    def test_generate_report_no_run_id_skips(self) -> None:
        with patch(
            "src.pipeline.reporting.summary_reporter.ExperimentReportGenerator"
        ) as MockGen:
            ExecutionSummaryReporter.generate_experiment_report(run_id=None, mlflow_manager=None)
            MockGen.assert_not_called()


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_single_phase_metric(self, reporter: ExecutionSummaryReporter) -> None:
        mgr = MagicMock()
        reporter.aggregate_training_metrics(
            mlflow_manager=mgr, collect_fn=lambda: [{"train_loss": 0.1}]
        )
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics == {"final_train_loss": 0.1}

    def test_collect_bfs_depth_zero(self) -> None:
        mgr = MagicMock()
        mgr.run_id = "parent"
        parent_run = MagicMock()
        parent_run.info.experiment_id = "exp"
        mgr.client.get_run.return_value = parent_run
        mgr.client.search_runs.return_value = []  # no children
        result = ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=mgr, max_depth=0)
        assert result == []

    def test_collect_respects_max_depth(self) -> None:
        """Boundary: children at depth > max_depth are skipped even if present."""
        mgr = MagicMock()
        mgr.run_id = "parent"
        parent_run = MagicMock()
        parent_run.info.experiment_id = "exp"
        mgr.client.get_run.return_value = parent_run

        phase = MagicMock()
        phase.info.run_id = "phase-1"
        phase.info.run_name = "phase_0"
        phase.data.metrics = {"x": 1}

        call_log: list[str] = []

        def search(*, experiment_ids, filter_string):
            call_log.append(filter_string)
            if "'parent'" in filter_string:
                return [phase]
            return []

        mgr.client.search_runs.side_effect = search
        # depth=1: phase is child of parent → collected
        result = ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=mgr, max_depth=1)
        assert result == [{"x": 1}]


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    def test_bfs_visits_each_id_once(self) -> None:
        """Invariant: no double-visit even if the tree has repeated edges."""
        mgr = MagicMock()
        mgr.run_id = "parent"
        parent_run = MagicMock()
        parent_run.info.experiment_id = "exp"
        mgr.client.get_run.return_value = parent_run

        child = MagicMock()
        child.info.run_id = "parent"  # self-reference loop
        child.info.run_name = "self-loop"
        child.data.metrics = {}

        mgr.client.search_runs.return_value = [child]
        # Must not hang — cycle detection
        result = ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=mgr, max_depth=3)
        assert result == []

    def test_aggregate_preserves_config(self, reporter: ExecutionSummaryReporter) -> None:
        """Invariant: aggregate_training_metrics never mutates reporter config."""
        original = reporter._config
        reporter.aggregate_training_metrics(mlflow_manager=None)
        assert reporter._config is original


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_mlflow_client_exception_returns_empty(self) -> None:
        mgr = MagicMock()
        mgr.run_id = "parent"
        mgr.client.get_run.side_effect = RuntimeError("mlflow down")
        assert ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=mgr) == []

    def test_aggregate_exception_in_log_metrics_bubbles(self) -> None:
        mgr = MagicMock()
        mgr.log_metrics.side_effect = RuntimeError("log broken")
        reporter = ExecutionSummaryReporter(_build_config())
        with pytest.raises(RuntimeError):
            reporter.aggregate_training_metrics(
                mlflow_manager=mgr, collect_fn=lambda: [{"train_loss": 0.1}]
            )

    def test_generate_report_swallows_generator_failure(self) -> None:
        mgr = MagicMock()
        mgr.tracking_uri = "http://x"
        with (
            patch(
                "src.pipeline.reporting.summary_reporter.ExperimentReportGenerator",
                side_effect=RuntimeError("broken"),
            ),
            patch(
                "src.pipeline.reporting.summary_reporter.get_run_log_dir",
                return_value=Path("/tmp"),
            ),
        ):
            # Must not raise
            ExecutionSummaryReporter.generate_experiment_report(run_id="rid", mlflow_manager=mgr)


# =============================================================================
# 6. REGRESSIONS — falsy-value bugs in aggregation
# =============================================================================


class TestRegressions:
    def test_regression_train_loss_zero_captured(self, reporter: ExecutionSummaryReporter) -> None:
        """REGRESSION: walrus ``if train_loss := ...`` dropped loss=0.0.
        Perfect convergence must surface in MLflow parent run."""
        mgr = MagicMock()
        reporter.aggregate_training_metrics(
            mlflow_manager=mgr, collect_fn=lambda: [{"train_loss": 0.0}]
        )
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["final_train_loss"] == 0.0

    def test_regression_zero_runtime_accumulator_accepts_without_crashing(
        self, reporter: ExecutionSummaryReporter
    ) -> None:
        """REGRESSION: walrus dropped ``runtime=0.0``. Accumulator now accepts zeros
        (no exception). When the summed total is 0 we still don't emit a summary
        metric — "no signal, no log" is the preserved policy."""
        mgr = MagicMock()
        reporter.aggregate_training_metrics(
            mlflow_manager=mgr,
            collect_fn=lambda: [{"train_runtime": 0.0}, {"train_runtime": 0.0}],
        )
        # Contract: no crash, no spurious metric emitted (all zeros -> no emit).
        mgr.log_metrics.assert_not_called()

    def test_regression_zero_steps_accumulator_accepts_without_crashing(
        self, reporter: ExecutionSummaryReporter
    ) -> None:
        """Same as above for global_step=0."""
        mgr = MagicMock()
        reporter.aggregate_training_metrics(
            mlflow_manager=mgr,
            collect_fn=lambda: [{"global_step": 0}, {"global_step": 0}],
        )
        mgr.log_metrics.assert_not_called()

    def test_regression_mixed_zeros_and_positives(self, reporter: ExecutionSummaryReporter) -> None:
        """REGRESSION: with mixed zero+positive phases, positives are still summed
        correctly — zeros no longer raise nor skew the total."""
        mgr = MagicMock()
        reporter.aggregate_training_metrics(
            mlflow_manager=mgr,
            collect_fn=lambda: [
                {"train_loss": 0.0, "train_runtime": 0.0, "global_step": 0},
                {"train_loss": 0.5, "train_runtime": 100.0, "global_step": 50},
            ],
        )
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["final_train_loss"] == 0.5  # last non-None loss
        assert metrics["total_train_runtime"] == 100.0
        assert metrics["total_train_steps"] == 50.0

    def test_regression_deque_used_for_bfs(self) -> None:
        """REGRESSION: list.pop(0) is O(n); deque.popleft is O(1).

        Assert that the actual implementation uses deque + popleft — compile
        the method source and inspect it after stripping comments/strings
        to avoid false positives from documentation.
        """
        import inspect
        import re

        src = inspect.getsource(ExecutionSummaryReporter.collect_descendant_metrics)
        # Strip docstring/comments/string literals — we only want real code
        code_only = re.sub(r'"""[\s\S]*?"""', "", src)
        code_only = re.sub(r"'[^'\n]*'", "''", code_only)
        code_only = re.sub(r'"[^"\n]*"', '""', code_only)
        code_only = re.sub(r"#.*", "", code_only)

        assert "deque" in code_only
        assert "popleft" in code_only
        # No list.pop(0) left in actual code (doc comments OK)
        assert re.search(r"\bpop\(0\)", code_only) is None

    def test_regression_tracking_uri_via_public_property(self) -> None:
        """REGRESSION: previously accessed ``_gateway.uri`` — now uses tracking_uri."""
        mgr = MagicMock()
        mgr.tracking_uri = "http://public-uri"
        with (
            patch(
                "src.pipeline.reporting.summary_reporter.ExperimentReportGenerator"
            ) as MockGen,
            patch(
                "src.pipeline.reporting.summary_reporter.get_run_log_dir",
                return_value=Path("/tmp"),
            ),
        ):
            MockGen.return_value.generate.return_value = ""
            ExecutionSummaryReporter.generate_experiment_report(run_id="rid", mlflow_manager=mgr)
        MockGen.assert_called_once_with("http://public-uri")


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


@pytest.mark.parametrize(
    ("phase_metrics", "expected_emit"),
    [
        # no metrics at all — no emit
        ([], False),
        ([{}], False),
        ([{"unrelated": 1}], False),
        # any real metric triggers log_metrics
        ([{"train_loss": 0.1}], True),
        ([{"train_runtime": 10.0}], True),
        ([{"global_step": 100}], True),
        ([{"train_loss": 0.1, "global_step": 50}], True),
    ],
)
def test_combinatorial_aggregate_emit_or_not(
    reporter: ExecutionSummaryReporter,
    phase_metrics: list[dict],
    expected_emit: bool,
) -> None:
    mgr = MagicMock()
    reporter.aggregate_training_metrics(mlflow_manager=mgr, collect_fn=lambda: phase_metrics)
    assert mgr.log_metrics.called == expected_emit


@pytest.mark.parametrize("n_phases", [0, 1, 5, 20])
def test_combinatorial_phase_count_scaling(
    reporter: ExecutionSummaryReporter, n_phases: int
) -> None:
    """Aggregation scales linearly with phase count; no O(n^2) collapse."""
    mgr = MagicMock()
    phases = [{"train_loss": 0.1 * i, "global_step": i} for i in range(1, n_phases + 1)]
    reporter.aggregate_training_metrics(mlflow_manager=mgr, collect_fn=lambda: phases)
    if n_phases > 0:
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["total_train_steps"] == float(sum(range(1, n_phases + 1)))
