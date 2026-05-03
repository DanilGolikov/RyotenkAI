"""Unit tests for ExecutionSummaryReporter.

Focus areas:
- aggregate_training_metrics: no manager / empty / aggregates loss/steps/runtime
- aggregate_training_metrics honours injected collect_fn (orchestrator seam)
- collect_descendant_metrics: no manager / no parent run / BFS over phase_* children
- generate_experiment_report: missing run_id / calls generator / swallows exceptions
- print_summary: smoke test each section via fake console
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.reporting.summary_reporter import ExecutionSummaryReporter
from src.pipeline.stages import StageNames


# -----------------------------------------------------------------------------
# Config fixture
# -----------------------------------------------------------------------------


def _build_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "gpt2"
    cfg.training.type = "sft"
    cfg.training.get_effective_load_in_4bit.return_value = False
    cfg.training.hyperparams.per_device_train_batch_size = 4
    cfg.training.get_strategy_chain.return_value = []
    cfg.get_adapter_config.side_effect = ValueError("no adapter")
    primary_ds = MagicMock()
    primary_ds.get_source_type.return_value = "local"
    primary_ds.source_hf = None
    primary_ds.source_local = None
    primary_ds.adapter_type = "auto"
    cfg.get_primary_dataset.return_value = primary_ds
    return cfg


@pytest.fixture
def reporter() -> ExecutionSummaryReporter:
    return ExecutionSummaryReporter(_build_config())


# -----------------------------------------------------------------------------
# aggregate_training_metrics
# -----------------------------------------------------------------------------


def test_aggregate_no_op_when_manager_none(reporter: ExecutionSummaryReporter) -> None:
    reporter.aggregate_training_metrics(mlflow_manager=None)  # should not crash


def test_aggregate_no_op_when_no_metrics(reporter: ExecutionSummaryReporter) -> None:
    mgr = MagicMock()
    reporter.aggregate_training_metrics(mlflow_manager=mgr, collect_fn=lambda: [])
    mgr.log_metrics.assert_not_called()


def test_aggregate_logs_final_loss_steps_runtime(reporter: ExecutionSummaryReporter) -> None:
    mgr = MagicMock()
    phase_metrics = [
        {"train_loss": 0.5, "train_runtime": 100.0, "global_step": 100},
        {"train_loss": 0.3, "train_runtime": 200.0, "global_step": 100},
    ]
    reporter.aggregate_training_metrics(mlflow_manager=mgr, collect_fn=lambda: phase_metrics)
    mgr.log_metrics.assert_called_once()
    metrics = mgr.log_metrics.call_args.args[0]
    assert metrics["final_train_loss"] == 0.3  # last wins
    assert metrics["total_train_steps"] == 200.0
    assert metrics["total_train_runtime"] == 300.0


def test_aggregate_honours_injected_collect_fn(reporter: ExecutionSummaryReporter) -> None:
    mgr = MagicMock()
    calls = []

    def collect() -> list[dict]:
        calls.append(1)
        return [{"train_loss": 0.1}]

    reporter.aggregate_training_metrics(mlflow_manager=mgr, collect_fn=collect)
    assert calls == [1]
    mgr.log_metrics.assert_called_once()


def test_aggregate_skips_when_no_metrics_extracted(reporter: ExecutionSummaryReporter) -> None:
    mgr = MagicMock()
    reporter.aggregate_training_metrics(
        mlflow_manager=mgr, collect_fn=lambda: [{"unrelated": 1}]
    )
    # No known keys → nothing logged
    mgr.log_metrics.assert_not_called()


# -----------------------------------------------------------------------------
# collect_descendant_metrics
# -----------------------------------------------------------------------------


def test_collect_returns_empty_when_no_manager() -> None:
    assert ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=None) == []


def test_collect_returns_empty_when_no_run_id() -> None:
    mgr = MagicMock()
    mgr.run_id = ""
    mgr._run_id = ""
    assert ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=mgr) == []


def test_collect_finds_phase_children_bfs() -> None:
    mgr = MagicMock()
    mgr.run_id = "parent"
    parent_run = MagicMock()
    parent_run.info.experiment_id = "exp-1"
    mgr.client.get_run.return_value = parent_run

    # One child named "strategy", one grandchild named "phase_0"
    strategy = MagicMock()
    strategy.info.run_id = "child-1"
    strategy.info.run_name = "strategy"
    strategy.data.metrics = {}

    phase = MagicMock()
    phase.info.run_id = "grand-1"
    phase.info.run_name = "phase_0"
    phase.data.metrics = {"train_loss": 0.5}

    def fake_search(*, experiment_ids: list[str], filter_string: str) -> list[MagicMock]:
        if "'parent'" in filter_string:
            return [strategy]
        if "'child-1'" in filter_string:
            return [phase]
        return []

    mgr.client.search_runs.side_effect = fake_search

    result = ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=mgr, max_depth=2)
    assert result == [{"train_loss": 0.5}]


def test_collect_returns_empty_on_client_error() -> None:
    mgr = MagicMock()
    mgr.run_id = "parent"
    mgr.client.get_run.side_effect = RuntimeError("mlflow down")
    assert ExecutionSummaryReporter.collect_descendant_metrics(mlflow_manager=mgr) == []


# -----------------------------------------------------------------------------
# generate_experiment_report
# -----------------------------------------------------------------------------


def test_report_warns_when_no_run_id() -> None:
    with patch("src.pipeline.reporting.summary_reporter.ExperimentReportGenerator") as gen:
        ExecutionSummaryReporter.generate_experiment_report(run_id=None, mlflow_manager=None)
        gen.assert_not_called()


def test_report_calls_generator_and_swallows_exceptions(tmp_path: Path) -> None:
    mgr = MagicMock()
    mgr._gateway.uri = "http://tracking"

    # Failure path — must not raise
    with patch(
        "src.pipeline.reporting.summary_reporter.ExperimentReportGenerator",
        side_effect=RuntimeError("generator down"),
    ):
        ExecutionSummaryReporter.generate_experiment_report(run_id="rid-abc", mlflow_manager=mgr)


def test_report_success_invokes_generate() -> None:
    mgr = MagicMock()
    mgr._gateway.uri = "http://tracking"
    generator_inst = MagicMock()
    generator_inst.generate.return_value = "report-content"
    with (
        patch(
            "src.pipeline.reporting.summary_reporter.ExperimentReportGenerator",
            return_value=generator_inst,
        ),
        patch(
            "src.pipeline.reporting.summary_reporter.get_run_log_dir",
            return_value=Path("/tmp/logs"),
        ),
    ):
        ExecutionSummaryReporter.generate_experiment_report(run_id="rid-abc", mlflow_manager=mgr)
    generator_inst.generate.assert_called_once_with(
        run_id="rid-abc", local_logs_dir=Path("/tmp/logs")
    )


# -----------------------------------------------------------------------------
# print_summary (smoke)
# -----------------------------------------------------------------------------


def test_print_summary_smoke(reporter: ExecutionSummaryReporter) -> None:
    """Happy-path smoke — patched console captures all print calls."""
    fake_console = MagicMock()
    with patch("src.pipeline.reporting.summary_reporter.console", fake_console):
        reporter.print_summary(
            context={
                StageNames.DATASET_VALIDATOR: {"sample_count": 1000, "avg_length": 45},
                StageNames.GPU_DEPLOYER: {
                    "provider_name": "runpod",
                    "provider_type": "cloud",
                    "gpu_type": "A100",
                    "gpu_count": 1,
                    "cost_per_hr": 2.5,
                    "pod_startup_seconds": 30,
                    "upload_duration_seconds": 10,
                },
                StageNames.TRAINING_MONITOR: {
                    "training_duration_seconds": 600,
                    "training_info": {"final_loss": 0.1, "final_accuracy": 0.95},
                },
                StageNames.MODEL_RETRIEVER: {
                    "local_model_path": "/tmp/m",
                    "hf_repo_id": "org/m",
                    "model_size_mb": 500,
                },
                StageNames.MODEL_EVALUATOR: {"metrics": {"accuracy": 0.95, "f1": 0.85}},
                StageNames.INFERENCE_DEPLOYER: {"inference_scripts": {"chat": "chat.py"}},
            }
        )
    # Multiple sections printed
    assert fake_console.print.call_count > 5


def test_print_summary_handles_missing_stages(reporter: ExecutionSummaryReporter) -> None:
    fake_console = MagicMock()
    with patch("src.pipeline.reporting.summary_reporter.console", fake_console):
        reporter.print_summary(context={})
    # Still prints configuration / dataset / empty placeholders
    assert fake_console.print.call_count > 3
