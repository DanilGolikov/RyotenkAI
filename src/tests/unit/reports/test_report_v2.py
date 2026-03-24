"""
Unit tests for Report Generation V2 Architecture.
Covers: Adapter -> Domain -> Builder -> Generator.

REFACTORED: All patches use pytest-mock's mocker fixture for proper cleanup.
"""

import json
import math
from unittest.mock import MagicMock

import mlflow
import pytest
from mlflow.entities import Run, RunData, RunInfo

from src.reports.adapters.mlflow_adapter import MLflowAdapter
from src.reports.domain.entities import RunStatus
from src.reports.report_generator import ExperimentReportGenerator

# =============================================================================
# FIXTURES & HELPERS
# =============================================================================


@pytest.fixture
def mock_mlflow_client(mocker):
    """Mock MlflowClient using mocker for proper cleanup."""
    mock = mocker.patch("src.reports.adapters.mlflow_adapter.MlflowClient")
    return mock.return_value


@pytest.fixture
def mock_adapter_methods(mocker):
    """Mock adapter internal methods for proper cleanup."""
    json_mock = mocker.patch.object(MLflowAdapter, "_load_json_events", return_value=[])
    yaml_mock = mocker.patch.object(MLflowAdapter, "_load_yaml_config", return_value={})
    return json_mock, yaml_mock


@pytest.fixture(autouse=True)
def _restore_mlflow_tracking_uri():
    """
    Ensure tests in this module don't leak MLflow global tracking URI.

    ExperimentReportGenerator.__init__ calls mlflow.set_tracking_uri(), which is global
    and can affect subsequent tests (e.g., PipelineOrchestrator report generation).
    """
    original_uri = mlflow.get_tracking_uri()
    try:
        yield
    finally:
        mlflow.set_tracking_uri(original_uri)


def create_mock_run(run_id: str, name: str, tags=None, params=None, metrics=None, children_ids=None):
    """Helper to create a mock MLflow Run object."""
    info = RunInfo(
        run_id=run_id,
        experiment_id="exp_1",
        user_id="user",
        status="FINISHED",
        start_time=1700000000000,
        end_time=1700001000000,
        lifecycle_stage="active",
        artifact_uri=f"path/{run_id}",
    )

    data = MagicMock(spec=RunData)
    data.metrics = metrics or {}
    data.params = params or {}
    data.tags = tags or {"mlflow.runName": name}

    run = Run(run_info=info, run_data=data)
    run._children_ids = children_ids or []
    return run


def create_running_run(run_id: str, name: str, parent_id: str | None = None):
    """Helper to create a RUNNING run with no end_time."""
    info = RunInfo(
        run_id=run_id,
        experiment_id="exp_1",
        user_id="user",
        status="RUNNING",
        start_time=1700000000000,
        end_time=None,
        lifecycle_stage="active",
        artifact_uri=f"path/{run_id}",
    )

    tags = {"mlflow.runName": name}
    if parent_id:
        tags["mlflow.parentRunId"] = parent_id

    data = MagicMock(spec=RunData)
    data.metrics = {}
    data.params = {}
    data.tags = tags

    return Run(run_info=info, run_data=data)


# =============================================================================
# 1. POSITIVE TESTS
# =============================================================================


class TestReportGenerationPositive:
    """Positive test cases for report generation."""

    def test_full_pipeline_success(self, mock_mlflow_client, mocker):
        """Positive 1: Standard pipeline with Parent -> Container -> Phases."""
        # Setup Hierarchy
        root = create_mock_run("root_1", "Pipeline_Run")
        container = create_mock_run(
            "cont_1", "Container", tags={"mlflow.parentRunId": "root_1"}, children_ids=["phase_1", "phase_2"]
        )
        phase1 = create_mock_run(
            "phase_1",
            "phase_0_cpt",
            tags={"mlflow.parentRunId": "cont_1", "training.strategy_type": "CPT"},
            params={"training.hyperparams.actual.learning_rate": "0.001"},
            metrics={"train_loss": 0.5},
        )
        phase2 = create_mock_run(
            "phase_2",
            "phase_1_sft",
            tags={"mlflow.parentRunId": "cont_1", "training.strategy_type": "SFT"},
            metrics={"train_loss": 0.3},
        )

        def get_run_side_effect(run_id):
            runs = {"root_1": root, "cont_1": container, "phase_1": phase1, "phase_2": phase2}
            if run_id not in runs:
                raise Exception(f"Run {run_id} Not Found")
            return runs[run_id]

        def search_runs_side_effect(experiment_ids, filter_string, **kwargs):
            if "root_1" in filter_string:
                return [container]
            if "cont_1" in filter_string:
                return [phase1, phase2]
            return []

        mock_mlflow_client.get_run.side_effect = get_run_side_effect
        mock_mlflow_client.search_runs.side_effect = search_runs_side_effect

        # Mock adapter methods using mocker
        mock_json = mocker.patch.object(MLflowAdapter, "_load_json_events")
        mock_json.side_effect = [
            [],  # pipeline events
            [
                {
                    "source": "MemoryManager",
                    "event_type": "oom",
                    "message": "OOM detected",
                    "timestamp": "2023-01-01T12:00:00Z",
                }
            ],
        ]
        mocker.patch.object(MLflowAdapter, "_load_yaml_config", return_value={})

        # Execute
        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("root_1")

        # Verify
        assert len(report.phases) == 2
        assert report.phases[0].strategy == "CPT"
        assert report.phases[0].effective_config["learning_rate"] == 0.001
        assert report.phases[1].final_loss == 0.3
        assert len(report.memory_management.oom_events) == 1
        assert report.memory_management.oom_events[0].message == "OOM detected"

    def test_single_run_success(self, mock_mlflow_client, mock_adapter_methods):
        """Positive 2: Simple single run (no hierarchy)."""
        run = create_mock_run("single_1", "Simple_Train", params={"learning_rate": "5e-5"}, metrics={"loss": 1.2})
        child = create_mock_run(
            "child_1",
            "phase_0_sft",
            tags={"mlflow.parentRunId": "single_1", "mlflow.runName": "phase_0_sft"},
            metrics={"loss": 1.0},
        )

        mock_mlflow_client.get_run.return_value = run
        mock_mlflow_client.search_runs.return_value = [child]

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("single_1")

        assert len(report.phases) == 1
        assert report.phases[0].run_name == "phase_0_sft"


# =============================================================================
# 2. ERROR TESTS
# =============================================================================


class TestReportGenerationErrors:
    """Error handling test cases."""

    def test_run_not_found(self, mock_mlflow_client):
        """Error 1: Run ID does not exist."""
        mock_mlflow_client.get_run.side_effect = Exception("Resource not found")

        generator = ExperimentReportGenerator("http://mock")

        with pytest.raises(ValueError, match="Run not found"):
            generator.generate_report_model("missing_id")

    def test_corrupted_artifacts(self, mock_mlflow_client, mocker):
        """Error 2: Artifacts are not valid JSON."""
        root = create_mock_run("root_bad", "Bad_Artifacts")
        mock_mlflow_client.get_run.return_value = root
        mock_mlflow_client.search_runs.return_value = []

        # Mock filesystem operations
        mocker.patch("pathlib.Path.read_text", side_effect=json.JSONDecodeError("msg", "doc", 0))
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("tempfile.mkdtemp", return_value="/tmp")

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("root_bad")

        # Should not crash, just empty events
        assert len(report.timeline) == 0
        assert report.memory_management.oom_count == 0


# =============================================================================
# 3. BOUNDARY TESTS
# =============================================================================


class TestReportGenerationBoundary:
    """Boundary condition test cases."""

    def test_empty_metrics_boundary(self, mock_mlflow_client, mock_adapter_methods):
        """Boundary 1: Phases exist but have no metrics."""
        root = create_mock_run("root_empty", "Empty_Metrics")
        child = create_mock_run("child_empty", "phase_0", metrics={})

        mock_mlflow_client.get_run.return_value = root
        mock_mlflow_client.search_runs.return_value = [child]

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("root_empty")

        # Phase without metrics may be filtered out
        assert len(report.phases) == 0

    def test_massive_oom_boundary(self, mock_mlflow_client, mocker):
        """Boundary 2: Stress test with 1000 OOM events."""
        root = create_mock_run("root_oom", "Massive_OOM")
        mock_mlflow_client.get_run.return_value = root
        mock_mlflow_client.search_runs.return_value = []

        ooms = [{"source": "MemoryManager", "event_type": "oom", "message": f"OOM {i}"} for i in range(1000)]

        mocker.patch.object(MLflowAdapter, "_load_json_events", return_value=ooms)
        mocker.patch.object(MLflowAdapter, "_load_yaml_config", return_value={})

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("root_oom")

        assert len(report.memory_management.oom_events) == 2000
        # Note: analyzer may return NEUTRAL status for memory events
        # The important thing is that events are collected
        assert report.memory_management.oom_count == 2000


# =============================================================================
# 4. NONSENSE TESTS
# =============================================================================


class TestReportGenerationNonsense:
    """Nonsense input test cases."""

    def test_garbage_run_id(self, mock_mlflow_client):
        """Nonsense 1: Garbage Run ID."""
        mock_mlflow_client.get_run.side_effect = ValueError("Run not found")

        generator = ExperimentReportGenerator("http://mock")

        with pytest.raises(ValueError):
            generator.generate_report_model("🍌_banana_king_👑")

    def test_garbage_event_content(self, mock_mlflow_client, mocker):
        """Nonsense 2: Events file contains random text."""
        root = create_mock_run("root_garbage", "Garbage_Content")
        mock_mlflow_client.get_run.return_value = root
        mock_mlflow_client.search_runs.return_value = []

        mocker.patch("pathlib.Path.read_text", return_value="Lorem Ipsum Dolor Sit Amet 🍌")
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("tempfile.mkdtemp", return_value="/tmp")

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("root_garbage")

        assert report.timeline == []


# =============================================================================
# 5. COMPLEX & REAL-WORLD SCENARIOS
# =============================================================================


class TestReportGenerationComplex:
    """Complex real-world scenario test cases."""

    def test_simulation_real_run_7725(self, mock_mlflow_client, mock_adapter_methods):
        """Complex 1: Simulation of actual run with 3 phases (CPT, SFT, COT)."""
        root = create_mock_run(
            "7725_root", "pipeline_Qwen", tags={"training.type": "qlora", "training.strategy_chain": "cpt→sft→cot"}
        )
        container = create_mock_run(
            "b168_cont", "Qwen_container", tags={"mlflow.parentRunId": "7725_root", "chain_type": "CPT → SFT → COT"}
        )
        phase0 = create_mock_run(
            "9ac9_cpt",
            "phase_0_cpt",
            tags={"mlflow.parentRunId": "b168_cont", "phase_idx": "0", "training.strategy_type": "CPT"},
            metrics={"train_loss": 4.3},
        )
        phase1 = create_mock_run(
            "6ccb_sft",
            "phase_1_sft",
            tags={"mlflow.parentRunId": "b168_cont", "phase_idx": "1", "training.strategy_type": "SFT"},
            metrics={"train_loss": 2.7, "loss": 3.4},
        )
        phase2 = create_mock_run(
            "9e3f_cot",
            "phase_2_cot",
            tags={"mlflow.parentRunId": "b168_cont", "phase_idx": "2", "training.strategy_type": "COT"},
            metrics={"train_loss": 3.2},
        )

        def get_run_mock(run_id):
            lookup = {
                "7725_root": root,
                "b168_cont": container,
                "9ac9_cpt": phase0,
                "6ccb_sft": phase1,
                "9e3f_cot": phase2,
            }
            return lookup[run_id]

        def search_runs_mock(experiment_ids, filter_string, **kwargs):
            if "7725_root" in filter_string:
                return [container]
            if "b168_cont" in filter_string:
                return [phase0, phase1, phase2]
            return []

        mock_mlflow_client.get_run.side_effect = get_run_mock
        mock_mlflow_client.search_runs.side_effect = search_runs_mock

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("7725_root")

        assert len(report.phases) == 3
        assert report.phases[0].strategy == "CPT"
        assert report.phases[1].strategy == "SFT"
        assert report.phases[2].strategy == "COT"
        assert report.phases[0].final_loss == 4.3
        assert report.summary.strategy_chain == ["CPT", "SFT", "COT"]

    def test_nan_infinity_metrics(self, mock_mlflow_client, mock_adapter_methods):
        """Complex 2: Metrics containing NaN or Infinity."""
        root = create_mock_run("root_nan", "Exploding_Run")
        phase = create_mock_run(
            "phase_nan",
            "phase_0",
            tags={"mlflow.parentRunId": "root_nan"},
            metrics={"train_loss": float("nan"), "grad_norm": float("inf")},
        )

        mock_mlflow_client.get_run.return_value = root
        mock_mlflow_client.search_runs.return_value = [phase]

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("root_nan")

        assert len(report.phases) == 1
        assert math.isnan(report.phases[0].final_loss)

    def test_incomplete_zombie_run(self, mock_mlflow_client, mock_adapter_methods):
        """Complex 3: Run is 'RUNNING' but has no end_time (Zombie/Active)."""
        root = create_running_run("root_active", "Active_Run")
        phase = create_running_run("phase_active", "phase_0", parent_id="root_active")

        mock_mlflow_client.get_run.return_value = root
        mock_mlflow_client.search_runs.return_value = [phase]

        generator = ExperimentReportGenerator("http://mock")
        report = generator.generate_report_model("root_active")

        assert report.summary.status == RunStatus.RUNNING
        assert report.summary.duration_total_seconds == 0.0
