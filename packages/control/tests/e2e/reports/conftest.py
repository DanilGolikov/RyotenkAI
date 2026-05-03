"""
Fixtures for e2e report tests.

Builds ExperimentData mocks for different scenarios.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.pipeline.artifacts.base import StageArtifactEnvelope
from src.reports.domain.entities import (
    ExperimentData,
    MemoryEvent,
    MetricHistory,
    PhaseData,
    RunStatus,
)


@pytest.fixture
def reports_output_dir() -> Path:
    """
    Directory for saved reports.
    Uses runs/tests_report/ at repo root so that:
    - reports are easy to inspect after a test run
    - runs/ is not cluttered with loose files

    Example: .../ryotenkai/runs/tests_report
    """
    # Repo root: .../ryotenkai (conftest lives under src/tests/e2e/reports/)
    repo_root = Path(__file__).resolve().parents[4]
    output_dir = repo_root / "runs" / "tests_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def base_timestamp():
    """Base time for fixtures."""
    return datetime(2025, 1, 15, 10, 0, 0)


def create_metric_history(
    key: str,
    values: list[float],
    start_step: int = 0,
    base_ts: datetime | None = None,
) -> MetricHistory:
    """Build metric history."""
    if base_ts is None:
        base_ts = datetime.now()

    steps = list(range(start_step, start_step + len(values)))
    timestamps = [int((base_ts + timedelta(seconds=i * 10)).timestamp() * 1000) for i in range(len(values))]

    return MetricHistory(key=key, values=values, steps=steps, timestamps=timestamps)


@pytest.fixture
def positive_experiment_data(base_timestamp):
    """
    POSITIVE: successful run, healthy (GREEN).
    - RunStatus.FINISHED
    - Loss decreases
    - No WARN/ERROR
    """
    start_time = base_timestamp
    end_time = start_time + timedelta(minutes=30)

    phase = PhaseData(
        idx=0,
        name="test-phase-0",
        strategy="SFT",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=1800,
        config={
            "learning_rate": 0.0001,
            "batch_size": 8,
            "grad_accum": 2,
        },
        metrics={
            "train_loss": 0.5,
            "epoch": 3.0,
            "global_step": 100,
            "train_samples_per_second": 12.5,
        },
        history={
            "train_loss": create_metric_history("train_loss", [1.5, 1.2, 0.8, 0.5], base_ts=start_time),
            "loss": create_metric_history("loss", [1.5, 1.2, 0.8, 0.5], base_ts=start_time),
        },
    )

    return ExperimentData(
        run_id="test-run-positive-001",
        run_name="positive_test_run",
        experiment_name="e2e_test",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=1800,
        phases=[phase],
        stage_envelopes=[],
        memory_events=[],
        source_config={"model": {"name": "test-model"}, "training": {"learning_rate": 0.0001}},
        root_params={"model_name": "test-model"},
        gpu_info={"name": "NVIDIA RTX 4090", "tier": "consumer_high", "total_vram_gb_raw": 24.0},
        resource_history={},
    )


@pytest.fixture
def negative_experiment_data(base_timestamp):
    """
    NEGATIVE: failed run, many errors (RED).
    - RunStatus.FAILED
    - 2+ ERROR events
    - Loss increases
    """
    start_time = base_timestamp
    end_time = start_time + timedelta(minutes=10)

    phase = PhaseData(
        idx=0,
        name="test-phase-0",
        strategy="SFT",
        status=RunStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=600,
        config={},
        metrics={"train_loss": 2.5, "epoch": 1.0, "global_step": 20},
        history={
            "train_loss": create_metric_history("train_loss", [0.5, 1.2, 1.8, 2.5], base_ts=start_time),
        },
    )

    return ExperimentData(
        run_id="test-run-negative-002",
        run_name="negative_test_run",
        experiment_name="e2e_test",
        status=RunStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=600,
        phases=[phase],
        stage_envelopes=[
            StageArtifactEnvelope(
                stage="GPU Deployer",
                status="failed",
                started_at=(start_time + timedelta(minutes=5)).isoformat(),
                duration_seconds=300.0,
                error="CUDA OOM during GPU deployment",
                data={},
            ),
            StageArtifactEnvelope(
                stage="Training Monitor",
                status="failed",
                started_at=(start_time + timedelta(minutes=8)).isoformat(),
                duration_seconds=120.0,
                error="Training failed: loss divergence",
                data={},
            ),
        ],
        memory_events=[
            MemoryEvent(
                timestamp=start_time + timedelta(minutes=5),
                event_type="oom",
                message="OOM during forward pass",
                operation="forward",
            ),
        ],
        source_config={},
        root_params={},
        gpu_info={},
        resource_history={},
    )


@pytest.fixture
def boundary_3_warnings_data(base_timestamp):
    """
    BOUNDARY: exactly 3 WARN (YELLOW).
    - RunStatus.FINISHED
    - Exactly 3 WARN events
    """
    start_time = base_timestamp
    end_time = start_time + timedelta(minutes=20)

    phase = PhaseData(
        idx=0,
        name="test-phase-0",
        strategy="CPT",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=1200,
        config={},
        metrics={"train_loss": 1.0},
        history={},
    )

    return ExperimentData(
        run_id="test-run-boundary-3warn-003",
        run_name="boundary_3_warnings",
        experiment_name="e2e_test",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=1200,
        phases=[phase],
        stage_envelopes=[
            StageArtifactEnvelope(
                stage="Dataset Validator",
                status="interrupted",
                started_at=start_time.isoformat(),
                duration_seconds=120.0,
                error=None,
                data={},
            ),
            StageArtifactEnvelope(
                stage="Inference Deployer",
                status="interrupted",
                started_at=(start_time + timedelta(minutes=5)).isoformat(),
                duration_seconds=60.0,
                error=None,
                data={},
            ),
            StageArtifactEnvelope(
                stage="Model Evaluator",
                status="interrupted",
                started_at=(start_time + timedelta(minutes=10)).isoformat(),
                duration_seconds=30.0,
                error=None,
                data={},
            ),
        ],
        missing_artifacts=[],
        memory_events=[],
        source_config={},
        root_params={},
        gpu_info={},
        resource_history={},
    )


@pytest.fixture
def boundary_5_warnings_data(base_timestamp):
    """
    BOUNDARY: exactly 5 WARN (RED).
    - RunStatus.FINISHED
    - Exactly 5 WARN events
    """
    start_time = base_timestamp
    end_time = start_time + timedelta(minutes=20)

    phase = PhaseData(
        idx=0,
        name="test-phase-0",
        strategy="DPO",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=1200,
        config={},
        metrics={"train_loss": 0.8},
        history={},
    )

    interrupted_envelopes = [
        StageArtifactEnvelope(
            stage=f"Stage {i + 1}",
            status="interrupted",
            started_at=(start_time + timedelta(minutes=i * 3)).isoformat(),
            duration_seconds=60.0,
            error=None,
            data={},
        )
        for i in range(5)
    ]

    return ExperimentData(
        run_id="test-run-boundary-5warn-004",
        run_name="boundary_5_warnings",
        experiment_name="e2e_test",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=1200,
        phases=[phase],
        stage_envelopes=interrupted_envelopes,
        missing_artifacts=[],
        memory_events=[],
        source_config={},
        root_params={},
        gpu_info={},
        resource_history={},
    )


@pytest.fixture
def crazy_empty_data(base_timestamp):
    """
    EDGE: almost empty data.
    - Minimal fields
    - Empty lists
    - None values where allowed
    """
    return ExperimentData(
        run_id="test-run-crazy-empty-005",
        run_name="crazy_empty_run",
        experiment_name="e2e_test",
        status=RunStatus.UNKNOWN,
        start_time=None,
        end_time=None,
        duration_seconds=0,
        phases=[],
        stage_envelopes=[],
        memory_events=[],
        source_config={},
        root_params={},
        gpu_info={},
        resource_history={},
    )


@pytest.fixture
def crazy_missing_fields_data(base_timestamp):
    """
    EDGE: missing required-like fields in PhaseData.
    - Phase without metric history
    - Empty config
    """
    start_time = base_timestamp

    phase = PhaseData(
        idx=0,
        name="",
        strategy="",
        status=RunStatus.UNKNOWN,
        start_time=None,
        end_time=None,
        duration_seconds=0,
        config={},
        metrics={},
        history={},
    )

    return ExperimentData(
        run_id="test-run-crazy-missing-006",
        run_name="",
        experiment_name="",
        status=RunStatus.UNKNOWN,
        start_time=start_time,
        end_time=start_time,
        duration_seconds=0,
        phases=[phase],
        stage_envelopes=[],
        memory_events=[],
        source_config={},
        root_params={},
        gpu_info={},
        resource_history={},
    )


@pytest.fixture
def crazy_mixed_severities_data(base_timestamp):
    """
    EDGE: mixed severities, 2 WARN + 10 INFO (should stay GREEN).
    - INFO must not affect health
    """
    start_time = base_timestamp
    end_time = start_time + timedelta(minutes=15)

    phase = PhaseData(
        idx=0,
        name="test-phase-0",
        strategy="RL",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=900,
        config={},
        metrics={"train_loss": 1.2},
        history={},
    )

    return ExperimentData(
        run_id="test-run-crazy-mixed-007",
        run_name="crazy_mixed_severities",
        experiment_name="e2e_test",
        status=RunStatus.FINISHED,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=900,
        phases=[phase],
        stage_envelopes=[
            StageArtifactEnvelope(
                stage="Dataset Validator",
                status="interrupted",
                started_at=(start_time + timedelta(minutes=10)).isoformat(),
                duration_seconds=30.0,
                error=None,
                data={},
            ),
            StageArtifactEnvelope(
                stage="Inference Deployer",
                status="interrupted",
                started_at=(start_time + timedelta(minutes=12)).isoformat(),
                duration_seconds=30.0,
                error=None,
                data={},
            ),
        ],
        memory_events=[],
        source_config={},
        root_params={},
        gpu_info={},
        resource_history={},
    )
