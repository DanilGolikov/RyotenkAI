"""
Domain Entities.

Pure data structures representing experiment data.
Independent of source (MLflow) and view (Report).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

    from src.pipeline.artifacts import StageArtifactEnvelope

# =============================================================================
# ENUMS
# =============================================================================


class RunStatus(str, Enum):
    """Run status enum."""

    FINISHED = "FINISHED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"
    KILLED = "KILLED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_success(self) -> bool:
        return self == RunStatus.FINISHED

    @property
    def emoji(self) -> str:
        """Get emoji for status."""
        return {
            RunStatus.FINISHED: "✅",
            RunStatus.FAILED: "❌",
            RunStatus.RUNNING: "🔄",
            RunStatus.KILLED: "⛔",
            RunStatus.UNKNOWN: "❓",
        }.get(self, "❓")


class GPUTier(str, Enum):
    """GPU tier classification."""

    MINIMAL = "minimal"
    CONSUMER_LOW = "consumer_low"
    CONSUMER_MID = "consumer_mid"
    CONSUMER_HIGH = "consumer_high"
    PROFESSIONAL = "professional"
    DATACENTER = "datacenter"
    UNKNOWN = "unknown"


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class MetricHistory:
    """Raw history of a metric."""

    key: str
    values: list[float]
    steps: list[int]
    timestamps: list[int]  # Unix timestamps (ms)

    @property
    def first(self) -> float | None:
        return self.values[0] if self.values else None

    @property
    def last(self) -> float | None:
        return self.values[-1] if self.values else None

    @property
    def min_value(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def max_value(self) -> float | None:
        return max(self.values) if self.values else None

    @property
    def count(self) -> int:
        return len(self.values)


@dataclass
class MetricTrend:
    """Calculated trend for a metric."""

    first: float | None = None
    last: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    change_pct: float | None = None
    direction: str = "unknown"  # decreased, increased, stable, unknown
    data_points: int = 0

    @property
    def improved(self) -> bool:
        return self.direction == "decreased"  # Usually used for loss


# =============================================================================
# MEMORY
# =============================================================================


@dataclass
class MemoryEvent:
    """Memory management event."""

    timestamp: datetime
    event_type: str  # "cache_clear", "oom", "warning", "critical"
    message: str
    source: str = "MemoryManager"
    phase: str | None = None
    freed_mb: int | None = None
    utilization_percent: float | None = None
    operation: str | None = None


@dataclass
class MemoryStats:
    """Snapshot of memory state."""

    used_mb: int
    total_mb: int
    utilization_percent: float
    fragmentation_ratio: float = 0.0


# =============================================================================
# VALIDATION
# =============================================================================


@dataclass
class ValidationPluginResults:
    """Result of a single validation plugin execution."""

    id: str
    plugin_name: str
    status: str  # passed, failed
    duration_ms: float
    description: str = ""  # Plugin description
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)  # Plugin configuration params
    thresholds: dict[str, Any] = field(default_factory=dict)  # Plugin pass/fail criteria
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)  # Recommendations (if any)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass
class DatasetValidation:
    """Validation report for a single dataset."""

    dataset_name: str
    dataset_path: str | None
    sample_count: int | None
    status: str  # passed, failed
    total_plugins: int
    passed_plugins: int
    failed_plugins: int
    critical_failures: int = 0  # Threshold for critical failures (0 = never critical)
    plugin_results: list[ValidationPluginResults] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def has_partial_failure(self) -> bool:
        """Check if dataset has failures but below critical threshold."""
        return 0 < self.failed_plugins < self.critical_failures


@dataclass
class ValidationReport:
    """Aggregated validation report for all datasets."""

    datasets: list[DatasetValidation] = field(default_factory=list)

    @property
    def total_datasets(self) -> int:
        return len(self.datasets)

    @property
    def passed_datasets(self) -> int:
        return sum(1 for d in self.datasets if d.passed)

    @property
    def failed_datasets(self) -> int:
        return self.total_datasets - self.passed_datasets

    @property
    def overall_status(self) -> str:
        """Overall status across all datasets."""
        if not self.datasets:
            return "unknown"
        return "passed" if all(d.passed for d in self.datasets) else "failed"


# =============================================================================
# EVALUATION
# =============================================================================


@dataclass
class EvalPluginResult:
    """Result of a single evaluation plugin (builder domain entity)."""

    name: str
    passed: bool
    plugin_name: str = ""
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    sample_count: int = 0
    failed_samples: int = 0


@dataclass
class EvaluationReport:
    """Aggregated evaluation report (builder domain entity)."""

    overall_passed: bool
    sample_count: int
    duration_seconds: float
    skipped_plugins: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    plugins: list[EvalPluginResult] = field(default_factory=list)


# =============================================================================
# EXPERIMENT STRUCTURE
# =============================================================================


@dataclass
class PhaseData:
    """
    Data for a single training phase.
    Contains already normalized keys (e.g., 'learning_rate' instead of 'training.hyperparams...').
    """

    idx: int
    name: str  # e.g., "sft", "dpo"
    strategy: str  # "SFT", "DPO"
    status: RunStatus
    duration_seconds: float
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Normalized Config
    config: dict[str, Any] = field(default_factory=dict)

    # Normalized Metrics (Scalar final values)
    metrics: dict[str, float] = field(default_factory=dict)

    # Metric Histories (Time series)
    history: dict[str, MetricHistory] = field(default_factory=dict)


@dataclass
class ExperimentData:
    """
    Complete data for an experiment.
    The "Clean Plate" for the ReportBuilder.
    """

    run_id: str
    run_name: str
    experiment_name: str
    status: RunStatus
    start_time: datetime | None
    end_time: datetime | None
    duration_seconds: float

    # Structure
    phases: list[PhaseData] = field(default_factory=list)

    # Global Artifacts
    source_config: dict[str, Any] = field(default_factory=dict)  # raw yaml
    root_params: dict[str, Any] = field(default_factory=dict)  # MLflow Root Params

    # Events (aggregated from training logs only — pipeline events moved to stage_envelopes)
    memory_events: list[MemoryEvent] = field(default_factory=list)

    # Stage artifacts: one StageArtifactEnvelope per pipeline stage
    # Populated by MLflowAdapter from *_results.json files
    stage_envelopes: list[StageArtifactEnvelope] = field(default_factory=list)

    # Parsed data payloads extracted from stage envelopes (typed views for builder)
    validation_results: dict[str, Any] | None = None  # ValidationArtifactData
    evaluation_results: dict[str, Any] | None = None  # EvalArtifactData
    deployment_results: dict[str, Any] | None = None  # DeploymentArtifactData
    training_stage_results: dict[str, Any] | None = None  # TrainingArtifactData
    model_results: dict[str, Any] | None = None  # ModelArtifactData
    inference_results: dict[str, Any] | None = None  # InferenceArtifactData

    # Hardware (detected)
    gpu_info: dict[str, Any] = field(default_factory=dict)  # name, vram, count

    # Global Resource History (from Training Run)
    resource_history: dict[str, MetricHistory] = field(default_factory=dict)

    # Metadata & Warnings
    missing_artifacts: list[str] = field(default_factory=list)  # e.g., ["training_events.json"]
