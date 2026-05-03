"""
Experiment Report Models.

Layer 2: Processed data ready for rendering.
All fields are typed and have sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

# Re-export domain entities to avoid duplication
from src.reports.domain.entities import (
    EvaluationReport,
    MemoryEvent,
    MetricTrend,
    RunStatus,
    ValidationReport,
)

if TYPE_CHECKING:
    from datetime import datetime


class ExperimentHealth(str, Enum):
    """Experiment health status."""

    GREEN = "green"  # All good
    YELLOW = "yellow"  # Some warnings (e.g., loss increased)
    RED = "red"  # Failed or critical issues

    @property
    def emoji(self) -> str:
        """Get emoji for health."""
        return {
            ExperimentHealth.GREEN: "🟢",
            ExperimentHealth.YELLOW: "🟡",
            ExperimentHealth.RED: "🔴",
        }.get(self, "⚪")


class MetricStatus(str, Enum):
    """Status of a specific metric analysis."""

    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    NEUTRAL = "neutral"

    @property
    def emoji(self) -> str:
        return {
            MetricStatus.GOOD: "✅",
            MetricStatus.WARNING: "⚠️",
            MetricStatus.BAD: "❌",
            MetricStatus.NEUTRAL: "i",
        }.get(self, "i")


@dataclass
class PercentileStats:
    """
    Statistical summary with percentiles.

    Used for GPU/VRAM/CPU utilization.
    """

    avg: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    p95: float | None = None
    p99: float | None = None
    data_points: int = 0

    def __bool__(self) -> bool:
        """True if has data."""
        return self.data_points > 0


@dataclass
class MetricAnalysis:
    """
    Detailed analysis of a specific training metric.

    Contains interpretation of the raw numbers.
    """

    name: str  # e.g., "train_loss"
    display_name: str  # e.g., "Training Loss"
    description: str  # Short explanation
    trend: MetricTrend
    status: MetricStatus
    verdict: str  # e.g., "Smooth convergence" or "High volatility"


@dataclass
class ReportSummary:
    """High-level summary of the experiment."""

    experiment_name: str
    run_name: str
    run_id: str
    status: RunStatus
    health: ExperimentHealth
    health_explanation: str  # NEW: reason for status (e.g. "RED: Run failed; 2 ERROR issues")
    duration_total_seconds: float
    # Time
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_formatted: str = "—"
    # Stats
    total_strategies: int = 0
    strategy_chain: list[str] = field(default_factory=list)
    # Aggregated metrics (e.g. from last phase)
    final_loss: float | None = None
    avg_loss_reduction_pct: float | None = None
    total_epochs: int = 0
    total_steps: int = 0
    samples_per_second: float | None = None
    steps_per_second: float | None = None


@dataclass
class PhaseInfo:
    """Detailed information about a training phase (strategy)."""

    phase_idx: int
    run_id: str
    run_name: str
    strategy: str
    status: RunStatus
    duration_seconds: float

    # Core Training Metrics
    final_loss: float | None = None
    loss_trend: MetricTrend = field(default_factory=MetricTrend)
    train_loss_trend: MetricTrend = field(default_factory=MetricTrend)
    entropy_trend: MetricTrend = field(default_factory=MetricTrend)
    accuracy_trend: MetricTrend = field(default_factory=MetricTrend)
    grad_norm_trend: MetricTrend = field(default_factory=MetricTrend)

    # NEW: Detailed Analysis Table
    metrics_analysis: list[MetricAnalysis] = field(default_factory=list)

    # Progress
    epochs: float | None = None
    steps: int | None = None
    total_steps: int | None = None
    samples_per_second: float | None = None

    # Resources (Averages)
    gpu_utilization: float | None = None
    gpu_memory_mb: float | None = None
    gpu_memory_percent: float | None = None
    cpu_utilization: float | None = None
    system_memory_mb: float | None = None
    system_memory_percent: float | None = None

    # Effective Configuration (from training.hyperparams.actual.*)
    effective_config: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Get formatted phase name."""
        return f"Phase {self.phase_idx}: {self.strategy.upper()}"


@dataclass
class ResourcesInfo:
    """Resource utilization stats (from system metrics)."""

    # GPU Info
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    gpu_tier: str | None = None
    # Utilization stats
    gpu_utilization: PercentileStats = field(default_factory=PercentileStats)
    gpu_memory_mb: PercentileStats = field(default_factory=PercentileStats)
    gpu_memory_percent: PercentileStats = field(default_factory=PercentileStats)
    cpu_utilization: PercentileStats = field(default_factory=PercentileStats)
    system_memory_mb: PercentileStats = field(default_factory=PercentileStats)
    system_memory_percent: PercentileStats = field(default_factory=PercentileStats)


@dataclass
class TimelineEvent:
    """Significant event during pipeline execution."""

    timestamp: datetime | None = None
    event_type: str = ""
    message: str = ""
    source: str = ""
    origin: str = "unknown"  # pipeline, training
    category: str = ""
    severity: str = "INFO"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about the model configuration."""

    name: str = ""
    training_type: str = "unknown"
    total_parameters: int | None = None
    trainable_parameters: int | None = None
    trainable_percent: float | None = None
    loading_time_seconds: float | None = None
    model_size_mb: float | None = None
    adapter_path: str | None = None
    quantization: str | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    target_modules: list[str] | None = None


@dataclass
class ConfigInfo:
    """Training configuration parameters."""

    batch_size: int | None = None
    grad_accum: int | None = None
    learning_rate: float | None = None
    num_epochs: int | None = None
    max_steps: int | None = None
    optimizer: str | None = None
    scheduler: str | None = None
    warmup_ratio: float | None = None
    weight_decay: float | None = None
    max_seq_length: int | None = None
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    full_config: dict[str, Any] = field(default_factory=dict)
    source_config: dict[str, Any] | None = None
    params_config: dict[str, str] | None = None


@dataclass
class MemoryPhaseStats:
    """Memory statistics for a training phase."""

    phase_idx: int
    strategy: str
    peak_mb: int
    peak_percent: float
    avg_mb: int
    avg_percent: float
    p95_mb: int
    p99_mb: int


@dataclass
class MemoryAnalysis:
    """Analysis result for memory health."""

    status: MetricStatus
    verdict: str  # Short summary (e.g. "Critical Fragmentation")
    insights: list[str]  # Detailed explanations
    recommendations: list[str]  # Actionable fixes
    overhead_seconds: float  # Time lost to memory management
    fragmentation_warnings: int  # Count of high fragmentation events
    oom_count: int
    efficiency_score: int  # 0-100 score


@dataclass
class MemoryManagementInfo:
    """MemoryManager configuration and events."""

    # GPU Configuration
    gpu_name: str | None = None
    gpu_tier: str | None = None
    total_vram_gb: float | None = None
    memory_margin_mb: int | None = None
    critical_threshold: float | None = None
    warning_threshold: float | None = None
    max_retries: int | None = None

    # Recommendations
    max_model: str | None = None
    notes: str | None = None
    actual_model_size: str | None = None  # Actual model size from config (e.g., "0.5B")

    # Config Comparison Warnings (from events)
    config_warnings: list[str] = field(default_factory=list)

    # Memory Events
    cache_clears: list[MemoryEvent] = field(default_factory=list)
    oom_events: list[MemoryEvent] = field(default_factory=list)
    memory_warnings: list[MemoryEvent] = field(default_factory=list)

    # Memory Timeline by Phase
    phase_stats: list[MemoryPhaseStats] = field(default_factory=list)

    # NEW: Structured Analysis (optional, filled by builder)
    analysis: MemoryAnalysis | None = None

    @property
    def total_cache_cleared_mb(self) -> int:
        """Total MB freed by cache clears."""
        return sum((e.freed_mb if e.freed_mb is not None else 0) for e in self.cache_clears)

    @property
    def oom_count(self) -> int:
        """Number of OOM events."""
        return len(self.oom_events)

    @property
    def warning_count(self) -> int:
        """Number of memory warnings."""
        return len(self.memory_warnings)


@dataclass
class Issue:
    """Detected issue or warning."""

    severity: str  # ERROR, WARN, INFO
    message: str
    context: str | None = None  # e.g. "Phase 1"


@dataclass
class ExperimentReport:
    """
    Root Report Object.

    Aggregates all sections of the report.
    """

    generated_at: datetime
    summary: ReportSummary
    model: ModelInfo
    config: ConfigInfo
    phases: list[PhaseInfo]
    resources: ResourcesInfo
    timeline: list[TimelineEvent]
    issues: list[Issue]
    memory_management: MemoryManagementInfo | None = None
    validation: ValidationReport | None = None  # Dataset Validation info
    evaluation: EvaluationReport | None = None  # Model Evaluation info
