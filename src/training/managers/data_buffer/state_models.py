"""
State model dataclasses and enums for DataBuffer.

Contains the pure data layer: PhaseStatus, PhaseState, PipelineState.
These objects have no dependency on DataBuffer itself.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.training.managers.constants import (
    KEY_COMPLETED_AT,
    KEY_PHASES,
    KEY_STARTED_AT,
    KEY_STATUS,
)
from src.training.metrics_models import TrainingMetricsSnapshot
from src.utils.logger import logger


class PhaseStatus(Enum):
    """Status of a training phase."""

    PENDING = "pending"  # Not started yet
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    SKIPPED = "skipped"  # Skipped (e.g., resume from later phase)
    INTERRUPTED = "interrupted"  # Interrupted by SIGINT/SIGTERM


@dataclass
class PhaseState:
    """
    State of a single training phase.

    Tracks execution status, paths, and metrics for one phase
    in the multi-phase training pipeline.
    """

    phase_idx: int
    strategy_type: str
    status: PhaseStatus = PhaseStatus.PENDING
    output_dir: str | None = None
    checkpoint_path: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metrics: TrainingMetricsSnapshot = field(default_factory=TrainingMetricsSnapshot)

    # Config that was used for this phase
    epochs: int = 1
    learning_rate: float | None = None
    dataset_name: str | None = None

    # Adapter cache state (populated when adapter_cache.enabled=true)
    adapter_cache_hit: bool = False
    adapter_cache_tag: str | None = None
    adapter_cache_upload_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data[KEY_STATUS] = self.status.value
        data["metrics"] = self.metrics.to_dict()
        # Convert datetime to ISO string
        if self.started_at:
            data[KEY_STARTED_AT] = self.started_at.isoformat()
        if self.completed_at:
            data[KEY_COMPLETED_AT] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseState:
        """Create from dictionary (JSON deserialization).

        Handles forward/backward compatibility by filtering unknown fields.
        Accepts either a ``TrainingMetricsSnapshot`` or a legacy dict payload
        for ``metrics``.
        """
        # FIX BUG-001: Filter only known fields to handle version migrations
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        # Warn about unknown fields (helps debugging version mismatches)
        unknown_fields = set(data.keys()) - known_fields
        if unknown_fields:
            logger.debug(f"[DB:COMPAT] Ignoring unknown fields in PhaseState: {unknown_fields}")

        # Convert status string to enum
        if KEY_STATUS in filtered_data:
            try:
                filtered_data[KEY_STATUS] = PhaseStatus(filtered_data[KEY_STATUS])
            except ValueError:
                logger.warning(f"Unknown status '{filtered_data[KEY_STATUS]}', defaulting to PENDING")
                filtered_data[KEY_STATUS] = PhaseStatus.PENDING

        # Convert ISO strings to datetime
        if filtered_data.get(KEY_STARTED_AT):
            filtered_data[KEY_STARTED_AT] = datetime.fromisoformat(filtered_data[KEY_STARTED_AT])
        if filtered_data.get(KEY_COMPLETED_AT):
            filtered_data[KEY_COMPLETED_AT] = datetime.fromisoformat(filtered_data[KEY_COMPLETED_AT])

        # Accept legacy dict or an already-typed snapshot
        raw_metrics = filtered_data.get("metrics")
        if isinstance(raw_metrics, TrainingMetricsSnapshot):
            pass
        elif isinstance(raw_metrics, dict):
            filtered_data["metrics"] = TrainingMetricsSnapshot.from_dict(raw_metrics)
        elif raw_metrics is None:
            filtered_data.pop("metrics", None)
        else:
            logger.warning(
                f"[DB:COMPAT] Unexpected metrics type {type(raw_metrics).__name__}, resetting to empty snapshot"
            )
            filtered_data["metrics"] = TrainingMetricsSnapshot()

        return cls(**filtered_data)

    @property
    def is_complete(self) -> bool:
        """Check if phase is successfully completed."""
        return self.status == PhaseStatus.COMPLETED

    @property
    def is_running(self) -> bool:
        """Check if phase is currently running."""
        return self.status == PhaseStatus.RUNNING

    @property
    def duration_seconds(self) -> float | None:
        """Get phase duration in seconds (if completed)."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class PipelineState:
    """
    Overall state of the multi-phase training pipeline.

    Tracks all phases and provides resume/recovery capabilities.
    """

    run_id: str
    base_output_dir: str
    base_model_path: str
    total_phases: int
    current_phase: int = 0
    phases: list[PhaseState] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str = "pending"  # pending, running, completed, failed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "base_output_dir": self.base_output_dir,
            "base_model_path": self.base_model_path,
            "total_phases": self.total_phases,
            "current_phase": self.current_phase,
            KEY_PHASES: [p.to_dict() for p in self.phases],
            KEY_STARTED_AT: self.started_at.isoformat() if self.started_at else None,
            KEY_COMPLETED_AT: (self.completed_at.isoformat() if self.completed_at else None),
            KEY_STATUS: self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        """Create from dictionary (JSON deserialization).

        Handles forward/backward compatibility and corrupted data.
        """
        # FIX BUG-003: Handle null/None phases gracefully
        phases_data = data.pop(KEY_PHASES, [])
        if phases_data is None:
            logger.warning("[DB:COMPAT] 'phases' is null in state file, using empty list")
            phases_data = []

        phases = [PhaseState.from_dict(p) for p in phases_data]

        # FIX BUG-001: Filter only known fields
        known_fields = {f.name for f in dataclasses.fields(cls)} - {KEY_PHASES}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        unknown_fields = set(data.keys()) - known_fields
        if unknown_fields:
            logger.debug(f"[DB:COMPAT] Ignoring unknown fields in PipelineState: {unknown_fields}")

        if filtered_data.get(KEY_STARTED_AT):
            filtered_data[KEY_STARTED_AT] = datetime.fromisoformat(filtered_data[KEY_STARTED_AT])
        if filtered_data.get(KEY_COMPLETED_AT):
            filtered_data[KEY_COMPLETED_AT] = datetime.fromisoformat(filtered_data[KEY_COMPLETED_AT])

        return cls(**filtered_data, phases=phases)

    @property
    def completed_phases(self) -> list[PhaseState]:
        """Get list of completed phases."""
        return [p for p in self.phases if p.is_complete]

    @property
    def failed_phases(self) -> list[PhaseState]:
        """Get list of failed phases."""
        return [p for p in self.phases if p.status == PhaseStatus.FAILED]

    @property
    def progress_percent(self) -> float:
        """Get pipeline progress as percentage."""
        if self.total_phases == 0:
            return 0.0
        return (len(self.completed_phases) / self.total_phases) * 100


__all__ = ["PhaseStatus", "PhaseState", "PipelineState"]
