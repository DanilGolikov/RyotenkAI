"""Typed DTOs for training metrics snapshots and aggregates.

Shared module to avoid the circular import between
`training.orchestrator` (producer) and `training.managers.data_buffer`
(persistent store).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _coerce_scalar(value: Any) -> Any:
    """Coerce numpy/torch scalars to plain Python numbers for JSON safety."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@dataclass
class TrainingMetricsSnapshot:
    """Final metrics extracted from a TRL/HF Trainer after a training run.

    All fields are optional because different strategies/configurations emit
    different subsets. Absence is represented as ``None``.
    """

    train_loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None
    train_runtime: float | None = None
    train_samples_per_second: float | None = None
    train_steps_per_second: float | None = None
    global_step: int | None = None
    epoch: float | None = None
    peak_memory_gb: float | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization. Keeps ``None`` values for stability of the contract."""
        return asdict(self)

    def numeric_kwargs(self) -> dict[str, int | float]:
        """Non-None numeric fields only. Used to log training metrics to MLflow as params/metrics."""
        return {
            f.name: v
            for f in fields(self)
            if isinstance((v := getattr(self, f.name)), (int, float))
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrainingMetricsSnapshot:
        """Build from dict, tolerating unknown keys and partial payloads.

        Numpy/torch scalars are coerced to plain Python numbers so the
        snapshot remains JSON-serialisable.
        """
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: _coerce_scalar(v) for k, v in data.items() if k in known})


@dataclass
class PhasesMetricsAggregate:
    """Aggregated metrics across multiple training phases."""

    total_phases: int = 0
    total_steps: int = 0
    total_runtime_seconds: float = 0.0
    final_loss: float | None = None
    per_phase: list[TrainingMetricsSnapshot] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.total_phases == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_phases": self.total_phases,
            "total_steps": self.total_steps,
            "total_runtime_seconds": self.total_runtime_seconds,
            "final_loss": self.final_loss,
            "per_phase": [p.to_dict() for p in self.per_phase],
        }


__all__ = ["TrainingMetricsSnapshot", "PhasesMetricsAggregate"]
