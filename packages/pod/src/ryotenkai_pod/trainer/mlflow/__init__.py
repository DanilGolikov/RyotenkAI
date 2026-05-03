"""
src/training/mlflow — MLflow subcomponent library.

Exports:
  - Subcomponent classes extracted from the MLflowManager God Object refactoring.
  - IMLflowManager Protocol (canonical home; re-exported from src/utils/container.py
    for backward compatibility so existing consumers don't need to change imports).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.training.mlflow.autolog import MLflowAutologManager
from src.training.mlflow.dataset_logger import MLflowDatasetLogger
from src.training.mlflow.domain_logger import MLflowDomainLogger
from src.training.mlflow.event_log import MLflowEventLog
from src.training.mlflow.model_registry import MLflowModelRegistry
from src.training.mlflow.primitives import IMLflowPrimitives
from src.training.mlflow.run_analytics import MLflowRunAnalytics


@runtime_checkable
class IMLflowManager(Protocol):
    """
    Interface for MLflow experiment tracking.

    Canonical home: src/training/mlflow/__init__.py
    Re-exported from src/utils/container.py for backward compatibility.

    Allows mocking MLflow in tests without actual server.
    Supports nested runs for multi-phase training.
    """

    @property
    def is_active(self) -> bool:
        """Check if MLflow is operational (setup succeeded)."""
        ...

    @property
    def run_id(self) -> str | None:
        """Get current run ID."""
        ...

    @property
    def parent_run_id(self) -> str | None:
        """Get parent run ID (for nested runs)."""
        ...

    def setup(self, timeout: float = 5.0, max_retries: int = 3, disable_system_metrics: bool = False) -> bool:
        """Initialize MLflow connection."""
        ...

    def start_run(self, run_name: str | None = None, description: str | None = None) -> Any:
        """Start a parent MLflow run (context manager)."""
        ...

    def start_nested_run(self, run_name: str, tags: dict[str, str] | None = None) -> Any:
        """Start a nested (child) run (context manager)."""
        ...

    def get_runtime_tracking_uri(self) -> str:
        """Get tracking URI used by the current runtime role."""
        ...

    def delete_run_tree(self, root_run_id: str) -> list[str]:
        """Soft-delete a root MLflow run and all nested descendants."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to current run."""
        ...

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on current run."""
        ...

    def log_dataset_info(
        self,
        name: str,
        path: str | None = None,
        source: str | None = None,
        version: str | None = None,
        num_rows: int = 0,
        num_samples: int | None = None,
        num_features: int | None = None,
        context: str = "training",
        extra_info: dict[str, Any] | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        """Log dataset metadata."""
        ...

    def create_mlflow_dataset(
        self,
        data: Any,
        name: str,
        source: str,
        targets: str | None = None,
    ) -> Any:
        """Create MLflow Dataset object from data."""
        ...

    def log_dataset_input(
        self,
        dataset: Any,
        context: str = "training",
    ) -> bool:
        """Link dataset to current MLflow run."""
        ...

    # Event Logging Methods
    def log_event(
        self,
        event_type: str,
        message: str,
        *,
        category: str = "info",
        source: str = "system",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Log a generic event."""
        ...

    def log_event_start(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log start event."""
        ...

    def log_event_complete(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log completion event."""
        ...

    def log_event_error(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log error event."""
        ...

    def log_event_warning(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log warning event."""
        ...

    def log_event_info(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log info event."""
        ...

    def log_event_checkpoint(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log checkpoint event."""
        ...

    def log_pipeline_initialized(self, run_id: str, total_phases: int, strategy_chain: list[str]) -> None:
        """Log pipeline initialization."""
        ...

    def log_state_saved(self, run_id: str, path: str) -> None:
        """Log state saved event."""
        ...

    def log_checkpoint_cleanup(self, cleaned_count: int, freed_mb: int) -> None:
        """Log checkpoint cleanup event."""
        ...

    # Memory Logging Methods
    def log_gpu_detection(self, name: str, vram_gb: float, tier: str) -> None:
        """Log GPU detection event."""
        ...

    def log_cache_cleared(self, freed_mb: int) -> None:
        """Log cache cleared event."""
        ...

    def log_memory_warning(
        self,
        utilization_percent: float,
        used_mb: int,
        total_mb: int,
        is_critical: bool,
    ) -> None:
        """Log memory warning event."""
        ...

    def log_oom(self, operation: str, free_mb: int | None) -> None:
        """Log OOM event."""
        ...

    def log_oom_recovery(self, operation: str, attempt: int, max_attempts: int) -> None:
        """Log OOM recovery attempt."""
        ...


__all__ = [
    "IMLflowManager",
    "IMLflowPrimitives",
    "MLflowAutologManager",
    "MLflowDatasetLogger",
    "MLflowDomainLogger",
    "MLflowEventLog",
    "MLflowModelRegistry",
    "MLflowRunAnalytics",
]
