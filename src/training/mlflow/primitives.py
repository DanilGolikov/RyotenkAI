"""
IMLflowPrimitives — internal Protocol for MLflow logging primitives.

Used by MLflowDomainLogger and MLflowDatasetLogger to depend on an
abstraction rather than the concrete MLflowManager.  This keeps the
subcomponents testable in isolation.
"""

from __future__ import annotations

from typing import Any, Protocol


class IMLflowPrimitives(Protocol):
    """
    Protocol for basic MLflow logging primitives.

    Implemented by MLflowManager (the facade).
    Used by MLflowDomainLogger and MLflowDatasetLogger as their only
    dependency on the outer manager — satisfying Dependency Inversion.
    """

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to current run."""
        ...

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on current run."""
        ...

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str, run_id: str | None = None) -> bool:
        """Log dict as JSON artifact."""
        ...


__all__ = ["IMLflowPrimitives"]
