"""
Value objects and callback contracts for ModelRetriever.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


# Keys in DataBuffer `pipeline_state.json` used for README generation.
_PIPELINE_STATE_STARTED_AT = "started_at"
_PIPELINE_STATE_COMPLETED_AT = "completed_at"
_PHASE_IDX = "phase_idx"
_STRATEGY_TYPE = "strategy_type"
_STATUS = "status"
_METRICS = "metrics"


@dataclass
class ModelRetrieverEventCallbacks:
    """
    Callbacks for ModelRetriever events (SOLID-compliant event collection).

    Used to integrate ModelRetriever with MLflow or other logging systems.
    """

    on_hf_upload_started: Callable[[str], None] | None = None
    on_hf_upload_completed: Callable[[str, float], None] | None = None
    on_hf_upload_failed: Callable[[str, str], None] | None = None
    on_local_download_started: Callable[[float], None] | None = None
    on_local_download_completed: Callable[[str], None] | None = None
    on_local_download_failed: Callable[[str], None] | None = None
    on_retrieval_completed: Callable[[bool, str | None], None] | None = None


@dataclass(frozen=True)
class ModelCardContext:
    """
    Data required to generate a HuggingFace model card (README.md).

    Intentionally small: ModelRetriever should NOT re-run expensive computations
    while generating the model card.
    """

    phase_metrics: list[dict[str, Any]]
    datasets: list[str]
    dataset_source_type: str | None = None
    training_started_at: str | None = None
    training_completed_at: str | None = None


@dataclass(frozen=True)
class PhaseMetricsResult:
    """Return value of _extract_phase_metrics."""

    phase_metrics: list[dict[str, Any]]
    training_started_at: str | None
    training_completed_at: str | None


__all__ = [
    "ModelRetrieverEventCallbacks",
    "ModelCardContext",
    "PhaseMetricsResult",
    "_PIPELINE_STATE_STARTED_AT",
    "_PIPELINE_STATE_COMPLETED_AT",
    "_PHASE_IDX",
    "_STRATEGY_TYPE",
    "_STATUS",
    "_METRICS",
]
