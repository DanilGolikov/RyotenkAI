"""
Value objects for ModelRetriever.

Phase 4 (event-system unification, 2026-05-16): the legacy
``ModelRetrieverEventCallbacks`` dataclass was removed. The stage now
emits typed ``ryotenkai.control.model.*`` envelopes through an
:class:`IEventEmitter` (see :mod:`ryotenkai_control.pipeline.stages.model_retriever.retriever`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Keys in DataBuffer `pipeline_state.json` used for README generation.
_PIPELINE_STATE_STARTED_AT = "started_at"
_PIPELINE_STATE_COMPLETED_AT = "completed_at"
_PHASE_IDX = "phase_idx"
_STRATEGY_TYPE = "strategy_type"
_STATUS = "status"
_METRICS = "metrics"


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
    "ModelCardContext",
    "PhaseMetricsResult",
    "_PIPELINE_STATE_STARTED_AT",
    "_PIPELINE_STATE_COMPLETED_AT",
    "_PHASE_IDX",
    "_STRATEGY_TYPE",
    "_STATUS",
    "_METRICS",
]
