"""
Interfaces for Report Generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.reports.domain.entities import ExperimentData, MetricTrend


class IExperimentDataProvider(Protocol):
    """
    Provider of experiment data.
    Implementations (Adapters) fetch data from sources like MLflow.
    """

    def load(self, run_id: str) -> ExperimentData:
        """Load complete experiment data."""
        ...


class IMetricAnalyzer(Protocol):
    """Analyzes a metric trend."""

    def analyze(self, key: str, trend: MetricTrend) -> Any:
        """Analyze metric trend."""
        ...


class IPercentileCalculator(Protocol):
    """Calculates percentiles."""

    def calculate(self, values: list[float]) -> Any: ...
