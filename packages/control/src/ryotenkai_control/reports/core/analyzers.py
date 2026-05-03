"""
Metric Analyzers and Calculators.

Separated from ReportBuilder following Single Responsibility Principle.
Each class has one focused job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.reports.core.constants import MetricThresholds
from src.reports.core.metrics_registry import METRIC_DESCRIPTIONS
from src.reports.domain.interfaces import IMetricAnalyzer, IPercentileCalculator
from src.reports.models.report import MetricAnalysis, MetricStatus, PercentileStats

DIR_STABLE = "Stable"
DIR_DECREASED = "decreased"
DIR_INCREASED = "increased"
WARMUP_MARGIN = 1.1
KL_THRESHOLD = 10.0
PFLOP_SCALE = 1e15
TFLOP_SCALE = 1e12
GFLOP_SCALE = 1e9
PERCENTILE_95 = 0.95
PERCENTILE_99 = 0.99

if TYPE_CHECKING:
    from src.reports.domain.entities import MetricTrend

# ============================================================================
# METRIC ANALYZER (Strategy Pattern)
# ============================================================================


class MetricAnalyzer(IMetricAnalyzer):
    """
    Analyzes training metrics and provides health assessment.

    Uses Strategy pattern internally - different logic for different metric types.
    Implements IMetricAnalyzer protocol.
    """

    @staticmethod
    def can_analyze(_key: str) -> bool:
        """Check if we have analysis logic for this metric."""
        # Now more inclusive as we support RL metrics
        return True  # We have a fallback for unknown metrics

    def analyze(self, key: str, trend: MetricTrend) -> MetricAnalysis | None:
        """
        Analyze metric health and generate detailed report.

        Args:
            key: Metric key (e.g., "train_loss", "grad_norm")
            trend: Metric trend data

        Returns:
            MetricAnalysis with status and verdict, or None if no data
        """
        if not trend:
            return None

        # Get description from registry (or fallback)
        name, desc = METRIC_DESCRIPTIONS.get(key, (key, "Description unavailable."))

        # Determine status and verdict using strategy dispatch
        status, verdict = self._analyze_by_type(key, trend)

        return MetricAnalysis(
            name=key,
            display_name=name,
            description=desc,
            trend=trend,
            status=status,
            verdict=verdict,
        )

    def _analyze_by_type(self, key: str, trend: MetricTrend) -> tuple[MetricStatus, str]:
        """
        Dispatch to appropriate analysis strategy based on metric type.

        This is the Strategy pattern - different algorithms for different metric types.
        """
        key_lower = key.lower()

        if "loss" in key_lower:
            return self._analyze_loss(trend)
        elif "accuracy" in key_lower or "accuracies" in key_lower:
            return self._analyze_accuracy(trend)
        elif "grad_norm" in key_lower:
            return self._analyze_grad_norm(trend)
        elif "learning_rate" in key_lower:
            return self._analyze_learning_rate(trend)
        elif "per_second" in key_lower:
            return self._analyze_speed(trend)
        elif "entropy" in key_lower:
            return self._analyze_entropy(trend)
        elif "flos" in key_lower:
            return self._analyze_flos(trend)
        # --- New RL / Preference Learning Metrics ---
        # Important: order matters. Some DPO metrics include the substring "rewards/"
        # (e.g., "rewards/margins") and must not be misrouted to the generic reward analyzer.
        elif "margin" in key_lower:
            return self._analyze_margin(trend)
        elif "reward" in key_lower:
            return self._analyze_reward(trend)
        elif "kl" in key_lower:
            return self._analyze_kl(trend)
        elif "length" in key_lower:
            return self._analyze_length(trend)
        elif "logp" in key_lower:
            return self._analyze_logp(key, trend)
        else:
            return MetricStatus.NEUTRAL, DIR_STABLE

    @staticmethod
    def _analyze_loss(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze loss metric - should decrease."""
        status = MetricStatus.NEUTRAL
        verdict = DIR_STABLE

        if trend.direction == DIR_DECREASED:
            status = MetricStatus.GOOD
            change = abs(trend.change_pct or 0)
            verdict = f"Improved by {change:.1f}%"
        elif trend.direction == DIR_INCREASED:
            change = trend.change_pct or 0
            if change < MetricThresholds.LOSS_SIGNIFICANT_INCREASE:
                status = MetricStatus.WARNING
                verdict = "Small increase (noise?)"
            else:
                status = MetricStatus.BAD
                verdict = "Degradation"

        # Check for volatility (spikes)
        if trend.max_val is not None and trend.min_val is not None and trend.min_val > 0:
            ratio = trend.max_val / trend.min_val
            if ratio > 10:
                # Check if high ratio is just due to excellent convergence
                # If max_val is roughly equal to first value (start of training),
                # then this is monotonic decrease, not volatility.
                # Allow 10% margin for initial warmup noise.
                is_start_peak = False
                if trend.first is not None:
                    is_start_peak = trend.max_val <= (trend.first * WARMUP_MARGIN)

                # Only flag volatility if peak was NOT at the start
                # (i.e. we had a huge spike in the middle)
                if not is_start_peak:
                    status = MetricStatus.WARNING
                    verdict += " (high volatility)"

        return status, verdict

    @staticmethod
    def _analyze_accuracy(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze accuracy metric - should increase."""
        if trend.direction == DIR_INCREASED:
            return MetricStatus.GOOD, "Improving"
        elif trend.direction == DIR_DECREASED:
            return MetricStatus.BAD, "Worsening"
        return MetricStatus.NEUTRAL, DIR_STABLE

    @staticmethod
    def _analyze_grad_norm(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze gradient norm - check for spikes."""
        if trend.max_val is not None and trend.first is not None:
            if trend.max_val > (trend.first * MetricThresholds.GRAD_NORM_WARNING):
                return MetricStatus.WARNING, "Spikes detected"
            return MetricStatus.GOOD, "Stable gradients"
        return MetricStatus.NEUTRAL, "Insufficient data"

    @staticmethod
    def _analyze_learning_rate(_trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze learning rate - should follow schedule."""
        return MetricStatus.NEUTRAL, "Follows schedule"

    @staticmethod
    def _analyze_speed(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze processing speed."""
        if trend.last is not None:
            return MetricStatus.NEUTRAL, f"~{trend.last:.1f} samples/sec"
        return MetricStatus.NEUTRAL, "N/A"

    @staticmethod
    def _analyze_entropy(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze entropy - should decrease (model becomes more confident)."""
        if trend.direction == DIR_DECREASED:
            return MetricStatus.GOOD, "Model becoming more confident"
        elif trend.direction == DIR_INCREASED:
            return MetricStatus.WARNING, "Rising model uncertainty"
        return MetricStatus.NEUTRAL, "Stable confidence"

    @staticmethod
    def _analyze_flos(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze FLOPs - informational metric."""
        if trend.last is not None:
            # Format large numbers nicely
            flos = trend.last
            if flos >= PFLOP_SCALE:
                return MetricStatus.NEUTRAL, f"{flos / PFLOP_SCALE:.2f} PFLOPs"
            elif flos >= TFLOP_SCALE:
                return MetricStatus.NEUTRAL, f"{flos / TFLOP_SCALE:.2f} TFLOPs"
            elif flos >= GFLOP_SCALE:
                return MetricStatus.NEUTRAL, f"{flos / GFLOP_SCALE:.2f} GFLOPs"
            return MetricStatus.NEUTRAL, f"{flos:.0f} FLOPs"
        return MetricStatus.NEUTRAL, "N/A"

    # --- New RL Analysis Methods ---

    @staticmethod
    def _analyze_reward(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze reward - should increase."""
        if trend.direction == DIR_INCREASED:
            return MetricStatus.GOOD, "Reward increasing (training progressing)"
        elif trend.direction == DIR_DECREASED:
            return MetricStatus.BAD, "Reward decreasing (degradation)"
        return MetricStatus.NEUTRAL, "Stable reward"

    @staticmethod
    def _analyze_kl(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze KL divergence - should be stable or grow slowly."""
        # If KL explodes, it's bad. If it's stable, it's good.
        if trend.max_val is not None and trend.max_val > KL_THRESHOLD:
            return MetricStatus.WARNING, "High divergence (mode collapse risk)"

        if trend.direction == DIR_INCREASED:
            return MetricStatus.NEUTRAL, "Divergence increasing (normal)"
        return MetricStatus.GOOD, DIR_STABLE

    @staticmethod
    def _analyze_length(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze completion length - informational."""
        if trend.last is not None:
            return MetricStatus.NEUTRAL, f"~{trend.last:.0f} tokens"
        return MetricStatus.NEUTRAL, "N/A"

    @staticmethod
    def _analyze_margin(trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze reward margin (DPO) - should increase."""
        if trend.direction == DIR_INCREASED:
            return MetricStatus.GOOD, "Margin increasing (confidence)"
        elif trend.direction == DIR_DECREASED:
            return MetricStatus.WARNING, "Margin decreasing"
        return MetricStatus.NEUTRAL, DIR_STABLE

    @staticmethod
    def _analyze_logp(key: str, trend: MetricTrend) -> tuple[MetricStatus, str]:
        """Analyze log probs (DPO)."""
        key_lower = key.lower()
        if "chosen" in key_lower and trend.direction == DIR_DECREASED:
            # LogP chosen usually drops slightly or stays stable as model drifts from SFT
            return MetricStatus.NEUTRAL, "Decreasing (expected for DPO)"
        if "rejected" in key_lower and trend.direction == DIR_DECREASED:
            # LogP rejected should drop significantly
            return MetricStatus.GOOD, "Model picks rejected less often"

        return MetricStatus.NEUTRAL, DIR_STABLE


# ============================================================================
# PERCENTILE CALCULATOR
# ============================================================================


class PercentileCalculator(IPercentileCalculator):
    """
    Calculates percentile statistics from a list of values.

    Implements IPercentileCalculator protocol.
    Stateless - can be shared across the application.
    """

    def calculate(self, values: list[float]) -> PercentileStats:
        """
        Calculate percentile statistics.

        Args:
            values: List of numeric values

        Returns:
            PercentileStats with avg, min, max, p95, p99
        """
        if not values:
            return PercentileStats()

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return PercentileStats(
            avg=sum(sorted_vals) / n,
            min_val=sorted_vals[0],
            max_val=sorted_vals[-1],
            p95=sorted_vals[min(int(n * PERCENTILE_95), n - 1)],
            p99=sorted_vals[min(int(n * PERCENTILE_99), n - 1)],
            data_points=n,
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "METRIC_DESCRIPTIONS",
    "MetricAnalyzer",
    "PercentileCalculator",
]
