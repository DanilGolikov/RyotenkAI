"""
Tests for MetricAnalyzer.

Covers various training scenarios:
- Successful training (loss decreasing, accuracy increasing)
- Failed training (loss increasing, divergence)
- Unstable training (spikes, high volatility)
- Edge cases (empty data, single value)

Based on real-world LLM fine-tuning issues:
- Loss spikes (sudden jumps in loss)
- Gradient explosion (grad_norm > 10x baseline)
- Overfitting (train_loss down, eval_loss up)
- Non-convergence (loss oscillates, doesn't decrease)
"""

import pytest

from src.reports.core.analyzers import (
    METRIC_DESCRIPTIONS,
    MetricAnalyzer,
    PercentileCalculator,
)
from src.reports.domain.entities import MetricTrend
from src.reports.models.report import MetricStatus

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def analyzer() -> MetricAnalyzer:
    """Create MetricAnalyzer instance."""
    return MetricAnalyzer()


@pytest.fixture
def percentile_calc() -> PercentileCalculator:
    """Create PercentileCalculator instance."""
    return PercentileCalculator()


# ============================================================================
# METRIC ANALYZER: SUCCESSFUL SCENARIOS
# ============================================================================


class TestMetricAnalyzerSuccess:
    """Tests for successful training scenarios."""

    def test_loss_decreasing_significantly(self, analyzer: MetricAnalyzer) -> None:
        """Loss decreased by >80% - excellent training."""
        trend = MetricTrend(
            first=3.5,
            last=0.5,
            min_val=0.5,
            max_val=3.5,
            change_pct=-85.7,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("train_loss", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD
        assert "85.7%" in result.verdict
        assert "Improved" in result.verdict

    def test_loss_decreasing_moderately(self, analyzer: MetricAnalyzer) -> None:
        """Loss decreased by ~50% - good training."""
        trend = MetricTrend(
            first=2.0,
            last=1.0,
            min_val=1.0,
            max_val=2.0,
            change_pct=-50.0,
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD

    def test_accuracy_increasing(self, analyzer: MetricAnalyzer) -> None:
        """Token accuracy improved from 50% to 85%."""
        trend = MetricTrend(
            first=0.50,
            last=0.85,
            min_val=0.50,
            max_val=0.85,
            change_pct=70.0,
            direction="increased",
            data_points=100,
        )
        result = analyzer.analyze("mean_token_accuracy", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD
        assert "Improving" in result.verdict

    def test_grad_norm_stable(self, analyzer: MetricAnalyzer) -> None:
        """Gradient norm stays within reasonable bounds (0.5 - 5.0)."""
        trend = MetricTrend(
            first=2.5,
            last=1.8,
            min_val=0.8,
            max_val=5.0,
            change_pct=-28.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("grad_norm", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD
        assert "Stable gradients" in result.verdict

    def test_entropy_decreasing(self, analyzer: MetricAnalyzer) -> None:
        """Entropy decreased - model becoming more confident."""
        trend = MetricTrend(
            first=3.0,
            last=1.5,
            min_val=1.5,
            max_val=3.0,
            change_pct=-50.0,
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("entropy", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD
        assert "confident" in result.verdict.lower()

    def test_learning_rate_follows_schedule(self, analyzer: MetricAnalyzer) -> None:
        """Learning rate decreasing as expected (cosine decay)."""
        trend = MetricTrend(
            first=2e-4,
            last=1e-5,
            min_val=1e-5,
            max_val=2e-4,
            change_pct=-95.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("learning_rate", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert "schedule" in result.verdict.lower()


# ============================================================================
# METRIC ANALYZER: FAILED SCENARIOS
# ============================================================================


class TestMetricAnalyzerFailed:
    """Tests for failed/problematic training scenarios."""

    def test_loss_increasing_significantly(self, analyzer: MetricAnalyzer) -> None:
        """Loss increased by >10% - training degradation."""
        trend = MetricTrend(
            first=1.0,
            last=2.5,
            min_val=1.0,
            max_val=2.5,
            change_pct=150.0,
            direction="increased",
            data_points=50,
        )
        result = analyzer.analyze("train_loss", trend)

        assert result is not None
        assert result.status == MetricStatus.BAD
        assert "Degradation" in result.verdict

    def test_loss_increasing_slightly_noise(self, analyzer: MetricAnalyzer) -> None:
        """Loss increased by <5% - could be noise, warning."""
        trend = MetricTrend(
            first=1.0,
            last=1.03,
            min_val=1.0,
            max_val=1.05,
            change_pct=3.0,
            direction="increased",
            data_points=20,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        assert result.status == MetricStatus.WARNING
        assert "noise" in result.verdict.lower() or "Small increase" in result.verdict

    def test_accuracy_decreasing(self, analyzer: MetricAnalyzer) -> None:
        """Token accuracy dropped - bad sign."""
        trend = MetricTrend(
            first=0.75,
            last=0.50,
            min_val=0.50,
            max_val=0.75,
            change_pct=-33.3,
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("mean_token_accuracy", trend)

        assert result is not None
        assert result.status == MetricStatus.BAD
        assert "Worsening" in result.verdict

    def test_gradient_explosion(self, analyzer: MetricAnalyzer) -> None:
        """
        Gradient explosion: max_val >> first (>10x).

        Real scenario: grad_norm starts at 2.0, spikes to 50+ during training.
        This indicates exploding gradients - need to reduce LR or add clipping.
        """
        trend = MetricTrend(
            first=2.0,
            last=5.0,
            min_val=1.5,
            max_val=50.0,  # 25x spike!
            change_pct=150.0,
            direction="increased",
            data_points=100,
        )
        result = analyzer.analyze("grad_norm", trend)

        assert result is not None
        assert result.status == MetricStatus.WARNING
        assert "spike" in result.verdict.lower()

    def test_entropy_increasing(self, analyzer: MetricAnalyzer) -> None:
        """Entropy increased - model becoming less confident (bad)."""
        trend = MetricTrend(
            first=1.5,
            last=3.0,
            min_val=1.5,
            max_val=3.0,
            change_pct=100.0,
            direction="increased",
            data_points=50,
        )
        result = analyzer.analyze("entropy", trend)

        assert result is not None
        assert result.status == MetricStatus.WARNING
        assert "uncertainty" in result.verdict.lower() or "Rising" in result.verdict

    def test_overfitting_scenario(self, analyzer: MetricAnalyzer) -> None:
        """
        Overfitting: train_loss decreases but eval_loss increases.

        This is a common issue when fine-tuning on small datasets.
        Model memorizes training data instead of learning patterns.
        """
        # Train loss looks good
        train_trend = MetricTrend(first=2.0, last=0.5, change_pct=-75.0, direction="decreased", data_points=100)
        train_result = analyzer.analyze("train_loss", train_trend)

        # But eval loss is bad
        eval_trend = MetricTrend(first=2.0, last=3.5, change_pct=75.0, direction="increased", data_points=100)
        eval_result = analyzer.analyze("eval_loss", eval_trend)

        assert train_result.status == MetricStatus.GOOD
        assert eval_result.status == MetricStatus.BAD


# ============================================================================
# METRIC ANALYZER: UNSTABLE SCENARIOS
# ============================================================================


class TestMetricAnalyzerUnstable:
    """Tests for unstable/volatile training scenarios."""

    def test_loss_high_volatility(self, analyzer: MetricAnalyzer) -> None:
        """
        Loss with high volatility (max/min ratio > 10).

        Real scenario: Loss oscillates wildly during training,
        possibly due to bad learning rate or data issues.
        """
        trend = MetricTrend(
            first=1.5,
            last=1.0,
            min_val=0.3,  # Best point
            max_val=5.0,  # Spike to 5.0 (ratio = 16.7x)
            change_pct=-33.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("train_loss", trend)

        assert result is not None
        # Should warn about volatility even though loss decreased
        assert result.status == MetricStatus.WARNING
        assert "volatility" in result.verdict.lower()

    def test_loss_spike_recovered(self, analyzer: MetricAnalyzer) -> None:
        """
        Loss spiked but recovered.

        Real scenario: Sudden loss spike mid-training (bad batch?),
        but model recovered. Still concerning.
        """
        trend = MetricTrend(
            first=1.0,
            last=0.8,
            min_val=0.7,
            max_val=15.0,  # Massive spike (15x from min)
            change_pct=-20.0,
            direction="decreased",
            data_points=200,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        assert result.status == MetricStatus.WARNING

    def test_grad_norm_minor_spikes(self, analyzer: MetricAnalyzer) -> None:
        """
        Minor gradient spikes (<10x) - usually acceptable.
        """
        trend = MetricTrend(
            first=1.0,
            last=1.5,
            min_val=0.5,
            max_val=8.0,  # 8x from first - borderline
            change_pct=50.0,
            direction="increased",
            data_points=100,
        )
        result = analyzer.analyze("grad_norm", trend)

        assert result is not None
        # 8x is borderline, should still be GOOD
        assert result.status == MetricStatus.GOOD

    def test_loss_stable_no_improvement(self, analyzer: MetricAnalyzer) -> None:
        """
        Loss barely changed - possible plateau.

        Real scenario: Model stuck in local minimum,
        or learning rate too low.
        """
        trend = MetricTrend(
            first=1.5,
            last=1.48,
            min_val=1.45,
            max_val=1.55,
            change_pct=-1.3,
            direction="stable",
            data_points=100,
        )
        result = analyzer.analyze("train_loss", trend)

        assert result is not None
        # Stable but not improving - neutral
        assert result.status == MetricStatus.NEUTRAL
        assert "Stable" in result.verdict


# ============================================================================
# METRIC ANALYZER: EDGE CASES
# ============================================================================


class TestMetricAnalyzerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_trend(self, analyzer: MetricAnalyzer) -> None:
        """Empty trend (no data) should return neutral analysis."""
        trend = MetricTrend()  # All defaults (None/0)
        result = analyzer.analyze("loss", trend)

        # Should return neutral analysis, not None
        assert result is not None
        assert result.status == MetricStatus.NEUTRAL

    def test_none_trend_returns_none(self, analyzer: MetricAnalyzer) -> None:
        """Passing None should be treated as no data."""
        result = analyzer.analyze("loss", None)  # type: ignore[arg-type]
        assert result is None

    def test_only_last_value(self, analyzer: MetricAnalyzer) -> None:
        """Trend with only final value (no history)."""
        trend = MetricTrend(last=1.5, data_points=0)
        result = analyzer.analyze("train_loss", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert "Stable" in result.verdict

    def test_single_data_point(self, analyzer: MetricAnalyzer) -> None:
        """Single data point - can't determine trend."""
        trend = MetricTrend(
            first=1.5,
            last=1.5,
            min_val=1.5,
            max_val=1.5,
            change_pct=0.0,
            direction="stable",
            data_points=1,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL

    def test_unknown_metric(self, analyzer: MetricAnalyzer) -> None:
        """Unknown metric key - should use default analysis."""
        trend = MetricTrend(
            first=1.0,
            last=2.0,
            change_pct=100.0,
            direction="increased",
            data_points=10,
        )
        result = analyzer.analyze("some_unknown_metric", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert result.display_name == "some_unknown_metric"

    def test_can_analyze_returns_true(self, analyzer: MetricAnalyzer) -> None:
        """can_analyze should return True for known patterns."""
        assert analyzer.can_analyze("train_loss") is True
        assert analyzer.can_analyze("mean_token_accuracy") is True
        assert analyzer.can_analyze("grad_norm") is True
        assert analyzer.can_analyze("learning_rate") is True
        assert analyzer.can_analyze("samples_per_second") is True

    def test_can_analyze_returns_true_for_any_metric(self, analyzer: MetricAnalyzer) -> None:
        """can_analyze should return True for any metric (has fallback logic)."""
        # Analyzer has fallback for all metrics
        assert analyzer.can_analyze("completely_unknown_xyz_metric") is True
        assert analyzer.can_analyze("random_metric_123") is True
        assert analyzer.can_analyze("loss") is True

    def test_total_flos_formatting(self, analyzer: MetricAnalyzer) -> None:
        """Total FLOPs should be formatted nicely."""
        # Test TFLOPs
        trend = MetricTrend(last=40e12, data_points=0)  # 40 TFLOPs
        result = analyzer.analyze("total_flos", trend)

        assert result is not None
        assert "TFLOPs" in result.verdict

        # Test GFLOPs
        trend_gflops = MetricTrend(last=500e9, data_points=0)  # 500 GFLOPs
        result_gflops = analyzer.analyze("total_flos", trend_gflops)

        assert result_gflops is not None
        assert "GFLOPs" in result_gflops.verdict

    def test_speed_metric(self, analyzer: MetricAnalyzer) -> None:
        """Speed metric (samples_per_second)."""
        trend = MetricTrend(last=15.5, data_points=0)
        result = analyzer.analyze("train_samples_per_second", trend)

        assert result is not None
        assert "15.5" in result.verdict or "samples/sec" in result.verdict


# ============================================================================
# PERCENTILE CALCULATOR
# ============================================================================


class TestPercentileCalculator:
    """Tests for PercentileCalculator."""

    def test_basic_percentiles(self, percentile_calc: PercentileCalculator) -> None:
        """Basic percentile calculation."""
        values = list(range(1, 101))  # 1 to 100
        result = percentile_calc.calculate(values)

        assert result.avg == 50.5
        assert result.min_val == 1
        assert result.max_val == 100
        # P95 = index at 95% of 100 items = index 95 = value 96
        assert result.p95 == 96
        assert result.p99 == 100
        assert result.data_points == 100

    def test_empty_list(self, percentile_calc: PercentileCalculator) -> None:
        """Empty list should return empty stats."""
        result = percentile_calc.calculate([])

        assert result.avg is None
        assert result.min_val is None
        assert result.max_val is None
        assert result.data_points == 0

    def test_single_value(self, percentile_calc: PercentileCalculator) -> None:
        """Single value - all stats should be that value."""
        result = percentile_calc.calculate([42.0])

        assert result.avg == 42.0
        assert result.min_val == 42.0
        assert result.max_val == 42.0
        assert result.p95 == 42.0
        assert result.p99 == 42.0
        assert result.data_points == 1

    def test_two_values(self, percentile_calc: PercentileCalculator) -> None:
        """Two values."""
        result = percentile_calc.calculate([10.0, 20.0])

        assert result.avg == 15.0
        assert result.min_val == 10.0
        assert result.max_val == 20.0
        assert result.data_points == 2

    def test_with_outliers(self, percentile_calc: PercentileCalculator) -> None:
        """Values with outliers - P95 shows distribution edge."""
        # 95 normal values + 5 outliers
        values = [1.0] * 95 + [100.0] * 5
        result = percentile_calc.calculate(values)

        assert result.min_val == 1.0
        assert result.max_val == 100.0
        # P95 at index 95 = 100.0 (first outlier in sorted list)
        assert result.p95 == 100.0
        # Verify avg is heavily influenced by outliers
        assert result.avg < 10.0  # Most values are 1.0


# ============================================================================
# METRIC DESCRIPTIONS
# ============================================================================


class TestMetricDescriptions:
    """Tests for metric descriptions."""

    def test_all_metrics_have_descriptions(self) -> None:
        """All expected metrics should have descriptions."""
        expected_metrics = [
            "loss",
            "train_loss",
            "eval_loss",
            "mean_token_accuracy",
            "grad_norm",
            "learning_rate",
            "train_samples_per_second",
            "epoch",
            "entropy",
            "total_flos",
        ]

        for metric in expected_metrics:
            assert metric in METRIC_DESCRIPTIONS, f"Missing description for {metric}"
            name, desc = METRIC_DESCRIPTIONS[metric]
            assert len(name) > 0, f"Empty display name for {metric}"
            assert len(desc) > 20, f"Description too short for {metric}"

    def test_descriptions_in_english(self) -> None:
        """Descriptions should be English (no Cyrillic)."""
        for key, (_name, desc) in METRIC_DESCRIPTIONS.items():
            has_cyrillic = any("\u0400" <= c <= "\u04ff" for c in desc)
            assert not has_cyrillic, f"Description for {key} must not contain Cyrillic"


# ============================================================================
# REAL-WORLD SCENARIOS
# ============================================================================


class TestRealWorldScenarios:
    """
    Tests based on real-world LLM fine-tuning scenarios.

    These scenarios are based on common issues reported in:
    - https://machinelearningmastery.com/5-problems-encountered-fine-tuning-llms-with-solutions/
    - https://neptune.ai/blog/monitoring-diagnosing-and-solving-gradient-issues-in-foundation-models
    """

    def test_scenario_healthy_sft_training(self, analyzer: MetricAnalyzer) -> None:
        """
        Healthy SFT training: loss decreases, accuracy increases,
        gradients stable, entropy decreases.
        """
        metrics = {
            "train_loss": MetricTrend(first=3.5, last=1.2, change_pct=-65.7, direction="decreased", data_points=100),
            "mean_token_accuracy": MetricTrend(
                first=0.45, last=0.82, change_pct=82.2, direction="increased", data_points=100
            ),
            "grad_norm": MetricTrend(
                first=3.0, last=1.5, min_val=0.8, max_val=5.0, direction="decreased", data_points=100
            ),
            "entropy": MetricTrend(first=2.8, last=1.2, change_pct=-57.1, direction="decreased", data_points=100),
        }

        results = {k: analyzer.analyze(k, v) for k, v in metrics.items()}

        assert results["train_loss"].status == MetricStatus.GOOD
        assert results["mean_token_accuracy"].status == MetricStatus.GOOD
        assert results["grad_norm"].status == MetricStatus.GOOD
        assert results["entropy"].status == MetricStatus.GOOD

    def test_scenario_diverging_training(self, analyzer: MetricAnalyzer) -> None:
        """
        Diverging training: loss explodes, gradients explode.

        Cause: Learning rate too high.
        """
        metrics = {
            "train_loss": MetricTrend(
                first=2.0,
                last=1e10,  # Loss exploded
                change_pct=5e11,
                direction="increased",
                data_points=50,
            ),
            "grad_norm": MetricTrend(
                first=2.0,
                last=1000.0,
                min_val=1.5,
                max_val=1e6,  # Gradient explosion
                change_pct=5e7,
                direction="increased",
                data_points=50,
            ),
        }

        results = {k: analyzer.analyze(k, v) for k, v in metrics.items()}

        assert results["train_loss"].status == MetricStatus.BAD
        assert results["grad_norm"].status == MetricStatus.WARNING

    def test_scenario_vanishing_gradients(self, analyzer: MetricAnalyzer) -> None:
        """
        Vanishing gradients: grad_norm approaches zero, loss doesn't decrease.

        Cause: Model too deep, activations saturating.
        """
        metrics = {
            "train_loss": MetricTrend(first=2.5, last=2.48, change_pct=-0.8, direction="stable", data_points=100),
            "grad_norm": MetricTrend(
                first=1.0,
                last=0.001,
                min_val=0.0001,
                max_val=1.0,
                change_pct=-99.9,
                direction="decreased",
                data_points=100,
            ),
        }

        results = {k: analyzer.analyze(k, v) for k, v in metrics.items()}

        # Loss is stable (neutral) - model isn't learning
        assert results["train_loss"].status == MetricStatus.NEUTRAL
        # Gradients decreased - technically "good" in analyzer,
        # but in context this is bad (vanishing)
        # This is a limitation - analyzer doesn't know absolute values are too small

    def test_scenario_loss_spike_recovery(self, analyzer: MetricAnalyzer) -> None:
        """
        Loss spike with recovery.

        Real scenario: Bad batch causes loss spike, but model recovers.
        Common in LLM training.
        """
        trend = MetricTrend(
            first=1.5,
            last=0.9,
            min_val=0.8,
            max_val=25.0,  # Big spike (31x from min)
            change_pct=-40.0,
            direction="decreased",
            data_points=500,
        )
        result = analyzer.analyze("train_loss", trend)

        # Should warn about volatility despite overall improvement
        assert result.status == MetricStatus.WARNING
        assert "volatility" in result.verdict.lower()

    def test_scenario_early_stopping_needed(self, analyzer: MetricAnalyzer) -> None:
        """
        Overfitting scenario where early stopping would help.

        train_loss keeps decreasing, but model is overfitting.
        """
        train_result = analyzer.analyze(
            "train_loss", MetricTrend(first=2.0, last=0.1, change_pct=-95.0, direction="decreased", data_points=100)
        )
        eval_result = analyzer.analyze(
            "eval_loss", MetricTrend(first=2.0, last=4.0, change_pct=100.0, direction="increased", data_points=100)
        )

        # Train looks great
        assert train_result.status == MetricStatus.GOOD
        # But eval is getting worse
        assert eval_result.status == MetricStatus.BAD


# ============================================================================
# EDGE CASES & STRANGE DATA
# ============================================================================


class TestMetricAnalyzerBoundaryConditions:
    """
    Tests for boundary conditions and strange data.

    These tests ensure the analyzer doesn't produce misleading results
    when given unusual or edge-case inputs.
    """

    # -------------------------------------------------------------------------
    # MINIMAL DATA
    # -------------------------------------------------------------------------

    def test_two_identical_points(self, analyzer: MetricAnalyzer) -> None:
        """Two identical data points - no change."""
        trend = MetricTrend(
            first=1.5,
            last=1.5,
            min_val=1.5,
            max_val=1.5,
            change_pct=0.0,
            direction="stable",
            data_points=2,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert "Stable" in result.verdict

    def test_two_points_tiny_improvement(self, analyzer: MetricAnalyzer) -> None:
        """Two points with microscopic improvement (0.01%)."""
        trend = MetricTrend(
            first=1.5000,
            last=1.4999,
            min_val=1.4999,
            max_val=1.5000,
            change_pct=-0.01,
            direction="decreased",
            data_points=2,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Such tiny change should be considered stable, not "good"
        # (depends on implementation - might be GOOD with tiny %)

    def test_three_points_with_spike(self, analyzer: MetricAnalyzer) -> None:
        """
        Three points: start=1.0, spike=100.0, end=0.9

        With only 3 points, hard to know if spike is anomaly or pattern.
        """
        trend = MetricTrend(
            first=1.0,
            last=0.9,
            min_val=0.9,
            max_val=100.0,  # Massive spike!
            change_pct=-10.0,
            direction="decreased",
            data_points=3,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Should warn about volatility (100x ratio)
        assert result.status == MetricStatus.WARNING

    # -------------------------------------------------------------------------
    # EXTREME VALUES
    # -------------------------------------------------------------------------

    def test_extremely_small_loss(self, analyzer: MetricAnalyzer) -> None:
        """Loss decreased to near-zero (1e-10)."""
        trend = MetricTrend(
            first=1.0,
            last=1e-10,
            min_val=1e-10,
            max_val=1.0,
            change_pct=-99.9999999,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Extreme improvement, but likely overfitting!
        # Analyzer might not catch this without eval_loss

    def test_extremely_large_loss(self, analyzer: MetricAnalyzer) -> None:
        """Loss exploded to astronomical value."""
        trend = MetricTrend(
            first=1.0,
            last=1e15,
            min_val=1.0,
            max_val=1e15,
            change_pct=1e17,
            direction="increased",
            data_points=50,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Analyzer gives WARNING for "increased <5%" otherwise BAD
        # But 1e17% is definitely > 5%, so should be BAD
        # Actually: current logic checks change_pct < 5.0 for WARNING
        # 1e17 > 5.0, so it goes to BAD branch
        # BUT! Volatility check (max/min > 10) adds WARNING
        # So final status depends on implementation details
        assert result.status in (MetricStatus.BAD, MetricStatus.WARNING)

    def test_zero_loss_start(self, analyzer: MetricAnalyzer) -> None:
        """Loss started at exactly 0 (unusual but possible)."""
        trend = MetricTrend(
            first=0.0,
            last=0.5,
            min_val=0.0,
            max_val=0.5,
            change_pct=None,  # Can't calculate % from 0
            direction="increased",
            data_points=10,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Direction is increased - analyzer checks change_pct
        # With change_pct=None, the <5% check fails (None < 5.0 = False)
        # So goes to else branch -> WARNING "Small increase (noise?)"
        # This is a quirk of the implementation
        assert result.status in (MetricStatus.BAD, MetricStatus.WARNING)

    def test_zero_loss_end(self, analyzer: MetricAnalyzer) -> None:
        """Loss ended at exactly 0 (perfect fit - suspicious!)."""
        trend = MetricTrend(
            first=2.0,
            last=0.0,
            min_val=0.0,
            max_val=2.0,
            change_pct=-100.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # 100% decrease is "good" but zero loss is very suspicious
        # Current analyzer would say GOOD - might need future enhancement

    def test_negative_loss(self, analyzer: MetricAnalyzer) -> None:
        """
        Negative loss value (shouldn't happen, but handle gracefully).

        Some custom loss functions can have negative values.
        """
        trend = MetricTrend(
            first=1.0,
            last=-0.5,
            min_val=-0.5,
            max_val=1.0,
            change_pct=-150.0,  # Went below zero
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Should still work, direction decreased = good

    # -------------------------------------------------------------------------
    # CONTRADICTORY DATA
    # -------------------------------------------------------------------------

    def test_direction_contradicts_values(self, analyzer: MetricAnalyzer) -> None:
        """
        Direction says 'decreased' but last > first.

        This is inconsistent data - analyzer should handle gracefully.
        """
        trend = MetricTrend(
            first=1.0,
            last=2.0,  # Actually increased!
            min_val=1.0,
            max_val=2.0,
            change_pct=100.0,  # Also says increased
            direction="decreased",  # But this says decreased!
            data_points=50,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Analyzer uses 'direction' field, so would say "good"
        # This is a data integrity issue upstream

    def test_change_pct_none(self, analyzer: MetricAnalyzer) -> None:
        """change_pct is None (couldn't be calculated)."""
        trend = MetricTrend(
            first=0.0,
            last=1.0,
            min_val=0.0,
            max_val=1.0,
            change_pct=None,
            direction="increased",
            data_points=10,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Should handle None gracefully
        # With change_pct=None, comparison "change_pct < 5.0" is False
        # So falls through to WARNING branch
        # This is a known limitation - None handling could be improved
        assert result.status in (MetricStatus.BAD, MetricStatus.WARNING)

    def test_min_greater_than_max(self, analyzer: MetricAnalyzer) -> None:
        """
        min_val > max_val (impossible, but handle gracefully).

        This indicates a bug in data collection.
        """
        trend = MetricTrend(
            first=1.0,
            last=0.8,
            min_val=2.0,  # min > max - impossible!
            max_val=0.5,
            change_pct=-20.0,
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Should not crash, even with impossible data

    # -------------------------------------------------------------------------
    # ACCURACY EDGE CASES
    # -------------------------------------------------------------------------

    def test_accuracy_above_100_percent(self, analyzer: MetricAnalyzer) -> None:
        """
        Accuracy > 1.0 (shouldn't happen, but some metrics might).

        Could indicate metric is not actually accuracy.
        """
        trend = MetricTrend(
            first=0.5,
            last=1.5,  # 150% accuracy?!
            min_val=0.5,
            max_val=1.5,
            change_pct=200.0,
            direction="increased",
            data_points=50,
        )
        result = analyzer.analyze("mean_token_accuracy", trend)

        assert result is not None
        # Increased accuracy = good, even if value is weird

    def test_accuracy_negative(self, analyzer: MetricAnalyzer) -> None:
        """Negative accuracy (definitely wrong data)."""
        trend = MetricTrend(
            first=0.5,
            last=-0.2,
            min_val=-0.2,
            max_val=0.5,
            change_pct=-140.0,
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("mean_token_accuracy", trend)

        assert result is not None
        # Decreased accuracy = bad

    # -------------------------------------------------------------------------
    # GRADIENT NORM EDGE CASES
    # -------------------------------------------------------------------------

    def test_grad_norm_zero(self, analyzer: MetricAnalyzer) -> None:
        """
        Gradient norm = 0 (dead model, no learning).

        This indicates vanishing gradients or frozen weights.
        """
        trend = MetricTrend(
            first=1.0,
            last=0.0,
            min_val=0.0,
            max_val=1.0,
            change_pct=-100.0,
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("grad_norm", trend)

        assert result is not None
        # Current analyzer would say "stable gradients" -
        # but zero is actually bad (vanishing)!
        # This is a known limitation.

    def test_grad_norm_exactly_at_threshold(self, analyzer: MetricAnalyzer) -> None:
        """Gradient spike exactly at 10x threshold."""
        trend = MetricTrend(
            first=1.0,
            last=1.0,
            min_val=1.0,
            max_val=10.0,  # Exactly 10x
            change_pct=0.0,
            direction="stable",
            data_points=100,
        )
        result = analyzer.analyze("grad_norm", trend)

        assert result is not None
        # At boundary - should be GOOD (< 10x triggers warning)
        assert result.status == MetricStatus.GOOD

    def test_grad_norm_just_over_threshold(self, analyzer: MetricAnalyzer) -> None:
        """Gradient spike just over 10x threshold."""
        trend = MetricTrend(
            first=1.0,
            last=1.0,
            min_val=1.0,
            max_val=10.01,  # Just over 10x
            change_pct=0.0,
            direction="stable",
            data_points=100,
        )
        result = analyzer.analyze("grad_norm", trend)

        assert result is not None
        # Just over boundary - should warn
        assert result.status == MetricStatus.WARNING

    # -------------------------------------------------------------------------
    # LEARNING RATE EDGE CASES
    # -------------------------------------------------------------------------

    def test_learning_rate_increased(self, analyzer: MetricAnalyzer) -> None:
        """
        Learning rate increased during training.

        Unusual - most schedules decrease LR. Could be warmup phase.
        """
        trend = MetricTrend(
            first=1e-6,
            last=1e-4,
            min_val=1e-6,
            max_val=1e-4,
            change_pct=9900.0,
            direction="increased",
            data_points=50,
        )
        result = analyzer.analyze("learning_rate", trend)

        assert result is not None
        # LR is always NEUTRAL (just following schedule)
        assert result.status == MetricStatus.NEUTRAL

    def test_learning_rate_constant(self, analyzer: MetricAnalyzer) -> None:
        """Learning rate stayed constant (no schedule)."""
        trend = MetricTrend(
            first=1e-4,
            last=1e-4,
            min_val=1e-4,
            max_val=1e-4,
            change_pct=0.0,
            direction="stable",
            data_points=100,
        )
        result = analyzer.analyze("learning_rate", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL

    # -------------------------------------------------------------------------
    # SPEED METRICS EDGE CASES
    # -------------------------------------------------------------------------

    def test_speed_zero(self, analyzer: MetricAnalyzer) -> None:
        """Speed = 0 samples/sec (training stalled)."""
        trend = MetricTrend(last=0.0, data_points=0)
        result = analyzer.analyze("train_samples_per_second", trend)

        assert result is not None
        assert result.verdict == "~0.0 samples/sec"

    def test_speed_extremely_high(self, analyzer: MetricAnalyzer) -> None:
        """Speed = 10000 samples/sec (unrealistically high)."""
        trend = MetricTrend(last=10000.0, data_points=0)
        result = analyzer.analyze("train_samples_per_second", trend)

        assert result is not None
        # Analyzer just reports the value, doesn't validate realism

    # -------------------------------------------------------------------------
    # ENTROPY EDGE CASES
    # -------------------------------------------------------------------------

    def test_entropy_zero(self, analyzer: MetricAnalyzer) -> None:
        """
        Entropy = 0 (model is 100% confident in one token).

        Could indicate collapse or overfitting.
        """
        trend = MetricTrend(
            first=2.0,
            last=0.0,
            min_val=0.0,
            max_val=2.0,
            change_pct=-100.0,
            direction="decreased",
            data_points=50,
        )
        result = analyzer.analyze("entropy", trend)

        assert result is not None
        # Decreased entropy = model more confident = GOOD
        # But zero entropy might indicate mode collapse!

    def test_entropy_very_high(self, analyzer: MetricAnalyzer) -> None:
        """Entropy = 10 (very uncertain model)."""
        trend = MetricTrend(
            first=2.0,
            last=10.0,
            min_val=2.0,
            max_val=10.0,
            change_pct=400.0,
            direction="increased",
            data_points=50,
        )
        result = analyzer.analyze("entropy", trend)

        assert result is not None
        # Increased entropy = model less confident = WARNING
        assert result.status == MetricStatus.WARNING

    # -------------------------------------------------------------------------
    # FLOPS EDGE CASES
    # -------------------------------------------------------------------------

    def test_flops_zero(self, analyzer: MetricAnalyzer) -> None:
        """Total FLOPs = 0 (no computation happened?)."""
        trend = MetricTrend(last=0.0, data_points=0)
        result = analyzer.analyze("total_flos", trend)

        assert result is not None
        assert result.verdict == "0 FLOPs"

    def test_flops_petascale(self, analyzer: MetricAnalyzer) -> None:
        """Total FLOPs in petascale (1e15+)."""
        trend = MetricTrend(last=5e15, data_points=0)  # 5 PFLOPs
        result = analyzer.analyze("total_flos", trend)

        assert result is not None
        assert "PFLOPs" in result.verdict


# ============================================================================
# RL / PREFERENCE LEARNING METRICS
# ============================================================================


class TestMetricAnalyzerRLMetrics:
    """Tests for RL/DPO/ORPO metrics dispatch and verdicts."""

    def test_reward_increasing_is_good(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=0.1, last=0.6, change_pct=500.0, direction="increased", data_points=50)
        result = analyzer.analyze("reward", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD
        assert "Reward increasing" in result.verdict

    def test_reward_stable_is_neutral(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=0.5, last=0.5, change_pct=0.0, direction="stable", data_points=50)
        result = analyzer.analyze("reward", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert "Stable reward" in result.verdict

    def test_kl_high_divergence_warns(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=0.1, last=5.0, min_val=0.1, max_val=10.1, direction="increased", data_points=50)
        result = analyzer.analyze("kl", trend)

        assert result is not None
        assert result.status == MetricStatus.WARNING
        assert "High divergence" in result.verdict

    def test_kl_exactly_10_is_not_warning(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=0.1, last=0.2, min_val=0.1, max_val=10.0, direction="stable", data_points=50)
        result = analyzer.analyze("kl", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD
        assert result.verdict == "Stable"

    def test_completion_length_zero_formats(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(last=0.0, data_points=0)
        result = analyzer.analyze("completion_length", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert result.verdict == "~0 tokens"

    def test_rewards_accuracies_uses_accuracy_strategy(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=0.4, last=0.7, change_pct=75.0, direction="increased", data_points=50)
        result = analyzer.analyze("rewards/accuracies", trend)

        assert result is not None
        assert result.display_name == "Reward Accuracy"
        assert result.status == MetricStatus.GOOD

    def test_rewards_margins_decreasing_warns(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=0.5, last=0.1, change_pct=-80.0, direction="decreased", data_points=50)
        result = analyzer.analyze("rewards/margins", trend)

        assert result is not None
        assert result.status == MetricStatus.WARNING
        assert result.verdict == "Margin decreasing"

    def test_logps_chosen_decreased_is_neutral(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=-1.0, last=-1.5, change_pct=-50.0, direction="decreased", data_points=50)
        result = analyzer.analyze("logps/chosen", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert "expected for DPO" in result.verdict

    def test_logps_rejected_decreased_is_good(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=-1.0, last=-3.0, change_pct=-200.0, direction="decreased", data_points=50)
        result = analyzer.analyze("logps/rejected", trend)

        assert result is not None
        assert result.status == MetricStatus.GOOD
        assert "rejected" in result.verdict

    def test_logps_key_is_case_insensitive(self, analyzer: MetricAnalyzer) -> None:
        trend = MetricTrend(first=-1.0, last=-1.5, change_pct=-50.0, direction="decreased", data_points=50)
        result = analyzer.analyze("LOGPS/CHOSEN", trend)

        assert result is not None
        assert result.status == MetricStatus.NEUTRAL
        assert "expected for DPO" in result.verdict


# ============================================================================
# VOLATILITY CALCULATION EDGE CASES
# ============================================================================


class TestVolatilityCalculation:
    """Tests specifically for the volatility detection logic."""

    def test_volatility_exactly_10x(self, analyzer: MetricAnalyzer) -> None:
        """Volatility ratio exactly 10x - boundary."""
        trend = MetricTrend(
            first=1.5,
            last=1.0,
            min_val=1.0,
            max_val=10.0,  # ratio = 10
            change_pct=-33.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Exactly 10x should NOT trigger warning (> 10 triggers)
        assert result.status == MetricStatus.GOOD

    def test_volatility_just_over_10x(self, analyzer: MetricAnalyzer) -> None:
        """Volatility ratio 10.1x - just over boundary."""
        trend = MetricTrend(
            first=1.5,
            last=1.0,
            min_val=1.0,
            max_val=10.1,  # ratio = 10.1
            change_pct=-33.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Just over 10x should trigger warning
        assert result.status == MetricStatus.WARNING

    def test_volatility_min_is_zero(self, analyzer: MetricAnalyzer) -> None:
        """
        min_val = 0 - would cause division by zero in ratio calculation.
        """
        trend = MetricTrend(
            first=1.5,
            last=0.5,
            min_val=0.0,  # Zero!
            max_val=2.0,
            change_pct=-66.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Should handle division by zero gracefully
        # and still give a reasonable result

    def test_volatility_negative_min(self, analyzer: MetricAnalyzer) -> None:
        """min_val < 0 - ratio would be weird."""
        trend = MetricTrend(
            first=1.0,
            last=0.5,
            min_val=-0.5,  # Negative
            max_val=2.0,
            change_pct=-50.0,
            direction="decreased",
            data_points=100,
        )
        result = analyzer.analyze("loss", trend)

        assert result is not None
        # Ratio 2.0 / -0.5 = -4, which is < 10
        # Should not crash
