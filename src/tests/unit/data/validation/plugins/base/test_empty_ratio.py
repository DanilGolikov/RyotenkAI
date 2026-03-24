"""Tests for EmptyRatioValidator plugin."""

from datasets import Dataset

from src.data.validation.plugins.base.empty_ratio import EmptyRatioValidator


class TestEmptyRatioValidator:
    """Test suite for EmptyRatioValidator."""

    def test_positive_case_low_empty_ratio(self):
        """Positive: Dataset with low empty ratio."""
        texts = ["Valid text content"] * 10
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.2})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["empty_ratio"] <= 0.2
        assert len(result.errors) == 0

    def test_negative_case_high_empty_ratio(self):
        """Negative: Dataset with too many empty samples."""
        texts = ["", "", "", "valid", ""]
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.2})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["empty_ratio"] > 0.2
        assert len(result.errors) > 0
        assert "empty examples" in result.errors[0].lower()

    def test_boundary_case_exact_threshold(self):
        """Boundary: Empty ratio exactly at threshold."""
        # 10% empty (1 of 10) — short text counts as empty
        texts = ["valid text content"] * 9 + ["x"]  # "x" = 1 char < 10 (default min_chars)
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.1})
        result = plugin.validate(dataset)

        # Should pass as it's exactly at threshold
        assert result.passed is True
        assert result.metrics["empty_ratio"] == 0.1

    def test_edge_case_all_empty(self):
        """Edge: All samples empty."""
        texts = ["", "   ", "\t\n"]
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.5})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["empty_ratio"] == 1.0
        assert result.metrics["empty_count"] == 3

    def test_edge_case_no_empty(self):
        """Edge: No empty samples."""
        texts = ["valid text"] * 10
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.1})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["empty_ratio"] == 0.0
        assert result.metrics["empty_count"] == 0

    def test_custom_min_chars(self):
        """Test: Custom min_chars parameter."""
        # Text with 5 chars should be empty if min_chars=10
        texts = ["short"] * 5 + ["long enough text"] * 5
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(params={"min_chars": 10}, thresholds={"max_ratio": 0.6})
        result = plugin.validate(dataset)

        # 50% are "empty" (less than 10 chars)
        assert result.passed is True
        assert result.metrics["empty_ratio"] == 0.5

    def test_large_dataset_sampling(self, large_dataset):
        """Test: Large dataset uses sampling."""
        plugin = EmptyRatioValidator(params={"sample_size": 100}, thresholds={"max_ratio": 0.5})
        result = plugin.validate(large_dataset)

        # Large datasets are sampled
        assert result.metrics["total_checked"] == 1000
        assert result.passed is True

    def test_streaming_dataset_support(self):
        """Test: Works with streaming datasets."""
        from datasets import IterableDataset

        def gen():
            for i in range(50):
                yield {"text": f"Sample {i} with valid content"}

        dataset = IterableDataset.from_generator(gen)

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.1})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["total_checked"] > 0

    def test_warning_close_to_threshold(self):
        """Test: Warning when ratio close to threshold."""
        # 7% empty, near 10% threshold — short texts below min_chars
        texts = ["valid text content"] * 93 + ["short"] * 7  # "short" = 5 chars < 10
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.1})
        result = plugin.validate(dataset)

        # Should have warning as 7% is in range (5%-10%)
        assert result.passed is True
        assert len(result.warnings) > 0
        assert "close to the threshold" in result.warnings[0].lower()

    def test_recommendations_on_failure(self):
        """Test: Recommendations provided on failure."""
        texts = [""] * 8 + ["valid"] * 2
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.2})
        result = plugin.validate(dataset)

        assert result.passed is False

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 4
        assert any("empty examples" in rec.lower() for rec in recommendations)
        assert any("remove" in rec.lower() for rec in recommendations)

    def test_recommendations_critical_high_ratio(self):
        """Test: Critical warning for very high empty ratio."""
        texts = [""] * 25 + ["valid"] * 5
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.5})
        result = plugin.validate(dataset)

        assert result.passed is False

        recommendations = plugin.get_recommendations(result)
        # Should have critical warning (>20%)
        assert any("CRITICAL" in rec for rec in recommendations)

    def test_default_parameters(self):
        """Test: Default max_ratio (0.1) and min_chars (10)."""
        texts = ["valid text"] * 10
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator({})
        result = plugin.validate(dataset)

        assert result.thresholds["max_ratio"] == 0.1

    def test_execution_time_recorded(self):
        """Test: Execution time is recorded."""
        texts = ["sample"] * 20
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.1})
        result = plugin.validate(dataset)

        assert result.execution_time_ms >= 0
        assert result.execution_time_ms < 1000  # Should be fast

    def test_whitespace_only_counts_as_empty(self):
        """Test: Whitespace-only text counts as empty."""
        texts = ["   ", "\t\t", "\n\n", "valid text"]
        dataset = Dataset.from_dict({"text": texts})

        plugin = EmptyRatioValidator(thresholds={"max_ratio": 0.5})
        result = plugin.validate(dataset)

        # 3 out of 4 are empty (whitespace only)
        assert result.metrics["empty_count"] == 3
        assert result.metrics["empty_ratio"] == 0.75
