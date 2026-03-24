"""Tests for MinSamplesValidator plugin."""

from datasets import Dataset

from src.data.validation.plugins.base.min_samples import MinSamplesValidator


class TestMinSamplesValidator:
    """Test suite for MinSamplesValidator."""

    def test_positive_case_pass(self, small_dataset):
        """Positive: Dataset has enough samples."""
        plugin = MinSamplesValidator(thresholds={"threshold": 5})
        result = plugin.validate(small_dataset)

        assert result.passed is True
        assert result.metrics["sample_count"] == 10
        assert result.thresholds["threshold"] == 5
        assert len(result.errors) == 0

    def test_positive_case_exact_threshold(self):
        """Positive: Dataset has exactly threshold samples."""
        dataset = Dataset.from_dict({"text": ["sample"] * 10})
        plugin = MinSamplesValidator(thresholds={"threshold": 10})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["sample_count"] == 10

    def test_negative_case_insufficient_samples(self):
        """Negative: Dataset has too few samples."""
        dataset = Dataset.from_dict({"text": ["sample"] * 5})
        plugin = MinSamplesValidator(thresholds={"threshold": 100})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["sample_count"] == 5
        assert result.thresholds["threshold"] == 100
        assert len(result.errors) == 1
        assert "Not enough examples" in result.errors[0]

    def test_boundary_case_one_sample(self):
        """Boundary: Dataset with single sample."""
        dataset = Dataset.from_dict({"text": ["only one"]})
        plugin = MinSamplesValidator(thresholds={"threshold": 1})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["sample_count"] == 1

    def test_boundary_case_zero_samples(self):
        """Boundary: Empty dataset."""
        dataset = Dataset.from_dict({"text": []})
        plugin = MinSamplesValidator(thresholds={"threshold": 1})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["sample_count"] == 0

    def test_edge_case_large_threshold(self, small_dataset):
        """Edge: Very large threshold."""
        plugin = MinSamplesValidator(thresholds={"threshold": 1000000})
        result = plugin.validate(small_dataset)

        assert result.passed is False
        assert result.thresholds["threshold"] == 1000000

    def test_default_threshold(self, small_dataset):
        """Test: Default threshold (100)."""
        plugin = MinSamplesValidator({})
        result = plugin.validate(small_dataset)

        assert result.passed is False  # small_dataset has only 10 samples
        assert result.thresholds["threshold"] == 100

    def test_recommendations_critical(self):
        """Test: Critical recommendations when very few samples."""
        dataset = Dataset.from_dict({"text": ["a", "b"]})
        plugin = MinSamplesValidator(thresholds={"threshold": 100})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 4
        assert any("CRITICAL" in rec for rec in recommendations)

    def test_recommendations_normal(self):
        """Test: Normal recommendations."""
        dataset = Dataset.from_dict({"text": ["sample"] * 80})
        plugin = MinSamplesValidator(thresholds={"threshold": 100})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) >= 4
        assert "Add 20 more examples" in " ".join(recommendations)

    def test_crazy_case_negative_threshold(self):
        """Crazy: Negative threshold."""
        plugin = MinSamplesValidator(thresholds={"threshold": -10})
        dataset = Dataset.from_dict({"text": ["a"]})
        result = plugin.validate(dataset)

        # Should pass since 1 > -10
        assert result.passed is True

    def test_crazy_case_zero_threshold(self, small_dataset):
        """Crazy: Zero threshold."""
        plugin = MinSamplesValidator(thresholds={"threshold": 0})
        result = plugin.validate(small_dataset)

        assert result.passed is True

    def test_execution_time_recorded(self, small_dataset):
        """Test: Execution time is recorded."""
        plugin = MinSamplesValidator(thresholds={"threshold": 5})
        result = plugin.validate(small_dataset)

        assert result.execution_time_ms >= 0
        assert result.execution_time_ms < 1000  # Should be fast

    def test_different_thresholds(self, large_dataset):
        """Test: Various threshold values."""
        thresholds = [10, 50, 100, 500, 1000, 2000]

        for threshold in thresholds:
            plugin = MinSamplesValidator(thresholds={"threshold": threshold})
            result = plugin.validate(large_dataset)

            if threshold <= 1000:
                assert result.passed is True
            else:
                assert result.passed is False
