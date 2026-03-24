"""Tests for AvgLengthValidator plugin."""

from datasets import Dataset

from src.data.validation.plugins.base.avg_length import AvgLengthValidator


class TestAvgLengthValidator:
    """Test suite for AvgLengthValidator."""

    def test_positive_case_normal_length(self, small_dataset):
        """Positive: Average length within range."""
        plugin = AvgLengthValidator(thresholds={"min": 10, "max": 100})
        result = plugin.validate(small_dataset)

        assert result.passed is True
        assert 10 <= result.metrics["avg_length"] <= 100
        assert len(result.errors) == 0

    def test_negative_case_too_short(self, short_samples_dataset):
        """Negative: Average length below minimum."""
        plugin = AvgLengthValidator(thresholds={"min": 50, "max": 2048})
        result = plugin.validate(short_samples_dataset)

        assert result.passed is False
        assert result.metrics["avg_length"] < 50
        assert any("too low" in error.lower() for error in result.errors)

    def test_negative_case_too_long(self):
        """Negative: Average length above maximum."""
        long_texts = ["x" * 1000 for _ in range(10)]
        dataset = Dataset.from_dict({"text": long_texts})

        plugin = AvgLengthValidator(thresholds={"min": 10, "max": 500})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["avg_length"] > 500
        assert any("too high" in error.lower() for error in result.errors)

    def test_boundary_case_exact_min(self):
        """Boundary: Average exactly at minimum."""
        dataset = Dataset.from_dict({"text": ["x" * 50] * 10})
        plugin = AvgLengthValidator(thresholds={"min": 50, "max": 100})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["avg_length"] == 50

    def test_boundary_case_exact_max(self):
        """Boundary: Average exactly at maximum."""
        dataset = Dataset.from_dict({"text": ["x" * 100] * 10})
        plugin = AvgLengthValidator(thresholds={"min": 50, "max": 100})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["avg_length"] == 100

    def test_edge_case_empty_strings(self):
        """Edge: All empty strings."""
        dataset = Dataset.from_dict({"text": ["", "", ""]})
        plugin = AvgLengthValidator(thresholds={"min": 1, "max": 100})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["avg_length"] == 0

    def test_messages_format(self, messages_format_dataset):
        """Test: Messages format extraction."""
        plugin = AvgLengthValidator(thresholds={"min": 5, "max": 200})
        result = plugin.validate(messages_format_dataset)

        assert result.passed is True
        assert result.metrics["avg_length"] > 0

    def test_default_parameters(self, small_dataset):
        """Test: Default min/max values."""
        plugin = AvgLengthValidator({})
        result = plugin.validate(small_dataset)

        assert result.thresholds["min"] == 50
        assert result.thresholds["max"] == 8192

    def test_warning_close_to_min(self):
        """Test: Warning when close to minimum."""
        dataset = Dataset.from_dict({"text": ["x" * 55] * 10})
        plugin = AvgLengthValidator(thresholds={"min": 50, "max": 1000})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert len(result.warnings) > 0
        assert "close to the minimum" in result.warnings[0].lower()

    def test_recommendations_too_short(self, short_samples_dataset):
        """Test: Recommendations for too short text."""
        plugin = AvgLengthValidator(thresholds={"min": 100, "max": 2048})
        result = plugin.validate(short_samples_dataset)

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 3
        assert any("truncated" in rec.lower() for rec in recommendations)

    def test_recommendations_too_long(self):
        """Test: Recommendations for too long text."""
        long_texts = ["x" * 10000 for _ in range(5)]
        dataset = Dataset.from_dict({"text": long_texts})

        plugin = AvgLengthValidator(thresholds={"min": 10, "max": 1000})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 3
        assert any("duplicated" in rec.lower() or "truncat" in rec.lower() for rec in recommendations)

    def test_crazy_case_inverted_min_max(self):
        """Crazy: Min > Max."""
        dataset = Dataset.from_dict({"text": ["text"] * 10})
        plugin = AvgLengthValidator(thresholds={"min": 1000, "max": 10})
        result = plugin.validate(dataset)

        # Will always fail since can't be both > 1000 and < 10
        assert result.passed is False

    def test_crazy_case_negative_thresholds(self):
        """Crazy: Negative thresholds."""
        dataset = Dataset.from_dict({"text": ["test"]})
        plugin = AvgLengthValidator(thresholds={"min": -100, "max": -10})
        result = plugin.validate(dataset)

        # Avg length is ~4, which is > -10, so will fail max check
        assert result.passed is False

    def test_multiple_formats_mixed(self):
        """Test: Mixed formats in dataset."""
        dataset = Dataset.from_dict(
            {
                "text": ["plain text"],
                "messages": [[{"role": "user", "content": "msg"}]],
                "input": ["input1"],
                "output": ["output1"],
            }
        )

        plugin = AvgLengthValidator(thresholds={"min": 1, "max": 100})
        # Should handle gracefully
        result = plugin.validate(dataset)
        assert result.metrics["avg_length"] > 0

    def test_sample_size_large_dataset(self, large_dataset):
        """Test: Sample size parameter for large datasets."""
        plugin = AvgLengthValidator(params={"sample_size": 100}, thresholds={"min": 10, "max": 200})
        result = plugin.validate(large_dataset)

        # For datasets < 100k, all samples are checked
        # For datasets >= 100k, sample_size is used
        # large_dataset fixture has 1000 samples (< 100k), so all are checked
        assert result.metrics["samples_checked"] == 1000
