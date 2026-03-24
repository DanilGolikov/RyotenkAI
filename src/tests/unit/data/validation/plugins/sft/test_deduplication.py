"""Tests for DeduplicationValidator plugin."""

from datasets import Dataset

from src.data.validation.plugins.sft.deduplication import DeduplicationValidator


class TestDeduplicationValidator:
    """Test suite for DeduplicationValidator."""

    def test_positive_case_no_duplicates(self, small_dataset):
        """Positive: No duplicates in dataset."""
        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.1})
        result = plugin.validate(small_dataset)

        assert result.passed is True
        assert result.metrics["duplicate_ratio"] == 0.0
        assert result.metrics["duplicate_count"] == 0
        assert len(result.errors) == 0

    def test_negative_case_many_duplicates(self, duplicate_dataset):
        """Negative: Too many duplicates."""
        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.1})
        result = plugin.validate(duplicate_dataset)

        assert result.passed is False
        assert result.metrics["duplicate_ratio"] > 0.1
        assert result.metrics["duplicate_count"] > 1
        assert any("Too many duplicates" in error for error in result.errors)

    def test_boundary_case_exact_threshold(self):
        """Boundary: Duplicate ratio exactly at threshold."""
        # 10 samples, 1 duplicate = 10% ratio
        dataset = Dataset.from_dict(
            {
                "text": [
                    "unique1",
                    "unique2",
                    "duplicate",
                    "unique3",
                    "unique4",
                    "unique5",
                    "unique6",
                    "unique7",
                    "unique8",
                    "duplicate",
                ]
            }
        )

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.1})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["duplicate_ratio"] == 0.1

    def test_edge_case_all_duplicates(self):
        """Edge: All samples are identical."""
        dataset = Dataset.from_dict({"text": ["same text"] * 10})

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.1})
        result = plugin.validate(dataset)

        assert result.passed is False
        # 9 out of 10 are duplicates = 90%
        assert result.metrics["duplicate_ratio"] == 0.9
        assert result.metrics["duplicate_count"] == 9

    def test_edge_case_one_sample(self):
        """Edge: Single sample (can't have duplicates)."""
        dataset = Dataset.from_dict({"text": ["only one"]})

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.0})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["duplicate_count"] == 0

    def test_default_threshold(self, duplicate_dataset):
        """Test: Default max_duplicate_ratio (0.1)."""
        plugin = DeduplicationValidator({})
        result = plugin.validate(duplicate_dataset)

        assert result.thresholds["max_duplicate_ratio"] == 0.1

    def test_warning_close_to_threshold(self):
        """Test: Warning when close to threshold."""
        # 8% duplicates (close to 10% threshold)
        dataset = Dataset.from_dict(
            {"text": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9", "u10", "u11", "u12", "dup"] * 4 + ["dup"]}
        )

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.1})
        result = plugin.validate(dataset)

        # Should have warning
        if result.passed:
            assert len(result.warnings) > 0

    def test_recommendations_high_duplicates(self):
        """Test: Recommendations for high duplicate ratio."""
        dataset = Dataset.from_dict({"text": ["dup"] * 8 + ["u1", "u2"]})

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.1})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 4
        assert any("Remove duplicate examples" in rec for rec in recommendations)

    def test_recommendations_critical(self):
        """Test: Critical recommendations for very high duplicates."""
        dataset = Dataset.from_dict({"text": ["dup"] * 10})

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.1})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert any("CRITICAL" in rec for rec in recommendations)

    def test_crazy_case_zero_tolerance(self, small_dataset):
        """Crazy: Zero tolerance for duplicates."""
        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.0})
        result = plugin.validate(small_dataset)

        # Should pass since small_dataset has no duplicates
        assert result.passed is True

    def test_crazy_case_100_percent_allowed(self):
        """Crazy: 100% duplicates allowed."""
        dataset = Dataset.from_dict({"text": ["dup"] * 100})

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 1.0})
        result = plugin.validate(dataset)

        assert result.passed is True  # Everything is allowed

    def test_similar_but_not_identical(self):
        """Test: Similar texts that are not identical."""
        dataset = Dataset.from_dict(
            {
                "text": [
                    "The quick brown fox",
                    "The quick brown dog",
                    "The slow brown fox",
                    "A quick brown fox",
                ]
            }
        )

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.0})
        result = plugin.validate(dataset)

        # MD5 hashing means these are NOT duplicates
        assert result.passed is True
        assert result.metrics["duplicate_count"] == 0

    def test_whitespace_differences(self):
        """Test: Texts differing only in whitespace."""
        dataset = Dataset.from_dict(
            {
                "text": [
                    "text",
                    "text ",  # Trailing space
                    " text",  # Leading space
                    "text",  # Exact duplicate
                ]
            }
        )

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.5})
        result = plugin.validate(dataset)

        # MD5 will detect whitespace differences, so only 1 duplicate
        assert result.metrics["duplicate_count"] >= 1

    def test_case_sensitivity(self):
        """Test: Case sensitivity in duplicate detection."""
        dataset = Dataset.from_dict({"text": ["TEXT", "text", "Text", "TEXT"]})

        plugin = DeduplicationValidator(thresholds={"max_duplicate_ratio": 0.5})
        result = plugin.validate(dataset)

        # MD5 is case-sensitive, so only "TEXT" is duplicated
        assert result.metrics["duplicate_count"] == 1
