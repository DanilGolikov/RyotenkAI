"""Tests for IdenticalPairsValidator plugin (DPO)."""

from datasets import Dataset

from src.data.validation.plugins.dpo.identical_pairs import IdenticalPairsValidator


class TestIdenticalPairsValidator:
    """Test suite for IdenticalPairsValidator."""

    def test_positive_case_no_identical_pairs(self, dpo_valid_dataset):
        """Positive: No identical chosen/rejected pairs."""
        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dpo_valid_dataset)

        assert result.passed is True
        assert result.metrics["identical_ratio"] == 0.0
        assert result.metrics["identical_count"] == 0
        assert len(result.errors) == 0

    def test_negative_case_many_identical(self, dpo_identical_pairs_dataset):
        """Negative: Too many identical pairs."""
        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dpo_identical_pairs_dataset)

        assert result.passed is False
        assert result.metrics["identical_count"] == 2  # 2 out of 3
        assert result.metrics["identical_ratio"] > 0.5
        assert any("Too many identical pairs" in error for error in result.errors)

    def test_boundary_case_exact_threshold(self):
        """Boundary: Identical ratio exactly at threshold."""
        # 1 identical out of 100 = 1%
        dataset = Dataset.from_dict(
            {
                "chosen": ["good"] * 99 + ["same"],
                "rejected": ["bad"] * 99 + ["same"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["identical_ratio"] == 0.01

    def test_edge_case_all_identical(self):
        """Edge: All pairs are identical."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["same text"] * 10,
                "rejected": ["same text"] * 10,
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["identical_ratio"] == 1.0
        assert result.metrics["identical_count"] == 10

    def test_edge_case_single_pair_identical(self):
        """Edge: Single pair that is identical."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["same"],
                "rejected": ["same"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.0})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert result.metrics["identical_count"] == 1

    def test_edge_case_single_pair_different(self):
        """Edge: Single pair that is different."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["good"],
                "rejected": ["bad"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.0})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["identical_count"] == 0

    def test_default_threshold(self, dpo_valid_dataset):
        """Test: Default max_identical_ratio (0.01 = 1%)."""
        plugin = IdenticalPairsValidator({})
        result = plugin.validate(dpo_valid_dataset)

        assert result.thresholds["max_identical_ratio"] == 0.01

    def test_case_insensitive_comparison(self):
        """Test: Comparison is case-insensitive."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["SAME TEXT", "different"],
                "rejected": ["same text", "other"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.0})
        result = plugin.validate(dataset)

        # Should detect "SAME TEXT" == "same text" (after normalization)
        assert result.metrics["identical_count"] == 1

    def test_whitespace_normalization(self):
        """Test: Whitespace is normalized."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["  text  ", "different"],
                "rejected": ["text", "other"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.0})
        result = plugin.validate(dataset)

        # Should detect "  text  " == "text" (after strip)
        assert result.metrics["identical_count"] == 1

    def test_warning_some_identical(self):
        """Test: Warning when some pairs identical but still passing."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["good"] * 199 + ["same"],
                "rejected": ["bad"] * 199 + ["same"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert len(result.warnings) > 0
        assert "identical pairs" in result.warnings[0].lower()

    def test_recommendations_moderate(self):
        """Test: Recommendations for moderate identical ratio."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["same"] * 3 + ["good"] * 7,
                "rejected": ["same"] * 3 + ["bad"] * 7,
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 4
        assert any("Remove examples where" in rec for rec in recommendations)

    def test_recommendations_critical(self):
        """Test: Critical recommendations for >5% identical."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["same"] * 10 + ["good"] * 90,
                "rejected": ["same"] * 10 + ["bad"] * 90,
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert any("CRITICAL" in rec for rec in recommendations)

    def test_crazy_case_100_percent_allowed(self):
        """Crazy: 100% identical pairs allowed."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["same"] * 100,
                "rejected": ["same"] * 100,
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 1.0})
        result = plugin.validate(dataset)

        assert result.passed is True

    def test_dict_format_messages(self):
        """Test: Dict format with content field."""
        dataset = Dataset.from_dict(
            {
                "chosen": [{"content": "good"}, {"content": "same"}],
                "rejected": [{"content": "bad"}, {"content": "same"}],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.5})
        result = plugin.validate(dataset)

        # Should detect second pair as identical
        assert result.metrics["identical_count"] == 1

    def test_list_format_messages(self):
        """Test: List format (multiple messages)."""
        dataset = Dataset.from_dict(
            {
                "chosen": [
                    [{"content": "msg1"}, {"content": "msg2"}],
                    [{"content": "same1"}, {"content": "same2"}],
                ],
                "rejected": [
                    [{"content": "diff1"}, {"content": "diff2"}],
                    [{"content": "same1"}, {"content": "same2"}],
                ],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.5})
        result = plugin.validate(dataset)

        # Should detect second pair as identical
        assert result.metrics["identical_count"] == 1

    def test_missing_fields_skipped(self):
        """Test: Samples missing fields are skipped."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["good", None, "different"],
                "rejected": ["bad", "something", "other"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.5})
        result = plugin.validate(dataset)

        # Should only check samples with both fields
        assert result.metrics["total_checked"] <= 3

    def test_examples_in_errors(self):
        """Test: Structured grouped errors include example indices."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["same", "same", "good"],
                "rejected": ["same", "same", "bad"],
            }
        )

        plugin = IdenticalPairsValidator(thresholds={"max_identical_ratio": 0.01})
        result = plugin.validate(dataset)

        assert len(result.errors) == 1
        assert len(result.error_groups) == 1
        assert result.error_groups[0].error_type == "identical_pair"
        assert result.error_groups[0].sample_indices == [0, 1]
