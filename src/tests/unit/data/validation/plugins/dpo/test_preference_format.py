"""Tests for PreferenceFormatValidator plugin (DPO)."""

from datasets import Dataset

from src.data.validation.plugins.dpo.preference_format import PreferenceFormatValidator


class TestPreferenceFormatValidator:
    """Test suite for PreferenceFormatValidator."""

    def test_positive_case_valid_format(self, dpo_valid_dataset):
        """Positive: All samples have valid DPO format."""
        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.95})
        result = plugin.validate(dpo_valid_dataset)

        assert result.passed is True
        assert result.metrics["valid_ratio"] == 1.0
        assert result.metrics["invalid_count"] == 0
        assert len(result.errors) == 0

    def test_negative_case_missing_field(self, dpo_invalid_format_dataset):
        """Negative: Dataset missing required fields."""
        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.95})
        result = plugin.validate(dpo_invalid_format_dataset)

        assert result.passed is False
        assert result.metrics["valid_ratio"] < 0.95
        assert result.metrics["invalid_count"] > 0
        assert any("Too many invalid examples" in error for error in result.errors)

    def test_boundary_case_exact_threshold(self):
        """Boundary: Valid ratio exactly at threshold."""
        # Create dataset with exactly 95% valid (95 out of 100)
        # Use dataset with missing 'rejected' field in 5 samples
        samples = []
        for _ in range(95):
            samples.append({"chosen": "good", "rejected": "bad"})
        for _ in range(5):
            samples.append({"chosen": "good"})  # Missing rejected field

        from datasets import Dataset

        dataset = Dataset.from_list(samples)

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.95})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["valid_ratio"] == 0.95

    def test_edge_case_all_invalid(self):
        """Edge: All samples are invalid."""
        dataset = Dataset.from_dict(
            {
                "wrong_field": ["data"] * 10,
                # Missing both chosen and rejected
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.0})
        result = plugin.validate(dataset)

        assert result.metrics["valid_ratio"] == 0.0
        assert result.metrics["invalid_count"] == 10

    def test_edge_case_single_sample_valid(self):
        """Edge: Single valid sample."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["good"],
                "rejected": ["bad"],
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 1.0})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["valid_count"] == 1

    def test_edge_case_single_sample_invalid(self):
        """Edge: Single invalid sample."""
        dataset = Dataset.from_dict({"prompt": ["question"]})  # Missing chosen/rejected

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.0})
        result = plugin.validate(dataset)

        assert result.metrics["invalid_count"] == 1

    def test_default_parameters(self, dpo_valid_dataset):
        """Test: Default min_valid_ratio (0.95)."""
        plugin = PreferenceFormatValidator({})
        result = plugin.validate(dpo_valid_dataset)

        assert result.thresholds["min_valid_ratio"] == 0.95

    def test_custom_required_fields(self):
        """Test: Custom required fields."""
        dataset = Dataset.from_dict(
            {
                "prompt": ["Q1", "Q2"],
                "chosen": ["A1", "A2"],
                "rejected": ["B1", "B2"],
            }
        )

        plugin = PreferenceFormatValidator(
            params={"required_fields": ["prompt", "chosen", "rejected"]},
        )
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["valid_ratio"] == 1.0

    def test_warning_some_invalid(self):
        """Test: Warning when some samples invalid."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["good"] * 9 + ["bad"],
                "rejected": ["bad"] * 9 + [None],  # Last one has None
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.8})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert len(result.warnings) > 0
        assert "invalid examples" in result.warnings[0].lower()

    def test_recommendations_low_validity(self):
        """Test: Recommendations for low validity ratio."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["good"] * 5,
                # Missing rejected field
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.95})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 4
        assert any("Check DPO dataset format" in rec for rec in recommendations)

    def test_recommendations_critical(self):
        """Test: Critical recommendations when <80% valid."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["good"] * 7,
                "rejected": [None] * 7,  # Invalid types
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.95})
        result = plugin.validate(dataset)

        recommendations = plugin.get_recommendations(result)
        assert any("CRITICAL" in rec for rec in recommendations)

    def test_crazy_case_zero_tolerance(self):
        """Crazy: Zero invalid samples allowed."""
        # 99 valid + 1 invalid (missing rejected field)
        samples = []
        for _ in range(99):
            samples.append({"chosen": "good", "rejected": "bad"})
        samples.append({"chosen": "good"})  # Missing rejected

        from datasets import Dataset

        dataset = Dataset.from_list(samples)

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 1.0})
        result = plugin.validate(dataset)

        assert result.passed is False  # 99% < 100%

    def test_crazy_case_negative_threshold(self):
        """Crazy: Negative threshold."""
        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": -0.5})
        dataset = Dataset.from_dict({"chosen": ["a"], "rejected": ["b"]})
        result = plugin.validate(dataset)

        # Should pass since any ratio > -0.5
        assert result.passed is True

    def test_dict_format_values(self):
        """Test: Dict format for chosen/rejected."""
        dataset = Dataset.from_dict(
            {
                "chosen": [{"content": "good1"}, {"content": "good2"}],
                "rejected": [{"content": "bad1"}, {"content": "bad2"}],
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.95})
        result = plugin.validate(dataset)

        assert result.passed is True

    def test_list_format_values(self):
        """Test: List format for chosen/rejected."""
        dataset = Dataset.from_dict(
            {
                "chosen": [["msg1", "msg2"], ["msg3"]],
                "rejected": [["msg4"], ["msg5", "msg6"]],
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.95})
        result = plugin.validate(dataset)

        assert result.passed is True

    def test_mixed_invalid_types(self):
        """Test: Various invalid type combinations."""
        dataset = Dataset.from_dict(
            {
                "chosen": ["valid", "none_str", "123_str", "valid", "empty_list"],
                "rejected": ["valid", "valid", "valid", "none_str", "dict_str"],
            }
        )

        plugin = PreferenceFormatValidator(thresholds={"min_valid_ratio": 0.5})
        result = plugin.validate(dataset)

        # All are valid strings now, so should pass
        assert result.metrics["valid_count"] == 5
