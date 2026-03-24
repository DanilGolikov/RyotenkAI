"""Tests for custom plugin creation."""

from datasets import Dataset

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry


class TestCustomPluginCreation:
    """Test suite for creating custom validation plugins."""

    def test_create_simple_custom_plugin(self):
        """Test: Create and use a simple custom plugin."""

        @ValidationPluginRegistry.register
        class CustomSimplePlugin(ValidationPlugin):
            """Simple custom plugin that always passes."""

            name = "custom_simple"

            @classmethod
            def get_description(cls) -> str:
                return "Custom plugin for testing"

            def validate(self, dataset):
                return ValidationResult(
                    plugin_name=self.name,
                    passed=True,
                    params=dict(self.params),
                    thresholds=dict(self.thresholds),
                    metrics={"custom_metric": 100.0},
                    warnings=[],
                    errors=[],
                    execution_time_ms=1.0,
                )

            def get_recommendations(self, result):
                return ["Custom recommendation"]

        # Use the plugin
        dataset = Dataset.from_dict({"text": ["test"]})
        plugin = ValidationPluginRegistry.get_plugin("custom_simple", {})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["custom_metric"] == 100.0

        recommendations = plugin.get_recommendations(result)
        assert "Custom recommendation" in recommendations

    def test_create_parameterized_custom_plugin(self):
        """Test: Custom plugin with parameters."""

        @ValidationPluginRegistry.register
        class CustomParamsPlugin(ValidationPlugin):
            """Custom plugin with configurable threshold."""

            name = "custom_params"

            @classmethod
            def get_description(cls) -> str:
                return "Custom plugin with parameters"

            def validate(self, dataset):
                threshold = self.thresholds.get("threshold", 50)
                value = len(dataset)

                passed = value >= threshold

                return ValidationResult(
                    plugin_name=self.name,
                    passed=passed,
                    params=dict(self.params),
                    thresholds={"threshold": float(threshold)},
                    metrics={"value": float(value)},
                    warnings=[],
                    errors=[] if passed else [f"Value {value} < threshold {threshold}"],
                    execution_time_ms=0.5,
                )

            def get_recommendations(self, result):
                if not result.passed:
                    return ["Increase dataset size", "Lower threshold"]
                return []

        # Test with different thresholds
        dataset = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})

        # Should pass with low threshold
        plugin1 = ValidationPluginRegistry.get_plugin("custom_params", {}, {"threshold": 3})
        result1 = plugin1.validate(dataset)
        assert result1.passed is True

        # Should fail with high threshold
        plugin2 = ValidationPluginRegistry.get_plugin("custom_params", {}, {"threshold": 10})
        result2 = plugin2.validate(dataset)
        assert result2.passed is False

        recommendations = plugin2.get_recommendations(result2)
        assert len(recommendations) == 2

    def test_create_field_checking_plugin(self):
        """Test: Custom plugin that checks specific fields."""

        @ValidationPluginRegistry.register
        class CustomFieldCheckPlugin(ValidationPlugin):
            """Checks if dataset has specific fields."""

            name = "custom_field_check"
            required_fields = ["id", "content"]

            @classmethod
            def get_description(cls) -> str:
                return "Checks for required fields"

            def validate(self, dataset):
                required_fields = self.params.get("fields", self.required_fields)
                dataset_fields = dataset.column_names

                missing = [f for f in required_fields if f not in dataset_fields]

                passed = len(missing) == 0

                return ValidationResult(
                    plugin_name=self.name,
                    passed=passed,
                    params=dict(self.params),
                    thresholds=dict(self.thresholds),
                    metrics={
                        "required_count": float(len(required_fields)),
                        "missing_count": float(len(missing)),
                    },
                    warnings=[],
                    errors=[f"Missing fields: {missing}"] if missing else [],
                    execution_time_ms=0.1,
                )

            def get_recommendations(self, result):
                if not result.passed:
                    return [
                        "Add missing fields to dataset",
                        "Check data pipeline for field mapping issues",
                    ]
                return []

        # Test with correct fields
        dataset_ok = Dataset.from_dict({"id": [1, 2], "content": ["a", "b"]})
        plugin = ValidationPluginRegistry.get_plugin("custom_field_check", {})
        result = plugin.validate(dataset_ok)

        assert result.passed is True

        # Test with missing fields
        dataset_bad = Dataset.from_dict({"id": [1, 2]})  # Missing 'content'
        result_bad = plugin.validate(dataset_bad)

        assert result_bad.passed is False
        assert "content" in str(result_bad.errors)

    def test_create_statistical_plugin(self):
        """Test: Custom plugin with statistical checks."""

        @ValidationPluginRegistry.register
        class CustomStatsPlugin(ValidationPlugin):
            """Checks statistical properties of numeric field."""

            name = "custom_stats"

            @classmethod
            def get_description(cls) -> str:
                return "Statistical validation plugin"

            def validate(self, dataset):
                field = self.params.get("field", "value")
                min_mean = self.thresholds.get("min_mean", 0)
                max_mean = self.thresholds.get("max_mean", 100)

                if field not in dataset.column_names:
                    return ValidationResult(
                        plugin_name=self.name,
                        passed=False,
                        params=dict(self.params),
                        thresholds=dict(self.thresholds),
                        metrics={},
                        warnings=[],
                        errors=[f"Field '{field}' not found"],
                        execution_time_ms=0.1,
                    )

                values = dataset[field]
                mean = sum(values) / len(values) if values else 0

                passed = min_mean <= mean <= max_mean

                return ValidationResult(
                    plugin_name=self.name,
                    passed=passed,
                    params=dict(self.params),
                    thresholds={
                        "min_mean": float(min_mean),
                        "max_mean": float(max_mean),
                    },
                    metrics={
                        "mean": mean,
                    },
                    warnings=[],
                    errors=[] if passed else [f"Mean {mean} not in range [{min_mean}, {max_mean}]"],
                    execution_time_ms=0.5,
                )

            def get_recommendations(self, result):
                return ["Check data distribution", "Review preprocessing"]

        # Test with numeric dataset
        dataset = Dataset.from_dict({"value": [10, 20, 30, 40, 50]})
        plugin = ValidationPluginRegistry.get_plugin(
            "custom_stats",
            {"field": "value"},
            {"min_mean": 20, "max_mean": 40},
        )
        result = plugin.validate(dataset)

        assert result.passed is True
        assert result.metrics["mean"] == 30.0

    def test_create_complex_custom_plugin(self):
        """Test: Complex custom plugin with multiple checks."""

        @ValidationPluginRegistry.register
        class CustomComplexPlugin(ValidationPlugin):
            """Complex plugin with multiple validation rules."""

            name = "custom_complex"
            priority = 100
            expensive = True

            @classmethod
            def get_description(cls) -> str:
                return "Multi-rule custom validator"

            def validate(self, dataset):
                errors = []
                warnings = []
                metrics = {}

                # Rule 1: Check sample count
                min_samples = self.thresholds.get("min_samples", 10)
                sample_count = len(dataset)
                metrics["sample_count"] = float(sample_count)

                if sample_count < min_samples:
                    errors.append(f"Too few samples: {sample_count} < {min_samples}")

                # Rule 2: Check field presence
                required_field = self.params.get("required_field", "text")
                if required_field not in dataset.column_names:
                    errors.append(f"Missing required field: {required_field}")
                else:
                    # Rule 3: Check non-empty values
                    values = dataset[required_field]
                    empty_count = sum(1 for v in values if not v or str(v).strip() == "")
                    empty_ratio = empty_count / len(values) if values else 0
                    metrics["empty_ratio"] = empty_ratio

                    if empty_ratio > 0.1:
                        warnings.append(f"High empty ratio: {empty_ratio:.2%}")

                passed = len(errors) == 0

                return ValidationResult(
                    plugin_name=self.name,
                    passed=passed,
                    params=dict(self.params),
                    thresholds=dict(self.thresholds),
                    metrics=metrics,
                    warnings=warnings,
                    errors=errors,
                    execution_time_ms=2.0,
                )

            def get_recommendations(self, result):
                recommendations = []
                if result.metrics.get("sample_count", 0) < 10:
                    recommendations.append("Add more training samples")
                if result.metrics.get("empty_ratio", 0) > 0.1:
                    recommendations.append("Remove or fix empty values")
                return recommendations

        # Test complex plugin
        dataset = Dataset.from_dict({"text": ["a", "b", "", "d"]})
        plugin = ValidationPluginRegistry.get_plugin(
            "custom_complex",
            {"required_field": "text"},
            {"min_samples": 3},
        )
        result = plugin.validate(dataset)

        assert result.passed is True
        assert len(result.warnings) > 0  # Should warn about empty ratio

        recommendations = plugin.get_recommendations(result)
        assert len(recommendations) > 0

    def test_plugin_inheritance(self):
        """Test: Plugin with inheritance from another plugin."""
        from src.data.validation.plugins.base.min_samples import MinSamplesValidator

        @ValidationPluginRegistry.register
        class CustomExtendedPlugin(MinSamplesValidator):
            """Extended version of MinSamplesValidator."""

            name = "custom_extended"

            @classmethod
            def get_description(cls) -> str:
                return "Extended min samples validator"

            def validate(self, dataset):
                # Call parent validation
                result = super().validate(dataset)

                # Add custom metric
                result.metrics["custom_check"] = 1.0

                return result

        dataset = Dataset.from_dict({"text": ["a"] * 10})
        plugin = ValidationPluginRegistry.get_plugin("custom_extended", {}, {"threshold": 5})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert "sample_count" in result.metrics  # From parent
        assert "custom_check" in result.metrics  # From custom
